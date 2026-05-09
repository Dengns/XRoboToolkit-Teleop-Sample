from __future__ import annotations

import argparse
import csv
import json
import math
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


HAND_ORDER = ("thumb", "index", "middle", "ring", "pinky")


@dataclass(frozen=True)
class CandidateConfig:
    name: str
    hidden_layer_sizes: tuple[int, ...]
    activation: str
    solver: str
    alpha: float
    max_iter: int


@dataclass(frozen=True)
class DatasetSplit:
    train_end: int
    val_end: int
    total: int

    @property
    def train_slice(self) -> slice:
        return slice(0, self.train_end)

    @property
    def val_slice(self) -> slice:
        return slice(self.train_end, self.val_end)

    @property
    def test_slice(self) -> slice:
        return slice(self.val_end, self.total)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="按手指分别训练外骨骼相对角到人手相对角的 MLP 映射模型。"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("outputs/clean_hand_mapping/per_finger"),
        help="每指训练 CSV 所在目录，默认 outputs/clean_hand_mapping/per_finger。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/hand_mapping_models/mlp"),
        help="模型与报告输出目录，默认 outputs/hand_mapping_models/mlp。",
    )
    parser.add_argument(
        "--fingers",
        nargs="*",
        default=list(HAND_ORDER),
        help="要训练的手指列表，默认训练 thumb/index/middle/ring/pinky。",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="按时间顺序划分训练集占比，默认 0.7。",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="按时间顺序划分验证集占比，默认 0.15。",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="神经网络初始化随机种子，默认 42。",
    )
    return parser.parse_args()


def read_training_rows(path: Path) -> tuple[list[str], list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    if not rows:
        raise RuntimeError(f"训练文件为空：{path}")
    x_columns = [name for name in fieldnames if name.startswith("x")]
    y_columns = [name for name in fieldnames if name.startswith("y")]
    if not x_columns or not y_columns:
        raise RuntimeError(f"训练文件缺少 x/y 列：{path}")
    return x_columns, y_columns, rows


def rows_to_numpy(rows: list[dict[str, str]], x_columns: list[str], y_columns: list[str]) -> tuple[np.ndarray, np.ndarray]:
    x_matrix = np.asarray(
        [[float(row[column]) for column in x_columns] for row in rows],
        dtype=np.float64,
    )
    y_matrix = np.asarray(
        [[float(row[column]) for column in y_columns] for row in rows],
        dtype=np.float64,
    )
    if not np.isfinite(x_matrix).all() or not np.isfinite(y_matrix).all():
        raise RuntimeError("训练数据存在 NaN 或无穷值，无法直接训练。")
    return x_matrix, y_matrix


def build_candidate_configs(output_dim: int) -> list[CandidateConfig]:
    width = max(48, output_dim * 24)
    deep_width = max(96, output_dim * 32)
    return [
        CandidateConfig(
            name="relu_lbfgs_wide",
            hidden_layer_sizes=(width * 2, width),
            activation="relu",
            solver="lbfgs",
            alpha=1e-4,
            max_iter=6000,
        ),
        CandidateConfig(
            name="tanh_lbfgs_compact",
            hidden_layer_sizes=(width, width // 2),
            activation="tanh",
            solver="lbfgs",
            alpha=1e-4,
            max_iter=6000,
        ),
        CandidateConfig(
            name="relu_adam_deep",
            hidden_layer_sizes=(width * 2, width, width // 2),
            activation="relu",
            solver="adam",
            alpha=5e-4,
            max_iter=5000,
        ),
        CandidateConfig(
            name="relu_lbfgs_deeper",
            hidden_layer_sizes=(deep_width * 2, deep_width, deep_width // 2),
            activation="relu",
            solver="lbfgs",
            alpha=5e-4,
            max_iter=8000,
        ),
        CandidateConfig(
            name="tanh_lbfgs_deeper",
            hidden_layer_sizes=(deep_width, deep_width, deep_width // 2),
            activation="tanh",
            solver="lbfgs",
            alpha=1e-3,
            max_iter=8000,
        ),
        CandidateConfig(
            name="relu_adam_very_deep",
            hidden_layer_sizes=(deep_width * 2, deep_width * 2, deep_width, deep_width // 2),
            activation="relu",
            solver="adam",
            alpha=1e-3,
            max_iter=6000,
        ),
        CandidateConfig(
            name="tanh_adam_very_deep",
            hidden_layer_sizes=(deep_width * 2, deep_width, deep_width, deep_width // 2),
            activation="tanh",
            solver="adam",
            alpha=1e-3,
            max_iter=6000,
        ),
    ]


def build_split(total_rows: int, train_ratio: float, val_ratio: float) -> DatasetSplit:
    if total_rows < 30:
        raise RuntimeError(f"样本数过少（{total_rows}），当前脚本要求至少 30 条样本。")
    if not (0.0 < train_ratio < 1.0) or not (0.0 <= val_ratio < 1.0):
        raise RuntimeError("train_ratio / val_ratio 超出有效范围。")
    if train_ratio + val_ratio >= 1.0:
        raise RuntimeError("train_ratio + val_ratio 必须小于 1.0。")

    train_end = max(1, int(round(total_rows * train_ratio)))
    val_size = max(1, int(round(total_rows * val_ratio)))
    val_end = min(total_rows - 1, train_end + val_size)

    if val_end >= total_rows:
        val_end = total_rows - 1
    if train_end >= val_end:
        train_end = max(1, val_end - 1)
    if train_end < 1 or val_end <= train_end or total_rows - val_end < 1:
        raise RuntimeError("数据划分失败，无法同时保留训练/验证/测试集合。")

    return DatasetSplit(train_end=train_end, val_end=val_end, total=total_rows)


def make_model(config: CandidateConfig, random_state: int) -> MLPRegressor:
    return MLPRegressor(
        hidden_layer_sizes=config.hidden_layer_sizes,
        activation=config.activation,
        solver=config.solver,
        alpha=config.alpha,
        max_iter=config.max_iter,
        random_state=random_state,
    )


def fit_with_scalers(
    x_train: np.ndarray,
    y_train: np.ndarray,
    config: CandidateConfig,
    random_state: int,
) -> tuple[StandardScaler, StandardScaler, MLPRegressor]:
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    x_train_scaled = x_scaler.fit_transform(x_train)
    y_train_scaled = y_scaler.fit_transform(y_train)
    model = make_model(config, random_state=random_state)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        model.fit(x_train_scaled, y_train_scaled)
    return x_scaler, y_scaler, model


def predict_with_scalers(
    model: MLPRegressor,
    x_scaler: StandardScaler,
    y_scaler: StandardScaler,
    x_values: np.ndarray,
) -> np.ndarray:
    prediction_scaled = model.predict(x_scaler.transform(x_values))
    prediction_scaled = np.asarray(prediction_scaled, dtype=np.float64)
    if prediction_scaled.ndim == 1:
        prediction_scaled = prediction_scaled.reshape(-1, 1)
    return y_scaler.inverse_transform(prediction_scaled)


def summarize_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    residual = y_pred - y_true
    per_output_rmse = np.sqrt(np.mean(np.square(residual), axis=0))
    per_output_mae = np.mean(np.abs(residual), axis=0)
    per_output_r2 = r2_score(y_true, y_pred, multioutput="raw_values")
    return {
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred, multioutput="uniform_average")),
        "per_output_rmse": [float(value) for value in per_output_rmse],
        "per_output_mae": [float(value) for value in per_output_mae],
        "per_output_r2": [float(value) for value in np.asarray(per_output_r2, dtype=np.float64)],
    }


def select_best_candidate(
    finger: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    random_state: int,
    output_dim: int,
) -> dict[str, Any]:
    candidate_reports: list[dict[str, Any]] = []
    for candidate in build_candidate_configs(output_dim):
        print(f"[{finger}] 开始候选 {candidate.name}，结构={candidate.hidden_layer_sizes}，solver={candidate.solver}")
        started_at = time.perf_counter()
        x_scaler, y_scaler, model = fit_with_scalers(
            x_train,
            y_train,
            candidate,
            random_state=random_state,
        )
        val_pred = predict_with_scalers(model, x_scaler, y_scaler, x_val)
        metrics = summarize_metrics(y_val, val_pred)
        elapsed = time.perf_counter() - started_at
        print(
            f"[{finger}] 完成候选 {candidate.name}："
            f"val_RMSE={metrics['rmse']:.6f} rad, "
            f"val_R2={metrics['r2']:.6f}, "
            f"耗时={elapsed:.2f}s"
        )
        candidate_reports.append(
            {
                "config": asdict(candidate),
                "val_metrics": metrics,
                "elapsed_seconds": float(elapsed),
            }
        )

    best_report = min(candidate_reports, key=lambda item: item["val_metrics"]["rmse"])
    return {"best": best_report, "all": candidate_reports}


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def train_one_finger(
    finger: str,
    input_path: Path,
    output_dir: Path,
    train_ratio: float,
    val_ratio: float,
    random_state: int,
) -> dict[str, Any]:
    print(f"[{finger}] 读取训练数据：{input_path}")
    x_columns, y_columns, rows = read_training_rows(input_path)
    x_all, y_all = rows_to_numpy(rows, x_columns, y_columns)
    split = build_split(len(rows), train_ratio=train_ratio, val_ratio=val_ratio)
    print(
        f"[{finger}] 样本数={len(rows)}，"
        f"train={split.train_end}，"
        f"val={split.val_end - split.train_end}，"
        f"test={split.total - split.val_end}"
    )

    x_train = x_all[split.train_slice]
    y_train = y_all[split.train_slice]
    x_val = x_all[split.val_slice]
    y_val = y_all[split.val_slice]
    x_test = x_all[split.test_slice]
    y_test = y_all[split.test_slice]

    candidate_result = select_best_candidate(
        finger,
        x_train,
        y_train,
        x_val,
        y_val,
        random_state=random_state,
        output_dim=len(y_columns),
    )
    best_config = CandidateConfig(**candidate_result["best"]["config"])

    x_fit = x_all[: split.val_end]
    y_fit = y_all[: split.val_end]
    x_scaler, y_scaler, model = fit_with_scalers(
        x_fit,
        y_fit,
        best_config,
        random_state=random_state,
    )
    test_pred = predict_with_scalers(model, x_scaler, y_scaler, x_test)
    test_metrics = summarize_metrics(y_test, test_pred)

    finger_output_dir = output_dir / finger
    finger_output_dir.mkdir(parents=True, exist_ok=True)

    artifact = {
        "finger": finger,
        "feature_columns": x_columns,
        "target_columns": y_columns,
        "split": {
            "train_count": split.train_end,
            "val_count": split.val_end - split.train_end,
            "test_count": split.total - split.val_end,
            "total_count": split.total,
        },
        "best_candidate": candidate_result["best"],
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "model": model,
    }
    joblib.dump(artifact, finger_output_dir / f"{finger}_mlp.joblib")

    prediction_rows: list[dict[str, Any]] = []
    for index, (source_row, y_true_row, y_pred_row) in enumerate(
        zip(rows[split.test_slice], y_test, test_pred, strict=True),
        start=1,
    ):
        output_row: dict[str, Any] = {
            "test_row_index": index,
            "clean_sample_id": source_row.get("clean_sample_id", ""),
            "source_sample_id": source_row.get("source_sample_id", ""),
            "recorded_at_local": source_row.get("recorded_at_local", ""),
        }
        for column_index, column_name in enumerate(x_columns):
            output_row[column_name] = source_row[column_name]
        for column_index, column_name in enumerate(y_columns):
            output_row[f"{column_name}_true"] = float(y_true_row[column_index])
            output_row[f"{column_name}_pred"] = float(y_pred_row[column_index])
            output_row[f"{column_name}_abs_err"] = float(abs(y_pred_row[column_index] - y_true_row[column_index]))
        prediction_rows.append(output_row)

    prediction_fields = list(prediction_rows[0].keys()) if prediction_rows else []
    if prediction_fields:
        write_csv(
            finger_output_dir / f"{finger}_test_predictions.csv",
            prediction_fields,
            prediction_rows,
        )

    report = {
        "finger": finger,
        "input_path": str(input_path),
        "model_path": str(finger_output_dir / f"{finger}_mlp.joblib"),
        "prediction_path": str(finger_output_dir / f"{finger}_test_predictions.csv"),
        "feature_columns": x_columns,
        "target_columns": y_columns,
        "sample_count": len(rows),
        "split": artifact["split"],
        "candidate_search": candidate_result["all"],
        "best_candidate": candidate_result["best"],
        "test_metrics": test_metrics,
    }
    (finger_output_dir / f"{finger}_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def main() -> None:
    args = parse_args()
    selected_fingers = [finger for finger in args.fingers if finger in HAND_ORDER]
    if not selected_fingers:
        raise RuntimeError("没有可训练的手指，请从 thumb/index/middle/ring/pinky 中选择。")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    aggregate_report: dict[str, Any] = {
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "random_state": args.random_state,
        "fingers": {},
    }

    for finger in selected_fingers:
        input_path = args.input_dir / f"{finger}_training.csv"
        if not input_path.exists():
            print(f"跳过 {finger}：未找到训练文件 {input_path}")
            continue
        report = train_one_finger(
            finger,
            input_path=input_path,
            output_dir=args.output_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            random_state=args.random_state,
        )
        aggregate_report["fingers"][finger] = report
        metrics = report["test_metrics"]
        print(
            f"{finger} 训练完成："
            f"RMSE={metrics['rmse']:.6f} rad, "
            f"MAE={metrics['mae']:.6f} rad, "
            f"R2={metrics['r2']:.6f}"
        )

    summary_path = args.output_dir / "training_summary.json"
    summary_path.write_text(
        json.dumps(aggregate_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"汇总报告已写入：{summary_path}")


if __name__ == "__main__":
    main()
