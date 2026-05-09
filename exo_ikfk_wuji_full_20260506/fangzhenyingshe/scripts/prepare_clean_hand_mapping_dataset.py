from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any


HAND_ORDER = ("thumb", "index", "middle", "ring", "pinky")
FOUR_FINGER_ORDER = ("index", "middle", "ring", "pinky")
FOUR_FINGER_SIDE_SWING_EXO_Q = 1
FOUR_FINGER_SIDE_SWING_HUMAN_Q = 4
FINGER_DOF = {
    "thumb": {"exo": 5, "human": 4},
    "index": {"exo": 4, "human": 4},
    "middle": {"exo": 4, "human": 4},
    "ring": {"exo": 4, "human": 4},
    "pinky": {"exo": 4, "human": 4},
}
Q_INDICES = range(1, 6)


def uses_four_finger_side_passthrough(finger: str, kind: str, q_index: int) -> bool:
    return (
        finger in FOUR_FINGER_ORDER
        and kind == "human"
        and q_index == FOUR_FINGER_SIDE_SWING_HUMAN_Q
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="把实时五指映射记录转换为基于标定零点的干净相对角训练数据。"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/live_hand_mapping_record.csv"),
        help="实时采集 CSV，默认 outputs/live_hand_mapping_record.csv。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/clean_hand_mapping"),
        help="清洗结果输出目录，默认 outputs/clean_hand_mapping。",
    )
    parser.add_argument(
        "--tip-error-threshold-mm",
        type=float,
        default=3.0,
        help="保留样本的最大指尖误差，默认 3.0 mm。",
    )
    parser.add_argument(
        "--calibration-selection",
        choices=("latest", "all"),
        default="latest",
        help="未指定 calibration-id 时使用最新一次标定或全部标定，默认 latest。",
    )
    parser.add_argument(
        "--calibration-id",
        default="",
        help="只使用指定 calibration_id 的标定行；留空时按 calibration-selection 选择。",
    )
    parser.add_argument(
        "--exo-angle-space",
        default="absolute",
        help="要求输入外骨骼角度空间，默认 absolute。",
    )
    parser.add_argument(
        "--exo-delta-mode",
        choices=("wrapped", "direct"),
        default="wrapped",
        help="外骨骼相对角计算方式；wrapped 使用最短角差，direct 直接相减。",
    )
    parser.add_argument(
        "--positive-eps-rad",
        type=float,
        default=1e-6,
        help="统计正负方向时忽略的近零阈值，默认 1e-6 rad。",
    )
    return parser.parse_args()


def parse_float(value: Any) -> float:
    if value is None:
        return math.nan
    text = str(value).strip()
    if not text:
        return math.nan
    try:
        number = float(text)
    except ValueError:
        return math.nan
    return number if math.isfinite(number) else math.nan


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "是"}


def format_float(value: float) -> str:
    if not math.isfinite(value):
        return ""
    return f"{value:.12g}"


def json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def finite_values(rows: list[dict[str, str]], column: str) -> list[float]:
    return [value for value in (parse_float(row.get(column)) for row in rows) if math.isfinite(value)]


def arithmetic_mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return math.nan
    return sum(finite) / len(finite)


def circular_mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return math.nan
    sin_sum = sum(math.sin(value) for value in finite)
    cos_sum = sum(math.cos(value) for value in finite)
    if abs(sin_sum) < 1e-12 and abs(cos_sum) < 1e-12:
        return arithmetic_mean(finite)
    return math.atan2(sin_sum, cos_sum)


def wrapped_delta(value: float, baseline: float) -> float:
    delta = value - baseline
    return math.atan2(math.sin(delta), math.cos(delta))


def percentile(values: list[float], ratio: float) -> float:
    finite = sorted(value for value in values if math.isfinite(value))
    if not finite:
        return math.nan
    index = min(len(finite) - 1, max(0, math.ceil(len(finite) * ratio) - 1))
    return finite[index]


def summarize(values: list[float]) -> dict[str, float | int]:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return {"count": 0, "mean": math.nan, "median": math.nan, "p95": math.nan, "min": math.nan, "max": math.nan}
    return {
        "count": len(finite),
        "mean": sum(finite) / len(finite),
        "median": float(median(finite)),
        "p95": percentile(finite, 0.95),
        "min": min(finite),
        "max": max(finite),
    }


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    if not fieldnames:
        raise RuntimeError(f"输入 CSV 没有表头：{path}")
    return fieldnames, rows


def select_calibration_rows(
    rows: list[dict[str, str]],
    *,
    calibration_id: str,
    calibration_selection: str,
    exo_angle_space: str,
) -> tuple[str, list[dict[str, str]]]:
    calibration_rows = [
        row
        for row in rows
        if row.get("record_kind") == "calibration"
        and row.get("angle_unit") == "rad"
        and row.get("exo_angle_space") == exo_angle_space
    ]
    if not calibration_rows:
        raise RuntimeError("没有找到可用标定行：需要 record_kind=calibration、angle_unit=rad。")

    if calibration_id:
        selected = [row for row in calibration_rows if str(row.get("calibration_id", "")) == str(calibration_id)]
        if not selected:
            raise RuntimeError(f"没有找到 calibration_id={calibration_id} 的标定行。")
        return str(calibration_id), selected

    if calibration_selection == "all":
        return "all", calibration_rows

    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in calibration_rows:
        groups[str(row.get("calibration_id", ""))].append(row)
    latest_id = max(
        groups,
        key=lambda key: max(parse_float(row.get("recorded_at_unix")) for row in groups[key]),
    )
    return latest_id, groups[latest_id]


def build_baseline(
    calibration_rows: list[dict[str, str]],
    *,
    exo_delta_mode: str,
) -> tuple[dict[str, dict[str, list[float]]], dict[str, dict[str, list[dict[str, Any]]]]]:
    baseline: dict[str, dict[str, list[float]]] = {
        finger: {"exo": [math.nan] * 5, "human": [math.nan] * 5} for finger in HAND_ORDER
    }
    baseline_meta: dict[str, dict[str, list[dict[str, Any]]]] = {
        finger: {"exo": [], "human": []} for finger in HAND_ORDER
    }

    for finger in HAND_ORDER:
        finger_rows = [row for row in calibration_rows if row.get("finger") == finger]
        for kind in ("exo", "human"):
            for q_index in Q_INDICES:
                if uses_four_finger_side_passthrough(finger, kind, q_index):
                    source_q_index = FOUR_FINGER_SIDE_SWING_EXO_Q
                    source_values = finite_values(finger_rows, f"exo_q{source_q_index}")
                    source_baseline = baseline[finger]["exo"][source_q_index - 1]
                    baseline[finger][kind][q_index - 1] = 0.0
                    baseline_meta[finger][kind].append(
                        {
                            "q_index": q_index,
                            "calibration_value_count": len(source_values),
                            "raw_mean_rad": 0.0,
                            "circular_mean_rad": math.nan,
                            "baseline_rad": 0.0,
                            "baseline_source": f"passthrough exo_q{source_q_index} relative to exo baseline",
                            "source_q_index": source_q_index,
                            "source_baseline_rad": source_baseline,
                        }
                    )
                    continue

                column = f"{kind}_q{q_index}"
                values = finite_values(finger_rows, column)
                raw_mean = arithmetic_mean(values)
                circ_mean = circular_mean(values) if kind == "exo" else math.nan
                if kind == "exo" and exo_delta_mode == "wrapped":
                    used = circ_mean
                else:
                    used = raw_mean
                baseline[finger][kind][q_index - 1] = used
                baseline_meta[finger][kind].append(
                    {
                        "q_index": q_index,
                        "calibration_value_count": len(values),
                        "raw_mean_rad": raw_mean,
                        "circular_mean_rad": circ_mean,
                        "baseline_rad": used,
                        "baseline_source": "recorded",
                        "source_q_index": q_index,
                        "source_baseline_rad": used,
                    }
                )

    missing: list[str] = []
    for finger, dof_map in FINGER_DOF.items():
        for kind, dof in dof_map.items():
            for q_index in range(1, dof + 1):
                if not math.isfinite(baseline[finger][kind][q_index - 1]):
                    missing.append(f"{finger}:{kind}_q{q_index}")
    if missing:
        raise RuntimeError("标定基准缺失必要自由度：" + ", ".join(missing))

    return baseline, baseline_meta


def reject_reason(
    row: dict[str, str],
    *,
    exo_angle_space: str,
    tip_error_threshold_mm: float,
) -> str:
    if row.get("record_kind") != "sample":
        return "not_sample"
    if row.get("angle_unit") != "rad":
        return "angle_unit_not_rad"
    if row.get("exo_angle_space") != exo_angle_space:
        return "exo_angle_space_mismatch"
    if not parse_bool(row.get("ik_enabled")):
        return "ik_disabled"
    if not parse_bool(row.get("is_reachable")):
        return "unreachable"
    tip_error = parse_float(row.get("tip_error_mm"))
    if not math.isfinite(tip_error):
        return "tip_error_missing"
    if tip_error > tip_error_threshold_mm:
        return "tip_error_too_large"
    finger = row.get("finger", "")
    if finger not in FINGER_DOF:
        return "unknown_finger"
    return ""


def build_candidates(
    rows: list[dict[str, str]],
    baseline: dict[str, dict[str, list[float]]],
    *,
    exo_angle_space: str,
    exo_delta_mode: str,
    tip_error_threshold_mm: float,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    candidates: list[dict[str, Any]] = []
    rejects: Counter[str] = Counter()

    for row in rows:
        reason = reject_reason(
            row,
            exo_angle_space=exo_angle_space,
            tip_error_threshold_mm=tip_error_threshold_mm,
        )
        if reason:
            rejects[reason] += 1
            continue

        finger = row["finger"]
        exo_delta = [math.nan] * 5
        human_delta = [math.nan] * 5
        required_missing = False

        for q_index in Q_INDICES:
            exo_value = parse_float(row.get(f"exo_q{q_index}"))
            exo_base = baseline[finger]["exo"][q_index - 1]
            if math.isfinite(exo_value) and math.isfinite(exo_base):
                exo_delta[q_index - 1] = (
                    wrapped_delta(exo_value, exo_base)
                    if exo_delta_mode == "wrapped"
                    else exo_value - exo_base
                )

            human_value = parse_float(row.get(f"human_q{q_index}"))
            human_base = baseline[finger]["human"][q_index - 1]
            if uses_four_finger_side_passthrough(finger, "human", q_index):
                passthrough_delta = exo_delta[FOUR_FINGER_SIDE_SWING_EXO_Q - 1]
                if math.isfinite(passthrough_delta):
                    human_delta[q_index - 1] = passthrough_delta
            elif math.isfinite(human_value) and math.isfinite(human_base):
                human_delta[q_index - 1] = human_value - human_base

        for q_index in range(1, FINGER_DOF[finger]["exo"] + 1):
            if not math.isfinite(exo_delta[q_index - 1]):
                required_missing = True
        for q_index in range(1, FINGER_DOF[finger]["human"] + 1):
            if not math.isfinite(human_delta[q_index - 1]):
                required_missing = True
        if required_missing:
            rejects["required_angle_missing"] += 1
            continue

        candidates.append({"row": row, "finger": finger, "exo_delta": exo_delta, "human_delta": human_delta})

    return candidates, rejects


def choose_directions(
    candidates: list[dict[str, Any]],
    *,
    positive_eps_rad: float,
) -> tuple[dict[str, dict[str, list[int]]], dict[str, dict[str, list[dict[str, Any]]]]]:
    signs: dict[str, dict[str, list[int]]] = {
        finger: {"exo": [1] * 5, "human": [1] * 5} for finger in HAND_ORDER
    }
    sign_meta: dict[str, dict[str, list[dict[str, Any]]]] = {
        finger: {"exo": [], "human": []} for finger in HAND_ORDER
    }

    for finger in HAND_ORDER:
        finger_candidates = [candidate for candidate in candidates if candidate["finger"] == finger]
        for kind in ("exo", "human"):
            deltas_key = f"{kind}_delta"
            dof = FINGER_DOF[finger][kind]
            for q_index in Q_INDICES:
                values = [
                    candidate[deltas_key][q_index - 1]
                    for candidate in finger_candidates
                    if math.isfinite(candidate[deltas_key][q_index - 1])
                ]
                positive_count = sum(1 for value in values if value > positive_eps_rad)
                negative_count = sum(1 for value in values if value < -positive_eps_rad)
                zero_count = len(values) - positive_count - negative_count
                if uses_four_finger_side_passthrough(finger, kind, q_index):
                    sign = signs[finger]["exo"][FOUR_FINGER_SIDE_SWING_EXO_Q - 1]
                else:
                    sign = -1 if negative_count > positive_count else 1
                after_positive = positive_count if sign == 1 else negative_count
                nonzero = positive_count + negative_count
                positive_ratio_after = after_positive / nonzero if nonzero else math.nan
                if q_index <= dof:
                    signs[finger][kind][q_index - 1] = sign
                sign_meta[finger][kind].append(
                    {
                        "q_index": q_index,
                        "required": q_index <= dof,
                        "direction": sign if q_index <= dof else "",
                        "positive_count_before": positive_count,
                        "negative_count_before": negative_count,
                        "zero_count": zero_count,
                        "positive_ratio_after": positive_ratio_after,
                    }
                )

    return signs, sign_meta


def build_clean_row(
    clean_sample_id: int,
    candidate: dict[str, Any],
    baseline: dict[str, dict[str, list[float]]],
    signs: dict[str, dict[str, list[int]]],
    *,
    calibration_id_used: str,
    tip_error_threshold_mm: float,
    exo_delta_mode: str,
) -> dict[str, Any]:
    source = candidate["row"]
    finger = candidate["finger"]
    output: dict[str, Any] = {
        "clean_sample_id": clean_sample_id,
        "source_sample_id": source.get("sample_id", ""),
        "finger": finger,
        "recorded_at_local": source.get("recorded_at_local", ""),
        "recorded_at_unix": source.get("recorded_at_unix", ""),
        "calibration_id_used": calibration_id_used,
        "tip_error_threshold_mm": tip_error_threshold_mm,
        "exo_delta_mode": exo_delta_mode,
        "tip_error_mm": source.get("tip_error_mm", ""),
        "stream_hz": source.get("stream_hz", ""),
        "source_age_ms": source.get("source_age_ms", ""),
        "trigger": source.get("trigger", ""),
        "qpos_mode": source.get("qpos_mode", ""),
        "angle_unit": source.get("angle_unit", ""),
        "exo_angle_space": source.get("exo_angle_space", ""),
        "L1_mm": source.get("L1_mm", ""),
        "L2_mm": source.get("L2_mm", ""),
        "L3_mm": source.get("L3_mm", ""),
        "base_dx_mm": source.get("base_dx_mm", ""),
        "base_dy_mm": source.get("base_dy_mm", ""),
        "base_dz_mm": source.get("base_dz_mm", ""),
        "tip_dx_mm": source.get("tip_dx_mm", ""),
        "tip_dy_mm": source.get("tip_dy_mm", ""),
    }

    for q_index in Q_INDICES:
        exo_delta = candidate["exo_delta"][q_index - 1]
        human_delta = candidate["human_delta"][q_index - 1]
        exo_sign = signs[finger]["exo"][q_index - 1]
        human_sign = signs[finger]["human"][q_index - 1]
        output[f"source_exo_q{q_index}"] = source.get(f"exo_q{q_index}", "")
        if uses_four_finger_side_passthrough(finger, "human", q_index):
            output[f"source_human_q{q_index}"] = ""
        else:
            output[f"source_human_q{q_index}"] = source.get(f"human_q{q_index}", "")
        output[f"exo_baseline_q{q_index}"] = format_float(baseline[finger]["exo"][q_index - 1])
        output[f"human_baseline_q{q_index}"] = format_float(baseline[finger]["human"][q_index - 1])
        output[f"exo_direction_q{q_index}"] = exo_sign if q_index <= FINGER_DOF[finger]["exo"] else ""
        output[f"human_direction_q{q_index}"] = human_sign if q_index <= FINGER_DOF[finger]["human"] else ""
        output[f"human_source_q{q_index}"] = (
            f"passthrough_exo_q{FOUR_FINGER_SIDE_SWING_EXO_Q}_baseline_offset"
            if uses_four_finger_side_passthrough(finger, "human", q_index)
            else "recorded"
        )
        output[f"exo_delta_q{q_index}"] = format_float(exo_delta)
        output[f"human_delta_q{q_index}"] = format_float(human_delta)
        output[f"exo_rel_q{q_index}"] = format_float(exo_sign * exo_delta)
        output[f"human_rel_q{q_index}"] = format_float(human_sign * human_delta)

    return output


def clean_fieldnames() -> list[str]:
    fields = [
        "clean_sample_id",
        "source_sample_id",
        "finger",
        "recorded_at_local",
        "recorded_at_unix",
        "calibration_id_used",
        "tip_error_threshold_mm",
        "exo_delta_mode",
        "tip_error_mm",
        "stream_hz",
        "source_age_ms",
        "trigger",
        "qpos_mode",
        "angle_unit",
        "exo_angle_space",
        "L1_mm",
        "L2_mm",
        "L3_mm",
        "base_dx_mm",
        "base_dy_mm",
        "base_dz_mm",
        "tip_dx_mm",
        "tip_dy_mm",
    ]
    for prefix in (
        "source_exo_q",
        "source_human_q",
        "exo_baseline_q",
        "human_baseline_q",
        "exo_direction_q",
        "human_direction_q",
        "human_source_q",
        "exo_delta_q",
        "human_delta_q",
        "exo_rel_q",
        "human_rel_q",
    ):
        fields.extend(f"{prefix}{q_index}" for q_index in Q_INDICES)
    return fields


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_per_finger_training_files(output_dir: Path, rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    per_finger_dir = output_dir / "per_finger"
    for finger in HAND_ORDER:
        finger_rows = [row for row in rows if row["finger"] == finger]
        exo_dof = FINGER_DOF[finger]["exo"]
        human_dof = FINGER_DOF[finger]["human"]
        fields = [
            "clean_sample_id",
            "source_sample_id",
            "recorded_at_local",
            "tip_error_mm",
            "L1_mm",
            "L2_mm",
            "L3_mm",
            "base_dx_mm",
            "base_dy_mm",
            "base_dz_mm",
            "tip_dx_mm",
            "tip_dy_mm",
        ]
        fields.extend(f"x{i}" for i in range(1, exo_dof + 1))
        fields.extend(f"y{i}" for i in range(1, human_dof + 1))
        training_rows: list[dict[str, Any]] = []
        for row in finger_rows:
            training_row = {field: row.get(field, "") for field in fields}
            for q_index in range(1, exo_dof + 1):
                training_row[f"x{q_index}"] = row[f"exo_rel_q{q_index}"]
            for q_index in range(1, human_dof + 1):
                training_row[f"y{q_index}"] = row[f"human_rel_q{q_index}"]
            training_rows.append(training_row)
        write_csv(per_finger_dir / f"{finger}_training.csv", fields, training_rows)
        counts[finger] = len(training_rows)
    return counts


def build_baseline_rows(
    calibration_id_used: str,
    baseline_meta: dict[str, dict[str, list[dict[str, Any]]]],
    sign_meta: dict[str, dict[str, list[dict[str, Any]]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for finger in HAND_ORDER:
        for kind in ("exo", "human"):
            sign_by_q = {item["q_index"]: item for item in sign_meta[finger][kind]}
            for meta in baseline_meta[finger][kind]:
                q_index = meta["q_index"]
                sign_info = sign_by_q[q_index]
                rows.append(
                    {
                        "calibration_id_used": calibration_id_used,
                        "finger": finger,
                        "angle_kind": kind,
                        "q_index": q_index,
                        "required": sign_info["required"],
                        "calibration_value_count": meta["calibration_value_count"],
                        "raw_mean_rad": format_float(meta["raw_mean_rad"]),
                        "circular_mean_rad": format_float(meta["circular_mean_rad"]),
                        "baseline_rad": format_float(meta["baseline_rad"]),
                        "baseline_source": meta.get("baseline_source", "recorded"),
                        "source_q_index": meta.get("source_q_index", ""),
                        "source_baseline_rad": format_float(parse_float(meta.get("source_baseline_rad"))),
                        "direction": sign_info["direction"],
                        "positive_count_before": sign_info["positive_count_before"],
                        "negative_count_before": sign_info["negative_count_before"],
                        "zero_count": sign_info["zero_count"],
                        "positive_ratio_after": format_float(sign_info["positive_ratio_after"]),
                        "relative_at_baseline_rad": "0",
                    }
                )
    return rows


def build_report(
    *,
    input_path: Path,
    output_dir: Path,
    source_row_count: int,
    calibration_id_used: str,
    calibration_rows: list[dict[str, str]],
    candidates: list[dict[str, Any]],
    clean_rows: list[dict[str, Any]],
    rejects: Counter[str],
    per_finger_counts: dict[str, int],
    baseline_meta: dict[str, dict[str, list[dict[str, Any]]]],
    sign_meta: dict[str, dict[str, list[dict[str, Any]]]],
    tip_error_threshold_mm: float,
    exo_delta_mode: str,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "source_row_count": source_row_count,
        "calibration": {
            "calibration_id_used": calibration_id_used,
            "rows_used": len(calibration_rows),
            "rows_by_finger": Counter(row.get("finger", "") for row in calibration_rows),
        },
        "filter": {
            "tip_error_threshold_mm": tip_error_threshold_mm,
            "exo_delta_mode": exo_delta_mode,
            "reject_reasons": dict(rejects),
        },
        "clean_row_count": len(clean_rows),
        "per_finger_counts": per_finger_counts,
        "per_finger_quality": {},
        "baseline": baseline_meta,
        "directions": sign_meta,
    }

    for finger in HAND_ORDER:
        finger_candidates = [candidate for candidate in candidates if candidate["finger"] == finger]
        finger_rows = [row for row in clean_rows if row["finger"] == finger]
        report["per_finger_quality"][finger] = {
            "clean_count": len(finger_rows),
            "tip_error_mm": summarize([parse_float(row.get("tip_error_mm")) for row in finger_rows]),
            "exo_rel_abs_rad": summarize(
                [
                    abs(parse_float(row.get(f"exo_rel_q{q_index}")))
                    for row in finger_rows
                    for q_index in range(1, FINGER_DOF[finger]["exo"] + 1)
                ]
            ),
            "human_rel_abs_rad": summarize(
                [
                    abs(parse_float(row.get(f"human_rel_q{q_index}")))
                    for row in finger_rows
                    for q_index in range(1, FINGER_DOF[finger]["human"] + 1)
                ]
            ),
            "candidate_count_after_basic_filter": len(finger_candidates),
            "human_q4_passthrough": finger in FOUR_FINGER_ORDER,
        }
    return report


def main() -> None:
    args = parse_args()
    _, rows = read_rows(args.input)
    calibration_id_used, calibration_rows = select_calibration_rows(
        rows,
        calibration_id=args.calibration_id,
        calibration_selection=args.calibration_selection,
        exo_angle_space=args.exo_angle_space,
    )
    baseline, baseline_meta = build_baseline(calibration_rows, exo_delta_mode=args.exo_delta_mode)
    candidates, rejects = build_candidates(
        rows,
        baseline,
        exo_angle_space=args.exo_angle_space,
        exo_delta_mode=args.exo_delta_mode,
        tip_error_threshold_mm=args.tip_error_threshold_mm,
    )
    signs, sign_meta = choose_directions(candidates, positive_eps_rad=args.positive_eps_rad)

    clean_rows = [
        build_clean_row(
            index,
            candidate,
            baseline,
            signs,
            calibration_id_used=calibration_id_used,
            tip_error_threshold_mm=args.tip_error_threshold_mm,
            exo_delta_mode=args.exo_delta_mode,
        )
        for index, candidate in enumerate(candidates, start=1)
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    clean_path = args.output_dir / "clean_hand_mapping_dataset.csv"
    baseline_path = args.output_dir / "baseline_summary.csv"
    report_path = args.output_dir / "preprocess_report.json"

    write_csv(clean_path, clean_fieldnames(), clean_rows)
    baseline_rows = build_baseline_rows(calibration_id_used, baseline_meta, sign_meta)
    write_csv(
        baseline_path,
        [
            "calibration_id_used",
            "finger",
            "angle_kind",
            "q_index",
            "required",
            "calibration_value_count",
            "raw_mean_rad",
            "circular_mean_rad",
            "baseline_rad",
            "baseline_source",
            "source_q_index",
            "source_baseline_rad",
            "direction",
            "positive_count_before",
            "negative_count_before",
            "zero_count",
            "positive_ratio_after",
            "relative_at_baseline_rad",
        ],
        baseline_rows,
    )
    per_finger_counts = write_per_finger_training_files(args.output_dir, clean_rows)

    report = build_report(
        input_path=args.input,
        output_dir=args.output_dir,
        source_row_count=len(rows),
        calibration_id_used=calibration_id_used,
        calibration_rows=calibration_rows,
        candidates=candidates,
        clean_rows=clean_rows,
        rejects=rejects,
        per_finger_counts=per_finger_counts,
        baseline_meta=baseline_meta,
        sign_meta=sign_meta,
        tip_error_threshold_mm=args.tip_error_threshold_mm,
        exo_delta_mode=args.exo_delta_mode,
    )
    report_path.write_text(json.dumps(json_safe(report), ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"输入行数：{len(rows)}")
    print(f"使用标定：{calibration_id_used}，标定行数：{len(calibration_rows)}")
    print(f"干净样本：{len(clean_rows)}")
    for finger in HAND_ORDER:
        print(f"  {finger}: {per_finger_counts[finger]}")
    print(f"拒绝原因：{dict(rejects)}")
    print(f"输出：{clean_path}")
    print(f"基准：{baseline_path}")
    print(f"报告：{report_path}")


if __name__ == "__main__":
    main()
