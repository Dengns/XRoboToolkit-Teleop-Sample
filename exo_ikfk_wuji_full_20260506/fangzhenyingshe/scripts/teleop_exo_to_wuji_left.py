#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""外骨骼 IK/FK 反解结果直写 WuJi 左手。

使用方式：
1. 先把手张开到 WuJi 逻辑零位，执行 ``--calibrate`` 保存当前 IK/FK 人手角基点。
2. 正常运行时读取该基点，用 ``baseline - current`` 生成 WuJi 5x4 弧度目标。
3. 四指侧摆暂时恒为 0；拇指侧摆继续使用拇指 4DOF IK 结果。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from compare_hand_3d import (  # noqa: E402
    FINGER_CONFIGS,
    FINGER_ORDER,
    HAND_ORDER,
    THUMB_KEY,
    build_mapping_state,
    build_thumb_mapping_state,
)
from compare_hand_3d_live import (  # noqa: E402
    DEFAULT_EXTERNAL_REPO,
    RealtimeSkeletonStream,
    apply_live_qpos,
    build_default_finger_states,
    build_joint_handles,
    build_live_limit_overrides,
    load_external_bridge_runtime,
)
from delivery_core.sim_loader import DEFAULT_URDF, load_model_from_urdf  # noqa: E402
from delivery_core.thumb_mapping import make_thumb_reference  # noqa: E402


DEFAULT_CALIBRATION_PATH = REPO_ROOT / "outputs" / "exo_wuji_open_baseline.json"
DEFAULT_RATE_HZ = 100.0
DEFAULT_CALIBRATION_FRAME_COUNT = 30
DEFAULT_CUTOFF_HZ = 8.0
DEFAULT_GAIN = 1.0
FRESH_FRAME_TIMEOUT_S = 5.0
WUJI_SHAPE = (5, 4)
WUJI_SIDE_POLICY = "four_finger_yaw_zero"
DEFAULT_SEND_MODE = "unchecked"
DEFAULT_COMMAND_LOG_INTERVAL_S = 1.0
DEFAULT_THUMB_CM_ORDER = "direct"
WUJI_RELATIVE_DIRECTION = np.ones(WUJI_SHAPE, dtype=np.float64)
#WUJI_RELATIVE_DIRECTION[0, 0] = -1.0
#WUJI_RELATIVE_DIRECTION[0, 1] = -1.0

WUJI_LAYOUT = [
    "thumb: [CM pitch, CM yaw, MP pitch, IP pitch]",
    "index: [MP pitch, MP yaw, PIP pitch, DIP pitch]",
    "middle: [MP pitch, MP yaw, PIP pitch, DIP pitch]",
    "ring: [MP pitch, MP yaw, PIP pitch, DIP pitch]",
    "pinky: [MP pitch, MP yaw, PIP pitch, DIP pitch]",
]
WUJI_FINGER_LABELS = ("拇指", "食指", "中指", "无名指", "小指")
WUJI_JOINT_LABELS = (
    ("CM掌骨翻折/弯曲", "CM掌骨侧摆", "MP掌指弯曲", "IP指间弯曲"),
    ("MP掌指弯曲", "MP侧摆(暂置0)", "PIP近端弯曲", "DIP末端弯曲"),
    ("MP掌指弯曲", "MP侧摆(暂置0)", "PIP近端弯曲", "DIP末端弯曲"),
    ("MP掌指弯曲", "MP侧摆(暂置0)", "PIP近端弯曲", "DIP末端弯曲"),
    ("MP掌指弯曲", "MP侧摆(暂置0)", "PIP近端弯曲", "DIP末端弯曲"),
)
THUMB_CM_ORDER_DIRECT = "direct"
THUMB_CM_ORDER_SWAPPED = "swapped"


@dataclass
class MappingRuntime:
    external_runtime: Any
    model: mujoco.MjModel
    data: mujoco.MjData
    joint_handles: dict[str, Any]
    thumb_reference: Any
    finger_states: dict[str, dict[str, object]]
    bundle_urdf: Path


def now_local_text() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def coerce_wuji_matrix(value: Any, *, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != WUJI_SHAPE:
        raise ValueError(f"{name} 必须是 {WUJI_SHAPE} 矩阵，当前形状为 {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} 包含非有限值")
    return matrix


def mapping_states_to_human_matrix(
    mapping_states: dict[str, dict[str, object]],
    *,
    thumb_cm_order: str = DEFAULT_THUMB_CM_ORDER,
) -> np.ndarray:
    """把五指 IK/FK 人手角按 WuJi 5x4 行列布局组装成矩阵。"""
    matrix = np.zeros(WUJI_SHAPE, dtype=np.float64)

    thumb_angles = np.asarray(mapping_states[THUMB_KEY]["human_angles"], dtype=np.float64).reshape(-1)
    if thumb_angles.size != 4:
        raise ValueError(f"拇指人手角应为 4 维，实际 {thumb_angles.size} 维")
    if thumb_cm_order == THUMB_CM_ORDER_DIRECT:
        matrix[0, :] = thumb_angles
    elif thumb_cm_order == THUMB_CM_ORDER_SWAPPED:
        matrix[0, :] = (thumb_angles[1], thumb_angles[0], thumb_angles[2], thumb_angles[3])
    else:
        raise ValueError(f"未知拇指 CM 映射顺序：{thumb_cm_order}")

    for row, finger_key in enumerate(FINGER_ORDER, start=1):
        angles = np.asarray(mapping_states[finger_key]["human_angles"], dtype=np.float64).reshape(-1)
        if angles.size != 3:
            raise ValueError(f"{finger_key} 人手角应为 3 维，实际 {angles.size} 维")
        matrix[row, 0] = angles[0]
        matrix[row, 1] = 0.0
        matrix[row, 2] = angles[1]
        matrix[row, 3] = angles[2]

    return matrix


def human_matrices_to_wuji_relative_target(
    current_human_matrix: np.ndarray,
    human_baseline_matrix: np.ndarray,
    *,
    gain: float = DEFAULT_GAIN,
) -> np.ndarray:
    """把当前人手角与张开基点转换成 WuJi 相对目标角。"""
    current = coerce_wuji_matrix(current_human_matrix, name="current_human_matrix")
    baseline = coerce_wuji_matrix(human_baseline_matrix, name="human_baseline_matrix")
    target = (baseline - current) * float(gain)
    target *= WUJI_RELATIVE_DIRECTION
    target[1:, 1] = 0.0
    return target.astype(np.float64)


def mapping_states_to_wuji_target(
    mapping_states: dict[str, dict[str, object]],
    human_baseline_matrix: np.ndarray,
    *,
    gain: float = DEFAULT_GAIN,
    thumb_cm_order: str = DEFAULT_THUMB_CM_ORDER,
    lower_limits: np.ndarray | None = None,
    upper_limits: np.ndarray | None = None,
) -> np.ndarray:
    """用“张开基点 - 当前 IK/FK 角”生成 WuJi 正向弯曲目标。"""
    baseline = coerce_wuji_matrix(human_baseline_matrix, name="human_baseline_matrix")
    current = mapping_states_to_human_matrix(mapping_states, thumb_cm_order=thumb_cm_order)
    target = human_matrices_to_wuji_relative_target(current, baseline, gain=gain)

    if lower_limits is not None or upper_limits is not None:
        if lower_limits is None or upper_limits is None:
            raise ValueError("lower_limits 和 upper_limits 必须同时提供")
        lower = coerce_wuji_matrix(lower_limits, name="lower_limits")
        upper = coerce_wuji_matrix(upper_limits, name="upper_limits")
        target = np.clip(target, lower, upper)
        target[1:, 1] = 0.0

    return target.astype(np.float64)


def compute_wuji_target_debug(
    mapping_states: dict[str, dict[str, object]],
    human_baseline_matrix: np.ndarray,
    *,
    gain: float = DEFAULT_GAIN,
    thumb_cm_order: str = DEFAULT_THUMB_CM_ORDER,
    lower_limits: np.ndarray | None = None,
    upper_limits: np.ndarray | None = None,
) -> dict[str, Any]:
    """计算 WuJi 目标，并保留诊断用的中间矩阵。"""
    baseline = coerce_wuji_matrix(human_baseline_matrix, name="human_baseline_matrix")
    current = mapping_states_to_human_matrix(mapping_states, thumb_cm_order=thumb_cm_order)
    unclipped = human_matrices_to_wuji_relative_target(current, baseline, gain=gain)
    target = unclipped.copy()

    if lower_limits is not None or upper_limits is not None:
        if lower_limits is None or upper_limits is None:
            raise ValueError("lower_limits 和 upper_limits 必须同时提供")
        lower = coerce_wuji_matrix(lower_limits, name="lower_limits")
        upper = coerce_wuji_matrix(upper_limits, name="upper_limits")
        target = np.clip(target, lower, upper)
        target[1:, 1] = 0.0

    return {
        "current": current.astype(np.float64),
        "baseline": baseline.astype(np.float64),
        "unclipped": unclipped.astype(np.float64),
        "target": target.astype(np.float64),
    }


def save_human_baseline(
    path: Path,
    human_baseline_matrix: np.ndarray,
    *,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    baseline = coerce_wuji_matrix(human_baseline_matrix, name="human_baseline_matrix")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "created_at_local": now_local_text(),
        "created_at_unix": time.time(),
        "finger_order": list(HAND_ORDER),
        "wuji_layout": WUJI_LAYOUT,
        "side_policy": WUJI_SIDE_POLICY,
        "human_baseline_matrix": baseline.tolist(),
        "metadata": metadata,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def load_human_baseline_payload(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def remap_thumb_cm_order(matrix: np.ndarray, *, from_order: str, to_order: str) -> np.ndarray:
    """在 direct/swapped 两种拇指 CM 列语义之间转换 5x4 矩阵。"""
    remapped = coerce_wuji_matrix(matrix, name="matrix").copy()
    if from_order == to_order:
        return remapped
    valid_orders = {THUMB_CM_ORDER_DIRECT, THUMB_CM_ORDER_SWAPPED}
    if from_order not in valid_orders or to_order not in valid_orders:
        raise ValueError(f"未知拇指 CM 映射顺序转换：{from_order} -> {to_order}")
    remapped[0, 0], remapped[0, 1] = remapped[0, 1], remapped[0, 0]
    return remapped


def load_human_baseline(path: Path, *, thumb_cm_order: str = DEFAULT_THUMB_CM_ORDER) -> np.ndarray:
    payload = load_human_baseline_payload(path)
    metadata = payload.get("metadata", {})
    file_thumb_cm_order = str(metadata.get("thumb_cm_order", DEFAULT_THUMB_CM_ORDER))
    if "human_baseline_matrix" in payload:
        matrix = coerce_wuji_matrix(payload["human_baseline_matrix"], name="human_baseline_matrix")
        return remap_thumb_cm_order(matrix, from_order=file_thumb_cm_order, to_order=thumb_cm_order)
    if "human_baseline" in payload:
        matrix = coerce_wuji_matrix(payload["human_baseline"], name="human_baseline")
        return remap_thumb_cm_order(matrix, from_order=file_thumb_cm_order, to_order=thumb_cm_order)
    raise RuntimeError(f"标定文件缺少 human_baseline_matrix：{path}")


def build_runtime(args: argparse.Namespace) -> MappingRuntime:
    external_runtime = load_external_bridge_runtime(args.external_repo)
    model, data, bundle_urdf = load_model_from_urdf(args.urdf, limit_overrides=build_live_limit_overrides())
    joint_handles, _finger_handles, _qpos_adrs_map = build_joint_handles(model)
    for handle in joint_handles.values():
        data.qpos[handle.qpos_adr] = 0.0
    mujoco.mj_forward(model, data)

    return MappingRuntime(
        external_runtime=external_runtime,
        model=model,
        data=data,
        joint_handles=joint_handles,
        thumb_reference=make_thumb_reference(model),
        finger_states=build_default_finger_states(),
        bundle_urdf=Path(bundle_urdf),
    )


def seed_runtime_human_angles(runtime: MappingRuntime, human_matrix: np.ndarray) -> None:
    """用指定 5x4 人手角矩阵初始化下一帧 IK 的连续性初值。"""
    matrix = coerce_wuji_matrix(human_matrix, name="human_matrix")
    runtime.finger_states[THUMB_KEY]["human_angles"] = tuple(float(value) for value in matrix[0, :])
    for row, finger_key in enumerate(FINGER_ORDER, start=1):
        runtime.finger_states[finger_key]["human_angles"] = (
            float(matrix[row, 0]),
            float(matrix[row, 2]),
            float(matrix[row, 3]),
        )


def build_current_mapping_states(
    runtime: MappingRuntime,
    *,
    flexion_sign: int,
    plane_y_sign: int,
) -> dict[str, dict[str, object]]:
    mujoco.mj_forward(runtime.model, runtime.data)
    mapping_states: dict[str, dict[str, object]] = {}

    for finger_key in HAND_ORDER:
        finger_state = runtime.finger_states[finger_key]
        if finger_key == THUMB_KEY:
            state = build_thumb_mapping_state(
                runtime.model,
                runtime.data,
                runtime.thumb_reference,
                human_lengths=finger_state["human_lengths"],
                human_angles_initial=tuple(float(value) for value in finger_state["human_angles"]),
                base_offset=finger_state["base_offset"],
                tip_offset_local=finger_state["tip_offset_local"],
                ik_enabled=True,
            )
        else:
            state = build_mapping_state(
                runtime.model,
                runtime.data,
                FINGER_CONFIGS[finger_key],
                plane_y_sign=plane_y_sign,
                human_lengths=finger_state["human_lengths"],
                human_angles_initial=tuple(float(value) for value in finger_state["human_angles"]),
                base_offset=finger_state["base_offset"],
                tip_offset_local=finger_state["tip_offset_local"],
                ik_enabled=True,
                flexion_sign=flexion_sign,
            )

        mapping_states[finger_key] = state
        runtime.finger_states[finger_key]["human_angles"] = tuple(state["human_angles"])

    return mapping_states


def wait_for_fresh_snapshot(
    stream: RealtimeSkeletonStream,
    *,
    last_latest_time: float = 0.0,
    timeout_s: float = FRESH_FRAME_TIMEOUT_S,
) -> dict[str, Any]:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        snapshot = stream.snapshot()
        latest_time = float(snapshot["latest_time"])
        if snapshot["qpos_values"] and latest_time > last_latest_time:
            return snapshot
        error_text = str(snapshot["error_text"]).strip()
        if error_text:
            raise RuntimeError(f"实时 skeleton 回调错误：{error_text}")
        time.sleep(0.01)
    raise TimeoutError(f"{timeout_s:.1f}s 内没有收到新的外骨骼 skeleton 数据")


def apply_snapshot_and_solve(
    runtime: MappingRuntime,
    snapshot: dict[str, Any],
    *,
    flexion_sign: int,
    plane_y_sign: int,
) -> dict[str, dict[str, object]]:
    applied_count = apply_live_qpos(
        runtime.data,
        joint_handles=runtime.joint_handles,
        qpos_values=snapshot["qpos_values"],
    )
    if applied_count <= 0:
        raise RuntimeError("当前 skeleton 帧没有匹配到任何 MuJoCo 右手关节")
    return build_current_mapping_states(runtime, flexion_sign=flexion_sign, plane_y_sign=plane_y_sign)


def format_profile_details(
    *,
    current_human_matrix: np.ndarray,
    human_baseline_matrix: np.ndarray,
    target_matrix: np.ndarray,
    snapshot: dict[str, Any],
) -> str:
    source_age_ms = max(0.0, time.time() - float(snapshot["latest_time"])) * 1000.0
    return "\n".join(
        [
            f"源数据年龄(ms): {source_age_ms:.1f}",
            "当前 IK/FK 人手角矩阵(rad):",
            np.array2string(current_human_matrix, precision=4, suppress_small=False),
            "human baseline 矩阵(rad):",
            np.array2string(human_baseline_matrix, precision=4, suppress_small=False),
            "WuJi 目标矩阵(rad):",
            np.array2string(target_matrix, precision=4, suppress_small=False),
        ]
    )


def format_wuji_command_details(
    *,
    current_human_matrix: np.ndarray,
    human_baseline_matrix: np.ndarray,
    unclipped_target_matrix: np.ndarray,
    target_matrix: np.ndarray,
    snapshot: dict[str, Any] | None = None,
) -> str:
    """同一行输出拇指翻折/侧摆相对差值、原始 IK 值和发送值。"""
    current = coerce_wuji_matrix(current_human_matrix, name="current_human_matrix")
    baseline = coerce_wuji_matrix(human_baseline_matrix, name="human_baseline_matrix")
    target = coerce_wuji_matrix(target_matrix, name="target_matrix")

    thumb_relative = (baseline - current)[0, :2]
    thumb_raw = current[0, :2]
    thumb_sent = target[0, :2]
    return "\n".join(
        [
            (
                "拇指翻折侧摆相对差值(rad): "
                f"{np.array2string(thumb_relative, precision=4, suppress_small=False)} "
                "原始翻折侧摆(rad): "
                f"{np.array2string(thumb_raw, precision=4, suppress_small=False)} "
                "发送翻折侧摆(rad): "
                f"{np.array2string(thumb_sent, precision=4, suppress_small=False)}"
            ),
        ]
    )


def connect_wuji_hand(wujihandpy: Any, *, send_mode: str) -> Any:
    if send_mode == "unchecked" and hasattr(wujihandpy, "_core"):
        hand = wujihandpy._core.Hand()
        if hasattr(hand, "disable_thread_safe_check"):
            hand.disable_thread_safe_check()
        return hand
    return wujihandpy.Hand()


def write_wuji_enabled(hand: Any, enabled: bool) -> None:
    if enabled and hasattr(hand, "write_joint_enabled_unchecked"):
        try:
            hand.write_joint_enabled_unchecked(True)
            return
        except Exception:
            pass
    if not enabled:
        try:
            hand.write_joint_enabled(False, 2.0)
            return
        except TypeError:
            pass
    hand.write_joint_enabled(enabled)


def write_wuji_target(hand: Any, target: np.ndarray, *, send_mode: str) -> None:
    target64 = coerce_wuji_matrix(target, name="target").astype(np.float64)
    if send_mode == "unchecked" and hasattr(hand, "write_joint_target_position_unchecked"):
        hand.write_joint_target_position_unchecked(target64)
        return
    hand.write_joint_target_position(target64)


def calibrate(args: argparse.Namespace) -> int:
    runtime = build_runtime(args)
    stream = RealtimeSkeletonStream(
        runtime.external_runtime,
        hand_side="right",
        qpos_mode=args.qpos_mode,
        qpos_scale=args.qpos_scale,
    )

    frames: list[np.ndarray] = []
    latest_time = 0.0
    try:
        stream.start()
        print("已连接外骨骼 skeleton，开始采样张开姿态 IK/FK 基点。")
        for index in range(args.calibration_frame_count):
            snapshot = wait_for_fresh_snapshot(stream, last_latest_time=latest_time)
            latest_time = float(snapshot["latest_time"])
            states = apply_snapshot_and_solve(
                runtime,
                snapshot,
                flexion_sign=args.flexion_sign,
                plane_y_sign=args.plane_y_sign,
            )
            frames.append(mapping_states_to_human_matrix(states, thumb_cm_order=args.thumb_cm_order))
            print(f"标定采样 {index + 1}/{args.calibration_frame_count}")
    finally:
        stream.stop()

    baseline = np.mean(np.stack(frames, axis=0), axis=0)
    seed_runtime_human_angles(runtime, baseline)
    save_human_baseline(
        args.calibration_path,
        baseline,
        metadata={
            "frame_count": len(frames),
            "requested_frame_count": args.calibration_frame_count,
            "qpos_mode": args.qpos_mode,
            "qpos_scale": args.qpos_scale,
            "flexion_sign": args.flexion_sign,
            "plane_y_sign": args.plane_y_sign,
            "thumb_cm_order": args.thumb_cm_order,
            "urdf": str(Path(args.urdf).resolve()),
            "bundle_urdf": str(runtime.bundle_urdf),
            "external_repo": str(Path(args.external_repo).resolve()),
            "external_calibration_path": str(runtime.external_runtime.calibration_path),
            "rosbridge": f"ws://{runtime.external_runtime.rosbridge_host}:{runtime.external_runtime.rosbridge_port}",
            "skeleton_topic": runtime.external_runtime.skeleton_topic,
        },
    )
    print(f"已保存 WuJi IK/FK 相对基点：{args.calibration_path}")
    print(np.array2string(baseline, precision=5, suppress_small=False))
    return 0


def run_dry_once(
    args: argparse.Namespace,
    runtime: MappingRuntime,
    stream: RealtimeSkeletonStream,
    human_baseline_matrix: np.ndarray,
) -> None:
    snapshot = wait_for_fresh_snapshot(stream)
    states = apply_snapshot_and_solve(
        runtime,
        snapshot,
        flexion_sign=args.flexion_sign,
        plane_y_sign=args.plane_y_sign,
    )
    debug = compute_wuji_target_debug(
        states,
        human_baseline_matrix,
        gain=args.gain,
        thumb_cm_order=args.thumb_cm_order,
    )
    current = debug["current"]
    target = debug["target"]
    print(format_wuji_command_details(
        current_human_matrix=current,
        human_baseline_matrix=debug["baseline"],
        unclipped_target_matrix=debug["unclipped"],
        target_matrix=target,
        snapshot=snapshot,
    ))
    print(format_profile_details(
        current_human_matrix=current,
        human_baseline_matrix=human_baseline_matrix,
        target_matrix=target,
        snapshot=snapshot,
    ))


def import_wujihandpy() -> Any:
    try:
        import wujihandpy  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"导入 wujihandpy 失败，请确认使用包含 WuJi SDK 的虚拟环境：{exc}") from exc
    return wujihandpy


def run_teleop(args: argparse.Namespace) -> int:
    human_baseline_matrix = load_human_baseline(args.calibration_path, thumb_cm_order=args.thumb_cm_order)
    runtime = build_runtime(args)
    seed_runtime_human_angles(runtime, human_baseline_matrix)
    stream = RealtimeSkeletonStream(
        runtime.external_runtime,
        hand_side="right",
        qpos_mode=args.qpos_mode,
        qpos_scale=args.qpos_scale,
    )

    stream.start()
    try:
        print("已连接外骨骼 skeleton。")
        if args.dry_run:
            run_dry_once(args, runtime, stream, human_baseline_matrix)
            return 0

        # 先确认外骨骼有数据，再触碰 WuJi 实机，避免数据源错误时误使能。
        wait_for_fresh_snapshot(stream)
        wujihandpy = import_wujihandpy()
        hand = connect_wuji_hand(wujihandpy, send_mode=args.send_mode)
        controller = None
        try:
            write_wuji_enabled(hand, True)
            lower_limits = coerce_wuji_matrix(hand.read_joint_lower_limit(), name="lower_limits")
            upper_limits = coerce_wuji_matrix(hand.read_joint_upper_limit(), name="upper_limits")
            dt = 1.0 / max(args.rate_hz, 1e-6)
            loop_count = 0
            t_start = time.monotonic()
            last_log_at = 0.0
            print(
                f"已使能 WuJi，开始 {args.rate_hz:.1f} Hz IK/FK 直写"
                f"（发送模式：{args.send_mode}）。按 Ctrl+C 安全回零并失能。"
            )

            def run_loop(writer: Any, *, realtime_controller_active: bool = False) -> None:
                nonlocal loop_count, last_log_at
                while True:
                    loop_started = time.monotonic()
                    snapshot = stream.snapshot()
                    if snapshot["qpos_values"]:
                        states = apply_snapshot_and_solve(
                            runtime,
                            snapshot,
                            flexion_sign=args.flexion_sign,
                            plane_y_sign=args.plane_y_sign,
                        )
                        debug = compute_wuji_target_debug(
                            states,
                            human_baseline_matrix,
                            gain=args.gain,
                            thumb_cm_order=args.thumb_cm_order,
                            lower_limits=lower_limits,
                            upper_limits=upper_limits,
                        )
                        seed_runtime_human_angles(runtime, debug["current"])
                        target = debug["target"]
                        if realtime_controller_active:
                            writer.set_joint_target_position(target)
                        else:
                            write_wuji_target(writer, target, send_mode=args.send_mode)

                        now = time.monotonic()
                        if args.command_log_interval_s > 0.0 and now - last_log_at >= args.command_log_interval_s:
                            print(format_wuji_command_details(
                                current_human_matrix=debug["current"],
                                human_baseline_matrix=debug["baseline"],
                                unclipped_target_matrix=debug["unclipped"],
                                target_matrix=target,
                                snapshot=snapshot,
                            ))
                            if args.profile:
                                print(format_profile_details(
                                    current_human_matrix=debug["current"],
                                    human_baseline_matrix=debug["baseline"],
                                    target_matrix=target,
                                    snapshot=snapshot,
                                ))
                                compute_ms = (time.monotonic() - loop_started) * 1000.0
                                print(f"循环耗时(ms): {compute_ms:.2f}")
                            last_log_at = now

                    loop_count += 1
                    sleep_time = t_start + loop_count * dt - time.monotonic()
                    if sleep_time > 0.0:
                        time.sleep(sleep_time)

            if args.send_mode == "realtime":
                filter_instance = wujihandpy.filter.LowPass(cutoff_freq=args.cutoff_hz)
                with hand.realtime_controller(True, filter_instance) as controller:
                    run_loop(controller, realtime_controller_active=True)
            else:
                run_loop(hand)
        except KeyboardInterrupt:
            print("收到 Ctrl+C，准备安全回零并失能。")
        finally:
            zero_target = np.zeros(WUJI_SHAPE, dtype=np.float64)
            try:
                if controller is not None:
                    controller.set_joint_target_position(zero_target)
                else:
                    write_wuji_target(hand, zero_target, send_mode=args.send_mode)
                time.sleep(0.2)
            except Exception as exc:
                print(f"回零下发失败：{exc}")
                try:
                    hand.write_joint_target_position(zero_target)
                    time.sleep(0.2)
                    print("已通过普通写入回零。")
                except Exception as fallback_exc:
                    print(f"普通写入回零也失败：{fallback_exc}")
            try:
                write_wuji_enabled(hand, False)
                print("已失能 WuJi。")
            except Exception as exc:
                print(f"WuJi 失能失败：{exc}")
    finally:
        stream.stop()

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="外骨骼 IK/FK 反解结果直写 WuJi 左手。")
    parser.add_argument("--calibrate", action="store_true", help="采集当前张开姿态 IK/FK 人手角并保存为相对基点。")
    parser.add_argument("--dry-run", action="store_true", help="只计算并打印一帧目标矩阵，不连接 WuJi。")
    parser.add_argument("--profile", action="store_true", help="打印当前 IK/FK 角、基点和 WuJi 目标矩阵。")
    parser.add_argument("--calibration-path", type=Path, default=DEFAULT_CALIBRATION_PATH, help="相对基点 JSON 路径。")
    parser.add_argument("--calibration-frame-count", type=int, default=DEFAULT_CALIBRATION_FRAME_COUNT, help="标定时采样帧数。")
    parser.add_argument("--rate-hz", type=float, default=DEFAULT_RATE_HZ, help="实时控制频率。")
    parser.add_argument("--cutoff-hz", type=float, default=DEFAULT_CUTOFF_HZ, help="WuJi LowPass 截止频率。")
    parser.add_argument("--gain", type=float, default=DEFAULT_GAIN, help="IK/FK 相对角到 WuJi 目标的比例。")
    parser.add_argument(
        "--send-mode",
        choices=("unchecked", "realtime"),
        default=DEFAULT_SEND_MODE,
        help="WuJi 下发模式；unchecked 使用底层非阻塞写入，realtime 使用 realtime_controller。",
    )
    parser.add_argument(
        "--command-log-interval-s",
        type=float,
        default=DEFAULT_COMMAND_LOG_INTERVAL_S,
        help="终端中文逐关节发送角度打印间隔；设为 0 可关闭。",
    )
    parser.add_argument(
        "--thumb-cm-order",
        choices=(THUMB_CM_ORDER_SWAPPED, THUMB_CM_ORDER_DIRECT),
        default=DEFAULT_THUMB_CM_ORDER,
        help="拇指 IK 前两轴到 WuJi CM pitch/yaw 的列顺序；默认 direct: IK展收->J0 CM pitch、IK侧摆->J1 CM yaw。",
    )
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF, help="待加载的完整右手 URDF 路径。")
    parser.add_argument("--external-repo", type=Path, default=DEFAULT_EXTERNAL_REPO, help="外部机器人操控项目根目录。")
    parser.add_argument("--qpos-mode", choices=("relative", "absolute"), default="absolute", help="实时骨架值写回 MuJoCo 的方式。")
    parser.add_argument("--qpos-scale", type=float, default=1.0, help="写回 MuJoCo 前对实时 qpos 统一乘的缩放系数。")
    parser.add_argument("--flexion-sign", type=int, choices=(-1, 1), default=-1, help="四指二维映射的人手屈曲方向。")
    parser.add_argument("--plane-y-sign", type=int, choices=(-1, 1), default=1, help="四指二维映射平面的纵轴方向。")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.calibration_frame_count <= 0:
        raise ValueError("--calibration-frame-count 必须大于 0")
    if args.calibrate:
        return calibrate(args)
    return run_teleop(args)


if __name__ == "__main__":
    raise SystemExit(main())
