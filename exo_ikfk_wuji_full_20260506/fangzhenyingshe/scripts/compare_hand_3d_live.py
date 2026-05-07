#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""实时订阅外部外骨骼数据，在 MuJoCo 3D 中观察五指映射是否合理。

目标：
1. 复用 `compare_hand_3d.py` 现有的五指建模与 IK 叠加逻辑，不重写运动学。
2. 从外部 `机器人操控` 项目读取 rosbridge 实时 skeleton 数据。
3. 把实时外骨骼关节姿态写回当前 MuJoCo 模型，并叠加人手映射结果。

说明：
1. 默认使用 unwrap 后的实时绝对关节角驱动 MuJoCo，也就是 `qpos = current_unwrapped`，
   用于在仿真环境中尽量一比一复刻现实手套姿态。
2. 若要沿用“手掌伸直 = 映射零位”的观察方式，可传入 `--qpos-mode relative`，
   此时 `qpos = current_unwrapped - open_base`。
3. 实时模型会临时放宽右手手套关节限位，避免绝对角接近 `2π` 时被裁剪。
"""

from __future__ import annotations

import argparse
import csv
import importlib
import math
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import mujoco
import numpy as np

from compare_hand_3d import (
    FINGER_CONFIGS,
    FINGER_ORDER,
    HAND_ORDER,
    THUMB_HUMAN_LABELS,
    THUMB_KEY,
    THUMB_LABEL,
    append_mapping_overlay,
    append_selected_finger_markers,
    build_limit_overrides,
    build_mapping_state,
    build_thumb_mapping_state,
    format_summary,
    visible_angle_count,
)
from delivery_core.four_finger_mapping import (
    configure_chinese_font,
    four_finger_base_offset_from_mm,
    mm_to_m,
    recover_four_finger_base_input,
)
from delivery_core.sim_loader import DEFAULT_URDF, load_model_from_urdf
from delivery_core.thumb_mapping import THUMB_DOF, THUMB_JOINTS, ThumbReference, append_thumb_markers, get_joint_id, hide_left_hand_geoms, make_thumb_reference
from hand_mapping_params import THUMB_MAPPING_PARAMS


DEFAULT_EXTERNAL_REPO = Path(__file__).resolve().parents[2] / "external_robot_control"
LIVE_QPOS_LIMIT = 4.0 * math.pi


@dataclass(frozen=True)
class ExternalBridgeRuntime:
    repo_root: Path
    calibration_path: Path
    open_base: dict[str, float]
    rosbridge_host: str
    rosbridge_port: int
    skeleton_topic: str
    skeleton_topic_type: str
    roslibpy: Any
    skeleton_unwrapper_cls: type
    parse_hand_skeleton: Callable[..., dict[str, float]]


@dataclass(frozen=True)
class JointHandle:
    joint_name: str
    label: str
    joint_id: int
    qpos_adr: int
    lower: float
    upper: float


def build_live_record_columns() -> tuple[str, ...]:
    columns = [
        "sample_id",
        "finger",
        "record_kind",
        "recorded_at_local",
        "recorded_at_unix",
        "source_latest_time",
        "source_age_ms",
        "stream_hz",
        "qpos_mode",
        "ik_enabled",
        "selected_finger",
        "trigger",
        "angle_unit",
        "exo_angle_space",
        "calibration_id",
        "calibration_frame_count",
    ]
    for index in range(1, 6):
        columns.append(f"exo_q{index}")
    for index in range(1, 6):
        columns.append(f"human_q{index}")
    columns.extend(
        [
            "tip_error_mm",
            "is_reachable",
            "L1_mm",
            "L2_mm",
            "L3_mm",
            "base_dx_mm",
            "base_dy_mm",
            "base_dz_mm",
            "tip_dx_mm",
            "tip_dy_mm",
            "exo_delta_max_rad",
            "human_delta_max_rad",
        ]
    )
    return tuple(columns)


LIVE_RECORD_COLUMNS = build_live_record_columns()


def pad_vector(values: np.ndarray | list[float] | tuple[float, ...], length: int = 5) -> list[float]:
    padded = [math.nan] * length
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    for index, value in enumerate(array.tolist()):
        if index >= length:
            break
        padded[index] = float(value)
    return padded


def display_base_offset_mm(finger_key: str, base_offset: np.ndarray) -> tuple[float, float, float]:
    base_offset = np.asarray(base_offset, dtype=np.float64)
    if finger_key == THUMB_KEY:
        dx = float(base_offset[0] * 1000.0)
        dy = float(base_offset[1] * 1000.0)
        dz = float(base_offset[2] * 1000.0)
        return dx, dy, dz

    display = recover_four_finger_base_input(base_offset)
    return float(display[0] * 1000.0), float(display[1] * 1000.0), 0.0


def current_exo_q_row(
    data: mujoco.MjData,
    finger_handles: dict[str, list[JointHandle]],
    finger_key: str,
) -> list[float]:
    exo_q = [float(data.qpos[handle.qpos_adr]) for handle in finger_handles[finger_key]]
    return pad_vector(exo_q)


def build_live_record_row(
    sample_id: int,
    *,
    finger_key: str,
    snapshot: dict[str, Any],
    data: mujoco.MjData,
    finger_handles: dict[str, list[JointHandle]],
    finger_states: dict[str, dict[str, object]],
    mapping_states: dict[str, dict[str, object]],
    qpos_mode: str,
    ik_enabled: bool,
    selected_key: str,
    trigger_reason: str,
    exo_delta_max_rad: float,
    human_delta_max_rad: float,
    record_kind: str = "sample",
    calibration_id: int | str = "",
    calibration_frame_count: int = 1,
    exo_q_values: np.ndarray | list[float] | tuple[float, ...] | None = None,
    human_q_values: np.ndarray | list[float] | tuple[float, ...] | None = None,
    tip_error_mm: float | None = None,
    is_reachable: bool | None = None,
) -> dict[str, object]:
    recorded_at_unix = float(time.time())
    latest_time = float(snapshot["latest_time"])
    age_ms = max(0.0, recorded_at_unix - latest_time) * 1000.0 if latest_time > 0.0 else math.nan
    state = mapping_states[finger_key]
    finger_state = finger_states[finger_key]
    exo_q_row = (
        pad_vector(exo_q_values)
        if exo_q_values is not None
        else current_exo_q_row(data, finger_handles, finger_key)
    )
    human_q_row = (
        pad_vector(human_q_values)
        if human_q_values is not None
        else pad_vector(np.asarray(state["human_angles"], dtype=np.float64))
    )
    human_lengths = np.asarray(finger_state["human_lengths"], dtype=np.float64)
    base_dx_mm, base_dy_mm, base_dz_mm = display_base_offset_mm(
        finger_key,
        np.asarray(finger_state["base_offset"], dtype=np.float64),
    )
    tip_offset_local = np.asarray(finger_state["tip_offset_local"], dtype=np.float64)

    row: dict[str, object] = {
        "sample_id": int(sample_id),
        "finger": finger_key,
        "record_kind": record_kind,
        "recorded_at_local": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(recorded_at_unix)),
        "recorded_at_unix": recorded_at_unix,
        "source_latest_time": latest_time,
        "source_age_ms": age_ms,
        "stream_hz": float(snapshot["hz"]),
        "qpos_mode": qpos_mode,
        "ik_enabled": bool(ik_enabled),
        "selected_finger": selected_key,
        "trigger": trigger_reason,
        "angle_unit": "rad",
        "exo_angle_space": qpos_mode,
        "calibration_id": calibration_id,
        "calibration_frame_count": int(calibration_frame_count),
        "tip_error_mm": float(tip_error_mm) if tip_error_mm is not None else float(state["tip_error"]) * 1000.0,
        "is_reachable": bool(is_reachable) if is_reachable is not None else bool(state["is_reachable"]),
        "L1_mm": float(human_lengths[0] * 1000.0),
        "L2_mm": float(human_lengths[1] * 1000.0),
        "L3_mm": float(human_lengths[2] * 1000.0),
        "base_dx_mm": base_dx_mm,
        "base_dy_mm": base_dy_mm,
        "base_dz_mm": base_dz_mm,
        "tip_dx_mm": float(tip_offset_local[0] * 1000.0),
        "tip_dy_mm": float(tip_offset_local[1] * 1000.0),
        "exo_delta_max_rad": float(exo_delta_max_rad),
        "human_delta_max_rad": float(human_delta_max_rad),
    }

    for index, value in enumerate(exo_q_row, start=1):
        row[f"exo_q{index}"] = value
    for index, value in enumerate(human_q_row, start=1):
        row[f"human_q{index}"] = value

    return row


def nanmean_vector(frames: list[np.ndarray]) -> list[float]:
    if not frames:
        return [math.nan] * 5
    array = np.vstack([np.asarray(frame, dtype=np.float64).reshape(1, -1) for frame in frames])
    means: list[float] = []
    for index in range(array.shape[1]):
        values = array[:, index]
        values = values[np.isfinite(values)]
        means.append(float(np.mean(values)) if values.size else math.nan)
    return pad_vector(means)


def finite_mean(values: list[float]) -> float:
    finite_values = [float(value) for value in values if math.isfinite(float(value))]
    if not finite_values:
        return math.nan
    return float(np.mean(np.asarray(finite_values, dtype=np.float64)))


class LiveDataRecorder:
    """把实时遥操作过程中的外骨骼角与映射后人手角写入 CSV。"""

    def __init__(
        self,
        output_path: Path,
        *,
        min_interval: float,
        flush_every: int,
        auto_start: bool,
        exo_threshold_rad: float,
        human_threshold_rad: float,
        calibration_frame_count: int,
    ) -> None:
        self.output_path = Path(output_path)
        self.min_interval = max(0.0, float(min_interval))
        self.flush_every = max(1, int(flush_every))
        self.enabled = bool(auto_start)
        self.exo_threshold_rad = max(0.0, float(exo_threshold_rad))
        self.human_threshold_rad = max(0.0, float(human_threshold_rad))
        self.calibration_frame_count = max(1, int(calibration_frame_count))
        self.sample_count = 0
        self.last_source_time = 0.0
        self.last_recorded_at = 0.0
        self.last_calibration_at = 0.0
        self.last_calibration_frame_count = 0
        self.recorded_finger_count = 0
        self._pending_flush = 0
        self._file: Any | None = None
        self._writer: csv.DictWriter | None = None
        self._last_finger_exo_q: dict[str, np.ndarray] = {}
        self._last_finger_human_q: dict[str, np.ndarray] = {}
        self._calibration_active = False
        self._calibration_id = 0
        self._calibration_last_source_time = 0.0
        self._calibration_frames_seen = 0
        self._calibration_exo_q: dict[str, list[np.ndarray]] = {}
        self._calibration_human_q: dict[str, list[np.ndarray]] = {}
        self._calibration_tip_error_mm: dict[str, list[float]] = {}
        self._calibration_reachable: dict[str, list[bool]] = {}

        if self.enabled:
            self.start()

    def _ensure_file(self) -> None:
        if self._file is None:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self._file = self.output_path.open("w", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(self._file, fieldnames=LIVE_RECORD_COLUMNS)
            self._writer.writeheader()
            self._file.flush()

    def start(self) -> None:
        self._ensure_file()
        self.enabled = True

    def pause(self) -> None:
        self.enabled = False
        self.flush()

    def toggle(self) -> None:
        if self.enabled:
            self.pause()
        else:
            self.start()

    def flush(self) -> None:
        if self._file is not None:
            self._file.flush()
        self._pending_flush = 0

    def close(self) -> None:
        try:
            self.flush()
        finally:
            if self._file is not None:
                self._file.close()
        self._file = None
        self._writer = None

    def request_calibration(self) -> None:
        self._ensure_file()
        self._calibration_id += 1
        self._calibration_active = True
        self._calibration_last_source_time = 0.0
        self._calibration_frames_seen = 0
        self._calibration_exo_q = {finger_key: [] for finger_key in HAND_ORDER}
        self._calibration_human_q = {finger_key: [] for finger_key in HAND_ORDER}
        self._calibration_tip_error_mm = {finger_key: [] for finger_key in HAND_ORDER}
        self._calibration_reachable = {finger_key: [] for finger_key in HAND_ORDER}

    def maybe_collect_calibration(
        self,
        *,
        snapshot: dict[str, Any],
        stream_enabled: bool,
        data: mujoco.MjData,
        finger_handles: dict[str, list[JointHandle]],
        finger_states: dict[str, dict[str, object]],
        mapping_states: dict[str, dict[str, object]],
        qpos_mode: str,
        ik_enabled: bool,
        selected_key: str,
    ) -> bool:
        if not self._calibration_active or not stream_enabled or self._writer is None:
            return False

        latest_time = float(snapshot["latest_time"])
        if latest_time <= 0.0:
            return False
        if latest_time <= self._calibration_last_source_time + 1e-9:
            return False

        for finger_key in HAND_ORDER:
            current_exo_q = np.asarray(current_exo_q_row(data, finger_handles, finger_key), dtype=np.float64)
            current_human_q = np.asarray(
                pad_vector(np.asarray(mapping_states[finger_key]["human_angles"], dtype=np.float64)),
                dtype=np.float64,
            )
            self._calibration_exo_q[finger_key].append(current_exo_q.copy())
            self._calibration_human_q[finger_key].append(current_human_q.copy())
            self._calibration_tip_error_mm[finger_key].append(float(mapping_states[finger_key]["tip_error"]) * 1000.0)
            self._calibration_reachable[finger_key].append(bool(mapping_states[finger_key]["is_reachable"]))

        self._calibration_last_source_time = latest_time
        self._calibration_frames_seen += 1

        if self._calibration_frames_seen >= self.calibration_frame_count:
            self._write_calibration_rows(
                snapshot=snapshot,
                data=data,
                finger_handles=finger_handles,
                finger_states=finger_states,
                mapping_states=mapping_states,
                qpos_mode=qpos_mode,
                ik_enabled=ik_enabled,
                selected_key=selected_key,
            )
        return True

    def _write_calibration_rows(
        self,
        *,
        snapshot: dict[str, Any],
        data: mujoco.MjData,
        finger_handles: dict[str, list[JointHandle]],
        finger_states: dict[str, dict[str, object]],
        mapping_states: dict[str, dict[str, object]],
        qpos_mode: str,
        ik_enabled: bool,
        selected_key: str,
    ) -> None:
        if self._writer is None:
            return

        frame_count = self._calibration_frames_seen
        for finger_key in HAND_ORDER:
            exo_q = nanmean_vector(self._calibration_exo_q.get(finger_key, []))
            human_q = nanmean_vector(self._calibration_human_q.get(finger_key, []))
            tip_error_mm = finite_mean(self._calibration_tip_error_mm.get(finger_key, []))
            reachable_values = self._calibration_reachable.get(finger_key, [])
            is_reachable = bool(reachable_values) and all(reachable_values)
            row = build_live_record_row(
                self.sample_count + 1,
                finger_key=finger_key,
                snapshot=snapshot,
                data=data,
                finger_handles=finger_handles,
                finger_states=finger_states,
                mapping_states=mapping_states,
                qpos_mode=qpos_mode,
                ik_enabled=ik_enabled,
                selected_key=selected_key,
                trigger_reason="calibration_mean",
                exo_delta_max_rad=0.0,
                human_delta_max_rad=0.0,
                record_kind="calibration",
                calibration_id=self._calibration_id,
                calibration_frame_count=frame_count,
                exo_q_values=exo_q,
                human_q_values=human_q,
                tip_error_mm=tip_error_mm,
                is_reachable=is_reachable,
            )
            self._writer.writerow(row)
            self.sample_count += 1
            self.recorded_finger_count += 1

        self.last_calibration_at = float(time.time())
        self.last_calibration_frame_count = frame_count
        self._calibration_active = False
        self.flush()

    def maybe_record(
        self,
        *,
        snapshot: dict[str, Any],
        stream_enabled: bool,
        data: mujoco.MjData,
        finger_handles: dict[str, list[JointHandle]],
        finger_states: dict[str, dict[str, object]],
        mapping_states: dict[str, dict[str, object]],
        qpos_mode: str,
        ik_enabled: bool,
        selected_key: str,
    ) -> bool:
        if not self.enabled or not stream_enabled or self._writer is None:
            return False

        latest_time = float(snapshot["latest_time"])
        if latest_time <= 0.0:
            return False
        if latest_time <= self.last_source_time + 1e-9:
            return False
        if self.last_source_time > 0.0 and latest_time - self.last_source_time < self.min_interval:
            return False

        recorded_any = False
        for finger_key in HAND_ORDER:
            current_exo_q = np.asarray(current_exo_q_row(data, finger_handles, finger_key), dtype=np.float64)
            current_human_q = np.asarray(
                pad_vector(np.asarray(mapping_states[finger_key]["human_angles"], dtype=np.float64)),
                dtype=np.float64,
            )

            prev_exo_q = self._last_finger_exo_q.get(finger_key)
            prev_human_q = self._last_finger_human_q.get(finger_key)

            if prev_exo_q is None or prev_human_q is None:
                trigger_reason = "init"
                exo_delta_max = math.inf
                human_delta_max = math.inf
            else:
                exo_delta_max = float(np.nanmax(np.abs(current_exo_q - prev_exo_q)))
                human_delta_max = float(np.nanmax(np.abs(current_human_q - prev_human_q)))
                exo_triggered = exo_delta_max >= self.exo_threshold_rad
                human_triggered = human_delta_max >= self.human_threshold_rad
                if not exo_triggered and not human_triggered:
                    continue
                trigger_reason = "exo+human" if exo_triggered and human_triggered else ("exo" if exo_triggered else "human")

            row = build_live_record_row(
                self.sample_count + 1,
                finger_key=finger_key,
                snapshot=snapshot,
                data=data,
                finger_handles=finger_handles,
                finger_states=finger_states,
                mapping_states=mapping_states,
                qpos_mode=qpos_mode,
                ik_enabled=ik_enabled,
                selected_key=selected_key,
                trigger_reason=trigger_reason,
                exo_delta_max_rad=exo_delta_max,
                human_delta_max_rad=human_delta_max,
            )
            self._writer.writerow(row)
            self.sample_count += 1
            self.recorded_finger_count += 1
            self._last_finger_exo_q[finger_key] = current_exo_q.copy()
            self._last_finger_human_q[finger_key] = current_human_q.copy()
            recorded_any = True

        if not recorded_any:
            self.last_source_time = latest_time
            return False

        self.last_source_time = latest_time
        self.last_recorded_at = float(time.time())
        self._pending_flush += 1
        if self._pending_flush >= self.flush_every:
            self.flush()
        return recorded_any


def format_record_status(recorder: LiveDataRecorder) -> str:
    latest_text = "无"
    if recorder.last_recorded_at > 0.0:
        latest_text = time.strftime("%H:%M:%S", time.localtime(recorder.last_recorded_at))
    if recorder._calibration_active:
        calibration_text = f"采集中 {recorder._calibration_frames_seen}/{recorder.calibration_frame_count}"
    elif recorder.last_calibration_at > 0.0:
        calibration_time = time.strftime("%H:%M:%S", time.localtime(recorder.last_calibration_at))
        calibration_text = f"已写入 {recorder.last_calibration_frame_count} 帧均值 @ {calibration_time}"
    else:
        calibration_text = "未采集"
    return "\n".join(
        [
            "录制状态：",
            f"录制开关：{'开启' if recorder.enabled else '关闭'}",
            f"输出文件：{recorder.output_path}",
            f"已写样本：{recorder.sample_count}",
            f"已录手指条目：{recorder.recorded_finger_count}",
            "角度单位：rad",
            f"最小采样间隔(s)：{recorder.min_interval:.3f}",
            f"外骨骼阈值(rad)：{recorder.exo_threshold_rad:.4f}",
            f"人手阈值(rad)：{recorder.human_threshold_rad:.4f}",
            f"标定采集：{calibration_text}",
            f"最近写入：{latest_text}",
        ]
    )


def load_external_bridge_runtime(repo_root: Path) -> ExternalBridgeRuntime:
    """从外部 `机器人操控` 项目加载实时数据依赖。"""
    repo_root = repo_root.resolve()
    if not repo_root.exists():
        raise RuntimeError(f"外部项目目录不存在：{repo_root}")

    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

    try:
        roslibpy = importlib.import_module("roslibpy")
        exo_mapping = importlib.import_module("core.mapping.exo_mapping")
        runtime_assets = importlib.import_module("core.runtime.assets")
        site_config = importlib.import_module("core.runtime.site_config")
    except Exception as exc:
        raise RuntimeError(
            f"读取外部项目实时数据依赖失败，请确认 {repo_root} 完整可用，且已安装 roslibpy。原始错误：{exc}"
        ) from exc

    calibration_payload, calibration_path = runtime_assets.load_calibration_payload(required=True)
    open_base = dict(calibration_payload.get("open_base", {}))
    if not open_base:
        raise RuntimeError(
            f"外部项目校准文件中 open_base 为空：{calibration_path}，请先在外部项目完成校准。"
        )

    return ExternalBridgeRuntime(
        repo_root=repo_root,
        calibration_path=Path(calibration_path),
        open_base=open_base,
        rosbridge_host=str(site_config.ROSBRIDGE_HOST),
        rosbridge_port=int(site_config.ROSBRIDGE_PORT),
        skeleton_topic=str(site_config.SKELETON_TOPIC),
        skeleton_topic_type=str(site_config.SKELETON_TOPIC_TYPE),
        roslibpy=roslibpy,
        skeleton_unwrapper_cls=exo_mapping.SkeletonUnwrapper,
        parse_hand_skeleton=exo_mapping.parse_hand_skeleton,
    )


class RealtimeSkeletonStream:
    """后台订阅实时 skeleton 数据，并缓存最新一帧可供 viewer 线程读取。"""

    def __init__(
        self,
        runtime: ExternalBridgeRuntime,
        *,
        hand_side: str,
        qpos_mode: str,
        qpos_scale: float,
    ) -> None:
        if qpos_mode not in {"relative", "absolute"}:
            raise ValueError(f"未知 qpos_mode：{qpos_mode}")

        self.runtime = runtime
        self.hand_side = hand_side
        self.qpos_mode = qpos_mode
        self.qpos_scale = float(qpos_scale)

        self._unwrapper = runtime.skeleton_unwrapper_cls()
        self._lock = threading.Lock()
        self._raw_values: dict[str, float] = {}
        self._qpos_values: dict[str, float] = {}
        self._latest_time = 0.0
        self._error_text = ""
        self._hz = 0.0
        self._hz_count = 0
        self._hz_last = time.time()
        self._connected = False

        self._client: Any | None = None
        self._topic: Any | None = None

    def _value_to_qpos(self, joint_name: str, value: float) -> float:
        if self.qpos_mode == "absolute":
            return float(value) * self.qpos_scale
        base = self.runtime.open_base.get(joint_name, value)
        return float(value - base) * self.qpos_scale

    def _on_skeleton(self, msg: dict[str, Any]) -> None:
        try:
            values = self.runtime.parse_hand_skeleton(
                msg,
                hand_side=self.hand_side,
                unwrapper=self._unwrapper,
                reference_values=self.runtime.open_base,
            )
            if not values:
                return

            qpos_values = {
                joint_name: self._value_to_qpos(joint_name, value)
                for joint_name, value in values.items()
            }
            now = time.time()

            with self._lock:
                self._hz_count += 1
                elapsed = now - self._hz_last
                if elapsed >= 1.0:
                    self._hz = self._hz_count / elapsed
                    self._hz_count = 0
                    self._hz_last = now

                self._raw_values = values
                self._qpos_values = qpos_values
                self._latest_time = now
                self._error_text = ""
        except Exception as exc:
            with self._lock:
                self._error_text = str(exc)

    def start(self) -> None:
        client = self.runtime.roslibpy.Ros(
            host=self.runtime.rosbridge_host,
            port=self.runtime.rosbridge_port,
        )
        client.run()
        if not client.is_connected:
            raise RuntimeError(
                f"无法连接 rosbridge: ws://{self.runtime.rosbridge_host}:{self.runtime.rosbridge_port}"
            )

        topic = self.runtime.roslibpy.Topic(
            client,
            self.runtime.skeleton_topic,
            self.runtime.skeleton_topic_type,
        )
        topic.subscribe(self._on_skeleton)

        with self._lock:
            self._client = client
            self._topic = topic
            self._connected = True

    def stop(self) -> None:
        with self._lock:
            topic = self._topic
            client = self._client
            self._topic = None
            self._client = None
            self._connected = False

        try:
            if topic is not None:
                topic.unsubscribe()
        except Exception:
            pass
        try:
            if client is not None and client.is_connected:
                client.terminate()
        except Exception:
            pass

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "connected": self._connected,
                "raw_values": dict(self._raw_values),
                "qpos_values": dict(self._qpos_values),
                "latest_time": float(self._latest_time),
                "error_text": str(self._error_text),
                "hz": float(self._hz),
            }


def build_joint_handles(
    model: mujoco.MjModel,
) -> tuple[dict[str, JointHandle], dict[str, list[JointHandle]], dict[str, list[int]]]:
    """为实时写入 qpos 和按手指汇总摘要建立索引。"""
    joint_handles: dict[str, JointHandle] = {}
    finger_handles: dict[str, list[JointHandle]] = {key: [] for key in HAND_ORDER}
    qpos_adrs_map: dict[str, list[int]] = {key: [] for key in HAND_ORDER}

    for item in THUMB_JOINTS:
        joint_id = get_joint_id(model, item.joint_name)
        handle = JointHandle(
            joint_name=item.joint_name,
            label=item.tag,
            joint_id=joint_id,
            qpos_adr=int(model.jnt_qposadr[joint_id]),
            lower=float(model.jnt_range[joint_id][0]),
            upper=float(model.jnt_range[joint_id][1]),
        )
        joint_handles[item.joint_name] = handle
        finger_handles[THUMB_KEY].append(handle)
        qpos_adrs_map[THUMB_KEY].append(handle.qpos_adr)

    for finger_key in FINGER_ORDER:
        cfg = FINGER_CONFIGS[finger_key]
        for index, joint_name in enumerate(cfg.joints, start=1):
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                raise RuntimeError(f"模型中缺少关节：{joint_name}")
            handle = JointHandle(
                joint_name=joint_name,
                label=f"{cfg.label}{index}",
                joint_id=int(joint_id),
                qpos_adr=int(model.jnt_qposadr[joint_id]),
                lower=float(model.jnt_range[joint_id][0]),
                upper=float(model.jnt_range[joint_id][1]),
            )
            joint_handles[joint_name] = handle
            finger_handles[finger_key].append(handle)
            qpos_adrs_map[finger_key].append(handle.qpos_adr)

    return joint_handles, finger_handles, qpos_adrs_map


def build_live_limit_overrides() -> dict[str, tuple[float, float, float, float]]:
    """为实时复刻放宽右手手套关节限位，避免绝对传感器角被裁剪。"""
    overrides = build_limit_overrides()
    for item in THUMB_JOINTS:
        overrides[item.joint_name] = (-LIVE_QPOS_LIMIT, LIVE_QPOS_LIMIT, 100.0, 1.0)
    for cfg in FINGER_CONFIGS.values():
        for joint_name in cfg.joints:
            overrides[joint_name] = (-LIVE_QPOS_LIMIT, LIVE_QPOS_LIMIT, 100.0, 1.0)
    return overrides


def build_default_finger_states() -> dict[str, dict[str, object]]:
    finger_states: dict[str, dict[str, object]] = {
        THUMB_KEY: {
            "human_lengths": tuple(mm_to_m(value) for value in THUMB_MAPPING_PARAMS.lengths_mm),
            "human_angles": tuple(0.0 for _ in range(THUMB_DOF)),
            "base_offset": np.array([mm_to_m(value) for value in THUMB_MAPPING_PARAMS.base_mm], dtype=np.float64),
            "tip_offset_local": np.array([mm_to_m(value) for value in THUMB_MAPPING_PARAMS.tip_mm], dtype=np.float64),
        }
    }

    for finger_key in FINGER_ORDER:
        cfg = FINGER_CONFIGS[finger_key]
        finger_states[finger_key] = {
            "human_lengths": tuple(mm_to_m(value) for value in cfg.default_lengths_mm),
            "human_angles": (0.0, 0.0, 0.0),
            "base_offset": four_finger_base_offset_from_mm(
                cfg.default_base_mm[0],
                cfg.default_base_mm[1],
            ),
            "tip_offset_local": np.array(
                [mm_to_m(cfg.default_tip_mm[0]), mm_to_m(cfg.default_tip_mm[1])],
                dtype=np.float64,
            ),
        }
    return finger_states


def reset_finger_states(finger_states: dict[str, dict[str, object]]) -> None:
    defaults = build_default_finger_states()
    for finger_key in HAND_ORDER:
        finger_states[finger_key]["human_lengths"] = defaults[finger_key]["human_lengths"]
        finger_states[finger_key]["human_angles"] = defaults[finger_key]["human_angles"]
        finger_states[finger_key]["base_offset"] = np.asarray(defaults[finger_key]["base_offset"], dtype=np.float64)
        finger_states[finger_key]["tip_offset_local"] = np.asarray(defaults[finger_key]["tip_offset_local"], dtype=np.float64)


def apply_live_qpos(
    data: mujoco.MjData,
    *,
    joint_handles: dict[str, JointHandle],
    qpos_values: dict[str, float],
) -> int:
    """把实时 qpos 写回 MuJoCo，缺失关节保持当前值不变。"""
    applied_count = 0
    for joint_name, handle in joint_handles.items():
        if joint_name not in qpos_values:
            continue

        value = float(qpos_values[joint_name])
        if np.isfinite(handle.lower) and np.isfinite(handle.upper) and handle.lower < handle.upper:
            value = float(np.clip(value, handle.lower, handle.upper))
        data.qpos[handle.qpos_adr] = value
        applied_count += 1
    return applied_count


def format_live_status(
    *,
    runtime: ExternalBridgeRuntime,
    snapshot: dict[str, Any],
    stream_enabled: bool,
    qpos_mode: str,
    selected_key: str,
    finger_handles: dict[str, list[JointHandle]],
) -> str:
    mode_text = "相对 open_base（映射零位观察）" if qpos_mode == "relative" else "unwrap 绝对角（复刻手套姿态）"
    latest_time = float(snapshot["latest_time"])
    latest_age_text = "尚未收到数据"
    if latest_time > 0.0:
        latest_age_text = f"{max(0.0, time.time() - latest_time):.2f} s 前"

    lines = [
        f"实时数据源：{runtime.repo_root}",
        f"校准文件：{runtime.calibration_path}",
        f"rosbridge：ws://{runtime.rosbridge_host}:{runtime.rosbridge_port}  {runtime.skeleton_topic}",
        f"实时更新：{'开启' if stream_enabled else '暂停（保持最近一帧）'}",
        f"qpos 模式：{mode_text}",
        f"连接状态：{'已连接' if snapshot['connected'] else '未连接'} | 采样率：{float(snapshot['hz']):.1f} Hz | 最近数据：{latest_age_text}",
    ]

    error_text = str(snapshot["error_text"]).strip()
    if error_text:
        lines.append(f"最近回调错误：{error_text}")

    lines.append("")
    lines.append("当前选中手指实时外骨骼角：")
    raw_values = snapshot["raw_values"]
    qpos_values = snapshot["qpos_values"]
    for handle in finger_handles[selected_key]:
        raw = raw_values.get(handle.joint_name)
        qpos = qpos_values.get(handle.joint_name)
        base = runtime.open_base.get(handle.joint_name)
        if raw is None or qpos is None:
            lines.append(f"- {handle.label}: 暂无数据")
            continue
        if base is None:
            lines.append(f"- {handle.label}: qpos={qpos:+.4f} rad | raw={raw:.4f}")
            continue
        lines.append(
            f"- {handle.label}: qpos={qpos:+.4f} rad | raw={raw:.4f} | base={base:.4f}"
        )

    return "\n".join(lines)


def slider_viewer_live(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    runtime: ExternalBridgeRuntime,
    live_stream: RealtimeSkeletonStream,
    joint_handles: dict[str, JointHandle],
    finger_handles: dict[str, list[JointHandle]],
    finger_states: dict[str, dict[str, object]],
    thumb_reference: ThumbReference,
    initial_finger: str,
    ik_mode_init: bool,
    qpos_mode: str,
    flexion_sign: int,
    plane_y_sign: int,
    axis_length: float,
    dt: float,
    recorder: LiveDataRecorder,
) -> None:
    try:
        import matplotlib

        configure_chinese_font()
        matplotlib.use("TkAgg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, RadioButtons, Slider, TextBox
        import mujoco.viewer as viewer
    except ImportError as exc:
        raise RuntimeError("缺少实时观察依赖，请检查 matplotlib 和 mujoco.viewer。") from exc

    fig = plt.figure("实时外骨骼全手五指映射观察器", figsize=(12.4, 9.2))
    fig.subplots_adjust(left=0.28, right=0.98, top=0.95, bottom=0.07)

    human_slider_specs = [
        ("human_q1", -3.14, 3.14, [0.32, 0.88, 0.62, 0.028]),
        ("human_q2", -3.14, 3.14, [0.32, 0.835, 0.62, 0.028]),
        ("human_q3", -3.14, 3.14, [0.32, 0.79, 0.62, 0.028]),
        ("human_q4", -3.14, 3.14, [0.32, 0.745, 0.62, 0.028]),
        ("human_q5", -3.14, 3.14, [0.32, 0.70, 0.62, 0.028]),
    ]
    input_specs = [
        ("L1_mm", [0.32, 0.39, 0.18, 0.038]),
        ("L2_mm", [0.32, 0.345, 0.18, 0.038]),
        ("L3_mm", [0.32, 0.30, 0.18, 0.038]),
        ("base_dx_mm", [0.32, 0.235, 0.18, 0.038]),
        ("base_dy_mm", [0.32, 0.19, 0.18, 0.038]),
        ("base_dz_mm", [0.32, 0.145, 0.18, 0.038]),
        ("tip_dx_mm", [0.32, 0.09, 0.18, 0.038]),
        ("tip_dy_mm", [0.32, 0.045, 0.18, 0.038]),
    ]

    sliders: dict[str, Slider] = {}
    for label, vmin, vmax, rect in human_slider_specs:
        ax_slider = fig.add_axes(rect)
        sliders[label] = Slider(ax_slider, label, vmin, vmax, valinit=0.0)

    numeric_inputs: dict[str, float] = {}
    textboxes: dict[str, TextBox] = {}
    for label, rect in input_specs:
        ax_input = fig.add_axes(rect)
        numeric_inputs[label] = 0.0
        textboxes[label] = TextBox(ax_input, label, initial="0.00")

    selected = {"key": initial_finger}
    ik_state = {"enabled": ik_mode_init}
    live_state = {"enabled": True, "last_qpos_values": {}}

    selected_text = fig.text(0.04, 0.94, "", va="top", ha="left", fontsize=11)
    summary_text = fig.text(0.56, 0.94, "", va="top", ha="left", fontsize=9)

    radio_ax = fig.add_axes([0.04, 0.62, 0.20, 0.24])
    radio = RadioButtons(
        radio_ax,
        [THUMB_LABEL, *[FINGER_CONFIGS[key].label for key in FINGER_ORDER]],
        active=HAND_ORDER.index(initial_finger),
    )
    radio_ax.set_title("当前观察手指")

    reset_ax = fig.add_axes([0.56, 0.02, 0.12, 0.05])
    reset_button = Button(reset_ax, "重置人手参数")
    toggle_ik_ax = fig.add_axes([0.70, 0.02, 0.12, 0.05])
    toggle_ik_button = Button(toggle_ik_ax, f"IK: {'开' if ik_state['enabled'] else '关'}")
    toggle_live_ax = fig.add_axes([0.84, 0.02, 0.12, 0.05])
    toggle_live_button = Button(toggle_live_ax, "实时: 开")
    record_ax = fig.add_axes([0.56, 0.085, 0.12, 0.05])
    record_button = Button(record_ax, f"录制: {'开' if recorder.enabled else '关'}")
    calibration_ax = fig.add_axes([0.70, 0.085, 0.12, 0.05])
    calibration_button = Button(calibration_ax, "标定")

    label_to_key = {THUMB_LABEL: THUMB_KEY, **{cfg.label: cfg.key for cfg in FINGER_CONFIGS.values()}}

    def set_textbox_mm(name: str, value_mm: float) -> None:
        numeric_inputs[name] = float(value_mm)
        textboxes[name].set_val(f"{value_mm:.2f}")

    def set_human_slider_visibility(visible_count: int, labels: list[str]) -> None:
        for index in range(1, 6):
            slider = sliders[f"human_q{index}"]
            active = index <= visible_count
            slider.ax.set_visible(active)
            if active:
                slider.label.set_text(labels[index - 1])

    def sync_controls_from_selected() -> None:
        selected_key = selected["key"]
        finger_state = finger_states[selected_key]
        _exo_count, human_count = visible_angle_count(selected_key)

        if selected_key == THUMB_KEY:
            human_labels = [f"human_q{i} {name}" for i, name in enumerate(THUMB_HUMAN_LABELS, start=1)]
            selected_text.set_text(
                "当前手指：拇指\n"
                "外骨骼姿态来自实时数据；人手链使用独立 3D 拇指 IK/FK。"
            )
        else:
            cfg = FINGER_CONFIGS[selected_key]
            human_labels = [f"human_q{i} {cfg.label}" for i in range(1, 4)]
            selected_text.set_text(
                f"当前手指：{cfg.label}\n"
                "外骨骼姿态来自实时数据；人手链继续沿用同平面映射。"
            )

        set_human_slider_visibility(human_count, human_labels)

        human_angles = tuple(float(value) for value in finger_state["human_angles"])
        for index, angle in enumerate(human_angles, start=1):
            sliders[f"human_q{index}"].set_val(angle)
        for index in range(len(human_angles) + 1, 6):
            sliders[f"human_q{index}"].set_val(0.0)

        human_lengths = finger_state["human_lengths"]
        base_offset = np.asarray(finger_state["base_offset"], dtype=np.float64)
        tip_offset_local = np.asarray(finger_state["tip_offset_local"], dtype=np.float64)
        display_base_offset = (
            base_offset if selected_key == THUMB_KEY else recover_four_finger_base_input(base_offset)
        )
        set_textbox_mm("L1_mm", human_lengths[0] * 1000.0)
        set_textbox_mm("L2_mm", human_lengths[1] * 1000.0)
        set_textbox_mm("L3_mm", human_lengths[2] * 1000.0)
        set_textbox_mm("base_dx_mm", display_base_offset[0] * 1000.0)
        set_textbox_mm("base_dy_mm", display_base_offset[1] * 1000.0)
        set_textbox_mm("base_dz_mm", (base_offset[2] * 1000.0) if len(base_offset) > 2 else 0.0)
        set_textbox_mm("tip_dx_mm", tip_offset_local[0] * 1000.0)
        set_textbox_mm("tip_dy_mm", tip_offset_local[1] * 1000.0)
        textboxes["base_dz_mm"].ax.set_visible(selected_key == THUMB_KEY)

    def read_input_mm(label: str, *, positive: bool = False) -> float:
        try:
            value = float(textboxes[label].text.strip())
        except ValueError:
            value = numeric_inputs[label]
        if positive:
            value = max(value, 0.1)
        numeric_inputs[label] = value
        return value

    def reset_human_params(_event: object) -> None:
        reset_finger_states(finger_states)
        sync_controls_from_selected()

    def toggle_ik(_event: object) -> None:
        ik_state["enabled"] = not ik_state["enabled"]
        toggle_ik_button.label.set_text(f"IK: {'开' if ik_state['enabled'] else '关'}")

    def toggle_live(_event: object) -> None:
        live_state["enabled"] = not live_state["enabled"]
        toggle_live_button.label.set_text(f"实时: {'开' if live_state['enabled'] else '关'}")

    def toggle_record(_event: object) -> None:
        recorder.toggle()
        record_button.label.set_text(f"录制: {'开' if recorder.enabled else '关'}")

    def request_calibration(_event: object) -> None:
        recorder.request_calibration()
        calibration_button.label.set_text(f"标定 0/{recorder.calibration_frame_count}")

    def choose_finger(label: str) -> None:
        selected["key"] = label_to_key[label]
        sync_controls_from_selected()

    reset_button.on_clicked(reset_human_params)
    toggle_ik_button.on_clicked(toggle_ik)
    toggle_live_button.on_clicked(toggle_live)
    record_button.on_clicked(toggle_record)
    calibration_button.on_clicked(request_calibration)
    radio.on_clicked(choose_finger)
    sync_controls_from_selected()

    with viewer.launch_passive(model, data, show_left_ui=True, show_right_ui=True) as handle:
        handle.opt.frame = mujoco.mjtFrame.mjFRAME_NONE
        print("已打开实时全手五指映射观察窗口。")
        print("操作说明：")
        print("- 外骨骼姿态来自机器人操控项目的 rosbridge skeleton 数据。")
        print("- 默认使用 unwrap 后的绝对关节角驱动 MuJoCo，目标是在仿真中复刻现实手套姿态。")
        print("- 如需回到映射零位观察，可用 --qpos-mode relative 切换为相对 open_base。")
        print("- 左侧单选按钮切换当前观察手指；下方可切换 IK、实时更新和录制。")
        print(f"- 点击“标定”会采集约 {recorder.calibration_frame_count} 个新实时帧，写入每指一条 calibration 均值样本。")
        print("- 文本框单位均为毫米，修改后按回车生效。")
        print("- 四指 base_dx/base_dy 输入会在内部按 (x, y) -> (y, -x) 转换。")
        print(f"- 录制数据将写入：{recorder.output_path}")

        while handle.is_running() and plt.fignum_exists(fig.number):
            snapshot = live_stream.snapshot()
            if live_state["enabled"] and snapshot["qpos_values"]:
                live_state["last_qpos_values"] = dict(snapshot["qpos_values"])

            apply_live_qpos(
                data,
                joint_handles=joint_handles,
                qpos_values=live_state["last_qpos_values"],
            )

            selected_key = selected["key"]
            _exo_count, human_count = visible_angle_count(selected_key)

            finger_states[selected_key]["human_lengths"] = (
                mm_to_m(read_input_mm("L1_mm", positive=True)),
                mm_to_m(read_input_mm("L2_mm", positive=True)),
                mm_to_m(read_input_mm("L3_mm", positive=True)),
            )
            if selected_key == THUMB_KEY:
                finger_states[selected_key]["base_offset"] = np.array(
                    [
                        mm_to_m(read_input_mm("base_dx_mm")),
                        mm_to_m(read_input_mm("base_dy_mm")),
                        mm_to_m(read_input_mm("base_dz_mm")),
                    ],
                    dtype=np.float64,
                )
            else:
                finger_states[selected_key]["base_offset"] = four_finger_base_offset_from_mm(
                    read_input_mm("base_dx_mm"),
                    read_input_mm("base_dy_mm"),
                )
            finger_states[selected_key]["tip_offset_local"] = np.array(
                [mm_to_m(read_input_mm("tip_dx_mm")), mm_to_m(read_input_mm("tip_dy_mm"))],
                dtype=np.float64,
            )
            selected_human_angles = tuple(sliders[f"human_q{i}"].val for i in range(1, human_count + 1))

            mujoco.mj_forward(model, data)

            mapping_states: dict[str, dict[str, object]] = {}
            for finger_key in HAND_ORDER:
                if finger_key == THUMB_KEY:
                    human_angles_initial = (
                        selected_human_angles if finger_key == selected_key else finger_states[finger_key]["human_angles"]
                    )
                    state = build_thumb_mapping_state(
                        model,
                        data,
                        thumb_reference,
                        human_lengths=finger_states[finger_key]["human_lengths"],
                        human_angles_initial=tuple(float(value) for value in human_angles_initial),
                        base_offset=finger_states[finger_key]["base_offset"],
                        tip_offset_local=finger_states[finger_key]["tip_offset_local"],
                        ik_enabled=ik_state["enabled"],
                    )
                else:
                    cfg = FINGER_CONFIGS[finger_key]
                    human_angles_initial = (
                        selected_human_angles if finger_key == selected_key else finger_states[finger_key]["human_angles"]
                    )
                    state = build_mapping_state(
                        model,
                        data,
                        cfg,
                        plane_y_sign=plane_y_sign,
                        human_lengths=finger_states[finger_key]["human_lengths"],
                        human_angles_initial=tuple(float(value) for value in human_angles_initial),
                        base_offset=finger_states[finger_key]["base_offset"],
                        tip_offset_local=finger_states[finger_key]["tip_offset_local"],
                        ik_enabled=ik_state["enabled"],
                        flexion_sign=flexion_sign,
                    )
                mapping_states[finger_key] = state
                finger_states[finger_key]["human_angles"] = tuple(state["human_angles"])

            recorder.maybe_collect_calibration(
                snapshot=snapshot,
                stream_enabled=live_state["enabled"],
                data=data,
                finger_handles=finger_handles,
                finger_states=finger_states,
                mapping_states=mapping_states,
                qpos_mode=qpos_mode,
                ik_enabled=ik_state["enabled"],
                selected_key=selected_key,
            )
            if recorder._calibration_active:
                calibration_button.label.set_text(
                    f"标定 {recorder._calibration_frames_seen}/{recorder.calibration_frame_count}"
                )
            else:
                calibration_button.label.set_text("标定")

            recorder.maybe_record(
                snapshot=snapshot,
                stream_enabled=live_state["enabled"],
                data=data,
                finger_handles=finger_handles,
                finger_states=finger_states,
                mapping_states=mapping_states,
                qpos_mode=qpos_mode,
                ik_enabled=ik_state["enabled"],
                selected_key=selected_key,
            )

            summary_text.set_text(
                format_live_status(
                    runtime=runtime,
                    snapshot=snapshot,
                    stream_enabled=live_state["enabled"],
                    qpos_mode=qpos_mode,
                    selected_key=selected_key,
                    finger_handles=finger_handles,
                )
                + "\n\n"
                + format_summary(
                    mapping_states[selected_key],
                    ik_enabled=ik_state["enabled"],
                    flexion_sign=flexion_sign,
                    plane_y_sign=plane_y_sign,
                )
                + "\n\n"
                + format_record_status(recorder)
            )

            with handle.lock():
                handle.user_scn.ngeom = 0
                if selected_key == THUMB_KEY:
                    append_thumb_markers(model, data, handle.user_scn, axis_length=axis_length)
                else:
                    append_selected_finger_markers(
                        model,
                        data,
                        handle.user_scn,
                        FINGER_CONFIGS[selected_key],
                        axis_length=axis_length,
                    )
                for finger_key in HAND_ORDER:
                    append_mapping_overlay(
                        handle.user_scn,
                        mapping_states[finger_key],
                        axis_length=axis_length,
                        selected=(finger_key == selected_key),
                    )
            handle.sync()
            plt.pause(dt)

    recorder.flush()
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="实时订阅外部外骨骼数据，在 MuJoCo 3D 中观察五指映射。")
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF, help="待加载的完整右手 URDF 路径。")
    parser.add_argument(
        "--external-repo",
        type=Path,
        default=DEFAULT_EXTERNAL_REPO,
        help="包含 rosbridge 调用逻辑和校准文件的外部项目根目录。",
    )
    parser.add_argument(
        "--finger",
        choices=HAND_ORDER,
        default=THUMB_KEY,
        help="初始选中的手指。",
    )
    parser.add_argument(
        "--qpos-mode",
        choices=("relative", "absolute"),
        default="absolute",
        help="实时骨架值写回 MuJoCo 的方式：absolute=unwrap 绝对值复刻手套姿态，relative=相对 open_base 观察映射零位。",
    )
    parser.add_argument(
        "--qpos-scale",
        type=float,
        default=1.0,
        help="写回 MuJoCo 前对实时 qpos 统一乘的缩放系数。",
    )
    parser.add_argument(
        "--flexion-sign",
        type=int,
        choices=(-1, 1),
        default=-1,
        help="四指二维映射的人手屈曲方向；拇指 3D 映射不使用该参数。",
    )
    parser.add_argument(
        "--plane-y-sign",
        type=int,
        choices=(-1, 1),
        default=1,
        help="四指二维映射平面的纵轴方向；拇指 3D 映射不使用该参数。",
    )
    parser.add_argument("--no-ik", action="store_false", dest="ik", help="关闭 IK，改为手动调节当前人手角度。")
    parser.set_defaults(ik=True)
    parser.add_argument("--show-left-hand", action="store_true", help="默认隐藏左手几何；传入该参数时显示完整模型。")
    parser.add_argument("--axis-length", type=float, default=0.03, help="三维叠加坐标轴与箭头长度，单位米。")
    parser.add_argument("--dt", type=float, default=0.02, help="viewer 与控制窗口刷新时间步长。")
    parser.add_argument(
        "--record-output",
        type=Path,
        default=Path("outputs/live_hand_mapping_record.csv"),
        help="实时遥操作录制输出 CSV 路径。",
    )
    parser.add_argument(
        "--record-min-interval",
        type=float,
        default=0.0,
        help="相邻两条录制样本允许的最小源时间间隔，单位秒；0 表示每个新实时帧都记录。",
    )
    parser.add_argument(
        "--record-flush-every",
        type=int,
        default=20,
        help="每写入多少条样本强制 flush 一次 CSV。",
    )
    parser.add_argument(
        "--record-exo-threshold-rad",
        type=float,
        default=0.03,
        help="单根手指外骨骼角最大变化超过该阈值时才记录，单位 rad。",
    )
    parser.add_argument(
        "--record-human-threshold-rad",
        type=float,
        default=0.03,
        help="单根手指映射后人手角最大变化超过该阈值时才记录，单位 rad。",
    )
    parser.add_argument(
        "--calibration-frame-count",
        type=int,
        default=10,
        help="点击标定按钮后采集的新实时帧数量；完成后按手指写入 calibration 均值样本。",
    )
    parser.add_argument("--record-auto-start", action="store_true", help="启动窗口后立即开始录制。")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_chinese_font()

    runtime = load_external_bridge_runtime(args.external_repo)
    print(f"已加载外部实时数据项目：{runtime.repo_root}")
    print(f"已加载外部校准文件：{runtime.calibration_path}")
    print(
        f"实时数据源：ws://{runtime.rosbridge_host}:{runtime.rosbridge_port}"
        f"  {runtime.skeleton_topic}"
    )

    model, data, bundle_urdf = load_model_from_urdf(args.urdf, limit_overrides=build_live_limit_overrides())
    print(f"已加载模型：{bundle_urdf}")
    if not args.show_left_hand:
        hidden_count = hide_left_hand_geoms(model)
        print(f"已隐藏左手几何体：{hidden_count} 个；如需完整模型可传入 --show-left-hand。")

    joint_handles, finger_handles, _qpos_adrs_map = build_joint_handles(model)
    for handle in joint_handles.values():
        data.qpos[handle.qpos_adr] = 0.0
    mujoco.mj_forward(model, data)

    thumb_reference = make_thumb_reference(model)
    finger_states = build_default_finger_states()

    live_stream = RealtimeSkeletonStream(
        runtime,
        hand_side="right",
        qpos_mode=args.qpos_mode,
        qpos_scale=args.qpos_scale,
    )
    recorder = LiveDataRecorder(
        args.record_output,
        min_interval=args.record_min_interval,
        flush_every=args.record_flush_every,
        auto_start=args.record_auto_start,
        exo_threshold_rad=args.record_exo_threshold_rad,
        human_threshold_rad=args.record_human_threshold_rad,
        calibration_frame_count=args.calibration_frame_count,
    )

    try:
        live_stream.start()
        slider_viewer_live(
            model,
            data,
            runtime=runtime,
            live_stream=live_stream,
            joint_handles=joint_handles,
            finger_handles=finger_handles,
            finger_states=finger_states,
            thumb_reference=thumb_reference,
            initial_finger=args.finger,
            ik_mode_init=args.ik,
            qpos_mode=args.qpos_mode,
            flexion_sign=args.flexion_sign,
            plane_y_sign=args.plane_y_sign,
            axis_length=args.axis_length,
            dt=args.dt,
            recorder=recorder,
        )
    finally:
        recorder.close()
        live_stream.stop()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n用户中断，退出实时全手映射观察器。")
        raise SystemExit(130)
