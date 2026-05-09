from __future__ import annotations

"""外骨骼骨架数据到 O6 姿态的公共映射工具。

这个模块被 bridge / wizard / diagnostics 共同复用，职责很单一：
1. 处理传感器 0~2π 回绕带来的跳变；
2. 从 rosbridge 消息里抽取当前左右手关节值；
3. 把线性映射或 NN 预测结果统一换算成 O6 的 0-255 指令。
"""

import math
import os
from dataclasses import dataclass, field
from typing import Any

from core.profiles.hand_profiles import HandProfile
from core.runtime.paths import ensure_sdk_on_path

ensure_sdk_on_path()
from LinkerHand.utils.mapping import arc_to_range_right  # type: ignore


TWO_PI = 2.0 * math.pi
THUMB_YAW_SOFTMAX_ENABLED = os.getenv('EXO_THUMB_YAW_SOFTMAX', '1').strip().lower() not in {'0', 'false', 'no'}
THUMB_YAW_SOFTMAX_GAIN = float(os.getenv('EXO_THUMB_YAW_SOFTMAX_GAIN', '6.0'))
THUMB_YAW_SOFTMAX_INPUT_RATIO = float(os.getenv('EXO_THUMB_YAW_SOFTMAX_INPUT_RATIO', '0.35'))


@dataclass
class SkeletonUnwrapper:
    """按关节维度维护 unwrap 状态，消除跨零点时的 2π 跳变。"""

    prev_raw: dict[str, float] = field(default_factory=dict)
    unwrap_offset: dict[str, float] = field(default_factory=dict)

    def unwrap(self, name: str, raw: float, reference: float | None = None) -> float:
        """把单个关节的原始弧度转成连续弧度。"""
        if name not in self.prev_raw:
            self.prev_raw[name] = raw
            offset = 0.0
            # 首帧按校准时的 open_base 选最近的 2π 分支，避免跨会话后落到错误圈数。
            if reference is not None and math.isfinite(reference):
                offset = round((reference - raw) / TWO_PI) * TWO_PI
            self.unwrap_offset[name] = offset
            return raw + offset

        diff = raw - self.prev_raw[name]
        # 传感器值跨越内部零点时会突然从接近 2π 跳到 0，或反过来。
        if diff > math.pi:
            self.unwrap_offset[name] -= TWO_PI
        elif diff < -math.pi:
            self.unwrap_offset[name] += TWO_PI

        self.prev_raw[name] = raw
        return raw + self.unwrap_offset[name]


def parse_hand_skeleton(
    msg: dict[str, Any],
    *,
    hand_side: str,
    unwrapper: SkeletonUnwrapper | None = None,
    reference_values: dict[str, float] | None = None,
) -> dict[str, float]:
    """从 rosbridge skeleton 消息里抽取指定手侧的关节值。"""
    vals: dict[str, float] = {}
    match_kw = 'Right' if hand_side.lower() == 'right' else 'Left'
    match_kw_lower = match_kw.lower()
    if unwrapper is not None:
        parser = lambda name, raw: unwrapper.unwrap(
            name,
            raw,
            None if reference_values is None else reference_values.get(name),
        )
    else:
        parser = lambda name, raw: raw

    data = msg.get('data', [])
    if not isinstance(data, list):
        return vals

    for seg in data:
        if not isinstance(seg, dict):
            continue
        names = seg.get('name', [])
        positions = seg.get('position', [])
        if not isinstance(names, list) or not isinstance(positions, list):
            continue
        for name, pos in zip(names, positions):
            if not isinstance(name, str):
                continue
            if match_kw not in name and match_kw_lower not in name:
                continue
            try:
                vals[name] = parser(name, float(pos))
            except (TypeError, ValueError):
                continue
    return vals


def compute_signed_feature_delta(vals: dict[str, float], open_base: dict[str, float], joint_name: str) -> float:
    """计算单个关节相对张开基线的有符号偏移量。"""
    cur = vals.get(joint_name, 0.0)
    base = open_base.get(joint_name, cur)
    return base - cur


def compute_feature_delta(vals: dict[str, float], open_base: dict[str, float], joint_name: str) -> float:
    """计算单个关节相对张开基线的单边弯曲量。"""
    return max(0.0, compute_signed_feature_delta(vals, open_base, joint_name))


def compute_linear_slot_arc(
    profile: HandProfile,
    joint_map: dict[str, str],
    open_base: dict[str, float],
    vals: dict[str, float],
    slot_index: int,
) -> float:
    """把一个 O6 槽位映射成线性弧度值。"""
    slot_name = profile.slot_names[slot_index]
    exo_name = joint_map.get(slot_name)
    if not exo_name:
        return 0.0
    raw = vals.get(exo_name)
    if raw is None:
        return 0.0
    base = open_base.get(exo_name, raw)
    slot_spec = profile.slot_specs[slot_index]
    bend_rad = max(0.0, (base - raw) * slot_spec.sensor_direction)
    if slot_name == 'thumb_cmc_yaw' and slot_spec.arc_max > 0 and THUMB_YAW_SOFTMAX_ENABLED:
        # 拇指横摆实测输入范围偏小，这里把线性侧摆量按机器人最大摆幅做 softmax 风格拉伸。
        input_ratio = min(max(THUMB_YAW_SOFTMAX_INPUT_RATIO, 1e-3), 1.0)
        gain = max(0.0, THUMB_YAW_SOFTMAX_GAIN)
        ref_arc = slot_spec.arc_max * input_ratio
        normalized = min(max(bend_rad / ref_arc, 0.0), 1.0)
        if gain > 0.0:
            normalized = math.log1p(gain * normalized) / math.log1p(gain)
        bend_rad = normalized * slot_spec.arc_max
    return min(bend_rad, profile.slot_arc_max[slot_index])


def arcs_to_pose(arcs: list[float], hand_joint: str) -> list[int]:
    """统一把弧度列表转换成 SDK 需要的 0-255 指令。"""
    raw_pose = arc_to_range_right(arcs, hand_joint)
    return [int(round(v)) for v in raw_pose]


def compute_linear_pose(
    profile: HandProfile,
    joint_map: dict[str, str],
    open_base: dict[str, float],
    vals: dict[str, float],
) -> list[int]:
    """按 profile 槽位顺序执行线性映射。"""
    arcs = [
        compute_linear_slot_arc(profile, joint_map, open_base, vals, idx)
        for idx in range(len(profile.slot_specs))
    ]
    return arcs_to_pose(arcs, profile.hand_joint)


def compute_nn_pose(
    profile: HandProfile,
    joint_map: dict[str, str],
    open_base: dict[str, float],
    vals: dict[str, float],
    nn_models: dict[str, Any],
    nn_finger_joints: dict[str, list[str]],
) -> list[int]:
    """按槽位执行 NN 推理，缺模型时自动回退到线性映射。"""
    import numpy as np

    arcs: list[float] = []
    for idx, slot_name in enumerate(profile.slot_names):
        model = nn_models.get(slot_name)
        finger_joints = nn_finger_joints.get(slot_name)
        if model is None or finger_joints is None:
            # 某个槽位没有训练好时，只回退该槽位，其他槽位仍可继续走 NN。
            arcs.append(compute_linear_slot_arc(profile, joint_map, open_base, vals, idx))
            continue

        features = [
            compute_feature_delta(vals, open_base, joint_name)
            for joint_name in finger_joints
        ]
        pred_arc = float(model.predict(np.array([features], dtype=np.float64))[0])
        pred_arc = max(0.0, min(pred_arc, profile.slot_arc_max[idx]))
        arcs.append(pred_arc)
    return arcs_to_pose(arcs, profile.hand_joint)


def build_thumb_debug(
    profile: HandProfile,
    joint_map: dict[str, str],
    open_base: dict[str, float],
    vals: dict[str, float],
    pose: list[int],
    *,
    use_nn: bool,
    nn_finger_joints: dict[str, list[str]],
) -> dict[str, Any]:
    """构建拇指调试信息，方便比对当前槽位到底看到了哪些特征。"""
    thumb_slots = [slot.name for slot in profile.slot_specs[:2]]
    slot_to_joint_names: dict[str, list[str]] = {}
    if use_nn:
        for slot_name in thumb_slots:
            slot_to_joint_names[slot_name] = list(nn_finger_joints.get(slot_name, []))
    else:
        for slot_name in thumb_slots:
            exo_name = joint_map.get(slot_name)
            slot_to_joint_names[slot_name] = [exo_name] if exo_name else []

    debug: dict[str, Any] = {
        'mode': 'nn' if use_nn else 'linear',
        'pose': {slot_name: int(pose[idx]) for idx, slot_name in enumerate(thumb_slots)},
        'slots': {},
    }
    for slot_name, joint_names in slot_to_joint_names.items():
        feature_items = []
        for joint_name in joint_names:
            if not joint_name:
                continue
            feature_items.append((joint_name, compute_feature_delta(vals, open_base, joint_name)))
        feature_items.sort(key=lambda item: item[1], reverse=True)
        debug['slots'][slot_name] = {
            'joint_names': list(joint_names),
            'deltas': {name: round(delta, 6) for name, delta in feature_items},
            'top3': [(name, round(delta, 6)) for name, delta in feature_items[:3]],
        }
    return debug
