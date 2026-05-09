#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""手部驱动的外骨骼映射采集器。

流程：
1. 生成当前手指的人手目标角轨迹；
2. 反求该手指外骨骼角；
3. 调用现有映射逻辑得到映射后的人手角；
4. 用可达性、末端误差和相对零位碰撞基线过滤；
5. 实时刷新 MuJoCo 3D viewer 和状态面板；
6. 导出有效样本 CSV 与拒绝样本 JSONL。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from scipy import optimize

from compare_hand_3d import (
    FINGER_CONFIGS,
    FINGER_ORDER,
    HAND_ORDER,
    THUMB_KEY,
    THUMB_LABEL,
    append_mapping_overlay,
    append_selected_finger_markers,
    build_limit_overrides,
    build_mapping_state,
    build_thumb_mapping_state,
    lift_points_to_3d,
    make_thumb_reference,
)
from delivery_core.four_finger_mapping import (
    build_human_polyline,
    configure_chinese_font,
    four_finger_base_offset_from_mm,
    human_joint_bounds,
    mm_to_m,
)
from delivery_core.sim_loader import DEFAULT_URDF, load_model_from_urdf
from delivery_core.thumb_mapping import THUMB_DOF, THUMB_EXO_DOF, THUMB_JOINTS, append_thumb_markers, build_thumb_fk, get_joint_id, hide_left_hand_geoms, thumb_joint_bounds
from delivery_core.viewer_primitives import add_connector, add_sphere
from hand_mapping_params import THUMB_MAPPING_PARAMS


SEGMENT_FRAMES = 24
AXIS_LENGTH = 0.03
COLLISION_EPS_M = 0.0005
RIGHT_BODY_PREFIX = "link_RightSkeleton"
REJECT_REASON_COLLISION_NEW = "collision_new_pair"
REJECT_REASON_COLLISION_DEEPER = "collision_deeper_than_baseline"
REJECT_REASON_UNREACHABLE = "unreachable"
REJECT_REASON_TIP_ERROR = "tip_error_over_threshold"
REJECT_REASON_INVERSE_FAILED = "inverse_failed"
REJECT_REASON_ATTEMPT_BUDGET = "attempt_budget_exhausted"

TARGET_JOINT_RGBA = np.array([0.15, 0.92, 0.92, 0.56], dtype=np.float32)
TARGET_LINK_RGBA = np.array([0.12, 0.78, 0.88, 0.42], dtype=np.float32)
TARGET_TIP_RGBA = np.array([0.20, 1.00, 0.95, 0.68], dtype=np.float32)
ACCEPT_COLOR = "#1f9d55"
REJECT_COLOR = "#c0392b"
NEUTRAL_COLOR = "#2c3e50"

FOUR_FINGER_FAMILIES = ("open_close", "mcp_lead", "hook")
FOUR_FINGER_WEIGHTS = {
    "open_close": np.array([0.65, 1.00, 0.75], dtype=np.float64),
    "mcp_lead": np.array([0.85, 0.70, 0.50], dtype=np.float64),
    "hook": np.array([0.20, 1.00, 0.85], dtype=np.float64),
}
THUMB_FAMILY_SEQUENCE = ("flex", "abduct", "opposition", "explore")
THUMB_FAMILY_WEIGHTS = {
    "flex": np.array([0.10, 0.05, 0.80, 0.60], dtype=np.float64),
    "abduct": np.array([1.00, 0.70, 0.00, 0.00], dtype=np.float64),
    "opposition": np.array([0.60, 0.50, 0.60, 0.40], dtype=np.float64),
}
THUMB_EXPLORE_STEP_RAD = np.radians(np.array([4.0, 4.0, 6.0, 6.0], dtype=np.float64))

VALID_CSV_COLUMNS = [
    "sample_id",
    "finger",
    "trajectory_family",
    "trajectory_id",
    "step_idx",
    "target_human_q1",
    "target_human_q2",
    "target_human_q3",
    "target_human_q4",
    "target_human_q5",
    "mapped_human_q1",
    "mapped_human_q2",
    "mapped_human_q3",
    "mapped_human_q4",
    "mapped_human_q5",
    "exo_q1",
    "exo_q2",
    "exo_q3",
    "exo_q4",
    "exo_q5",
    "tip_error_mm",
    "is_reachable",
    "target_distance_mm",
    "max_reach_mm",
    "collision_new_pair_count",
    "collision_deeper_pair_count",
]


@dataclass
class FingerRuntimeState:
    key: str
    label: str
    exo_dof: int
    human_dof: int
    qpos_adrs: list[int]
    human_lengths: tuple[float, float, float]
    base_offset: np.ndarray
    tip_offset_local: np.ndarray
    prev_valid_exo_q: np.ndarray
    prev_valid_mapped_q: np.ndarray
    target_anchor_q: np.ndarray
    segment_counter: int = 0
    family_queue: list[str] | None = None
    thumb_cycle_index: int = 0
    init_exo_q: np.ndarray | None = None
    init_mapped_q: np.ndarray | None = None


@dataclass(frozen=True)
class InverseSolveResult:
    exo_q: np.ndarray
    mapping_state: dict[str, object] | None
    solver_success: bool
    solver_nfev: int


@dataclass(frozen=True)
class CollisionBaseline:
    pair_to_dist: dict[tuple[str, str], float]
    details: list[dict[str, object]]
    adjacent_pairs: set[tuple[str, str]]


@dataclass(frozen=True)
class CollisionCheck:
    new_pairs: list[tuple[str, str]]
    deeper_pairs: list[tuple[str, str]]
    contact_details: list[dict[str, object]]


@dataclass(frozen=True)
class InitializationResult:
    exo_q: np.ndarray
    mapping_state: dict[str, object] | None
    success: bool
    solver_nfev: int
    score: tuple[float, float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="采集手部角度到外骨骼角度的映射数据集。")
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF, help="待加载的 v4 外骨骼 URDF。")
    parser.add_argument(
        "--fingers",
        choices=("all", *HAND_ORDER),
        default="all",
        help="要采集的手指范围，默认依次处理五指。",
    )
    parser.add_argument("--valid-per-finger", type=int, default=1000, help="每根手指需要保留的有效样本数。")
    parser.add_argument("--tip-error-mm", type=float, default=3.0, help="有效样本允许的最大末端误差。")
    parser.add_argument("--seed", type=int, default=0, help="随机种子。")
    parser.add_argument("--preview-every", type=int, default=1, help="每隔多少次尝试刷新一次预览。")
    parser.add_argument("--dt", type=float, default=0.02, help="可视化刷新步长。")
    parser.add_argument(
        "--max-attempt-factor",
        type=int,
        default=30,
        help="每根手指的最大尝试次数 = valid_per_finger * max_attempt_factor。",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("outputs/hand_mapping_dataset"),
        help="输出前缀，不带文件后缀。",
    )
    parser.add_argument("--headless", action="store_true", help="关闭 viewer 和状态面板，仅做批量采集。")
    return parser.parse_args()


def choose_finger_order(fingers_arg: str) -> list[str]:
    if fingers_arg == "all":
        return list(HAND_ORDER)
    return [fingers_arg]


def build_qpos_adrs_map(model: mujoco.MjModel) -> dict[str, list[int]]:
    qpos_adrs_map: dict[str, list[int]] = {THUMB_KEY: []}
    for item in THUMB_JOINTS:
        joint_id = get_joint_id(model, item.joint_name)
        qpos_adrs_map[THUMB_KEY].append(int(model.jnt_qposadr[joint_id]))

    for key in FINGER_ORDER:
        cfg = FINGER_CONFIGS[key]
        qpos_adrs_map[key] = []
        for joint_name in cfg.joints:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                raise RuntimeError(f"模型中缺少关节：{joint_name}")
            qpos_adrs_map[key].append(int(model.jnt_qposadr[joint_id]))
    return qpos_adrs_map


def zero_all_hand_qpos(data: mujoco.MjData, qpos_adrs_map: dict[str, list[int]]) -> None:
    for qpos_adrs in qpos_adrs_map.values():
        for qpos_adr in qpos_adrs:
            data.qpos[qpos_adr] = 0.0


def set_finger_qpos(data: mujoco.MjData, qpos_adrs_map: dict[str, list[int]], finger_key: str, values: np.ndarray) -> None:
    q = np.asarray(values, dtype=np.float64)
    qpos_adrs = qpos_adrs_map[finger_key]
    if len(qpos_adrs) != q.shape[0]:
        raise ValueError(f"{finger_key} 自由度数量不匹配：期望 {len(qpos_adrs)}，实际 {q.shape[0]}")
    for qpos_adr, value in zip(qpos_adrs, q):
        data.qpos[qpos_adr] = float(value)


def build_runtime_states(qpos_adrs_map: dict[str, list[int]]) -> dict[str, FingerRuntimeState]:
    states: dict[str, FingerRuntimeState] = {
        THUMB_KEY: FingerRuntimeState(
            key=THUMB_KEY,
            label=THUMB_LABEL,
            exo_dof=THUMB_EXO_DOF,
            human_dof=THUMB_DOF,
            qpos_adrs=qpos_adrs_map[THUMB_KEY],
            human_lengths=tuple(mm_to_m(value) for value in THUMB_MAPPING_PARAMS.lengths_mm),
            base_offset=np.array([mm_to_m(value) for value in THUMB_MAPPING_PARAMS.base_mm], dtype=np.float64),
            tip_offset_local=np.array([mm_to_m(value) for value in THUMB_MAPPING_PARAMS.tip_mm], dtype=np.float64),
            prev_valid_exo_q=np.zeros(THUMB_EXO_DOF, dtype=np.float64),
            prev_valid_mapped_q=np.zeros(THUMB_DOF, dtype=np.float64),
            target_anchor_q=np.zeros(THUMB_DOF, dtype=np.float64),
        )
    }

    for key in FINGER_ORDER:
        cfg = FINGER_CONFIGS[key]
        states[key] = FingerRuntimeState(
            key=key,
            label=cfg.label,
            exo_dof=4,
            human_dof=3,
            qpos_adrs=qpos_adrs_map[key],
            human_lengths=tuple(mm_to_m(value) for value in cfg.default_lengths_mm),
            base_offset=four_finger_base_offset_from_mm(
                cfg.default_base_mm[0],
                cfg.default_base_mm[1],
            ),
            tip_offset_local=np.array([mm_to_m(cfg.default_tip_mm[0]), mm_to_m(cfg.default_tip_mm[1])], dtype=np.float64),
            prev_valid_exo_q=np.zeros(4, dtype=np.float64),
            prev_valid_mapped_q=np.zeros(3, dtype=np.float64),
            target_anchor_q=np.zeros(3, dtype=np.float64),
            family_queue=[],
        )
    return states


def smooth_blend(frame_idx: int, frame_count: int) -> float:
    if frame_count <= 1:
        return 1.0
    u = frame_idx / float(frame_count - 1)
    return float(np.sin(0.5 * np.pi * u) ** 2)


def interpolate_segment(start_q: np.ndarray, goal_q: np.ndarray, frame_count: int = SEGMENT_FRAMES) -> np.ndarray:
    start = np.asarray(start_q, dtype=np.float64)
    goal = np.asarray(goal_q, dtype=np.float64)
    frames = []
    for frame_idx in range(frame_count):
        s = smooth_blend(frame_idx, frame_count)
        frames.append(start + (goal - start) * s)
    return np.vstack(frames)


def four_finger_flex_magnitudes(flexion_sign: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lower, upper = human_joint_bounds(flexion_sign)
    if flexion_sign >= 0:
        magnitudes = np.abs(upper)
        direction = np.ones(3, dtype=np.float64)
    else:
        magnitudes = np.abs(lower)
        direction = -np.ones(3, dtype=np.float64)
    return lower, upper, direction * magnitudes


def next_four_finger_segment(rng: np.random.Generator, runtime: FingerRuntimeState, flexion_sign: int) -> tuple[str, np.ndarray]:
    if runtime.family_queue is None:
        runtime.family_queue = []
    if not runtime.family_queue:
        runtime.family_queue = list(rng.permutation(FOUR_FINGER_FAMILIES))

    family = runtime.family_queue.pop(0)
    lower, upper, signed_magnitudes = four_finger_flex_magnitudes(flexion_sign)
    scale = float(rng.uniform(0.35, 1.00))
    goal_q = scale * FOUR_FINGER_WEIGHTS[family] * signed_magnitudes
    goal_q = np.clip(goal_q, lower, upper)
    frames = interpolate_segment(runtime.target_anchor_q, goal_q, SEGMENT_FRAMES)
    runtime.target_anchor_q = goal_q.copy()
    runtime.segment_counter += 1
    return family, frames


def next_thumb_explore_segment(rng: np.random.Generator, runtime: FingerRuntimeState) -> np.ndarray:
    lower, upper = thumb_joint_bounds()
    current_q = runtime.target_anchor_q.copy()
    frames = []
    waypoint = rng.uniform(lower, upper)

    for _ in range(SEGMENT_FRAMES):
        if np.all(np.abs(waypoint - current_q) <= THUMB_EXPLORE_STEP_RAD + 1e-12):
            waypoint = rng.uniform(lower, upper)
        delta = waypoint - current_q
        step = np.clip(delta, -THUMB_EXPLORE_STEP_RAD, THUMB_EXPLORE_STEP_RAD)
        current_q = np.clip(current_q + step, lower, upper)
        frames.append(current_q.copy())

    runtime.target_anchor_q = current_q.copy()
    return np.vstack(frames)


def next_thumb_segment(rng: np.random.Generator, runtime: FingerRuntimeState) -> tuple[str, np.ndarray]:
    family = THUMB_FAMILY_SEQUENCE[runtime.thumb_cycle_index % len(THUMB_FAMILY_SEQUENCE)]
    runtime.thumb_cycle_index += 1
    runtime.segment_counter += 1

    if family == "explore":
        return family, next_thumb_explore_segment(rng, runtime)

    _, upper = thumb_joint_bounds()
    scale = float(rng.uniform(0.30, 0.95))
    goal_q = scale * THUMB_FAMILY_WEIGHTS[family] * upper
    frames = interpolate_segment(runtime.target_anchor_q, goal_q, SEGMENT_FRAMES)
    runtime.target_anchor_q = goal_q.copy()
    return family, frames


def evaluate_mapping_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    thumb_reference,
    runtime: FingerRuntimeState,
    *,
    human_angles_initial: np.ndarray,
    plane_y_sign: int,
    flexion_sign: int,
) -> dict[str, object]:
    if runtime.key == THUMB_KEY:
        return build_thumb_mapping_state(
            model,
            data,
            thumb_reference,
            human_lengths=runtime.human_lengths,
            human_angles_initial=tuple(float(value) for value in human_angles_initial),
            base_offset=runtime.base_offset,
            tip_offset_local=runtime.tip_offset_local,
            ik_enabled=True,
        )

    return build_mapping_state(
        model,
        data,
        FINGER_CONFIGS[runtime.key],
        plane_y_sign=plane_y_sign,
        human_lengths=runtime.human_lengths,
        human_angles_initial=tuple(float(value) for value in human_angles_initial),
        base_offset=runtime.base_offset,
        tip_offset_local=runtime.tip_offset_local,
        ik_enabled=True,
        flexion_sign=flexion_sign,
    )


def initialization_score(mapping_state: dict[str, object] | None, exo_q: np.ndarray) -> tuple[float, float, float]:
    if mapping_state is None:
        return (1.0, 1e9, 1e9)
    reachable_flag = 0.0 if bool(mapping_state["is_reachable"]) else 1.0
    tip_error = float(mapping_state["tip_error"])
    exo_norm = float(np.linalg.norm(np.asarray(exo_q, dtype=np.float64)))
    return (reachable_flag, tip_error, exo_norm)


def solve_initial_alignment(
    model: mujoco.MjModel,
    work_data: mujoco.MjData,
    qpos_adrs_map: dict[str, list[int]],
    thumb_reference,
    runtime: FingerRuntimeState,
    rng: np.random.Generator,
    *,
    plane_y_sign: int,
    flexion_sign: int,
) -> InitializationResult:
    lower = -np.pi * np.ones(runtime.exo_dof, dtype=np.float64)
    upper = np.pi * np.ones(runtime.exo_dof, dtype=np.float64)
    target_human_q = np.zeros(runtime.human_dof, dtype=np.float64)
    best_state: dict[str, object] | None = None
    best_exo_q = np.zeros(runtime.exo_dof, dtype=np.float64)
    best_success = False
    best_nfev = 0
    best_score = (1.0, 1e9, 1e9)

    def evaluate_state(exo_q: np.ndarray) -> dict[str, object]:
        zero_all_hand_qpos(work_data, qpos_adrs_map)
        set_finger_qpos(work_data, qpos_adrs_map, runtime.key, exo_q)
        mujoco.mj_forward(model, work_data)
        return evaluate_mapping_state(
            model,
            work_data,
            thumb_reference,
            runtime,
            human_angles_initial=target_human_q,
            plane_y_sign=plane_y_sign,
            flexion_sign=flexion_sign,
        )

    def residual(exo_q: np.ndarray) -> np.ndarray:
        try:
            state = evaluate_state(exo_q)
        except Exception:
            return np.full(runtime.human_dof + 3, 1e3, dtype=np.float64)

        mapped_human_q = np.asarray(state["human_angles"], dtype=np.float64)
        tip_error = float(state["tip_error"])
        reach_gap = max(0.0, float(state["target_distance"]) - float(state["max_reach"]))
        match_term = mapped_human_q - target_human_q
        tip_term = np.array([50.0 * tip_error], dtype=np.float64)
        reach_term = np.array([35.0 * reach_gap], dtype=np.float64)
        zero_bias = 0.02 * exo_q
        return np.concatenate([match_term, tip_term, reach_term, zero_bias])

    seed_list = [np.zeros(runtime.exo_dof, dtype=np.float64)]
    for _ in range(5):
        seed_list.append(rng.uniform(-0.45, 0.45, size=runtime.exo_dof))

    for seed in seed_list:
        result = optimize.least_squares(
            residual,
            x0=np.clip(seed, lower, upper),
            bounds=(lower, upper),
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10,
            max_nfev=120,
        )
        exo_q = np.asarray(result.x, dtype=np.float64)
        try:
            mapping_state = evaluate_state(exo_q)
        except Exception:
            mapping_state = None

        score = initialization_score(mapping_state, exo_q)
        if score < best_score:
            best_score = score
            best_state = mapping_state
            best_exo_q = exo_q.copy()
            best_success = bool(result.success and mapping_state is not None)
            best_nfev = int(result.nfev)

    return InitializationResult(
        exo_q=best_exo_q,
        mapping_state=best_state,
        success=best_success,
        solver_nfev=best_nfev,
        score=best_score,
    )


def initialize_runtime_alignments(
    model: mujoco.MjModel,
    work_data: mujoco.MjData,
    qpos_adrs_map: dict[str, list[int]],
    runtime_states: dict[str, FingerRuntimeState],
    thumb_reference,
    rng: np.random.Generator,
    *,
    plane_y_sign: int,
    flexion_sign: int,
) -> dict[str, InitializationResult]:
    results: dict[str, InitializationResult] = {}
    for finger_key in HAND_ORDER:
        runtime = runtime_states[finger_key]
        result = solve_initial_alignment(
            model,
            work_data,
            qpos_adrs_map,
            thumb_reference,
            runtime,
            rng,
            plane_y_sign=plane_y_sign,
            flexion_sign=flexion_sign,
        )
        results[finger_key] = result
        runtime.init_exo_q = result.exo_q.copy()
        runtime.prev_valid_exo_q = result.exo_q.copy()
        if result.mapping_state is not None:
            mapped_q = np.asarray(result.mapping_state["human_angles"], dtype=np.float64)
            runtime.init_mapped_q = mapped_q.copy()
            runtime.prev_valid_mapped_q = mapped_q.copy()
            runtime.target_anchor_q = mapped_q.copy()
        else:
            runtime.init_mapped_q = np.zeros(runtime.human_dof, dtype=np.float64)
            runtime.prev_valid_mapped_q = np.zeros(runtime.human_dof, dtype=np.float64)
            runtime.target_anchor_q = np.zeros(runtime.human_dof, dtype=np.float64)
    return results


def solve_inverse_exo(
    model: mujoco.MjModel,
    work_data: mujoco.MjData,
    qpos_adrs_map: dict[str, list[int]],
    thumb_reference,
    runtime: FingerRuntimeState,
    target_human_q: np.ndarray,
    *,
    plane_y_sign: int,
    flexion_sign: int,
) -> InverseSolveResult:
    target_human_q = np.asarray(target_human_q, dtype=np.float64)
    lower = -np.pi * np.ones(runtime.exo_dof, dtype=np.float64)
    upper = np.pi * np.ones(runtime.exo_dof, dtype=np.float64)
    initial = np.clip(runtime.prev_valid_exo_q, lower, upper)
    last_state: dict[str, object] | None = None

    def evaluate_state(exo_q: np.ndarray) -> dict[str, object]:
        zero_all_hand_qpos(work_data, qpos_adrs_map)
        set_finger_qpos(work_data, qpos_adrs_map, runtime.key, exo_q)
        mujoco.mj_forward(model, work_data)
        return evaluate_mapping_state(
            model,
            work_data,
            thumb_reference,
            runtime,
            human_angles_initial=runtime.prev_valid_mapped_q,
            plane_y_sign=plane_y_sign,
            flexion_sign=flexion_sign,
        )

    def residual(exo_q: np.ndarray) -> np.ndarray:
        nonlocal last_state
        try:
            state = evaluate_state(exo_q)
        except Exception:
            penalty = np.full(runtime.human_dof + runtime.exo_dof + runtime.exo_dof + 2, 1e3, dtype=np.float64)
            return penalty

        last_state = state
        mapped_human_q = np.asarray(state["human_angles"], dtype=np.float64)
        tip_error = float(state["tip_error"])
        reach_gap = max(0.0, float(state["target_distance"]) - float(state["max_reach"]))
        match_term = mapped_human_q - target_human_q
        smooth_term = 0.08 * (exo_q - runtime.prev_valid_exo_q)
        norm_term = 0.01 * exo_q
        tip_term = np.array([45.0 * tip_error], dtype=np.float64)
        reach_term = np.array([30.0 * reach_gap], dtype=np.float64)
        return np.concatenate([match_term, smooth_term, norm_term, tip_term, reach_term])

    result = optimize.least_squares(
        residual,
        x0=initial,
        bounds=(lower, upper),
        xtol=1e-10,
        ftol=1e-10,
        gtol=1e-10,
        max_nfev=150,
    )

    exo_q = np.asarray(result.x, dtype=np.float64)
    mapping_state: dict[str, object] | None = None
    try:
        mapping_state = evaluate_state(exo_q)
        last_state = mapping_state
    except Exception:
        mapping_state = last_state

    solver_success = bool(result.success and np.all(np.isfinite(exo_q)) and mapping_state is not None)
    return InverseSolveResult(
        exo_q=exo_q,
        mapping_state=mapping_state,
        solver_success=solver_success,
        solver_nfev=int(result.nfev),
    )


def pair_key(body_a: str, body_b: str) -> tuple[str, str]:
    return tuple(sorted((body_a, body_b)))


def collect_right_hand_body_names(model: mujoco.MjModel) -> dict[int, str]:
    body_names: dict[int, str] = {}
    for body_id in range(model.nbody):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        if body_name.startswith(RIGHT_BODY_PREFIX):
            body_names[body_id] = body_name
    return body_names


def build_adjacent_body_pairs(model: mujoco.MjModel, right_body_names: dict[int, str]) -> set[tuple[str, str]]:
    adjacent_pairs: set[tuple[str, str]] = set()
    for body_id, body_name in right_body_names.items():
        parent_id = int(model.body_parentid[body_id])
        parent_name = right_body_names.get(parent_id)
        if parent_name:
            adjacent_pairs.add(pair_key(body_name, parent_name))
    return adjacent_pairs


def collect_contact_pairs(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    right_body_names: dict[int, str],
    adjacent_pairs: set[tuple[str, str]],
) -> dict[tuple[str, str], float]:
    pair_to_min_dist: dict[tuple[str, str], float] = {}
    for contact_idx in range(data.ncon):
        contact = data.contact[contact_idx]
        body_a = int(model.geom_bodyid[int(contact.geom1)])
        body_b = int(model.geom_bodyid[int(contact.geom2)])
        name_a = right_body_names.get(body_a)
        name_b = right_body_names.get(body_b)
        if not name_a or not name_b:
            continue
        if name_a == name_b:
            continue
        key = pair_key(name_a, name_b)
        if key in adjacent_pairs:
            continue
        distance = float(contact.dist)
        if key not in pair_to_min_dist or distance < pair_to_min_dist[key]:
            pair_to_min_dist[key] = distance
    return pair_to_min_dist


def format_contact_details(pair_to_dist: dict[tuple[str, str], float]) -> list[dict[str, object]]:
    details: list[dict[str, object]] = []
    for body_pair in sorted(pair_to_dist):
        details.append(
            {
                "body_a": body_pair[0],
                "body_b": body_pair[1],
                "dist_mm": pair_to_dist[body_pair] * 1000.0,
            }
        )
    return details


def build_collision_baseline(model: mujoco.MjModel, data: mujoco.MjData, qpos_adrs_map: dict[str, list[int]]) -> CollisionBaseline:
    zero_all_hand_qpos(data, qpos_adrs_map)
    mujoco.mj_forward(model, data)
    right_body_names = collect_right_hand_body_names(model)
    adjacent_pairs = build_adjacent_body_pairs(model, right_body_names)
    pair_to_dist = collect_contact_pairs(model, data, right_body_names, adjacent_pairs)
    return CollisionBaseline(
        pair_to_dist=pair_to_dist,
        details=format_contact_details(pair_to_dist),
        adjacent_pairs=adjacent_pairs,
    )


def evaluate_collision_against_baseline(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    baseline: CollisionBaseline,
) -> CollisionCheck:
    right_body_names = collect_right_hand_body_names(model)
    current_pairs = collect_contact_pairs(model, data, right_body_names, baseline.adjacent_pairs)

    new_pairs: list[tuple[str, str]] = []
    deeper_pairs: list[tuple[str, str]] = []
    details: list[dict[str, object]] = []

    for body_pair in sorted(current_pairs):
        current_dist = current_pairs[body_pair]
        baseline_dist = baseline.pair_to_dist.get(body_pair)
        entry = {
            "body_a": body_pair[0],
            "body_b": body_pair[1],
            "dist_mm": current_dist * 1000.0,
            "baseline_dist_mm": (baseline_dist * 1000.0) if baseline_dist is not None else None,
        }

        if baseline_dist is None:
            if current_dist < -COLLISION_EPS_M:
                new_pairs.append(body_pair)
                entry["classification"] = REJECT_REASON_COLLISION_NEW
            else:
                entry["classification"] = "new_pair_but_not_penetrating"
        else:
            if current_dist < baseline_dist - COLLISION_EPS_M:
                deeper_pairs.append(body_pair)
                entry["classification"] = REJECT_REASON_COLLISION_DEEPER
            else:
                entry["classification"] = "within_baseline"

        details.append(entry)

    return CollisionCheck(new_pairs=new_pairs, deeper_pairs=deeper_pairs, contact_details=details)


def pad_vector(values: np.ndarray | list[float] | tuple[float, ...], length: int = 5) -> list[float]:
    padded = [math.nan] * length
    for index, value in enumerate(np.asarray(values, dtype=np.float64).tolist()):
        if index >= length:
            break
        padded[index] = float(value)
    return padded


def build_common_row(
    sample_id: int,
    finger_key: str,
    trajectory_family: str,
    trajectory_id: int,
    step_idx: int,
    target_human_q: np.ndarray,
    inverse_result: InverseSolveResult,
    collision_check: CollisionCheck | None,
) -> dict[str, object]:
    state = inverse_result.mapping_state
    mapped_human_q = np.asarray(state["human_angles"], dtype=np.float64) if state is not None else np.full(0, np.nan)
    target_q_row = pad_vector(target_human_q)
    mapped_q_row = pad_vector(mapped_human_q)
    exo_q_row = pad_vector(inverse_result.exo_q)

    row = {
        "sample_id": sample_id,
        "finger": finger_key,
        "trajectory_family": trajectory_family,
        "trajectory_id": trajectory_id,
        "step_idx": step_idx,
        "target_human_q1": target_q_row[0],
        "target_human_q2": target_q_row[1],
        "target_human_q3": target_q_row[2],
        "target_human_q4": target_q_row[3],
        "target_human_q5": target_q_row[4],
        "mapped_human_q1": mapped_q_row[0],
        "mapped_human_q2": mapped_q_row[1],
        "mapped_human_q3": mapped_q_row[2],
        "mapped_human_q4": mapped_q_row[3],
        "mapped_human_q5": mapped_q_row[4],
        "exo_q1": exo_q_row[0],
        "exo_q2": exo_q_row[1],
        "exo_q3": exo_q_row[2],
        "exo_q4": exo_q_row[3],
        "exo_q5": exo_q_row[4],
        "tip_error_mm": float(state["tip_error"]) * 1000.0 if state is not None else math.nan,
        "is_reachable": bool(state["is_reachable"]) if state is not None else False,
        "target_distance_mm": float(state["target_distance"]) * 1000.0 if state is not None else math.nan,
        "max_reach_mm": float(state["max_reach"]) * 1000.0 if state is not None else math.nan,
        "collision_new_pair_count": len(collision_check.new_pairs) if collision_check is not None else 0,
        "collision_deeper_pair_count": len(collision_check.deeper_pairs) if collision_check is not None else 0,
    }
    return row


def sanitize_for_json(value):
    if isinstance(value, dict):
        return {str(key): sanitize_for_json(sub_value) for key, sub_value in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return [sanitize_for_json(item) for item in value.tolist()]
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        if math.isnan(numeric) or math.isinf(numeric):
            return None
        return numeric
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def append_target_chain_overlay(
    scene: mujoco.MjvScene,
    current_state: dict[str, object],
    target_human_q: np.ndarray,
    runtime: FingerRuntimeState,
    thumb_reference,
) -> None:
    target_human_q = np.asarray(target_human_q, dtype=np.float64)
    if runtime.key == THUMB_KEY:
        target_points = build_thumb_fk(thumb_reference, target_human_q, runtime.human_lengths, runtime.base_offset)
    else:
        human_points_2d = build_human_polyline(runtime.base_offset, runtime.human_lengths, tuple(target_human_q))
        target_points = lift_points_to_3d(
            human_points_2d,
            np.asarray(current_state["plane_origin"], dtype=np.float64),
            np.asarray(current_state["plane_x"], dtype=np.float64),
            np.asarray(current_state["plane_y"], dtype=np.float64),
        )

    for point in target_points:
        add_sphere(scene, point, radius=0.0042, rgba=TARGET_JOINT_RGBA)
    for start, end in zip(target_points[:-1], target_points[1:]):
        add_connector(
            scene,
            start,
            end,
            width=0.0022,
            rgba=TARGET_LINK_RGBA,
            geom_type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        )
    add_sphere(scene, target_points[-1], radius=0.0050, rgba=TARGET_TIP_RGBA)


def make_status_text(
    *,
    finger_key: str,
    trajectory_family: str,
    trajectory_id: int,
    step_idx: int,
    accepted: dict[str, int],
    rejected: dict[str, int],
    target_human_q: np.ndarray,
    inverse_result: InverseSolveResult,
    collision_check: CollisionCheck | None,
    reject_reasons: list[str],
    attempted: dict[str, int],
) -> tuple[str, str]:
    state = inverse_result.mapping_state
    mapped_human_q = np.asarray(state["human_angles"], dtype=np.float64) if state is not None else np.full(0, np.nan)
    tip_error_mm = float(state["tip_error"]) * 1000.0 if state is not None else math.nan
    exo_q = np.asarray(inverse_result.exo_q, dtype=np.float64)

    if collision_check is None:
        collision_text = "未计算"
    elif collision_check.new_pairs or collision_check.deeper_pairs:
        collision_text = f"新接触={len(collision_check.new_pairs)}，更深接触={len(collision_check.deeper_pairs)}"
    else:
        collision_text = "通过"

    accepted_total = ", ".join(f"{key}:{accepted[key]}" for key in HAND_ORDER)
    rejected_total = ", ".join(f"{key}:{rejected[key]}" for key in HAND_ORDER)
    attempted_total = ", ".join(f"{key}:{attempted[key]}" for key in HAND_ORDER)

    if reject_reasons:
        status_line = f"本条状态：拒绝 | 原因：{', '.join(reject_reasons)}"
        color = REJECT_COLOR
    else:
        status_line = "本条状态：接受"
        color = ACCEPT_COLOR

    lines = [
        f"当前手指：{finger_key}",
        f"当前动作族：{trajectory_family}",
        f"当前段号/帧号：{trajectory_id} / {step_idx}",
        f"已尝试：{attempted_total}",
        f"已接受：{accepted_total}",
        f"已拒绝：{rejected_total}",
        f"目标手部角(rad)：{', '.join(f'{value:.3f}' for value in np.asarray(target_human_q, dtype=np.float64))}",
        f"求得外骨骼角(rad)：{', '.join(f'{value:.3f}' for value in exo_q)}",
        f"映射后手部角(rad)：{', '.join(f'{value:.3f}' for value in mapped_human_q)}" if mapped_human_q.size else "映射后手部角(rad)：无",
        f"tip_error_mm：{tip_error_mm:.3f}" if not math.isnan(tip_error_mm) else "tip_error_mm：无",
        f"is_reachable：{bool(state['is_reachable']) if state is not None else False}",
        f"碰撞判定：{collision_text}",
        status_line,
    ]
    return "\n".join(lines), color


def maybe_build_preview(headless: bool):
    if headless:
        return None

    import matplotlib

    configure_chinese_font()
    matplotlib.use("TkAgg", force=True)
    import matplotlib.pyplot as plt
    import mujoco.viewer as viewer

    fig = plt.figure("手部驱动映射采集状态", figsize=(8.8, 7.2))
    fig.subplots_adjust(left=0.06, right=0.98, top=0.97, bottom=0.05)
    ax = fig.add_axes([0.03, 0.03, 0.94, 0.94])
    ax.axis("off")
    text_artist = ax.text(0.0, 1.0, "", va="top", ha="left", fontsize=10, color=NEUTRAL_COLOR)
    return {"plt": plt, "viewer": viewer, "fig": fig, "text_artist": text_artist}


def print_progress_summary(title: str, accepted: dict[str, int], rejected: dict[str, int], attempted: dict[str, int]) -> None:
    print(title)
    for key in HAND_ORDER:
        print(f"- {key}: accepted={accepted[key]}, rejected={rejected[key]}, attempted={attempted[key]}")


def refresh_preview(
    preview,
    model: mujoco.MjModel,
    view_data: mujoco.MjData,
    qpos_adrs_map: dict[str, list[int]],
    zero_preview_states: dict[str, dict[str, object]],
    thumb_reference,
    runtime_states: dict[str, FingerRuntimeState],
    current_finger: str,
    current_state: dict[str, object] | None,
    current_exo_q: np.ndarray,
    target_human_q: np.ndarray,
    *,
    trajectory_family: str,
    trajectory_id: int,
    step_idx: int,
    accepted: dict[str, int],
    rejected: dict[str, int],
    attempted: dict[str, int],
    inverse_result: InverseSolveResult,
    collision_check: CollisionCheck | None,
    reject_reasons: list[str],
    dt: float,
) -> bool:
    if preview is None:
        return True

    plt = preview["plt"]
    fig = preview["fig"]
    text_artist = preview["text_artist"]
    if not plt.fignum_exists(fig.number):
        return False

    zero_all_hand_qpos(view_data, qpos_adrs_map)
    set_finger_qpos(view_data, qpos_adrs_map, current_finger, current_exo_q)
    mujoco.mj_forward(model, view_data)

    status_text, status_color = make_status_text(
        finger_key=current_finger,
        trajectory_family=trajectory_family,
        trajectory_id=trajectory_id,
        step_idx=step_idx,
        accepted=accepted,
        rejected=rejected,
        attempted=attempted,
        target_human_q=target_human_q,
        inverse_result=inverse_result,
        collision_check=collision_check,
        reject_reasons=reject_reasons,
    )
    text_artist.set_text(status_text)
    text_artist.set_color(status_color)

    current_overlay_state = current_state if current_state is not None else zero_preview_states[current_finger]
    handle = preview["handle"]
    if not handle.is_running():
        return False

    with handle.lock():
        handle.user_scn.ngeom = 0
        if current_finger == THUMB_KEY:
            append_thumb_markers(model, view_data, handle.user_scn, axis_length=AXIS_LENGTH)
        else:
            append_selected_finger_markers(
                model,
                view_data,
                handle.user_scn,
                FINGER_CONFIGS[current_finger],
                axis_length=AXIS_LENGTH,
            )

        for finger_key in HAND_ORDER:
            state = current_overlay_state if finger_key == current_finger else zero_preview_states[finger_key]
            append_mapping_overlay(
                handle.user_scn,
                state,
                axis_length=AXIS_LENGTH,
                selected=(finger_key == current_finger),
            )

        append_target_chain_overlay(
            handle.user_scn,
            current_overlay_state,
            target_human_q,
            runtime_states[current_finger],
            thumb_reference,
        )

    handle.sync()
    fig.canvas.draw_idle()
    plt.pause(dt)
    return True


def initialize_zero_preview_states(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos_adrs_map: dict[str, list[int]],
    thumb_reference,
    runtime_states: dict[str, FingerRuntimeState],
    *,
    plane_y_sign: int,
    flexion_sign: int,
) -> dict[str, dict[str, object]]:
    zero_all_hand_qpos(data, qpos_adrs_map)
    mujoco.mj_forward(model, data)
    preview_states: dict[str, dict[str, object]] = {}
    for finger_key in HAND_ORDER:
        preview_states[finger_key] = evaluate_mapping_state(
            model,
            data,
            thumb_reference,
            runtime_states[finger_key],
            human_angles_initial=np.zeros(runtime_states[finger_key].human_dof, dtype=np.float64),
            plane_y_sign=plane_y_sign,
            flexion_sign=flexion_sign,
        )
    return preview_states


def write_rejected_record(
    file_obj,
    baseline: CollisionBaseline,
    common_row: dict[str, object],
    reject_reasons: list[str],
    collision_check: CollisionCheck | None,
    inverse_result: InverseSolveResult,
) -> None:
    payload = dict(common_row)
    payload.update(
        {
            "reject_reasons": reject_reasons,
            "contact_details": collision_check.contact_details if collision_check is not None else [],
            "baseline_contact_details": baseline.details,
            "solver_success": inverse_result.solver_success,
            "solver_nfev": inverse_result.solver_nfev,
        }
    )
    file_obj.write(json.dumps(sanitize_for_json(payload), ensure_ascii=False) + "\n")
    file_obj.flush()


def main() -> int:
    args = parse_args()
    plane_y_sign = 1
    flexion_sign = -1
    rng = np.random.default_rng(args.seed)

    if args.valid_per_finger <= 0:
        raise ValueError("valid_per_finger 必须大于 0。")
    if args.preview_every <= 0:
        raise ValueError("preview_every 必须大于 0。")
    if args.max_attempt_factor <= 0:
        raise ValueError("max_attempt_factor 必须大于 0。")

    finger_order = choose_finger_order(args.fingers)
    output_prefix = args.output_prefix
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    valid_csv_path = Path(f"{output_prefix}_valid.csv")
    rejected_jsonl_path = Path(f"{output_prefix}_rejected.jsonl")

    model, view_data, bundle_urdf = load_model_from_urdf(args.urdf, limit_overrides=build_limit_overrides())
    work_data = mujoco.MjData(model)
    zero_data = mujoco.MjData(model)
    hidden_geom_count = hide_left_hand_geoms(model)

    qpos_adrs_map = build_qpos_adrs_map(model)
    runtime_states = build_runtime_states(qpos_adrs_map)
    thumb_reference = make_thumb_reference(model)
    init_results = initialize_runtime_alignments(
        model,
        work_data,
        qpos_adrs_map,
        runtime_states,
        thumb_reference,
        rng,
        plane_y_sign=plane_y_sign,
        flexion_sign=flexion_sign,
    )
    zero_preview_states = initialize_zero_preview_states(
        model,
        zero_data,
        qpos_adrs_map,
        thumb_reference,
        runtime_states,
        plane_y_sign=plane_y_sign,
        flexion_sign=flexion_sign,
    )
    baseline = build_collision_baseline(model, zero_data, qpos_adrs_map)

    print(f"已加载模型：{bundle_urdf}")
    print(f"默认隐藏左手几何数量：{hidden_geom_count}")
    print(f"零位右手碰撞基线对数：{len(baseline.pair_to_dist)}")
    print("初始化对齐结果：")
    for finger_key in HAND_ORDER:
        init_result = init_results[finger_key]
        init_tip_error = (
            float(init_result.mapping_state["tip_error"]) * 1000.0 if init_result.mapping_state is not None else math.nan
        )
        init_reachable = bool(init_result.mapping_state["is_reachable"]) if init_result.mapping_state is not None else False
        print(
            f"- {finger_key}: success={init_result.success}, reachable={init_reachable}, "
            f"tip_error_mm={init_tip_error:.3f}, score={tuple(round(v, 6) for v in init_result.score)}"
        )

    preview = maybe_build_preview(args.headless)
    if preview is not None:
        with preview["viewer"].launch_passive(model, view_data, show_left_ui=True, show_right_ui=True) as handle:
            handle.opt.frame = mujoco.mjtFrame.mjFRAME_NONE
            preview["handle"] = handle
            return run_collection(
                args=args,
                model=model,
                view_data=view_data,
                work_data=work_data,
                qpos_adrs_map=qpos_adrs_map,
                runtime_states=runtime_states,
                thumb_reference=thumb_reference,
                zero_preview_states=zero_preview_states,
                baseline=baseline,
                finger_order=finger_order,
                valid_csv_path=valid_csv_path,
                rejected_jsonl_path=rejected_jsonl_path,
                preview=preview,
                plane_y_sign=plane_y_sign,
                flexion_sign=flexion_sign,
                rng=rng,
            )

    return run_collection(
        args=args,
        model=model,
        view_data=view_data,
        work_data=work_data,
        qpos_adrs_map=qpos_adrs_map,
        runtime_states=runtime_states,
        thumb_reference=thumb_reference,
        zero_preview_states=zero_preview_states,
        baseline=baseline,
        finger_order=finger_order,
        valid_csv_path=valid_csv_path,
        rejected_jsonl_path=rejected_jsonl_path,
        preview=None,
        plane_y_sign=plane_y_sign,
        flexion_sign=flexion_sign,
        rng=rng,
    )


def run_collection(
    *,
    args: argparse.Namespace,
    model: mujoco.MjModel,
    view_data: mujoco.MjData,
    work_data: mujoco.MjData,
    qpos_adrs_map: dict[str, list[int]],
    runtime_states: dict[str, FingerRuntimeState],
    thumb_reference,
    zero_preview_states: dict[str, dict[str, object]],
    baseline: CollisionBaseline,
    finger_order: list[str],
    valid_csv_path: Path,
    rejected_jsonl_path: Path,
    preview,
    plane_y_sign: int,
    flexion_sign: int,
    rng: np.random.Generator,
) -> int:
    accepted = {key: 0 for key in HAND_ORDER}
    rejected = {key: 0 for key in HAND_ORDER}
    attempted = {key: 0 for key in HAND_ORDER}
    sample_id = 0
    early_stop = False

    with valid_csv_path.open("w", newline="", encoding="utf-8") as valid_file, rejected_jsonl_path.open(
        "w", encoding="utf-8"
    ) as rejected_file:
        valid_writer = csv.DictWriter(valid_file, fieldnames=VALID_CSV_COLUMNS)
        valid_writer.writeheader()
        valid_file.flush()

        for finger_key in finger_order:
            runtime = runtime_states[finger_key]
            max_attempts = args.valid_per_finger * args.max_attempt_factor

            while accepted[finger_key] < args.valid_per_finger and attempted[finger_key] < max_attempts:
                if finger_key == THUMB_KEY:
                    trajectory_family, segment_targets = next_thumb_segment(rng, runtime)
                else:
                    trajectory_family, segment_targets = next_four_finger_segment(rng, runtime, flexion_sign)
                trajectory_id = runtime.segment_counter

                for step_idx, target_human_q in enumerate(segment_targets):
                    if accepted[finger_key] >= args.valid_per_finger or attempted[finger_key] >= max_attempts:
                        break

                    attempted[finger_key] += 1
                    sample_id += 1

                    inverse_result = solve_inverse_exo(
                        model,
                        work_data,
                        qpos_adrs_map,
                        thumb_reference,
                        runtime,
                        np.asarray(target_human_q, dtype=np.float64),
                        plane_y_sign=plane_y_sign,
                        flexion_sign=flexion_sign,
                    )

                    reject_reasons: list[str] = []
                    collision_check: CollisionCheck | None = None
                    current_state = inverse_result.mapping_state
                    if not inverse_result.solver_success:
                        reject_reasons.append(REJECT_REASON_INVERSE_FAILED)

                    if current_state is not None:
                        if not bool(current_state["is_reachable"]):
                            reject_reasons.append(REJECT_REASON_UNREACHABLE)
                        if float(current_state["tip_error"]) * 1000.0 > args.tip_error_mm:
                            reject_reasons.append(REJECT_REASON_TIP_ERROR)
                        collision_check = evaluate_collision_against_baseline(model, work_data, baseline)
                        if collision_check.new_pairs:
                            reject_reasons.append(REJECT_REASON_COLLISION_NEW)
                        if collision_check.deeper_pairs:
                            reject_reasons.append(REJECT_REASON_COLLISION_DEEPER)

                    common_row = build_common_row(
                        sample_id=sample_id,
                        finger_key=finger_key,
                        trajectory_family=trajectory_family,
                        trajectory_id=trajectory_id,
                        step_idx=step_idx,
                        target_human_q=np.asarray(target_human_q, dtype=np.float64),
                        inverse_result=inverse_result,
                        collision_check=collision_check,
                    )

                    if reject_reasons:
                        rejected[finger_key] += 1
                        write_rejected_record(
                            rejected_file,
                            baseline,
                            common_row,
                            reject_reasons,
                            collision_check,
                            inverse_result,
                        )
                    else:
                        accepted[finger_key] += 1
                        valid_writer.writerow(common_row)
                        valid_file.flush()
                        runtime.prev_valid_exo_q = inverse_result.exo_q.copy()
                        runtime.prev_valid_mapped_q = np.asarray(current_state["human_angles"], dtype=np.float64).copy()

                    need_refresh = (
                        preview is not None
                        and (attempted[finger_key] % args.preview_every == 0 or not reject_reasons or accepted[finger_key] == args.valid_per_finger)
                    )
                    if need_refresh:
                        keep_running = refresh_preview(
                            preview,
                            model,
                            view_data,
                            qpos_adrs_map,
                            zero_preview_states,
                            thumb_reference,
                            runtime_states,
                            finger_key,
                            current_state,
                            inverse_result.exo_q,
                            np.asarray(target_human_q, dtype=np.float64),
                            trajectory_family=trajectory_family,
                            trajectory_id=trajectory_id,
                            step_idx=step_idx,
                            accepted=accepted,
                            rejected=rejected,
                            attempted=attempted,
                            inverse_result=inverse_result,
                            collision_check=collision_check,
                            reject_reasons=reject_reasons,
                            dt=args.dt,
                        )
                        if not keep_running:
                            early_stop = True
                            break

                if early_stop:
                    break

            if early_stop:
                break

            if accepted[finger_key] < args.valid_per_finger and attempted[finger_key] >= max_attempts:
                sample_id += 1
                rejected[finger_key] += 1
                failed_row = {
                    column: math.nan for column in VALID_CSV_COLUMNS
                }
                failed_row.update(
                    {
                        "sample_id": sample_id,
                        "finger": finger_key,
                        "trajectory_family": "budget_exhausted",
                        "trajectory_id": runtime.segment_counter,
                        "step_idx": -1,
                        "is_reachable": False,
                        "collision_new_pair_count": 0,
                        "collision_deeper_pair_count": 0,
                    }
                )
                write_rejected_record(
                    rejected_file,
                    baseline,
                    failed_row,
                    [REJECT_REASON_ATTEMPT_BUDGET],
                    None,
                    InverseSolveResult(
                        exo_q=np.full(runtime.exo_dof, math.nan, dtype=np.float64),
                        mapping_state=None,
                        solver_success=False,
                        solver_nfev=0,
                    ),
                )
                print_progress_summary("达到尝试预算后中止：", accepted, rejected, attempted)
                return 1

    if early_stop:
        print_progress_summary("用户关闭预览窗口，已提前刷盘：", accepted, rejected, attempted)
        return 130

    print_progress_summary("采集完成：", accepted, rejected, attempted)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n用户中断，采集结束。")
        raise SystemExit(130)
