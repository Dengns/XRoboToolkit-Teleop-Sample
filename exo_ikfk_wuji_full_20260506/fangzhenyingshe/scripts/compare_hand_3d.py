#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""在 MuJoCo 3D viewer 中验证右手四指的同平面映射。

目标：
1. 在完整右手模型中同时叠加食指、中指、无名指、小指的人手三连杆映射。
2. 保留外骨骼四关节滑块，并通过单选按钮切换当前操作的手指。
3. 每根手指使用独立的长度、根部偏移与末端局部偏移参数。
4. 初始状态统一为“手掌伸直”：外骨骼四关节为 0，人手三连杆为 0。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from delivery_core.four_finger_mapping import (
    build_human_polyline,
    configure_chinese_font,
    four_finger_base_offset_from_mm,
    human_joint_bounds,
    mm_to_m,
    project_point,
    reachability_summary,
    recover_four_finger_base_input,
    solve_human_ik,
    target_tip_from_local_offset,
)
from delivery_core.sim_loader import DEFAULT_URDF, load_model_from_urdf
from delivery_core.thumb_mapping import (
    THUMB_JOINTS,
    THUMB_DOF,
    THUMB_EXO_DOF,
    ThumbReference,
    build_thumb_fk,
    format_thumb_summary,
    get_joint_id,
    make_thumb_reference,
    print_thumb_headless_report,
    qpos_for_thumb,
    solve_thumb_ik,
    thumb_tip_position,
    thumb_tip_target_from_local_offset,
)
from delivery_core.viewer_primitives import add_connector, add_frame_axes, add_sphere, joint_axis_in_world
from hand_mapping_params import (
    THUMB_MAPPING_PARAMS,
    default_baseline_summary_lines,
    get_four_finger_params,
)
from delivery_core.thumb_mapping import append_thumb_markers


TIP_LOCAL_OFFSET = np.array([-0.0026, 0.005, -0.027], dtype=np.float64)

PLANE_AXIS_X_COLOR = np.array([0.25, 0.95, 0.60, 1.0], dtype=np.float32)
PLANE_AXIS_Y_COLOR = np.array([0.25, 0.55, 1.00, 1.0], dtype=np.float32)
PLANE_NORMAL_COLOR = np.array([1.00, 0.95, 0.25, 1.0], dtype=np.float32)
PROJECTION_COLOR = np.array([0.80, 0.80, 0.80, 0.70], dtype=np.float32)
ANCHOR_COLOR = np.array([0.70, 0.40, 0.98, 1.0], dtype=np.float32)
TARGET_COLOR = np.array([1.00, 0.55, 0.10, 1.0], dtype=np.float32)
MARKER_POINT_COLOR = np.array([1.0, 0.9, 0.2, 1.0], dtype=np.float32)
MARKER_AXIS_COLOR = np.array([1.0, 0.6, 0.0, 1.0], dtype=np.float32)
TIP_MARKER_COLOR = np.array([1.0, 0.0, 1.0, 1.0], dtype=np.float32)
THUMB_HUMAN_JOINT_COLOR = np.array([1.00, 0.30, 0.30, 1.0], dtype=np.float32)
THUMB_HUMAN_LINK_COLOR = np.array([0.92, 0.24, 0.24, 1.0], dtype=np.float32)
THUMB_TARGET_COLOR = np.array([1.00, 0.58, 0.10, 1.0], dtype=np.float32)
THUMB_ANCHOR_COLOR = np.array([0.70, 0.40, 1.00, 1.0], dtype=np.float32)
THUMB_KEY = "thumb"
THUMB_LABEL = "拇指"
THUMB_DEFAULT_LENGTHS_MM = THUMB_MAPPING_PARAMS.lengths_mm
THUMB_DEFAULT_BASE_MM = THUMB_MAPPING_PARAMS.base_mm
THUMB_DEFAULT_TIP_MM = THUMB_MAPPING_PARAMS.tip_mm
THUMB_HUMAN_LABELS = ("展收", "侧摆", "中屈", "末屈")


@dataclass(frozen=True)
class FingerConfig:
    key: str
    label: str
    joints: tuple[str, str, str, str]
    tip_joint: str
    tip_parent_body: str
    default_lengths_mm: tuple[float, float, float]
    default_base_mm: tuple[float, float]
    default_tip_mm: tuple[float, float]
    exo_joint_color: tuple[float, float, float]
    exo_link_color: tuple[float, float, float]
    human_joint_color: tuple[float, float, float]
    human_link_color: tuple[float, float, float]


FINGER_CONFIGS: dict[str, FingerConfig] = {
    "index": FingerConfig(
        key="index",
        label="食指",
        joints=(
            "joint_RightSkeletonIndex1",
            "joint_RightSkeletonIndex2",
            "joint_RightSkeletonIndex3",
            "joint_RightSkeletonIndex4",
        ),
        tip_joint="right_index_tip_joint",
        tip_parent_body="link_RightSkeletonIndex4",
        default_lengths_mm=get_four_finger_params("index").lengths_mm,
        default_base_mm=get_four_finger_params("index").base_mm,
        default_tip_mm=get_four_finger_params("index").tip_mm,
        exo_joint_color=(0.20, 0.88, 0.98),
        exo_link_color=(0.10, 0.62, 0.98),
        human_joint_color=(0.98, 0.30, 0.30),
        human_link_color=(0.92, 0.42, 0.42),
    ),
    "middle": FingerConfig(
        key="middle",
        label="中指",
        joints=(
            "joint_RightSkeletonMiddle1",
            "joint_RightSkeletonMiddle2",
            "joint_RightSkeletonMiddle3",
            "joint_RightSkeletonMiddle4",
        ),
        tip_joint="right_middle_tip_joint",
        tip_parent_body="link_RightSkeletonMiddle4",
        default_lengths_mm=get_four_finger_params("middle").lengths_mm,
        default_base_mm=get_four_finger_params("middle").base_mm,
        default_tip_mm=get_four_finger_params("middle").tip_mm,
        exo_joint_color=(0.28, 0.92, 0.55),
        exo_link_color=(0.12, 0.72, 0.42),
        human_joint_color=(1.00, 0.62, 0.18),
        human_link_color=(0.98, 0.50, 0.22),
    ),
    "ring": FingerConfig(
        key="ring",
        label="无名指",
        joints=(
            "joint_RightSkeletonRing1",
            "joint_RightSkeletonRing2",
            "joint_RightSkeletonRing3",
            "joint_RightSkeletonRing4",
        ),
        tip_joint="right_ring_tip_joint",
        tip_parent_body="link_RightSkeletonRing4",
        default_lengths_mm=get_four_finger_params("ring").lengths_mm,
        default_base_mm=get_four_finger_params("ring").base_mm,
        default_tip_mm=get_four_finger_params("ring").tip_mm,
        exo_joint_color=(0.98, 0.82, 0.22),
        exo_link_color=(0.86, 0.66, 0.10),
        human_joint_color=(0.90, 0.34, 0.92),
        human_link_color=(0.78, 0.28, 0.80),
    ),
    "pinky": FingerConfig(
        key="pinky",
        label="小指",
        joints=(
            "joint_RightSkeletonPinky1",
            "joint_RightSkeletonPinky2",
            "joint_RightSkeletonPinky3",
            "joint_RightSkeletonPinky4",
        ),
        tip_joint="right_pinky_tip_joint",
        tip_parent_body="link_RightSkeletonPinky4",
        default_lengths_mm=get_four_finger_params("pinky").lengths_mm,
        default_base_mm=get_four_finger_params("pinky").base_mm,
        default_tip_mm=get_four_finger_params("pinky").tip_mm,
        exo_joint_color=(0.72, 0.72, 0.72),
        exo_link_color=(0.56, 0.56, 0.56),
        human_joint_color=(1.00, 0.22, 0.58),
        human_link_color=(0.92, 0.20, 0.52),
    ),
}
FINGER_ORDER = ("index", "middle", "ring", "pinky")
HAND_ORDER = (THUMB_KEY, *FINGER_ORDER)


def rgb_to_rgba(rgb: tuple[float, float, float], alpha: float) -> np.ndarray:
    return np.array([rgb[0], rgb[1], rgb[2], alpha], dtype=np.float32)


def lift_points_to_3d(
    points_2d: np.ndarray,
    origin: np.ndarray,
    plane_x: np.ndarray,
    plane_y: np.ndarray,
) -> np.ndarray:
    points_2d = np.asarray(points_2d, dtype=np.float64)
    return origin[None, :] + points_2d[:, [0]] * plane_x[None, :] + points_2d[:, [1]] * plane_y[None, :]


def build_limit_overrides() -> dict[str, tuple[float, float, float, float]]:
    overrides: dict[str, tuple[float, float, float, float]] = {}
    for cfg in FINGER_CONFIGS.values():
        for joint_name in cfg.joints:
            overrides[joint_name] = (-3.14, 3.14, 100.0, 1.0)
    return overrides


def get_finger_landmarks(model: mujoco.MjModel, data: mujoco.MjData, cfg: FingerConfig) -> dict[str, np.ndarray]:
    landmarks: dict[str, np.ndarray] = {}
    for joint_name in cfg.joints:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise RuntimeError(f"模型中缺少关节：{joint_name}")
        landmarks[joint_name] = np.asarray(data.xanchor[joint_id], dtype=np.float64)

    tip_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cfg.tip_parent_body)
    if tip_body_id < 0:
        raise RuntimeError(f"模型中缺少末端父 body：{cfg.tip_parent_body}")
    tip_body_pos = np.asarray(data.xpos[tip_body_id], dtype=np.float64)
    tip_rot = data.xmat[tip_body_id].reshape(3, 3)
    landmarks[cfg.tip_joint] = tip_body_pos + tip_rot @ TIP_LOCAL_OFFSET
    return landmarks


def compute_finger_plane_frame(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    cfg: FingerConfig,
    landmarks: dict[str, np.ndarray],
    plane_y_sign: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if plane_y_sign not in (-1, 1):
        raise ValueError("plane_y_sign 只能是 -1 或 1")

    origin = landmarks[cfg.joints[0]]
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, cfg.joints[0])
    if joint_id < 0:
        raise RuntimeError(f"模型中不存在根部关节：{cfg.joints[0]}")
    body_id = int(model.jnt_bodyid[joint_id])
    rot = data.xmat[body_id].reshape(3, 3)

    plane_x = rot[:, 1].astype(np.float64)
    plane_y = plane_y_sign * rot[:, 2].astype(np.float64)
    plane_x /= np.linalg.norm(plane_x)
    plane_y /= np.linalg.norm(plane_y)
    return origin, plane_x, plane_y


def build_exo_projection(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    cfg: FingerConfig,
    plane_y_sign: int,
) -> dict[str, object]:
    landmarks = get_finger_landmarks(model, data, cfg)
    plane_origin, plane_x, plane_y = compute_finger_plane_frame(
        model,
        data,
        cfg,
        landmarks,
        plane_y_sign=plane_y_sign,
    )
    plane_normal = np.cross(plane_x, plane_y)
    plane_normal /= np.linalg.norm(plane_normal)

    ordered_names = [*cfg.joints, cfg.tip_joint]
    exo_points_3d = np.vstack([landmarks[name] for name in ordered_names])
    exo_points_2d = np.vstack([project_point(landmarks[name], plane_origin, plane_x, plane_y) for name in ordered_names])
    exo_points_projected_3d = lift_points_to_3d(exo_points_2d, plane_origin, plane_x, plane_y)
    exo_plane_distances = (exo_points_3d - plane_origin[None, :]) @ plane_normal

    return {
        "finger_key": cfg.key,
        "finger_label": cfg.label,
        "ordered_names": ordered_names,
        "plane_origin": plane_origin,
        "plane_x": plane_x,
        "plane_y": plane_y,
        "plane_normal": plane_normal,
        "exo_points_3d": exo_points_3d,
        "exo_points_2d": exo_points_2d,
        "exo_points_projected_3d": exo_points_projected_3d,
        "exo_plane_distances": exo_plane_distances,
    }


def build_mapping_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    cfg: FingerConfig,
    *,
    plane_y_sign: int,
    human_lengths: tuple[float, float, float],
    human_angles_initial: tuple[float, float, float],
    base_offset: np.ndarray,
    tip_offset_local: np.ndarray,
    ik_enabled: bool,
    flexion_sign: int,
) -> dict[str, object]:
    projection = build_exo_projection(model, data, cfg, plane_y_sign=plane_y_sign)

    plane_origin = np.asarray(projection["plane_origin"], dtype=np.float64)
    plane_x = np.asarray(projection["plane_x"], dtype=np.float64)
    plane_y = np.asarray(projection["plane_y"], dtype=np.float64)
    plane_normal = np.asarray(projection["plane_normal"], dtype=np.float64)
    exo_points_2d = np.asarray(projection["exo_points_2d"], dtype=np.float64)
    exo_points_3d = np.asarray(projection["exo_points_3d"], dtype=np.float64)
    exo_points_projected_3d = np.asarray(projection["exo_points_projected_3d"], dtype=np.float64)

    base_offset = np.asarray(base_offset, dtype=np.float64)
    tip_offset_local = np.asarray(tip_offset_local, dtype=np.float64)

    human_target_tip_2d = target_tip_from_local_offset(exo_points_2d, tip_offset_local)
    human_target_tip_3d = lift_points_to_3d(human_target_tip_2d[None, :], plane_origin, plane_x, plane_y)[0]

    if ik_enabled:
        solved_angles, tip_error = solve_human_ik(
            target_tip_xy=human_target_tip_2d,
            lengths=human_lengths,
            base_xy=base_offset,
            initial_angles=human_angles_initial,
            flexion_sign=flexion_sign,
        )
        human_angles = (float(solved_angles[0]), float(solved_angles[1]), float(solved_angles[2]))
    else:
        human_angles = human_angles_initial
        tip_error = float(
            np.linalg.norm(build_human_polyline(base_offset, human_lengths, human_angles)[-1] - human_target_tip_2d)
        )

    human_points_2d = build_human_polyline(base_offset, human_lengths, human_angles)
    human_points_3d = lift_points_to_3d(human_points_2d, plane_origin, plane_x, plane_y)
    human_plane_distances = (human_points_3d - plane_origin[None, :]) @ plane_normal
    target_distance, max_reach, is_reachable = reachability_summary(human_target_tip_2d, human_points_2d[0], human_lengths)

    projection.update(
        {
            "human_lengths": human_lengths,
            "human_angles": human_angles,
            "base_offset": base_offset,
            "tip_offset_local": tip_offset_local,
            "human_points_2d": human_points_2d,
            "human_points_3d": human_points_3d,
            "human_target_tip_2d": human_target_tip_2d,
            "human_target_tip_3d": human_target_tip_3d,
            "human_plane_distances": human_plane_distances,
            "tip_error": tip_error,
            "target_distance": target_distance,
            "max_reach": max_reach,
            "is_reachable": is_reachable,
        }
    )
    return projection


def build_thumb_mapping_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    reference: ThumbReference,
    *,
    human_lengths: tuple[float, float, float],
    human_angles_initial: tuple[float, ...],
    base_offset: np.ndarray,
    tip_offset_local: np.ndarray,
    ik_enabled: bool,
) -> dict[str, object]:
    exo_q = qpos_for_thumb(model, data)
    exo_tip = thumb_tip_position(model, data)
    exo_anchor_points = [np.asarray(data.xanchor[get_joint_id(model, item.joint_name)], dtype=np.float64) for item in THUMB_JOINTS]
    exo_points = np.vstack([*exo_anchor_points, exo_tip])
    target_tip = thumb_tip_target_from_local_offset(model, data, reference, tip_offset_local)
    fk_points = build_thumb_fk(reference, exo_q, human_lengths, base_offset)
    fk_error = float(np.linalg.norm(fk_points[-1] - target_tip))

    if ik_enabled:
        solved_angles, tip_error = solve_thumb_ik(reference, target_tip, human_lengths, base_offset, human_angles_initial)
        human_angles = tuple(float(value) for value in solved_angles)
    else:
        human_angles = tuple(float(value) for value in human_angles_initial)
        tip_error = float(np.linalg.norm(build_thumb_fk(reference, np.asarray(human_angles), human_lengths, base_offset)[-1] - target_tip))

    human_points = build_thumb_fk(reference, np.asarray(human_angles), human_lengths, base_offset)
    target_distance, max_reach, is_reachable = reachability_summary(target_tip, human_points[0], human_lengths)

    return {
        "finger_key": THUMB_KEY,
        "finger_label": THUMB_LABEL,
        "exo_q": exo_q,
        "exo_tip": exo_tip,
        "ordered_names": [item.tag for item in THUMB_JOINTS] + ["TIP"],
        "exo_points_3d": exo_points,
        "human_lengths": human_lengths,
        "human_angles": human_angles,
        "base_offset": np.asarray(base_offset, dtype=np.float64),
        "tip_offset_local": np.asarray(tip_offset_local, dtype=np.float64),
        "fk_points": fk_points,
        "fk_error": fk_error,
        "ik_q": np.asarray(human_angles, dtype=np.float64),
        "human_points_3d": human_points,
        "ik_points": human_points,
        "ik_error": tip_error,
        "human_target_tip_3d": target_tip,
        "target_tip": target_tip,
        "tip_error": tip_error,
        "target_distance": target_distance,
        "max_reach": max_reach,
        "is_reachable": is_reachable,
    }


def visible_angle_count(finger_key: str) -> tuple[int, int]:
    if finger_key == THUMB_KEY:
        return THUMB_EXO_DOF, THUMB_DOF
    return 4, 3


def parse_args_v2() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="在 MuJoCo 3D viewer 中统一预览右手五指映射。")
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF, help="待加载的完整右手 URDF 路径。")
    parser.add_argument("--finger", choices=HAND_ORDER, default=THUMB_KEY, help="初始选中的手指。")
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
    parser.add_argument("--no-ik", action="store_false", dest="ik", help="关闭 IK，改为手动调节当前手指人手角度。")
    parser.set_defaults(ik=True)
    parser.add_argument("--headless", action="store_true", help="只打印五指映射报告，不打开 viewer。")
    parser.add_argument("--axis-length", type=float, default=0.03, help="三维叠加坐标轴与箭头长度，单位米。")
    parser.add_argument("--dt", type=float, default=0.02, help="viewer 与控制窗口刷新时间步长。")
    return parser.parse_args()


def add_polyline_overlay(
    scene: mujoco.MjvScene,
    points: np.ndarray,
    *,
    joint_rgba: np.ndarray,
    link_rgba: np.ndarray,
    joint_radius: float,
    link_width: float,
) -> None:
    for point in points:
        add_sphere(scene, point, radius=joint_radius, rgba=joint_rgba)
    for start, end in zip(points[:-1], points[1:]):
        add_connector(
            scene,
            start,
            end,
            width=link_width,
            rgba=link_rgba,
            geom_type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        )


def append_selected_finger_markers(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    scene: mujoco.MjvScene,
    cfg: FingerConfig,
    axis_length: float,
) -> None:
    for joint_name in cfg.joints:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            continue
        body_id = int(model.jnt_bodyid[joint_id])
        joint_pos = np.asarray(data.xanchor[joint_id], dtype=np.float64)
        body_rot = data.xmat[body_id].reshape(3, 3)
        axis_world = joint_axis_in_world(model, data, joint_name)

        add_sphere(scene, joint_pos, radius=0.006, rgba=MARKER_POINT_COLOR)
        add_frame_axes(scene, joint_pos, body_rot, axis_length=axis_length, width=0.004)
        add_connector(
            scene,
            joint_pos,
            joint_pos + axis_world * axis_length * 1.25,
            width=0.006,
            rgba=MARKER_AXIS_COLOR,
            geom_type=mujoco.mjtGeom.mjGEOM_ARROW,
        )

    tip_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cfg.tip_parent_body)
    if tip_body_id >= 0:
        tip_body_pos = np.asarray(data.xpos[tip_body_id], dtype=np.float64)
        tip_rot = data.xmat[tip_body_id].reshape(3, 3)
        tip_pos = tip_body_pos + tip_rot @ TIP_LOCAL_OFFSET
        add_sphere(scene, tip_pos, radius=0.008, rgba=TIP_MARKER_COLOR)
        add_frame_axes(scene, tip_pos, tip_rot, axis_length=axis_length * 0.75, width=0.003)


def append_mapping_overlay(
    scene: mujoco.MjvScene,
    state: dict[str, object],
    *,
    axis_length: float,
    selected: bool,
) -> None:
    if state["finger_key"] == THUMB_KEY:
        human_points_3d = np.asarray(state["human_points_3d"], dtype=np.float64)
        target_tip = np.asarray(state["target_tip"], dtype=np.float64)
        alpha = 1.0 if selected else 0.38
        joint_rgba = np.array([THUMB_HUMAN_JOINT_COLOR[0], THUMB_HUMAN_JOINT_COLOR[1], THUMB_HUMAN_JOINT_COLOR[2], alpha], dtype=np.float32)
        link_rgba = np.array([THUMB_HUMAN_LINK_COLOR[0], THUMB_HUMAN_LINK_COLOR[1], THUMB_HUMAN_LINK_COLOR[2], alpha], dtype=np.float32)
        joint_radius = 0.0052 if selected else 0.0036
        link_width = 0.0034 if selected else 0.0020
        add_polyline_overlay(
            scene,
            human_points_3d,
            joint_rgba=joint_rgba,
            link_rgba=link_rgba,
            joint_radius=joint_radius,
            link_width=link_width,
        )
        add_sphere(
            scene,
            target_tip,
            radius=0.0062 if selected else 0.0040,
            rgba=np.array([THUMB_TARGET_COLOR[0], THUMB_TARGET_COLOR[1], THUMB_TARGET_COLOR[2], alpha], dtype=np.float32),
        )
        if selected:
            add_connector(
                scene,
                human_points_3d[0],
                target_tip,
                width=0.0018,
                rgba=THUMB_ANCHOR_COLOR,
                geom_type=mujoco.mjtGeom.mjGEOM_ARROW,
            )
        return

    cfg = FINGER_CONFIGS[state["finger_key"]]
    plane_origin = np.asarray(state["plane_origin"], dtype=np.float64)
    plane_x = np.asarray(state["plane_x"], dtype=np.float64)
    plane_y = np.asarray(state["plane_y"], dtype=np.float64)
    plane_normal = np.asarray(state["plane_normal"], dtype=np.float64)
    exo_points_3d = np.asarray(state["exo_points_3d"], dtype=np.float64)
    exo_points_projected_3d = np.asarray(state["exo_points_projected_3d"], dtype=np.float64)
    human_points_3d = np.asarray(state["human_points_3d"], dtype=np.float64)
    human_target_tip_3d = np.asarray(state["human_target_tip_3d"], dtype=np.float64)

    alpha = 1.0 if selected else 0.38
    joint_radius = 0.0048 if selected else 0.0032
    link_width = 0.0030 if selected else 0.0018
    exo_joint_rgba = rgb_to_rgba(cfg.exo_joint_color, alpha)
    exo_link_rgba = rgb_to_rgba(cfg.exo_link_color, alpha)
    human_joint_rgba = rgb_to_rgba(cfg.human_joint_color, alpha)
    human_link_rgba = rgb_to_rgba(cfg.human_link_color, alpha)

    if selected:
        add_connector(scene, plane_origin, plane_origin + plane_x * axis_length, 0.004, PLANE_AXIS_X_COLOR)
        add_connector(scene, plane_origin, plane_origin + plane_y * axis_length, 0.004, PLANE_AXIS_Y_COLOR)
        add_connector(
            scene,
            plane_origin,
            plane_origin + plane_normal * axis_length * 0.7,
            0.005,
            PLANE_NORMAL_COLOR,
            geom_type=mujoco.mjtGeom.mjGEOM_ARROW,
        )

    add_polyline_overlay(
        scene,
        exo_points_projected_3d,
        joint_rgba=exo_joint_rgba,
        link_rgba=exo_link_rgba,
        joint_radius=joint_radius,
        link_width=link_width,
    )
    add_polyline_overlay(
        scene,
        human_points_3d,
        joint_rgba=human_joint_rgba,
        link_rgba=human_link_rgba,
        joint_radius=joint_radius,
        link_width=link_width + 0.0004,
    )

    if selected:
        for actual_point, projected_point in zip(exo_points_3d[1:], exo_points_projected_3d[1:]):
            add_connector(
                scene,
                actual_point,
                projected_point,
                width=0.0013,
                rgba=PROJECTION_COLOR,
                geom_type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            )
        add_connector(
            scene,
            exo_points_projected_3d[0],
            human_points_3d[0],
            width=0.0022,
            rgba=ANCHOR_COLOR,
            geom_type=mujoco.mjtGeom.mjGEOM_ARROW,
        )
        add_sphere(scene, human_target_tip_3d, radius=0.0055, rgba=TARGET_COLOR)
        add_connector(
            scene,
            exo_points_projected_3d[-1],
            human_target_tip_3d,
            width=0.0022,
            rgba=TARGET_COLOR,
            geom_type=mujoco.mjtGeom.mjGEOM_ARROW,
        )
    else:
        add_sphere(scene, human_target_tip_3d, radius=0.0035, rgba=rgb_to_rgba((1.0, 0.65, 0.22), 0.32))


def format_summary(
    state: dict[str, object],
    *,
    ik_enabled: bool,
    flexion_sign: int,
    plane_y_sign: int,
) -> str:
    if state["finger_key"] == THUMB_KEY:
        return format_thumb_summary(
            state,
            lengths=tuple(np.asarray(state["human_lengths"], dtype=np.float64)),
            base_offset=np.asarray(state["base_offset"], dtype=np.float64),
            tip_offset_local=np.asarray(state["tip_offset_local"], dtype=np.float64),
            use_ik=ik_enabled,
        )

    cfg = FINGER_CONFIGS[state["finger_key"]]
    human_lengths = np.asarray(state["human_lengths"], dtype=np.float64)
    base_offset = np.asarray(state["base_offset"], dtype=np.float64)
    display_base_offset = recover_four_finger_base_input(base_offset)
    tip_offset_local = np.asarray(state["tip_offset_local"], dtype=np.float64)
    human_angles = np.asarray(state["human_angles"], dtype=np.float64)
    exo_plane_distances = np.asarray(state["exo_plane_distances"], dtype=np.float64)
    human_plane_distances = np.asarray(state["human_plane_distances"], dtype=np.float64)
    bounds_lower, bounds_upper = human_joint_bounds(flexion_sign)

    return "\n".join(
        [
            f"当前手指: {cfg.label}",
            f"IK: {'开' if ik_enabled else '关'}",
            f"向下屈曲方向: {'正向' if flexion_sign > 0 else '反向'}",
            f"平面纵轴: {'+z' if plane_y_sign > 0 else '-z'}",
            f"长度(mm): L1={human_lengths[0]*1000:.1f} L2={human_lengths[1]*1000:.1f} L3={human_lengths[2]*1000:.1f}",
            "四指 base 输入语义: (x, y) -> 内部 (y, -x)",
            f"基座偏移(mm): dx={display_base_offset[0]*1000:.1f} dy={display_base_offset[1]*1000:.1f}",
            f"末端局部偏移(mm): dx={tip_offset_local[0]*1000:.1f} dy={tip_offset_local[1]*1000:.1f}",
            f"人手角度(rad): q1={human_angles[0]:.3f} q2={human_angles[1]:.3f} q3={human_angles[2]:.3f}",
            "限位(deg): "
            f"MCP[{np.degrees(bounds_lower[0]):.0f},{np.degrees(bounds_upper[0]):.0f}] "
            f"PIP[{np.degrees(bounds_lower[1]):.0f},{np.degrees(bounds_upper[1]):.0f}] "
            f"DIP[{np.degrees(bounds_lower[2]):.0f},{np.degrees(bounds_upper[2]):.0f}]",
            f"目标距离/最大可达(mm): {float(state['target_distance'])*1000:.1f} / {float(state['max_reach'])*1000:.1f}",
            f"可达性: {'可达' if bool(state['is_reachable']) else '不可达，IK 会尽量伸直'}",
            f"末端误差(mm): {float(state['tip_error'])*1000:.2f}",
            f"外骨骼离平面最大偏差(mm): {np.max(np.abs(exo_plane_distances))*1000:.2f}",
            f"人手离平面最大偏差(mm): {np.max(np.abs(human_plane_distances))*1000:.6f}",
            "",
            "当前默认基线:",
            *default_baseline_summary_lines(),
        ]
    )


def print_headless_report(
    state: dict[str, object],
    *,
    ik_enabled: bool,
    flexion_sign: int,
    plane_y_sign: int,
) -> None:
    if state["finger_key"] == THUMB_KEY:
        print("=== 拇指 ===")
        print_thumb_headless_report(
            state,
            lengths=tuple(np.asarray(state["human_lengths"], dtype=np.float64)),
            base_offset=np.asarray(state["base_offset"], dtype=np.float64),
            tip_offset_local=np.asarray(state["tip_offset_local"], dtype=np.float64),
            use_ik=ik_enabled,
        )
        print()
        return

    cfg = FINGER_CONFIGS[state["finger_key"]]
    ordered_names = list(state["ordered_names"])
    exo_points_3d = np.asarray(state["exo_points_3d"], dtype=np.float64)
    exo_points_projected_3d = np.asarray(state["exo_points_projected_3d"], dtype=np.float64)
    human_points_3d = np.asarray(state["human_points_3d"], dtype=np.float64)
    human_target_tip_3d = np.asarray(state["human_target_tip_3d"], dtype=np.float64)
    exo_plane_distances = np.asarray(state["exo_plane_distances"], dtype=np.float64)
    human_plane_distances = np.asarray(state["human_plane_distances"], dtype=np.float64)

    print(f"=== {cfg.label} ===")
    for name, actual_point, projected_point, plane_distance in zip(
        ordered_names,
        exo_points_3d,
        exo_points_projected_3d,
        exo_plane_distances,
    ):
        print(
            f"- 外骨骼 {name}: 实际=({actual_point[0]:.5f}, {actual_point[1]:.5f}, {actual_point[2]:.5f}), "
            f"投影=({projected_point[0]:.5f}, {projected_point[1]:.5f}, {projected_point[2]:.5f}), "
            f"离平面={plane_distance*1000:.3f} mm"
        )
    for idx, (point, plane_distance) in enumerate(zip(human_points_3d, human_plane_distances)):
        print(
            f"- 人手 p{idx}: ({point[0]:.5f}, {point[1]:.5f}, {point[2]:.5f}), "
            f"离平面={plane_distance*1000:.6f} mm"
        )
    print(
        f"- 人手目标末端: ({human_target_tip_3d[0]:.5f}, {human_target_tip_3d[1]:.5f}, {human_target_tip_3d[2]:.5f})"
    )
    print(format_summary(state, ik_enabled=ik_enabled, flexion_sign=flexion_sign, plane_y_sign=plane_y_sign))
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="在 MuJoCo 3D viewer 中验证右手四指的同平面映射。")
    parser.add_argument(
        "--urdf",
        type=Path,
        default=DEFAULT_URDF,
        help="待加载的完整右手 URDF 路径。默认使用 blender_human_skeleton_v4.urdf。",
    )
    parser.add_argument(
        "--finger",
        choices=FINGER_ORDER,
        default="index",
        help="初始选中的手指。默认 index。",
    )
    parser.add_argument(
        "--flexion-sign",
        type=int,
        choices=(-1, 1),
        default=-1,
        help="人手向下屈曲方向。-1 表示向下屈曲落在负角度侧，1 表示落在正角度侧。",
    )
    parser.add_argument(
        "--plane-y-sign",
        type=int,
        choices=(-1, 1),
        default=1,
        help="映射平面纵轴方向。1 使用根部关节子 link 的 +z 方向，-1 使用 -z 方向。",
    )
    parser.add_argument(
        "--no-ik",
        action="store_false",
        dest="ik",
        help="关闭 IK，改为手动调节当前选中手指的人手三连杆角度。",
    )
    parser.set_defaults(ik=True)
    parser.add_argument("--headless", action="store_true", help="只打印四指映射报告，不打开 viewer。")
    parser.add_argument("--axis-length", type=float, default=0.03, help="三维叠加坐标轴与箭头长度，单位米。")
    parser.add_argument("--dt", type=float, default=0.02, help="viewer 和控制窗口刷新时间步长。")
    return parser.parse_args()


def slider_viewer(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    qpos_adrs_map: dict[str, list[int]],
    finger_states: dict[str, dict[str, object]],
    initial_finger: str,
    ik_mode_init: bool,
    flexion_sign: int,
    plane_y_sign: int,
    axis_length: float,
    dt: float,
) -> None:
    try:
        import matplotlib

        configure_chinese_font()
        matplotlib.use("TkAgg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, RadioButtons, Slider, TextBox
        import mujoco.viewer as viewer
    except ImportError as exc:
        raise RuntimeError("缺少三维观察所需依赖，请检查 matplotlib 和 mujoco.viewer") from exc

    fig = plt.figure("三维全手映射观察器", figsize=(10.6, 8.8))
    fig.subplots_adjust(left=0.28, right=0.98, top=0.95, bottom=0.07)

    slider_specs = [
        ("exo_q1", -3.14, 3.14, [0.32, 0.88, 0.62, 0.03]),
        ("exo_q2", -3.14, 3.14, [0.32, 0.83, 0.62, 0.03]),
        ("exo_q3", -3.14, 3.14, [0.32, 0.78, 0.62, 0.03]),
        ("exo_q4", -3.14, 3.14, [0.32, 0.73, 0.62, 0.03]),
        ("human_q1", -3.14, 3.14, [0.32, 0.65, 0.62, 0.03]),
        ("human_q2", -3.14, 3.14, [0.32, 0.60, 0.62, 0.03]),
        ("human_q3", -3.14, 3.14, [0.32, 0.55, 0.62, 0.03]),
    ]
    input_specs = [
        ("L1_mm", [0.32, 0.45, 0.18, 0.04]),
        ("L2_mm", [0.32, 0.40, 0.18, 0.04]),
        ("L3_mm", [0.32, 0.35, 0.18, 0.04]),
        ("base_dx_mm", [0.32, 0.26, 0.18, 0.04]),
        ("base_dy_mm", [0.32, 0.21, 0.18, 0.04]),
        ("tip_dx_mm", [0.32, 0.12, 0.18, 0.04]),
        ("tip_dy_mm", [0.32, 0.07, 0.18, 0.04]),
    ]

    sliders: dict[str, Slider] = {}
    for label, vmin, vmax, rect in slider_specs:
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

    summary_text = fig.text(0.55, 0.48, "", va="top", ha="left", fontsize=9)
    selected_text = fig.text(0.04, 0.92, "", va="top", ha="left", fontsize=11)

    radio_ax = fig.add_axes([0.04, 0.66, 0.18, 0.18])
    radio = RadioButtons(radio_ax, [FINGER_CONFIGS[key].label for key in FINGER_ORDER], active=FINGER_ORDER.index(initial_finger))
    radio_ax.set_title("当前控制手指")

    reset_ax = fig.add_axes([0.52, 0.02, 0.12, 0.05])
    reset_button = Button(reset_ax, "全部伸直")
    toggle_ax = fig.add_axes([0.66, 0.02, 0.12, 0.05])
    toggle_button = Button(toggle_ax, f"IK: {'开' if ik_state['enabled'] else '关'}")

    label_to_key = {cfg.label: cfg.key for cfg in FINGER_CONFIGS.values()}

    def set_textbox_mm(name: str, value_mm: float) -> None:
        numeric_inputs[name] = float(value_mm)
        textboxes[name].set_val(f"{value_mm:.2f}")

    def sync_controls_from_selected() -> None:
        key = selected["key"]
        cfg = FINGER_CONFIGS[key]
        finger_state = finger_states[key]
        selected_text.set_text(f"当前手指：{cfg.label}\n滑块控制外骨骼四关节与当前手指的人手三关节。")

        for index, qpos_adr in enumerate(qpos_adrs_map[key], start=1):
            slider = sliders[f"exo_q{index}"]
            slider.label.set_text(f"exo_q{index} {cfg.label}{index}")
            slider.set_val(float(data.qpos[qpos_adr]))

        human_angles = finger_state["human_angles"]
        for index, angle in enumerate(human_angles, start=1):
            slider = sliders[f"human_q{index}"]
            slider.label.set_text(f"human_q{index} {cfg.label}")
            slider.set_val(float(angle))

        human_lengths = finger_state["human_lengths"]
        base_offset = np.asarray(finger_state["base_offset"], dtype=np.float64)
        display_base_offset = recover_four_finger_base_input(base_offset)
        tip_offset_local = finger_state["tip_offset_local"]
        set_textbox_mm("L1_mm", human_lengths[0] * 1000.0)
        set_textbox_mm("L2_mm", human_lengths[1] * 1000.0)
        set_textbox_mm("L3_mm", human_lengths[2] * 1000.0)
        set_textbox_mm("base_dx_mm", display_base_offset[0] * 1000.0)
        set_textbox_mm("base_dy_mm", display_base_offset[1] * 1000.0)
        set_textbox_mm("tip_dx_mm", tip_offset_local[0] * 1000.0)
        set_textbox_mm("tip_dy_mm", tip_offset_local[1] * 1000.0)

    def read_input_mm(label: str, *, positive: bool = False) -> float:
        try:
            value = float(textboxes[label].text.strip())
        except ValueError:
            value = numeric_inputs[label]
        if positive:
            value = max(value, 0.1)
        numeric_inputs[label] = value
        return value

    def reset_all(_event) -> None:
        for key, cfg in FINGER_CONFIGS.items():
            for qpos_adr in qpos_adrs_map[key]:
                data.qpos[qpos_adr] = 0.0
            finger_states[key]["human_angles"] = (0.0, 0.0, 0.0)
            finger_states[key]["human_lengths"] = tuple(mm_to_m(value) for value in cfg.default_lengths_mm)
            finger_states[key]["base_offset"] = four_finger_base_offset_from_mm(
                cfg.default_base_mm[0],
                cfg.default_base_mm[1],
            )
            finger_states[key]["tip_offset_local"] = np.array(
                [mm_to_m(cfg.default_tip_mm[0]), mm_to_m(cfg.default_tip_mm[1])],
                dtype=np.float64,
            )
        mujoco.mj_forward(model, data)
        sync_controls_from_selected()

    def toggle_ik(_event) -> None:
        ik_state["enabled"] = not ik_state["enabled"]
        toggle_button.label.set_text(f"IK: {'开' if ik_state['enabled'] else '关'}")

    def choose_finger(label: str) -> None:
        selected["key"] = label_to_key[label]
        sync_controls_from_selected()

    reset_button.on_clicked(reset_all)
    toggle_button.on_clicked(toggle_ik)
    radio.on_clicked(choose_finger)

    sync_controls_from_selected()

    with viewer.launch_passive(model, data, show_left_ui=True, show_right_ui=True) as handle:
        handle.opt.frame = mujoco.mjtFrame.mjFRAME_NONE
        print("已打开三维全手映射观察窗口。")
        print("操作说明：")
        print("- 左侧单选按钮切换当前控制手指。")
        print("- exo_q1~exo_q4 只控制当前选中手指的外骨骼四关节。")
        print("- human_q1~human_q3 在 IK 关闭时手动控制当前选中手指的人手三连杆。")
        print("- 文本框中的长度、base 偏移、tip 偏移都属于当前选中手指，单位为毫米。")
        print("- 四指 base_dx/base_dy 输入会在内部按 (x, y) -> (y, -x) 转换。")
        print("- “全部伸直”会把四指一起恢复到掌面伸直初始状态。")

        while handle.is_running() and plt.fignum_exists(fig.number):
            selected_key = selected["key"]
            selected_cfg = FINGER_CONFIGS[selected_key]

            for index, qpos_adr in enumerate(qpos_adrs_map[selected_key], start=1):
                data.qpos[qpos_adr] = sliders[f"exo_q{index}"].val

            finger_states[selected_key]["human_lengths"] = (
                mm_to_m(read_input_mm("L1_mm", positive=True)),
                mm_to_m(read_input_mm("L2_mm", positive=True)),
                mm_to_m(read_input_mm("L3_mm", positive=True)),
            )
            finger_states[selected_key]["base_offset"] = four_finger_base_offset_from_mm(
                read_input_mm("base_dx_mm"),
                read_input_mm("base_dy_mm"),
            )
            finger_states[selected_key]["tip_offset_local"] = np.array(
                [mm_to_m(read_input_mm("tip_dx_mm")), mm_to_m(read_input_mm("tip_dy_mm"))],
                dtype=np.float64,
            )

            selected_human_angles = (
                sliders["human_q1"].val,
                sliders["human_q2"].val,
                sliders["human_q3"].val,
            )

            mujoco.mj_forward(model, data)

            mapping_states: dict[str, dict[str, object]] = {}
            for key in FINGER_ORDER:
                cfg = FINGER_CONFIGS[key]
                human_lengths = finger_states[key]["human_lengths"]
                base_offset = finger_states[key]["base_offset"]
                tip_offset_local = finger_states[key]["tip_offset_local"]
                if key == selected_key:
                    human_angles_initial = selected_human_angles
                else:
                    human_angles_initial = finger_states[key]["human_angles"]

                state = build_mapping_state(
                    model,
                    data,
                    cfg,
                    plane_y_sign=plane_y_sign,
                    human_lengths=human_lengths,
                    human_angles_initial=human_angles_initial,
                    base_offset=base_offset,
                    tip_offset_local=tip_offset_local,
                    ik_enabled=ik_state["enabled"],
                    flexion_sign=flexion_sign,
                )
                mapping_states[key] = state
                finger_states[key]["human_angles"] = tuple(state["human_angles"])

            summary_text.set_text(
                format_summary(
                    mapping_states[selected_key],
                    ik_enabled=ik_state["enabled"],
                    flexion_sign=flexion_sign,
                    plane_y_sign=plane_y_sign,
                )
            )

            with handle.lock():
                handle.user_scn.ngeom = 0
                append_selected_finger_markers(model, data, handle.user_scn, selected_cfg, axis_length=axis_length)
                for key in FINGER_ORDER:
                    append_mapping_overlay(
                        handle.user_scn,
                        FINGER_CONFIGS[key],
                        mapping_states[key],
                        axis_length=axis_length,
                        selected=(key == selected_key),
                    )
            handle.sync()
            plt.pause(dt)

    plt.close(fig)


def slider_viewer_v2(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    qpos_adrs_map: dict[str, list[int]],
    finger_states: dict[str, dict[str, object]],
    thumb_reference: ThumbReference,
    initial_finger: str,
    ik_mode_init: bool,
    flexion_sign: int,
    plane_y_sign: int,
    axis_length: float,
    dt: float,
) -> None:
    try:
        import matplotlib

        configure_chinese_font()
        matplotlib.use("TkAgg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, RadioButtons, Slider, TextBox
        import mujoco.viewer as viewer
    except ImportError as exc:
        raise RuntimeError("缺少三维观察依赖，请检查 matplotlib 和 mujoco.viewer") from exc

    fig = plt.figure("三维全手五指映射观察器", figsize=(11.6, 9.0))
    fig.subplots_adjust(left=0.28, right=0.98, top=0.95, bottom=0.07)

    slider_specs = [
        ("exo_q1", -3.14, 3.14, [0.32, 0.90, 0.62, 0.028]),
        ("exo_q2", -3.14, 3.14, [0.32, 0.855, 0.62, 0.028]),
        ("exo_q3", -3.14, 3.14, [0.32, 0.81, 0.62, 0.028]),
        ("exo_q4", -3.14, 3.14, [0.32, 0.765, 0.62, 0.028]),
        ("exo_q5", -3.14, 3.14, [0.32, 0.72, 0.62, 0.028]),
        ("human_q1", -3.14, 3.14, [0.32, 0.655, 0.62, 0.028]),
        ("human_q2", -3.14, 3.14, [0.32, 0.61, 0.62, 0.028]),
        ("human_q3", -3.14, 3.14, [0.32, 0.565, 0.62, 0.028]),
        ("human_q4", -3.14, 3.14, [0.32, 0.52, 0.62, 0.028]),
        ("human_q5", -3.14, 3.14, [0.32, 0.475, 0.62, 0.028]),
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
    for label, vmin, vmax, rect in slider_specs:
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

    summary_text = fig.text(0.56, 0.44, "", va="top", ha="left", fontsize=9)
    selected_text = fig.text(0.04, 0.93, "", va="top", ha="left", fontsize=11)

    radio_ax = fig.add_axes([0.04, 0.60, 0.20, 0.26])
    radio = RadioButtons(
        radio_ax,
        [THUMB_LABEL, *[FINGER_CONFIGS[key].label for key in FINGER_ORDER]],
        active=HAND_ORDER.index(initial_finger),
    )
    radio_ax.set_title("当前控制手指")

    reset_ax = fig.add_axes([0.56, 0.02, 0.12, 0.05])
    reset_button = Button(reset_ax, "全部伸直")
    toggle_ax = fig.add_axes([0.70, 0.02, 0.12, 0.05])
    toggle_button = Button(toggle_ax, f"IK: {'开' if ik_state['enabled'] else '关'}")

    label_to_key = {THUMB_LABEL: THUMB_KEY, **{cfg.label: cfg.key for cfg in FINGER_CONFIGS.values()}}

    def set_textbox_mm(name: str, value_mm: float) -> None:
        numeric_inputs[name] = float(value_mm)
        textboxes[name].set_val(f"{value_mm:.2f}")

    def set_slider_visibility(prefix: str, visible_count: int, labels: list[str]) -> None:
        for index in range(1, 6):
            slider = sliders[f"{prefix}_q{index}"]
            active = index <= visible_count
            slider.ax.set_visible(active)
            if active:
                slider.label.set_text(labels[index - 1])

    def sync_controls_from_selected() -> None:
        key = selected["key"]
        finger_state = finger_states[key]
        exo_count, human_count = visible_angle_count(key)
        if key == THUMB_KEY:
            selected_text.set_text("当前手指：拇指\n外骨骼保留 5 关节；人手映射删除根部屈伸。")
            exo_labels = [f"exo_q{i} {item.tag}" for i, item in enumerate(THUMB_JOINTS, start=1)]
            human_labels = [f"human_q{i} {name}" for i, name in enumerate(THUMB_HUMAN_LABELS, start=1)]
        else:
            cfg = FINGER_CONFIGS[key]
            selected_text.set_text(f"当前手指：{cfg.label}\n四指继续使用同平面映射逻辑。")
            exo_labels = [f"exo_q{i} {cfg.label}{i}" for i in range(1, 5)]
            human_labels = [f"human_q{i} {cfg.label}" for i in range(1, 4)]

        set_slider_visibility("exo", exo_count, exo_labels)
        set_slider_visibility("human", human_count, human_labels)

        for index, qpos_adr in enumerate(qpos_adrs_map[key], start=1):
            sliders[f"exo_q{index}"].set_val(float(data.qpos[qpos_adr]))
        for index in range(len(qpos_adrs_map[key]) + 1, 6):
            sliders[f"exo_q{index}"].set_val(0.0)

        human_angles = tuple(float(value) for value in finger_state["human_angles"])
        for index, angle in enumerate(human_angles, start=1):
            sliders[f"human_q{index}"].set_val(angle)
        for index in range(len(human_angles) + 1, 6):
            sliders[f"human_q{index}"].set_val(0.0)

        human_lengths = finger_state["human_lengths"]
        base_offset = np.asarray(finger_state["base_offset"], dtype=np.float64)
        tip_offset_local = finger_state["tip_offset_local"]
        display_base_offset = (
            base_offset if key == THUMB_KEY else recover_four_finger_base_input(base_offset)
        )
        set_textbox_mm("L1_mm", human_lengths[0] * 1000.0)
        set_textbox_mm("L2_mm", human_lengths[1] * 1000.0)
        set_textbox_mm("L3_mm", human_lengths[2] * 1000.0)
        set_textbox_mm("base_dx_mm", display_base_offset[0] * 1000.0)
        set_textbox_mm("base_dy_mm", display_base_offset[1] * 1000.0)
        set_textbox_mm("base_dz_mm", (base_offset[2] * 1000.0) if len(base_offset) > 2 else 0.0)
        set_textbox_mm("tip_dx_mm", tip_offset_local[0] * 1000.0)
        set_textbox_mm("tip_dy_mm", tip_offset_local[1] * 1000.0)
        textboxes["base_dz_mm"].ax.set_visible(key == THUMB_KEY)

    def read_input_mm(label: str, *, positive: bool = False) -> float:
        try:
            value = float(textboxes[label].text.strip())
        except ValueError:
            value = numeric_inputs[label]
        if positive:
            value = max(value, 0.1)
        numeric_inputs[label] = value
        return value

    def reset_all(_event) -> None:
        for key in HAND_ORDER:
            for qpos_adr in qpos_adrs_map[key]:
                data.qpos[qpos_adr] = 0.0
            if key == THUMB_KEY:
                finger_states[key]["human_angles"] = tuple(0.0 for _ in range(THUMB_DOF))
                finger_states[key]["human_lengths"] = tuple(mm_to_m(value) for value in THUMB_DEFAULT_LENGTHS_MM)
                finger_states[key]["base_offset"] = np.array([mm_to_m(v) for v in THUMB_DEFAULT_BASE_MM], dtype=np.float64)
                finger_states[key]["tip_offset_local"] = np.array([mm_to_m(v) for v in THUMB_DEFAULT_TIP_MM], dtype=np.float64)
            else:
                cfg = FINGER_CONFIGS[key]
                finger_states[key]["human_angles"] = (0.0, 0.0, 0.0)
                finger_states[key]["human_lengths"] = tuple(mm_to_m(value) for value in cfg.default_lengths_mm)
                finger_states[key]["base_offset"] = four_finger_base_offset_from_mm(
                    cfg.default_base_mm[0],
                    cfg.default_base_mm[1],
                )
                finger_states[key]["tip_offset_local"] = np.array([mm_to_m(cfg.default_tip_mm[0]), mm_to_m(cfg.default_tip_mm[1])], dtype=np.float64)
        mujoco.mj_forward(model, data)
        sync_controls_from_selected()

    def toggle_ik(_event) -> None:
        ik_state["enabled"] = not ik_state["enabled"]
        toggle_button.label.set_text(f"IK: {'开' if ik_state['enabled'] else '关'}")

    def choose_finger(label: str) -> None:
        selected["key"] = label_to_key[label]
        sync_controls_from_selected()

    reset_button.on_clicked(reset_all)
    toggle_button.on_clicked(toggle_ik)
    radio.on_clicked(choose_finger)
    sync_controls_from_selected()

    with viewer.launch_passive(model, data, show_left_ui=True, show_right_ui=True) as handle:
        handle.opt.frame = mujoco.mjtFrame.mjFRAME_NONE
        print("已打开三维全手五指映射观察窗口。")
        print("操作说明：")
        print("- 左侧单选按钮可切换拇指/食指/中指/无名指/小指。")
        print("- 四指继续沿用原同平面映射；拇指外骨骼保留 5 关节，人手 IK/FK 删除根部屈伸。")
        print("- 文本框单位均为毫米；修改后按回车生效。")
        print("- 四指 base_dx/base_dy 输入会在内部按 (x, y) -> (y, -x) 转换。")

        while handle.is_running() and plt.fignum_exists(fig.number):
            selected_key = selected["key"]
            exo_count, human_count = visible_angle_count(selected_key)

            for index, qpos_adr in enumerate(qpos_adrs_map[selected_key], start=1):
                data.qpos[qpos_adr] = sliders[f"exo_q{index}"].val

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
            for key in HAND_ORDER:
                if key == THUMB_KEY:
                    human_angles_initial = selected_human_angles if key == selected_key else finger_states[key]["human_angles"]
                    state = build_thumb_mapping_state(
                        model,
                        data,
                        thumb_reference,
                        human_lengths=finger_states[key]["human_lengths"],
                        human_angles_initial=tuple(float(v) for v in human_angles_initial),
                        base_offset=finger_states[key]["base_offset"],
                        tip_offset_local=finger_states[key]["tip_offset_local"],
                        ik_enabled=ik_state["enabled"],
                    )
                else:
                    cfg = FINGER_CONFIGS[key]
                    human_angles_initial = selected_human_angles if key == selected_key else finger_states[key]["human_angles"]
                    state = build_mapping_state(
                        model,
                        data,
                        cfg,
                        plane_y_sign=plane_y_sign,
                        human_lengths=finger_states[key]["human_lengths"],
                        human_angles_initial=tuple(float(v) for v in human_angles_initial),
                        base_offset=finger_states[key]["base_offset"],
                        tip_offset_local=finger_states[key]["tip_offset_local"],
                        ik_enabled=ik_state["enabled"],
                        flexion_sign=flexion_sign,
                    )
                mapping_states[key] = state
                finger_states[key]["human_angles"] = tuple(state["human_angles"])

            summary_text.set_text(
                format_summary(
                    mapping_states[selected_key],
                    ik_enabled=ik_state["enabled"],
                    flexion_sign=flexion_sign,
                    plane_y_sign=plane_y_sign,
                )
            )

            with handle.lock():
                handle.user_scn.ngeom = 0
                if selected_key == THUMB_KEY:
                    append_thumb_markers(model, data, handle.user_scn, axis_length=axis_length)
                else:
                    append_selected_finger_markers(model, data, handle.user_scn, FINGER_CONFIGS[selected_key], axis_length=axis_length)
                for key in HAND_ORDER:
                    append_mapping_overlay(
                        handle.user_scn,
                        mapping_states[key],
                        axis_length=axis_length,
                        selected=(key == selected_key),
                    )
            handle.sync()
            plt.pause(dt)

    plt.close(fig)


def main() -> int:
    args = parse_args()
    configure_chinese_font()

    model, data, bundle_urdf = load_model_from_urdf(args.urdf, limit_overrides=build_limit_overrides())
    print(f"已加载模型：{bundle_urdf}")

    qpos_adrs_map: dict[str, list[int]] = {}
    for key in FINGER_ORDER:
        cfg = FINGER_CONFIGS[key]
        qpos_adrs: list[int] = []
        for joint_name in cfg.joints:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                raise RuntimeError(f"模型中缺少关节：{joint_name}")
            qpos_adrs.append(int(model.jnt_qposadr[joint_id]))
            data.qpos[qpos_adrs[-1]] = 0.0
        qpos_adrs_map[key] = qpos_adrs
    mujoco.mj_forward(model, data)

    finger_states: dict[str, dict[str, object]] = {}
    for key in FINGER_ORDER:
        cfg = FINGER_CONFIGS[key]
        finger_states[key] = {
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

    mapping_states: dict[str, dict[str, object]] = {}
    for key in FINGER_ORDER:
        cfg = FINGER_CONFIGS[key]
        mapping_states[key] = build_mapping_state(
            model,
            data,
            cfg,
            plane_y_sign=args.plane_y_sign,
            human_lengths=finger_states[key]["human_lengths"],
            human_angles_initial=finger_states[key]["human_angles"],
            base_offset=finger_states[key]["base_offset"],
            tip_offset_local=finger_states[key]["tip_offset_local"],
            ik_enabled=args.ik,
            flexion_sign=args.flexion_sign,
        )
        finger_states[key]["human_angles"] = tuple(mapping_states[key]["human_angles"])

    if args.headless:
        print("三维全手映射报告：")
        print("初始状态：四指外骨骼角度全 0，人手三连杆全 0。")
        print()
        for key in FINGER_ORDER:
            print_headless_report(
                mapping_states[key],
                ik_enabled=args.ik,
                flexion_sign=args.flexion_sign,
                plane_y_sign=args.plane_y_sign,
            )
        return 0

    slider_viewer(
        model,
        data,
        qpos_adrs_map=qpos_adrs_map,
        finger_states=finger_states,
        initial_finger=args.finger,
        ik_mode_init=args.ik,
        flexion_sign=args.flexion_sign,
        plane_y_sign=args.plane_y_sign,
        axis_length=args.axis_length,
        dt=args.dt,
    )
    return 0


def main_v2() -> int:
    args = parse_args_v2()
    configure_chinese_font()

    model, data, bundle_urdf = load_model_from_urdf(args.urdf, limit_overrides=build_limit_overrides())
    print(f"已加载模型：{bundle_urdf}")

    qpos_adrs_map: dict[str, list[int]] = {}
    qpos_adrs_map[THUMB_KEY] = []
    for item in THUMB_JOINTS:
        joint_id = get_joint_id(model, item.joint_name)
        qpos_adrs_map[THUMB_KEY].append(int(model.jnt_qposadr[joint_id]))
        data.qpos[qpos_adrs_map[THUMB_KEY][-1]] = 0.0

    for key in FINGER_ORDER:
        cfg = FINGER_CONFIGS[key]
        qpos_adrs: list[int] = []
        for joint_name in cfg.joints:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                raise RuntimeError(f"模型中缺少关节：{joint_name}")
            qpos_adrs.append(int(model.jnt_qposadr[joint_id]))
            data.qpos[qpos_adrs[-1]] = 0.0
        qpos_adrs_map[key] = qpos_adrs
    mujoco.mj_forward(model, data)

    thumb_reference = make_thumb_reference(model)

    finger_states: dict[str, dict[str, object]] = {
        THUMB_KEY: {
            "human_lengths": tuple(mm_to_m(value) for value in THUMB_DEFAULT_LENGTHS_MM),
            "human_angles": tuple(0.0 for _ in range(THUMB_DOF)),
            "base_offset": np.array([mm_to_m(v) for v in THUMB_DEFAULT_BASE_MM], dtype=np.float64),
            "tip_offset_local": np.array([mm_to_m(v) for v in THUMB_DEFAULT_TIP_MM], dtype=np.float64),
        }
    }
    for key in FINGER_ORDER:
        cfg = FINGER_CONFIGS[key]
        finger_states[key] = {
            "human_lengths": tuple(mm_to_m(value) for value in cfg.default_lengths_mm),
            "human_angles": (0.0, 0.0, 0.0),
            "base_offset": four_finger_base_offset_from_mm(
                cfg.default_base_mm[0],
                cfg.default_base_mm[1],
            ),
            "tip_offset_local": np.array([mm_to_m(cfg.default_tip_mm[0]), mm_to_m(cfg.default_tip_mm[1])], dtype=np.float64),
        }

    mapping_states: dict[str, dict[str, object]] = {}
    mapping_states[THUMB_KEY] = build_thumb_mapping_state(
        model,
        data,
        thumb_reference,
        human_lengths=finger_states[THUMB_KEY]["human_lengths"],
        human_angles_initial=finger_states[THUMB_KEY]["human_angles"],
        base_offset=finger_states[THUMB_KEY]["base_offset"],
        tip_offset_local=finger_states[THUMB_KEY]["tip_offset_local"],
        ik_enabled=args.ik,
    )
    finger_states[THUMB_KEY]["human_angles"] = tuple(mapping_states[THUMB_KEY]["human_angles"])

    for key in FINGER_ORDER:
        cfg = FINGER_CONFIGS[key]
        mapping_states[key] = build_mapping_state(
            model,
            data,
            cfg,
            plane_y_sign=args.plane_y_sign,
            human_lengths=finger_states[key]["human_lengths"],
            human_angles_initial=finger_states[key]["human_angles"],
            base_offset=finger_states[key]["base_offset"],
            tip_offset_local=finger_states[key]["tip_offset_local"],
            ik_enabled=args.ik,
            flexion_sign=args.flexion_sign,
        )
        finger_states[key]["human_angles"] = tuple(mapping_states[key]["human_angles"])

    if args.headless:
        print("三维全手五指映射报告：")
        print("初始姿态：五指外骨骼角度全 0，人手链全部为伸直初始值。")
        print()
        for key in HAND_ORDER:
            print_headless_report(
                mapping_states[key],
                ik_enabled=args.ik,
                flexion_sign=args.flexion_sign,
                plane_y_sign=args.plane_y_sign,
            )
        return 0

    slider_viewer_v2(
        model,
        data,
        qpos_adrs_map=qpos_adrs_map,
        finger_states=finger_states,
        thumb_reference=thumb_reference,
        initial_finger=args.finger,
        ik_mode_init=args.ik,
        flexion_sign=args.flexion_sign,
        plane_y_sign=args.plane_y_sign,
        axis_length=args.axis_length,
        dt=args.dt,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main_v2())
    except KeyboardInterrupt:
        print("\n用户中断，退出三维全手映射观察器。")
        raise SystemExit(130)
