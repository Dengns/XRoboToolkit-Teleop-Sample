from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize
import mujoco

from delivery_core.four_finger_mapping import mm_to_m, reachability_summary
from delivery_core.viewer_primitives import joint_axis_in_world


THUMB_TIP_PARENT_BODY = "link_RightSkeletonThumb4"
THUMB_TIP_LOCAL_OFFSET = np.array([-0.0026, 0.005, -0.027], dtype=np.float64)
THUMB_IK_RETRY_ERROR_M = 1e-4
THUMB_IK_EQUAL_ERROR_TOL_M = 1e-7
THUMB_IK_REGULARIZATION = 5e-4
THUMB_ROOT_AXIS_LENGTH = 0.10


@dataclass(frozen=True)
class ThumbJointLabel:
    tag: str
    joint_name: str
    child_link: str
    semantic_name: str
    role: str


THUMB_JOINTS: tuple[ThumbJointLabel, ...] = (
    ThumbJointLabel("T0", "joint_RightSkeletonThumbBase", "link_RightSkeletonThumbBase", "关节1：掌根展收", "绕近似手掌/中指纵轴旋转，带动拇指整体翻出或收回。"),
    ThumbJointLabel("T1", "joint_RightSkeletonThumb1", "link_RightSkeletonThumb1", "关节2A：根部侧摆", "第二机械关节的侧摆自由度，主要改变拇指在掌面内的扫动方向。"),
    ThumbJointLabel("T2", "joint_RightSkeletonThumb2", "link_RightSkeletonThumb2", "关节2B：根部屈伸", "第二机械关节的屈伸自由度，开始形成拇指向掌心的弯曲。"),
    ThumbJointLabel("T3", "joint_RightSkeletonThumb3", "link_RightSkeletonThumb3", "关节3：中段屈伸", "后段屈伸自由度，和 T2/T4 轴向平行，负责继续卷曲。"),
    ThumbJointLabel("T4", "joint_RightSkeletonThumb4", "link_RightSkeletonThumb4", "关节4：末端屈伸", "末端屈伸/指尖补偿自由度，可按需要与 T3 耦合。"),
)
THUMB_HAND_ACTIVE_INDICES = (0, 1, 3, 4)
THUMB_EXO_DOF = len(THUMB_JOINTS)
THUMB_DOF = len(THUMB_HAND_ACTIVE_INDICES)


@dataclass(frozen=True)
class ThumbReference:
    base_origin: np.ndarray
    abduction_axis: np.ndarray
    side_axis: np.ndarray
    flex_axis: np.ndarray
    forward_axis: np.ndarray


def get_joint_id(model: mujoco.MjModel, joint_name: str) -> int:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        raise RuntimeError(f"模型中找不到关节：{joint_name}")
    return int(joint_id)


def get_body_id(model: mujoco.MjModel, body_name: str) -> int:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise RuntimeError(f"模型中找不到 body/link：{body_name}")
    return int(body_id)


def hide_left_hand_geoms(model: mujoco.MjModel) -> int:
    probe_data = mujoco.MjData(model)
    mujoco.mj_forward(model, probe_data)
    hidden_count = 0
    for geom_id in range(model.ngeom):
        body_id = int(model.geom_bodyid[geom_id])
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        is_left_link = body_name.startswith("link_Left") or body_name.startswith("left_")
        is_left_world_base = body_name == "world" and float(probe_data.geom_xpos[geom_id, 0]) < 0.0
        if is_left_link or is_left_world_base:
            model.geom_rgba[geom_id, 3] = 0.0
            hidden_count += 1
    return hidden_count


def thumb_tip_position(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    body_id = get_body_id(model, THUMB_TIP_PARENT_BODY)
    body_pos = np.asarray(data.xpos[body_id], dtype=np.float64)
    body_rot = data.xmat[body_id].reshape(3, 3)
    return body_pos + body_rot @ THUMB_TIP_LOCAL_OFFSET


def normalized(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        raise ValueError("无法归一化长度过小的向量。")
    return np.asarray(vector, dtype=np.float64) / norm


def rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = normalized(axis)
    x, y, z = axis
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    t = 1.0 - c
    return np.array(
        [
            [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
        ],
        dtype=np.float64,
    )


def make_thumb_reference(model: mujoco.MjModel) -> ThumbReference:
    ref_data = mujoco.MjData(model)
    for item in THUMB_JOINTS:
        joint_id = get_joint_id(model, item.joint_name)
        ref_data.qpos[int(model.jnt_qposadr[joint_id])] = 0.0
    mujoco.mj_forward(model, ref_data)
    joint_ids = [get_joint_id(model, item.joint_name) for item in THUMB_JOINTS]
    anchors = [np.asarray(ref_data.xanchor[joint_id], dtype=np.float64) for joint_id in joint_ids]
    axes = [joint_axis_in_world(model, ref_data, joint_id) for joint_id in joint_ids]
    forward = anchors[3] - anchors[2]
    if np.linalg.norm(forward) < 1e-12:
        forward = thumb_tip_position(model, ref_data) - anchors[2]
    return ThumbReference(
        base_origin=anchors[0],
        abduction_axis=normalized(axes[0]),
        side_axis=normalized(axes[1]),
        flex_axis=normalized(axes[3]),
        forward_axis=normalized(forward),
    )


def coerce_thumb_q(values: np.ndarray | tuple[float, ...] | list[float]) -> np.ndarray:
    q = np.asarray(values, dtype=np.float64).reshape(-1)
    if q.size == THUMB_DOF:
        return q
    if q.size == THUMB_EXO_DOF:
        return q[list(THUMB_HAND_ACTIVE_INDICES)]
    raise ValueError(f"拇指自由度数量不匹配：期望 {THUMB_DOF}，实际 {q.size}")


def qpos_for_thumb(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    values = []
    for item in THUMB_JOINTS:
        joint_id = get_joint_id(model, item.joint_name)
        values.append(float(data.qpos[int(model.jnt_qposadr[joint_id])]))
    return np.asarray(values, dtype=np.float64)


def thumb_joint_bounds() -> tuple[np.ndarray, np.ndarray]:
    lower_degrees = np.array([-90.0, -90.0, -120.0, -120.0], dtype=np.float64)
    # 删除 T2 根部屈伸后，T3/T4 继续保留少量正向补偿余量，避免位置可达时被 0° 上界卡死。
    upper_degrees = np.array([90.0, 90.0, 15.0, 15.0], dtype=np.float64)
    return np.radians(lower_degrees), np.radians(upper_degrees)


def thumb_ik_seed_candidates(lower: np.ndarray, upper: np.ndarray, initial_q: np.ndarray) -> list[np.ndarray]:
    midpoint = 0.5 * (lower + upper)
    raw_candidates = [
        initial_q,
        np.zeros(THUMB_DOF, dtype=np.float64),
        midpoint,
        np.array([0.0, 0.0, np.radians(-25.0), np.radians(-15.0)], dtype=np.float64),
        np.array([np.radians(20.0), np.radians(18.0), np.radians(-35.0), np.radians(-20.0)], dtype=np.float64),
        np.array([np.radians(-20.0), np.radians(18.0), np.radians(-30.0), np.radians(-18.0)], dtype=np.float64),
    ]
    candidates: list[np.ndarray] = []
    for candidate in raw_candidates:
        clipped = np.clip(np.asarray(candidate, dtype=np.float64), lower, upper)
        if not any(np.allclose(clipped, existing, atol=1e-9, rtol=0.0) for existing in candidates):
            candidates.append(clipped)
    return candidates


def build_thumb_fk(reference: ThumbReference, q: np.ndarray, lengths: tuple[float, float, float], base_offset: np.ndarray) -> np.ndarray:
    q = coerce_thumb_q(q)
    base_offset = np.asarray(base_offset, dtype=np.float64)
    origin = (
        reference.base_origin
        + base_offset[0] * reference.forward_axis
        + base_offset[1] * reference.side_axis
        + base_offset[2] * reference.flex_axis
    )
    rot = np.eye(3, dtype=np.float64)
    rot = rotation_matrix(reference.abduction_axis, float(q[0])) @ rot
    rot = rotation_matrix(rot @ reference.side_axis, float(q[1])) @ rot
    points = [origin]
    points.append(points[-1] + rot @ (reference.forward_axis * float(lengths[0])))
    for flex_angle, length in zip(q[2:], lengths[1:]):
        rot = rotation_matrix(rot @ reference.flex_axis, float(flex_angle)) @ rot
        points.append(points[-1] + rot @ (reference.forward_axis * float(length)))
    return np.vstack(points)


def thumb_tip_target_from_local_offset(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    reference: ThumbReference,
    tip_offset_local: np.ndarray,
) -> np.ndarray:
    tip_origin = thumb_tip_position(model, data)
    tip_offset_local = np.asarray(tip_offset_local, dtype=np.float64)
    if np.allclose(tip_offset_local, 0.0):
        return tip_origin
    tip_joint_id = get_joint_id(model, THUMB_JOINTS[-1].joint_name)
    local_x = tip_origin - np.asarray(data.xanchor[tip_joint_id], dtype=np.float64)
    if np.linalg.norm(local_x) < 1e-12:
        local_x = reference.forward_axis
    local_x = normalized(local_x)
    plane_normal = joint_axis_in_world(model, data, tip_joint_id)
    local_y = np.cross(plane_normal, local_x)
    if np.linalg.norm(local_y) < 1e-12:
        local_y = np.cross(reference.flex_axis, local_x)
    if np.linalg.norm(local_y) < 1e-12:
        local_y = reference.side_axis - np.dot(reference.side_axis, local_x) * local_x
    local_y = normalized(local_y)
    return tip_origin + tip_offset_local[0] * local_x + tip_offset_local[1] * local_y


def solve_thumb_ik(
    reference: ThumbReference,
    target_tip: np.ndarray,
    lengths: tuple[float, float, float],
    base_offset: np.ndarray,
    initial_q: np.ndarray,
) -> tuple[np.ndarray, float]:
    lower, upper = thumb_joint_bounds()
    initial = np.clip(coerce_thumb_q(initial_q), lower, upper)
    target_tip = np.asarray(target_tip, dtype=np.float64)

    def residual(q: np.ndarray) -> np.ndarray:
        points = build_thumb_fk(reference, q, lengths, base_offset)
        tip_error = points[-1] - target_tip
        regularization = THUMB_IK_REGULARIZATION * (q - initial)
        return np.concatenate([tip_error, regularization])

    def solve_from(seed: np.ndarray) -> tuple[np.ndarray, float, float, float]:
        result = optimize.least_squares(
            residual,
            x0=seed,
            bounds=(lower, upper),
            xtol=1e-12,
            ftol=1e-12,
            gtol=1e-12,
            max_nfev=300,
        )
        solved = np.asarray(result.x, dtype=np.float64)
        final_error = float(np.linalg.norm(build_thumb_fk(reference, solved, lengths, base_offset)[-1] - target_tip))
        continuity_cost = float(np.linalg.norm(solved - initial))
        objective_cost = float(np.hypot(final_error, THUMB_IK_REGULARIZATION * continuity_cost))
        return solved, final_error, continuity_cost, objective_cost

    best_q, best_error, best_continuity, best_objective = solve_from(initial)
    if best_error > THUMB_IK_RETRY_ERROR_M:
        for seed in thumb_ik_seed_candidates(lower, upper, initial)[1:]:
            solved, final_error, continuity_cost, objective_cost = solve_from(seed)
            if objective_cost < best_objective - THUMB_IK_EQUAL_ERROR_TOL_M or (
                abs(objective_cost - best_objective) <= THUMB_IK_EQUAL_ERROR_TOL_M
                and continuity_cost < best_continuity
            ):
                best_q = solved
                best_error = final_error
                best_continuity = continuity_cost
                best_objective = objective_cost
    return best_q, best_error


def format_thumb_summary(
    state: dict[str, object],
    lengths: tuple[float, float, float],
    base_offset: np.ndarray,
    tip_offset_local: np.ndarray,
    use_ik: bool,
) -> str:
    exo_q = np.asarray(state["exo_q"], dtype=np.float64)
    ik_q = np.asarray(state["ik_q"], dtype=np.float64)
    base_offset = np.asarray(base_offset, dtype=np.float64)
    tip_offset_local = np.asarray(tip_offset_local, dtype=np.float64)
    return "\n".join(
        [
            "简化拇指模型: 3D 展收 + 侧摆 + 中段/末端屈伸, 3连杆",
            "目标: 外骨骼 right_thumb_tip_joint 推导点 + 末端局部平面偏置",
            f"IK: {'开启' if use_ik else '关闭'}",
            f"长度(mm): L1={lengths[0]*1000:.1f} L2={lengths[1]*1000:.1f} L3={lengths[2]*1000:.1f}",
            f"根部偏移(mm): dx={base_offset[0]*1000:.1f} dy={base_offset[1]*1000:.1f} dz={base_offset[2]*1000:.1f}",
            f"末端局部偏移(mm): dx={tip_offset_local[0]*1000:.1f} dy={tip_offset_local[1]*1000:.1f}",
            f"外骨骼角(rad): {', '.join(f'{value:.3f}' for value in exo_q)}",
            f"IK角(rad): {', '.join(f'{value:.3f}' for value in ik_q)}",
            f"FK直接映射误差(mm): {float(state['fk_error'])*1000:.2f}",
            f"IK追踪误差(mm): {float(state['ik_error'])*1000:.2f}",
            f"目标距离/最大可达(mm): {float(state['target_distance'])*1000:.1f} / {float(state['max_reach'])*1000:.1f}",
            f"可达性: {'可达' if bool(state['is_reachable']) else '不可达，需要调整长度/基座偏移'}",
        ]
    )


def print_thumb_headless_report(
    state: dict[str, object],
    lengths: tuple[float, float, float],
    base_offset: np.ndarray,
    tip_offset_local: np.ndarray,
    use_ik: bool,
) -> None:
    print("v4 右手拇指简化人手模型映射报告：")
    print(format_thumb_summary(state, lengths=lengths, base_offset=base_offset, tip_offset_local=tip_offset_local, use_ik=use_ik))
    for label, points in (("FK", state["fk_points"]), ("IK", state["ik_points"])):
        print(f"{label} 人手拇指链：")
        for index, point in enumerate(np.asarray(points, dtype=np.float64)):
            print(f"- p{index}: ({point[0]:.5f}, {point[1]:.5f}, {point[2]:.5f})")
    exo_tip = np.asarray(state["exo_tip"], dtype=np.float64)
    target_tip = np.asarray(state["target_tip"], dtype=np.float64)
    print(f"- 外骨骼 tip: ({exo_tip[0]:.5f}, {exo_tip[1]:.5f}, {exo_tip[2]:.5f})")
    print(f"- 目标 tip: ({target_tip[0]:.5f}, {target_tip[1]:.5f}, {target_tip[2]:.5f})")


def append_thumb_markers(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    scene: mujoco.MjvScene,
    *,
    axis_length: float,
) -> None:
    from delivery_core.viewer_primitives import add_connector, add_frame_axes, add_sphere

    joint_point_colors = [
        np.array([1.00, 0.25, 0.20, 1.0], dtype=np.float32),
        np.array([1.00, 0.65, 0.15, 1.0], dtype=np.float32),
        np.array([0.20, 0.85, 0.35, 1.0], dtype=np.float32),
        np.array([0.20, 0.55, 1.00, 1.0], dtype=np.float32),
        np.array([0.75, 0.35, 1.00, 1.0], dtype=np.float32),
    ]
    tip_color = np.array([1.0, 0.0, 1.0, 1.0], dtype=np.float32)
    chain_color = np.array([0.95, 0.95, 0.95, 0.85], dtype=np.float32)
    axis_arrow_color = np.array([1.0, 0.95, 0.10, 1.0], dtype=np.float32)

    points: list[np.ndarray] = []
    for index, item in enumerate(THUMB_JOINTS):
        joint_id = get_joint_id(model, item.joint_name)
        body_id = int(model.jnt_bodyid[joint_id])
        joint_pos = np.asarray(data.xanchor[joint_id], dtype=np.float64)
        body_rot = data.xmat[body_id].reshape(3, 3)
        axis_world = joint_axis_in_world(model, data, joint_id)
        color = joint_point_colors[index]
        points.append(joint_pos)
        add_sphere(scene, joint_pos, radius=0.0075, rgba=color)
        joint_axis_length = THUMB_ROOT_AXIS_LENGTH if index == 0 else axis_length
        if index == 0:
            parent_body_id = int(model.body_parentid[body_id])
            frame_rot = data.xmat[parent_body_id].reshape(3, 3)
        else:
            frame_rot = body_rot
        add_frame_axes(scene, joint_pos, frame_rot, axis_length=joint_axis_length, width=0.0035)
        add_connector(
            scene,
            joint_pos,
            joint_pos + axis_world * axis_length * 1.35,
            width=0.006,
            rgba=axis_arrow_color,
            geom_type=mujoco.mjtGeom.mjGEOM_ARROW,
        )

    tip_pos = thumb_tip_position(model, data)
    points.append(tip_pos)
    add_sphere(scene, tip_pos, radius=0.0085, rgba=tip_color)
    for start, end in zip(points, points[1:]):
        add_connector(
            scene,
            start,
            end,
            width=0.003,
            rgba=chain_color,
            geom_type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        )
