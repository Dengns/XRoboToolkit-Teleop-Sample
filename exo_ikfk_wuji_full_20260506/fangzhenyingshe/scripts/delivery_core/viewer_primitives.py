from __future__ import annotations

import numpy as np
import mujoco


AXIS_COLORS = {
    "x": np.array([1.0, 0.2, 0.2, 1.0], dtype=np.float32),
    "y": np.array([0.2, 0.9, 0.2, 1.0], dtype=np.float32),
    "z": np.array([0.2, 0.4, 1.0, 1.0], dtype=np.float32),
}


def make_identity_mat() -> np.ndarray:
    return np.eye(3, dtype=np.float64).reshape(-1)


def joint_axis_in_world(model: mujoco.MjModel, data: mujoco.MjData, joint_ref: int | str) -> np.ndarray:
    if isinstance(joint_ref, str):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_ref)
        if joint_id < 0:
            raise ValueError(f"模型中不存在关节：{joint_ref}")
    else:
        joint_id = int(joint_ref)
    body_id = int(model.jnt_bodyid[joint_id])
    body_rot = data.xmat[body_id].reshape(3, 3)
    local_axis = np.asarray(model.jnt_axis[joint_id], dtype=np.float64)
    world_axis = body_rot @ local_axis
    norm = np.linalg.norm(world_axis)
    if norm < 1e-9:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return world_axis / norm


def add_sphere(scene: mujoco.MjvScene, pos: np.ndarray, radius: float, rgba: np.ndarray) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([radius, radius, radius], dtype=np.float64),
        np.asarray(pos, dtype=np.float64),
        make_identity_mat(),
        np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1


def add_connector(
    scene: mujoco.MjvScene,
    from_pos: np.ndarray,
    to_pos: np.ndarray,
    width: float,
    rgba: np.ndarray,
    geom_type: int = mujoco.mjtGeom.mjGEOM_ARROW,
) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        geom_type,
        np.array([0.0, 0.0, 0.0], dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        make_identity_mat(),
        np.asarray(rgba, dtype=np.float32),
    )
    mujoco.mjv_connector(
        geom,
        geom_type,
        width,
        np.asarray(from_pos, dtype=np.float64),
        np.asarray(to_pos, dtype=np.float64),
    )
    scene.ngeom += 1


def add_frame_axes(
    scene: mujoco.MjvScene,
    pos: np.ndarray,
    rot_mat: np.ndarray,
    axis_length: float,
    width: float,
) -> None:
    add_connector(scene, pos, pos + rot_mat[:, 0] * axis_length, width, AXIS_COLORS["x"])
    add_connector(scene, pos, pos + rot_mat[:, 1] * axis_length, width, AXIS_COLORS["y"])
    add_connector(scene, pos, pos + rot_mat[:, 2] * axis_length, width, AXIS_COLORS["z"])
