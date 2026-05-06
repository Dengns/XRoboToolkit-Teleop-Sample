#!/usr/bin/env python3
"""观察 SteamVR Tracker 本体坐标系，并验证初始化参考系下的相对位姿。"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np
import openvr

DEFAULT_RATE_HZ = 10.0
DEFAULT_DECIMALS = 4
R_REFERENCE_TO_TARGET = np.array(
    [
        [0.0, 0.0, -1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=float,
)


@dataclass(frozen=True)
class TrackerPose:
    index: int
    serial: str
    transform: np.ndarray

    @property
    def xyz(self) -> np.ndarray:
        return self.transform[:3, 3].astype(float)

    @property
    def rotation(self) -> np.ndarray:
        return self.transform[:3, :3].astype(float)

    @property
    def get_T(self) -> np.ndarray:
        return self.transform.astype(float)


def mat34_to_matrix(mat: object) -> np.ndarray:
    """将 OpenVR 3x4 位姿矩阵转换为 4x4 齐次矩阵。"""
    return np.array(
        [
            [mat[0][0], mat[0][1], mat[0][2], mat[0][3]],
            [mat[1][0], mat[1][1], mat[1][2], mat[1][3]],
            [mat[2][0], mat[2][1], mat[2][2], mat[2][3]],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def rotation_matrix_to_rpy(rotation_matrix: np.ndarray) -> np.ndarray:
    """将旋转矩阵转换为 roll/pitch/yaw，单位 rad。"""
    r = np.asarray(rotation_matrix, dtype=float)
    sy = float(np.hypot(r[0, 0], r[1, 0]))
    singular = sy < 1.0e-6

    if not singular:
        roll = np.arctan2(r[2, 1], r[2, 2])
        pitch = np.arctan2(-r[2, 0], sy)
        yaw = np.arctan2(r[1, 0], r[0, 0])
    else:
        roll = np.arctan2(-r[1, 2], r[1, 1])
        pitch = np.arctan2(-r[2, 0], sy)
        yaw = 0.0

    return np.asarray([roll, pitch, yaw], dtype=float)


def normalize_quaternion_wxyz(quat: np.ndarray) -> np.ndarray:
    """归一化四元数并固定到 w>=0，避免同一姿态出现正负号跳变。"""
    q = np.asarray(quat, dtype=float)
    norm = float(np.linalg.norm(q))
    if norm <= 1.0e-9 or not np.all(np.isfinite(q)):
        raise RuntimeError(f"四元数无效: {quat}")

    q = q / norm
    if q[0] < 0.0:
        q = -q
    return q


def rotation_matrix_to_quaternion_wxyz(rotation_matrix: np.ndarray) -> np.ndarray:
    """将旋转矩阵转换为四元数 [w, x, y, z]。"""
    r = np.asarray(rotation_matrix, dtype=float)
    trace = float(np.trace(r))

    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        quat = np.array(
            [
                0.25 * s,
                (r[2, 1] - r[1, 2]) / s,
                (r[0, 2] - r[2, 0]) / s,
                (r[1, 0] - r[0, 1]) / s,
            ],
            dtype=float,
        )
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        quat = np.array(
            [
                (r[2, 1] - r[1, 2]) / s,
                0.25 * s,
                (r[0, 1] + r[1, 0]) / s,
                (r[0, 2] + r[2, 0]) / s,
            ],
            dtype=float,
        )
    elif r[1, 1] > r[2, 2]:
        s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        quat = np.array(
            [
                (r[0, 2] - r[2, 0]) / s,
                (r[0, 1] + r[1, 0]) / s,
                0.25 * s,
                (r[1, 2] + r[2, 1]) / s,
            ],
            dtype=float,
        )
    else:
        s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
        quat = np.array(
            [
                (r[1, 0] - r[0, 1]) / s,
                (r[0, 2] + r[2, 0]) / s,
                (r[1, 2] + r[2, 1]) / s,
                0.25 * s,
            ],
            dtype=float,
        )

    return normalize_quaternion_wxyz(quat)


def quaternion_conjugate_wxyz(quat: np.ndarray) -> np.ndarray:
    """返回四元数共轭 [w, -x, -y, -z]。"""
    q = normalize_quaternion_wxyz(quat)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)


def quaternion_multiply_wxyz(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """计算四元数乘法 lhs * rhs，输入输出均为 [w, x, y, z]。"""
    w1, x1, y1, z1 = normalize_quaternion_wxyz(lhs)
    w2, x2, y2, z2 = normalize_quaternion_wxyz(rhs)
    quat = np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=float,
    )
    return normalize_quaternion_wxyz(quat)


def quaternion_to_rotvec_wxyz(quat: np.ndarray, eps: float = 1.0e-6) -> np.ndarray:
    """将四元数转换为旋转向量，单位 rad。"""
    q = normalize_quaternion_wxyz(quat)
    angle = 2.0 * np.arccos(np.clip(q[0], -1.0, 1.0))
    sin_half_angle = np.sin(angle / 2.0)
    if angle < eps or sin_half_angle < eps:
        return np.zeros(3, dtype=float)

    axis = q[1:] / sin_half_angle
    return axis * angle


def format_vector(name: str, vec: np.ndarray, decimals: int) -> str:
    values = np.round(np.asarray(vec, dtype=float), decimals).tolist()
    return f"{name}={values}"


def get_string_property(vrsystem: openvr.IVRSystem, index: int, prop: int) -> str | None:
    try:
        return vrsystem.getStringTrackedDeviceProperty(index, prop)
    except Exception:
        return None


def read_tracker_pose(vrsystem: openvr.IVRSystem, preferred_serial: str | None) -> TrackerPose:
    poses = vrsystem.getDeviceToAbsoluteTrackingPose(
        openvr.TrackingUniverseStanding,
        0.0,
        openvr.k_unMaxTrackedDeviceCount,
    )

    candidates: list[tuple[int, str, object]] = []
    for index in range(openvr.k_unMaxTrackedDeviceCount):
        if vrsystem.getTrackedDeviceClass(index) != openvr.TrackedDeviceClass_GenericTracker:
            continue

        serial = get_string_property(vrsystem, index, openvr.Prop_SerialNumber_String) or f"tracker_{index}"
        candidates.append((index, serial, poses[index]))

    if not candidates:
        raise RuntimeError("SteamVR 中未发现 GenericTracker，请确认 tracker 已在线。")

    if preferred_serial is not None:
        candidates = [item for item in candidates if item[1] == preferred_serial]
        if not candidates:
            raise RuntimeError(f"未找到指定 tracker serial: {preferred_serial}")

    for index, serial, pose in candidates:
        if not pose.bDeviceIsConnected or not pose.bPoseIsValid:
            continue
        return TrackerPose(index=index, serial=serial, transform=mat34_to_matrix(pose.mDeviceToAbsoluteTracking))

    serials = [serial for _, serial, _ in candidates]
    raise RuntimeError(f"找到 tracker 但当前没有有效 pose: {serials}")


def print_body_axes(pose: TrackerPose, decimals: int):
    rotation = pose.rotation
    quat = rotation_matrix_to_quaternion_wxyz(rotation)
    rpy_deg = np.rad2deg(rotation_matrix_to_rpy(rotation))

    # OpenVR 位姿矩阵列向量含义：tracker 本体轴在 SteamVR 世界坐标中的方向。
    body_x_in_world = rotation[:, 0]
    body_y_in_world = rotation[:, 1]
    body_z_in_world = rotation[:, 2]

    print(f"serial={pose.serial}, index={pose.index}")
    print("OpenVR standing universe 绝对位姿:")
    print(f"  {format_vector('world_xyz_m', pose.xyz, decimals)}")
    print(f"  {format_vector('world_rpy_deg', rpy_deg, 2)}")
    print(f"  {format_vector('world_quat_wxyz', quat, decimals)}")
    print("tracker 本体局部轴在 OpenVR 世界坐标中的方向:")
    print(f"  tracker +X -> {np.round(body_x_in_world, decimals).tolist()}")
    print(f"  tracker +Y -> {np.round(body_y_in_world, decimals).tolist()}")
    print(f"  tracker +Z -> {np.round(body_z_in_world, decimals).tolist()}")


def print_reference_delta(
    current_pose: TrackerPose,
    reference_pose: TrackerPose,
    decimals: int,
):
    delta_T = np.linalg.inv(reference_pose.get_T) @ current_pose.get_T
    delta_xyz_in_reference = delta_T[:3, 3].astype(float)
    delta_xyz_in_target = R_REFERENCE_TO_TARGET @ delta_xyz_in_reference
    relative_rotation = delta_T[:3, :3].astype(float)
    relative_quat = rotation_matrix_to_quaternion_wxyz(relative_rotation)
    relative_rotvec_deg = np.rad2deg(quaternion_to_rotvec_wxyz(relative_quat))
    relative_rpy_deg = np.rad2deg(rotation_matrix_to_rpy(relative_rotation))

    print("相对初始化参考系:")
    print(f"  {format_vector('delta_xyz_ref_m', delta_xyz_in_reference, decimals)}")
    print(f"  {format_vector('delta_xyz_target_m', delta_xyz_in_target, decimals)}")
    print(f"  {format_vector('relative_rpy_ref_deg', relative_rpy_deg, 2)}")
    print(f"  {format_vector('relative_rotvec_ref_deg', relative_rotvec_deg, 2)}")


def run_probe(serial: str | None, rate_hz: float, decimals: int):
    openvr.init(openvr.VRApplication_Other)
    try:
        vrsystem = openvr.VRSystem()
        reference_pose = read_tracker_pose(vrsystem, serial)

        print("[INFO] 已锁定初始化参考系。")
        print("[INFO] 建议此刻让 tracker 和机械臂 base 坐标系按你期望的同方向摆放。")
        print("[INFO] 后续 delta_xyz_ref_m 就是在这个初始化参考系下的相对位移。")
        print("[INFO] 判断 tracker 本体轴时，分别把 tracker 向自己认为的 +X/+Y/+Z 方向移动或转动。")
        print()
        print_body_axes(reference_pose, decimals)

        period = 1.0 / max(rate_hz, 1.0e-6)
        while True:
            start = time.time()
            pose = read_tracker_pose(vrsystem, reference_pose.serial)
            print("\033[2J\033[H", end="")
            print_body_axes(pose, decimals)
            print()
            print_reference_delta(pose, reference_pose, decimals)

            sleep_time = period - (time.time() - start)
            if sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        openvr.shutdown()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="观察 SteamVR Tracker 本体坐标系与初始化参考系相对位姿")
    parser.add_argument("--tracker-serial", default=None, help="指定 tracker serial；不填则使用第一个有效 GenericTracker")
    parser.add_argument("--rate", type=float, default=DEFAULT_RATE_HZ, help="打印频率 Hz，默认 10")
    parser.add_argument("--decimals", type=int, default=DEFAULT_DECIMALS, help="xyz/方向向量小数位数，默认 4")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        run_probe(serial=args.tracker_serial, rate_hz=args.rate, decimals=args.decimals)
    except KeyboardInterrupt:
        print("\n[INFO] 已退出 tracker 本体坐标系观察。")


if __name__ == "__main__":
    main()
