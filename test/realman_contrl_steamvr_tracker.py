#!/usr/bin/env python3
"""使用 SteamVR/OpenVR 单个 Tracker 相对位姿控制 RealMan RM75-B 末端 xyz+rpy。"""

from __future__ import annotations

import argparse
import os
import select
import sys
import termios
import time
import tty
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import openvr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from xrobotoolkit_teleop.hardware.interface.rm75b import RM75BInterface

# 与项目现有 XR 输入控制坐标保持一致：
# OpenVR 原始坐标 x右/y上/z后 -> 机械臂控制世界系
R_HEADSET_TO_WORLD = np.array(
    [
        [0, 0, -1],
        [-1, 0, 0],
        [0, 1, 0],
    ],
    dtype=float,
)
R_REFERENCE_TO_TARGET = np.array(
    [
        [0.0, -1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
    ],
    dtype=float,
)

DEFAULT_ARM_IP = "10.10.10.100"
DEFAULT_ARM_PORT = 8080
DEFAULT_CONTROL_RATE_HZ = 50                
DEFAULT_XYZ_SCALE_FACTOR = 1.0
SCALE_OPTIONS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
DEFAULT_XYZ_AXIS_MAP = (0, 1, 2)
DEFAULT_ROTVEC_AXIS_MAP = (1, 0, 2)
DEFAULT_ROTVEC_AXIS_SIGN = (-1.0, 1.0, 1.0)
DEFAULT_WORLD_YAW_OFFSET_DEG = 0.0
DEFAULT_MAX_DELTA_M = 0.25
DEFAULT_TOGGLE_KEY = "space"
DEFAULT_POSE_LOG_INTERVAL_S = 1.0


@dataclass(frozen=True)
class TeleopConfig:
    arm_ip: str
    arm_port: int
    control_rate_hz: int
    xyz_scale_factor: float
    xyz_axis_map: tuple[int, int, int]
    rotvec_axis_map: tuple[int, int, int]
    rotvec_axis_sign: tuple[float, float, float]
    world_yaw_offset_deg: float
    debug_pose: bool
    pose_log_interval_s: float


@dataclass(frozen=True)
class TrackerPoseSample:
    serial: str
    raw_transform: np.ndarray
    raw_xyz: np.ndarray
    raw_quat_wxyz: np.ndarray
    project_world_xyz: np.ndarray
    project_world_quat_wxyz: np.ndarray
    control_xyz: np.ndarray
    control_quat_wxyz: np.ndarray


@dataclass(frozen=True)
class KeyCommand:
    action: str
    value: int | None = None


def read_arm_pose(arm: RM75BInterface) -> np.ndarray:
    """读取当前机械臂末端位姿 [x, y, z, rx, ry, rz]。"""
    ret, state = arm.arm.rm_get_current_arm_state()
    if ret != 0:
        raise RuntimeError(f"读取机械臂状态失败，ret={ret}")

    pose = np.asarray(state["pose"], dtype=float)
    if pose.shape[0] < 6 or not np.all(np.isfinite(pose[:6])):
        raise RuntimeError(f"机械臂位姿无效: {state.get('pose')}")
    return pose[:6].copy()


def parse_axis_map(value: str) -> tuple[int, int, int]:
    """解析 xyz 通道映射，例如 1,0,2。"""
    try:
        axis_map = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("轴映射必须是逗号分隔的整数，例如 1,0,2") from exc

    if len(axis_map) != 3 or sorted(axis_map) != [0, 1, 2]:
        raise argparse.ArgumentTypeError("轴映射必须包含且只包含 0,1,2，例如 1,0,2")
    return axis_map


def parse_axis_sign(value: str) -> tuple[float, float, float]:
    """解析旋转轴方向，例如 1,-1,1。"""
    try:
        signs = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("轴方向必须是逗号分隔数字，例如 1,-1,1") from exc

    if len(signs) != 3 or any(sign not in (-1.0, 1.0) for sign in signs):
        raise argparse.ArgumentTypeError("轴方向只能由 1 或 -1 组成，例如 1,-1,1")
    return signs


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
    """将四元数转换为旋转向量（angle-axis vector）。"""
    q = normalize_quaternion_wxyz(quat)
    angle = 2.0 * np.arccos(np.clip(q[0], -1.0, 1.0))
    sin_half_angle = np.sin(angle / 2.0)
    if angle < eps or sin_half_angle < eps:
        return np.zeros(3, dtype=float)

    axis = q[1:] / sin_half_angle
    return axis * angle


def rotvec_to_quaternion_wxyz(rotvec: np.ndarray, eps: float = 1.0e-6) -> np.ndarray:
    """将旋转向量转换为四元数 [w, x, y, z]。"""
    vec = np.asarray(rotvec, dtype=float)
    angle = float(np.linalg.norm(vec))
    if angle < eps:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

    axis = vec / angle
    half_angle = angle / 2.0
    sin_half_angle = np.sin(half_angle)
    quat = np.array(
        [
            np.cos(half_angle),
            axis[0] * sin_half_angle,
            axis[1] * sin_half_angle,
            axis[2] * sin_half_angle,
        ],
        dtype=float,
    )
    return normalize_quaternion_wxyz(quat)


def quaternion_wxyz_to_rotation_matrix(quat: np.ndarray) -> np.ndarray:
    """将四元数 [w, x, y, z] 转换为旋转矩阵。"""
    w, x, y, z = normalize_quaternion_wxyz(quat)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def build_z_axis_rotation_matrix(yaw_deg: float) -> np.ndarray:
    """构造绕控制坐标系 z 轴旋转的 3x3 矩阵，用于水平面朝向校准。"""
    yaw_rad = np.deg2rad(yaw_deg)
    cos_yaw = float(np.cos(yaw_rad))
    sin_yaw = float(np.sin(yaw_rad))
    return np.array(
        [
            [cos_yaw, -sin_yaw, 0.0],
            [sin_yaw, cos_yaw, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def convert_openvr_pose(
    transform: np.ndarray,
    world_yaw_offset_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """将 OpenVR pose 依次转换到项目坐标系和最终控制坐标系。"""
    raw_xyz = transform[:3, 3].astype(float)
    raw_rotation = transform[:3, :3].astype(float)
    raw_quat_wxyz = rotation_matrix_to_quaternion_wxyz(raw_rotation)

    project_world_xyz = R_HEADSET_TO_WORLD @ raw_xyz
    project_world_rotation = R_HEADSET_TO_WORLD @ raw_rotation @ R_HEADSET_TO_WORLD.T
    project_world_quat_wxyz = rotation_matrix_to_quaternion_wxyz(project_world_rotation)

    world_alignment_rotation = build_z_axis_rotation_matrix(world_yaw_offset_deg)
    control_xyz = world_alignment_rotation @ project_world_xyz
    control_rotation = (
        world_alignment_rotation
        @ project_world_rotation
        @ world_alignment_rotation.T
    )
    control_quat_wxyz = rotation_matrix_to_quaternion_wxyz(control_rotation)
    return (
        raw_xyz,
        raw_quat_wxyz,
        project_world_xyz,
        project_world_quat_wxyz,
        control_xyz,
        control_quat_wxyz,
    )


def clip_delta(delta_xyz: np.ndarray) -> np.ndarray:
    """限制单次按住期间的最大相对位移，避免 tracker 异常跳变直接传给机械臂。"""
    return np.clip(delta_xyz, -DEFAULT_MAX_DELTA_M, DEFAULT_MAX_DELTA_M)


def format_scale_options() -> str:
    """格式化离散 scale 档位，便于日志展示。"""
    return ",".join(f"{scale:g}" for scale in SCALE_OPTIONS)


def find_nearest_scale_index(scale: float) -> int:
    """返回与当前 scale 最接近的离散档位索引。"""
    return min(range(len(SCALE_OPTIONS)), key=lambda index: abs(SCALE_OPTIONS[index] - scale))


class SteamVrTrackerReader:
    """读取 SteamVR/OpenVR 中第一个可用 tracker 的位姿。"""

    def __init__(self, world_yaw_offset_deg: float):
        openvr.init(openvr.VRApplication_Other)
        self.vrsystem = openvr.VRSystem()
        self.tracking_universe = openvr.TrackingUniverseStanding
        self.world_yaw_offset_deg = world_yaw_offset_deg

    def _get_string_property(self, index: int, prop: int) -> str | None:
        try:
            return self.vrsystem.getStringTrackedDeviceProperty(index, prop)
        except Exception:
            return None

    def _find_tracker_index(self, preferred_serial: str | None) -> tuple[int, str]:
        poses = self.vrsystem.getDeviceToAbsoluteTrackingPose(
            self.tracking_universe,
            0.0,
            openvr.k_unMaxTrackedDeviceCount,
        )

        trackers: list[tuple[int, str, object]] = []
        for index in range(openvr.k_unMaxTrackedDeviceCount):
            if self.vrsystem.getTrackedDeviceClass(index) != openvr.TrackedDeviceClass_GenericTracker:
                continue

            serial = self._get_string_property(index, openvr.Prop_SerialNumber_String) or f"tracker_{index}"
            trackers.append((index, serial, poses[index]))

        if not trackers:
            raise RuntimeError("SteamVR 中未发现任何 GenericTracker，请先确认 tracker 已在设备列表中在线。")

        if preferred_serial is not None:
            for index, serial, pose in trackers:
                if serial != preferred_serial:
                    continue
                if not pose.bDeviceIsConnected:
                    raise RuntimeError(f"tracker 已断开连接: {serial}")
                if not pose.bPoseIsValid:
                    raise RuntimeError(f"tracker 当前 pose 无效: {serial}")
                return index, serial

            raise RuntimeError(f"上一次使用的 tracker 不存在: {preferred_serial}")

        for index, serial, pose in trackers:
            if pose.bDeviceIsConnected and pose.bPoseIsValid:
                return index, serial

        connected = [serial for _, serial, pose in trackers if pose.bDeviceIsConnected]
        if connected:
            raise RuntimeError(f"SteamVR 已识别 tracker，但当前没有有效 pose。当前设备: {connected}")

        raise RuntimeError("SteamVR 中存在 tracker，但都处于未连接状态。")

    def read_tracker_pose(self, preferred_serial: str | None) -> TrackerPoseSample:
        index, serial = self._find_tracker_index(preferred_serial)
        poses = self.vrsystem.getDeviceToAbsoluteTrackingPose(
            self.tracking_universe,
            0.0,
            openvr.k_unMaxTrackedDeviceCount,
        )
        pose = poses[index]
        if not pose.bDeviceIsConnected:
            raise RuntimeError(f"tracker 已断开连接: {serial}")
        if not pose.bPoseIsValid:
            raise RuntimeError(f"tracker 当前 pose 无效: {serial}")

        transform = mat34_to_matrix(pose.mDeviceToAbsoluteTracking)
        (
            raw_xyz,
            raw_quat_wxyz,
            project_world_xyz,
            project_world_quat_wxyz,
            control_xyz,
            control_quat_wxyz,
        ) = convert_openvr_pose(transform, self.world_yaw_offset_deg)
        return TrackerPoseSample(
            serial=serial,
            raw_transform=transform,
            raw_xyz=raw_xyz,
            raw_quat_wxyz=raw_quat_wxyz,
            project_world_xyz=project_world_xyz,
            project_world_quat_wxyz=project_world_quat_wxyz,
            control_xyz=control_xyz,
            control_quat_wxyz=control_quat_wxyz,
        )

    def close(self):
        openvr.shutdown()


class TerminalKeyMonitor:
    """终端即时按键监听器。

    通过 cbreak 模式逐字节读取终端输入，不需要回车即可执行：
    - Space: 切换跟随启停
    - 上/下方向键: 调整 xyz scale 档位
    - 数字 0-9: 直接切到预设 xyz scale 档位
    """

    def __init__(self, toggle_key: str):
        if not sys.stdin.isatty():
            raise RuntimeError("当前标准输入不是 TTY，无法在终端模式下监听按键。")

        self.toggle_key = toggle_key
        self.should_exit = False
        self.fd = sys.stdin.fileno()
        self.original_termios = termios.tcgetattr(self.fd)
        self.buffer = bytearray()
        self.pending_commands: list[KeyCommand] = []
        tty.setcbreak(self.fd)

        print(
            "[INFO] 当前使用终端按键模式："
            "请保持终端焦点。"
            "空格切换开始/停止跟随，方向键上/下调节 xyz scale，数字键 1-9 对应 0.1-0.9，数字键 0 对应 1.0。"
        )

    def _matches(self, char: str) -> bool:
        if self.toggle_key == "space":
            return char == " "
        return char.strip().lower() == self.toggle_key

    def _append_command(self, action: str, value: int | None = None):
        self.pending_commands.append(KeyCommand(action=action, value=value))

    def _drain_buffer(self):
        while self.buffer:
            first = self.buffer[0]
            if first in (0x03, 0x04):
                del self.buffer[:1]
                self.should_exit = True
                return

            if first == 0x1B:
                if len(self.buffer) == 1:
                    return
                if self.buffer[1] not in (0x5B, 0x4F):
                    del self.buffer[:1]
                    continue
                if len(self.buffer) < 3:
                    return

                escape_code = self.buffer[2]
                del self.buffer[:3]
                if escape_code == ord("A"):
                    self._append_command("scale_up")
                elif escape_code == ord("B"):
                    self._append_command("scale_down")
                continue

            char = chr(first)
            del self.buffer[:1]
            if self._matches(char):
                self._append_command("toggle")
            elif "0" <= char <= "9":
                self._append_command("scale_digit", int(char))

    def poll(self):
        while True:
            ready, _, _ = select.select([sys.stdin], [], [], 0.0)
            if not ready:
                break
            chunk = os.read(self.fd, 64)
            if not chunk:
                break
            self.buffer.extend(chunk)
            self._drain_buffer()

    def get_commands(self) -> list[KeyCommand]:
        commands = self.pending_commands[:]
        self.pending_commands.clear()
        return commands

    def close(self):
        try:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.original_termios)
        except Exception:
            pass


class RealmanSteamVrTrackerTeleop:
    """空格切换启停，用 SteamVR 单个 tracker 相对位姿驱动 RM75-B。"""

    def __init__(self, config: TeleopConfig):
        self.config = config
        self.arm: RM75BInterface | None = None
        self.tracker_reader: SteamVrTrackerReader | None = None
        self.key_monitor: TerminalKeyMonitor | None = None

        self.active_tracker_serial: str | None = None
        self.tracker_origin_transform: np.ndarray | None = None
        self.tracker_origin_xyz: np.ndarray | None = None
        self.tracker_origin_quat_wxyz: np.ndarray | None = None
        self.arm_origin_xyz: np.ndarray | None = None
        self.arm_origin_quat_wxyz: np.ndarray | None = None
        self.teleop_enabled = False
        self.xyz_scale_factor = self.config.xyz_scale_factor
        self.was_active = False
        self.waiting_for_tracker_pose = False
        self.last_tracker_pose_error: str | None = None
        self.last_warn_time = 0.0
        self.last_pose_log_time = 0.0

        self.init_hardware()
        self.log_info(
            "初始化完成：按一次空格开始控制，再按一次空格停止跟随。"
            "开始控制瞬间的 tracker 本体坐标系会作为本次控制参考系。"
        )
        self.log_info(f"当前 tracker xyz scale={self.xyz_scale_factor:g}，预设档位: {format_scale_options()}")

    def log_info(self, message: str):
        print(f"[INFO] {message}")

    def log_warning(self, message: str):
        print(f"[WARN] {message}")

    def init_hardware(self):
        self.tracker_reader = SteamVrTrackerReader(self.config.world_yaw_offset_deg)
        self.key_monitor = TerminalKeyMonitor(DEFAULT_TOGGLE_KEY)
        self.arm = RM75BInterface(self.config.arm_ip, self.config.arm_port, enable_gripper=False)
        current_pose = read_arm_pose(self.arm)
        self.log_info(f"机械臂连接成功，当前末端位姿: {current_pose.tolist()}")
        self.log_info(
            f"当前控制参数: xyz_axis_map={self.config.xyz_axis_map}, "
            f"xyz_scale={self.xyz_scale_factor}, "
            f"rotvec_axis_map={self.config.rotvec_axis_map}, "
            f"rotvec_axis_sign={self.config.rotvec_axis_sign}, "
            f"xyz_reference_to_target={R_REFERENCE_TO_TARGET.tolist()}"
        )

    def is_control_active(self) -> bool:
        assert self.key_monitor is not None
        self.key_monitor.poll()
        if self.key_monitor.should_exit:
            raise KeyboardInterrupt
        for command in self.key_monitor.get_commands():
            self.handle_key_command(command)
        return self.teleop_enabled

    def handle_key_command(self, command: KeyCommand):
        if command.action == "toggle":
            self.teleop_enabled = not self.teleop_enabled
            state_text = "启动" if self.teleop_enabled else "停止"
            self.log_info(f"空格触发：{state_text} SteamVR Tracker 遥操。")
            return

        if command.action == "scale_up":
            self.adjust_scale_by_step(1, "方向键上")
            return

        if command.action == "scale_down":
            self.adjust_scale_by_step(-1, "方向键下")
            return

        if command.action == "scale_digit" and command.value is not None:
            self.set_scale_by_digit(command.value)

    def adjust_scale_by_step(self, direction: int, source: str):
        current_index = find_nearest_scale_index(self.xyz_scale_factor)
        next_index = max(0, min(len(SCALE_OPTIONS) - 1, current_index + direction))
        if next_index == current_index:
            self.log_info(f"{source}：tracker xyz scale 已在边界 {self.xyz_scale_factor:g}")
            return

        self.xyz_scale_factor = SCALE_OPTIONS[next_index]
        self.log_info(f"{source}：tracker xyz scale 调整为 {self.xyz_scale_factor:g}")

    def set_scale_by_digit(self, digit: int):
        if digit == 0:
            scale_index = len(SCALE_OPTIONS) - 1
        elif 1 <= digit <= min(9, len(SCALE_OPTIONS)):
            scale_index = digit - 1
        else:
            self.log_warning(f"数字键 {digit} 超出可用 scale 档位范围。")
            return

        self.xyz_scale_factor = SCALE_OPTIONS[scale_index]
        self.log_info(f"数字键 {digit}：tracker xyz scale 调整为 {self.xyz_scale_factor:g}")

    def read_tracker_pose(self) -> TrackerPoseSample:
        assert self.tracker_reader is not None
        sample = self.tracker_reader.read_tracker_pose(self.active_tracker_serial)
        self.active_tracker_serial = sample.serial
        return sample

    def activate_control(self, tracker_sample: TrackerPoseSample):
        assert self.arm is not None
        self.tracker_origin_transform = tracker_sample.raw_transform.copy()
        self.tracker_origin_xyz = tracker_sample.control_xyz.copy()
        self.tracker_origin_quat_wxyz = tracker_sample.control_quat_wxyz.copy()

        arm_pose = read_arm_pose(self.arm)
        self.arm_origin_xyz = arm_pose[:3].copy()
        self.arm_origin_quat_wxyz = normalize_quaternion_wxyz(
            np.asarray(self.arm.arm.rm_algo_euler2quaternion(arm_pose[3:6].tolist()), dtype=float)
        )
        self.was_active = True

        self.log_info(
            "tracker 控制已激活，记录控制原点: "
            f"serial={tracker_sample.serial}, "
            f"raw_xyz={tracker_sample.raw_xyz.tolist()}, "
            f"project_world_xyz={tracker_sample.project_world_xyz.tolist()}, "
            f"control_xyz={tracker_sample.control_xyz.tolist()}, "
            f"arm_xyz={self.arm_origin_xyz.tolist()}"
        )

    def should_log_pose_debug(self) -> bool:
        if not self.config.debug_pose:
            return False

        now = time.time()
        if now - self.last_pose_log_time < self.config.pose_log_interval_s:
            return False

        self.last_pose_log_time = now
        return True

    def log_pose_debug(
        self,
        tracker_sample: TrackerPoseSample,
        raw_delta_xyz: np.ndarray | None = None,
        mapped_delta_xyz: np.ndarray | None = None,
        raw_delta_rotvec: np.ndarray | None = None,
        mapped_delta_rotvec: np.ndarray | None = None,
    ):
        if not self.should_log_pose_debug():
            return

        raw_rpy_deg = np.rad2deg(
            rotation_matrix_to_rpy(quaternion_wxyz_to_rotation_matrix(tracker_sample.raw_quat_wxyz))
        )
        project_world_rpy_deg = np.rad2deg(
            rotation_matrix_to_rpy(quaternion_wxyz_to_rotation_matrix(tracker_sample.project_world_quat_wxyz))
        )
        control_rpy_deg = np.rad2deg(
            rotation_matrix_to_rpy(quaternion_wxyz_to_rotation_matrix(tracker_sample.control_quat_wxyz))
        )

        message_lines = [
            "tracker 位姿调试:",
            f"  serial={tracker_sample.serial}",
            f"  raw_openvr_xyz={np.round(tracker_sample.raw_xyz, 4).tolist()}, raw_openvr_rpy_deg={np.round(raw_rpy_deg, 2).tolist()}",
            f"  project_world_xyz={np.round(tracker_sample.project_world_xyz, 4).tolist()}, project_world_rpy_deg={np.round(project_world_rpy_deg, 2).tolist()}",
            f"  control_xyz={np.round(tracker_sample.control_xyz, 4).tolist()}, control_rpy_deg={np.round(control_rpy_deg, 2).tolist()}, world_yaw_offset_deg={self.config.world_yaw_offset_deg}",
            f"  rotvec_axis_map={self.config.rotvec_axis_map}, rotvec_axis_sign={self.config.rotvec_axis_sign}",
            f"  current_xyz_scale={self.xyz_scale_factor:g}",
        ]
        if raw_delta_xyz is not None and mapped_delta_xyz is not None:
            message_lines.append(
                f"  raw_delta_xyz={np.round(raw_delta_xyz, 4).tolist()}, mapped_delta_xyz={np.round(mapped_delta_xyz, 4).tolist()}"
            )
        if raw_delta_rotvec is not None and mapped_delta_rotvec is not None:
            message_lines.append(
                f"  raw_delta_rotvec_deg={np.round(np.rad2deg(raw_delta_rotvec), 2).tolist()}, "
                f"mapped_delta_rotvec_deg={np.round(np.rad2deg(mapped_delta_rotvec), 2).tolist()}"
            )
        self.log_info("\n".join(message_lines))

    def clear_control_state(self):
        self.tracker_origin_xyz = None
        self.tracker_origin_transform = None
        self.tracker_origin_quat_wxyz = None
        self.arm_origin_xyz = None
        self.arm_origin_quat_wxyz = None
        self.was_active = False

    def deactivate_control(self):
        assert self.arm is not None
        ret = self.arm.arm.rm_set_arm_slow_stop()
        if ret != 0:
            self.log_warning(f"停止跟随后缓停指令返回异常: ret={ret}")

        self.clear_control_state()
        self.waiting_for_tracker_pose = False
        self.last_tracker_pose_error = None
        self.log_info("已停止发送控制并清空 tracker 原点。")

    def fail_safe_stop(self, reason: str):
        assert self.arm is not None
        self.log_warning(f"控制异常，执行缓停并清空控制状态: {reason}")
        if self.was_active:
            ret = self.arm.arm.rm_set_arm_slow_stop()
            if ret != 0:
                self.log_warning(f"异常缓停指令返回异常: ret={ret}")
        self.clear_control_state()

    @staticmethod
    def is_tracker_pose_runtime_error(exc: Exception) -> bool:
        message = str(exc)
        return (
            "tracker 当前 pose 无效" in message
            or "tracker 已断开连接" in message
            or "当前没有有效 pose" in message
        )

    def pause_for_tracker_pose_loss(self, reason: str):
        assert self.arm is not None
        first_pause = not self.waiting_for_tracker_pose

        if first_pause and self.was_active:
            ret = self.arm.arm.rm_set_arm_slow_stop()
            if ret != 0:
                self.log_warning(f"tracker 丢追踪后的缓停指令返回异常: ret={ret}")

        if first_pause:
            self.clear_control_state()

        self.waiting_for_tracker_pose = True
        self.last_tracker_pose_error = reason

        now = time.time()
        if first_pause or now - self.last_warn_time > 1.0:
            self.log_warning(
                f"{reason}；当前暂停发送控制，等待 tracker 恢复有效 pose。"
                "恢复后若当前仍处于启用状态，将以恢复瞬间重新建立控制原点。"
            )
            self.last_warn_time = now

    def on_tracker_pose_recovered(self, tracker_sample: TrackerPoseSample):
        if not self.waiting_for_tracker_pose:
            return

        self.waiting_for_tracker_pose = False
        previous_error = self.last_tracker_pose_error
        self.last_tracker_pose_error = None
        self.log_info(
            "tracker 已恢复有效 pose，允许重新激活控制: "
            f"serial={tracker_sample.serial}, "
            f"control_xyz={tracker_sample.control_xyz.tolist()}, "
            f"上一次异常={previous_error}"
        )

    def send_target_pose(self, tracker_sample: TrackerPoseSample):
        assert self.arm is not None
        assert self.tracker_origin_transform is not None
        assert self.tracker_origin_xyz is not None
        assert self.tracker_origin_quat_wxyz is not None
        assert self.arm_origin_xyz is not None
        assert self.arm_origin_quat_wxyz is not None

        delta_transform = np.linalg.inv(self.tracker_origin_transform) @ tracker_sample.raw_transform
        raw_delta_xyz = delta_transform[:3, 3].astype(float)
        raw_delta_xyz = R_REFERENCE_TO_TARGET @ raw_delta_xyz
        delta_xyz = raw_delta_xyz[list(self.config.xyz_axis_map)] * self.xyz_scale_factor
        delta_xyz = clip_delta(delta_xyz)

        raw_delta_quat_wxyz = quaternion_multiply_wxyz(
            tracker_sample.control_quat_wxyz,
            quaternion_conjugate_wxyz(self.tracker_origin_quat_wxyz),
        )
        raw_delta_rotvec = quaternion_to_rotvec_wxyz(raw_delta_quat_wxyz)

        # SteamVR 相对旋转向量 -> 机械臂姿态增量的通道映射。
        delta_rotvec = raw_delta_rotvec[list(self.config.rotvec_axis_map)]
        delta_rotvec = delta_rotvec * np.asarray(self.config.rotvec_axis_sign, dtype=float)
        delta_quat_wxyz = rotvec_to_quaternion_wxyz(delta_rotvec)
        target_quat_wxyz = quaternion_multiply_wxyz(delta_quat_wxyz, self.arm_origin_quat_wxyz)

        self.log_pose_debug(
            tracker_sample=tracker_sample,
            raw_delta_xyz=raw_delta_xyz,
            mapped_delta_xyz=delta_xyz,
            raw_delta_rotvec=raw_delta_rotvec,
            mapped_delta_rotvec=delta_rotvec,
        )

        target_xyz = self.arm_origin_xyz + delta_xyz
        command_pose = np.concatenate([target_xyz, target_quat_wxyz])

        ret = self.arm.arm.rm_movep_canfd(command_pose.tolist(), follow=False)
        if ret != 0:
            now = time.time()
            if now - self.last_warn_time > 1.0:
                self.log_warning(f"rm_movep_canfd 返回异常: ret={ret}, target={command_pose.tolist()}")
                self.last_warn_time = now

    def control_loop(self):
        active = self.is_control_active()
        if not active:
            if self.was_active:
                self.deactivate_control()
            else:
                self.waiting_for_tracker_pose = False
                self.last_tracker_pose_error = None
            return

        try:
            tracker_sample = self.read_tracker_pose()
        except RuntimeError as exc:
            if self.is_tracker_pose_runtime_error(exc):
                self.pause_for_tracker_pose_loss(str(exc))
                return
            raise

        self.on_tracker_pose_recovered(tracker_sample)
        if not self.was_active:
            self.activate_control(tracker_sample)
        else:
            self.log_pose_debug(tracker_sample=tracker_sample)
        self.send_target_pose(tracker_sample)

    def run(self):
        period = 1.0 / float(self.config.control_rate_hz)
        while True:
            start_time = time.time()
            try:
                self.control_loop()
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                self.fail_safe_stop(str(exc))
                now = time.time()
                if now - self.last_warn_time > 1.0:
                    self.log_warning(f"控制循环异常: {exc}")
                    self.last_warn_time = now

            sleep_time = period - (time.time() - start_time)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def cleanup(self):
        if self.arm is not None:
            try:
                if self.was_active:
                    self.arm.arm.rm_set_arm_slow_stop()
                self.arm.close()
            except Exception as exc:
                self.log_warning(f"关闭机械臂连接失败: {exc}")
            self.arm = None

        if self.tracker_reader is not None:
            try:
                self.tracker_reader.close()
            except Exception as exc:
                self.log_warning(f"关闭 OpenVR 失败: {exc}")
            self.tracker_reader = None

        if self.key_monitor is not None:
            self.key_monitor.close()
            self.key_monitor = None


def parse_args(argv: list[str] | None = None) -> TeleopConfig:
    parser = argparse.ArgumentParser(description="SteamVR 单 tracker 相对位姿控制 RealMan RM75-B")
    parser.add_argument("--ip", default=DEFAULT_ARM_IP, help="RM75-B 控制器 IP")
    parser.add_argument("--port", type=int, default=DEFAULT_ARM_PORT, help="RM75-B 控制器端口")
    parser.add_argument("--rate", type=int, default=DEFAULT_CONTROL_RATE_HZ, help="控制频率 Hz")
    parser.add_argument("--xyz-scale", type=float, default=DEFAULT_XYZ_SCALE_FACTOR, help="tracker xyz 位移到机械臂 xyz 位移的比例")
    parser.add_argument("--xyz-axis-map", type=parse_axis_map, default=DEFAULT_XYZ_AXIS_MAP, help="xyz 通道映射，默认 0,1,2")
    parser.add_argument(
        "--rotvec-axis-map",
        type=parse_axis_map,
        default=DEFAULT_ROTVEC_AXIS_MAP,
        help="相对旋转向量通道映射，默认 1,0,2；若出现绕 z 实际绕 x，可在现场直接调这个参数",
    )
    parser.add_argument(
        "--rotvec-axis-sign",
        type=parse_axis_sign,
        default=DEFAULT_ROTVEC_AXIS_SIGN,
        help="相对旋转向量各轴方向，默认 1,1,1；若某个旋转方向反了可改成 -1",
    )
    parser.add_argument(
        "--world-yaw-offset-deg",
        type=float,
        default=DEFAULT_WORLD_YAW_OFFSET_DEG,
        help="在项目控制坐标系基础上，再绕 z 轴做一次水平朝向校准；若平移整体偏转可先试 90 或 -90",
    )
    parser.add_argument("--debug-pose", action="store_true", help="按低频打印 tracker 原始绝对位姿、项目坐标位姿和最终控制位姿")
    parser.add_argument(
        "--pose-log-interval",
        type=float,
        default=DEFAULT_POSE_LOG_INTERVAL_S,
        help="--debug-pose 打印周期，单位秒，默认 1.0",
    )
    args, _ = parser.parse_known_args(argv)

    return TeleopConfig(
        arm_ip=args.ip,
        arm_port=args.port,
        control_rate_hz=args.rate,
        xyz_scale_factor=args.xyz_scale,
        xyz_axis_map=args.xyz_axis_map,
        rotvec_axis_map=args.rotvec_axis_map,
        rotvec_axis_sign=args.rotvec_axis_sign,
        world_yaw_offset_deg=args.world_yaw_offset_deg,
        debug_pose=args.debug_pose,
        pose_log_interval_s=max(0.0, float(args.pose_log_interval)),
    )


def main(args: list[str] | None = None):
    config = parse_args(args)
    teleop = RealmanSteamVrTrackerTeleop(config)
    try:
        teleop.run()
    except KeyboardInterrupt:
        teleop.log_info("检测到退出信号，正在关闭。")
    finally:
        teleop.cleanup()


if __name__ == "__main__":
    main()
