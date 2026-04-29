#!/usr/bin/env python3
"""使用 Pico Motion Tracker 相对位姿控制 RealMan RM75-B 末端 xyz+rpy。"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from xrobotoolkit_teleop.common.xr_client import XrClient
from xrobotoolkit_teleop.hardware.interface.rm75b import RM75BInterface
from xrobotoolkit_teleop.utils.geometry import R_HEADSET_TO_WORLD

DEFAULT_ARM_IP = "192.168.5.154"
DEFAULT_ARM_PORT = 8080
DEFAULT_CONTROL_RATE_HZ = 50
SCALE_OPTIONS = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
DEFAULT_XYZ_SCALE_FACTOR = 1.0
DEFAULT_RPY_SCALE_FACTOR = 1.0
DEFAULT_RPY_AXIS_MAP = (1, 0, 2)
DEFAULT_RPY_AXIS_SIGN = (1.0, 1.0, 1.0)
DEFAULT_MAX_DELTA_M = 0.25
DEFAULT_RESET_ARM_POSE = (0.1166,0.0,0.7247,0.0,1.043,0.0)
DEFAULT_RESET_STREAM_DURATION_S = 0.5
JOYSTICK_SCALE_THRESHOLD = 0.6


@dataclass(frozen=True)
class TeleopConfig:
    arm_ip: str
    arm_port: int
    control_rate_hz: int
    xyz_scale_factor: float
    rpy_scale_factor: float
    rpy_axis_map: tuple[int, int, int]
    rpy_axis_sign: tuple[float, float, float]
    max_delta_m: float
    reset_arm_pose: tuple[float, ...]
    high_follow: bool
    slow_stop_on_release: bool
    enable_ros_publish: bool


def describe_robotic_arm_package() -> str:
    """返回 RealMan Python 包来源，用于区分 pip 安装和本地源码导入。"""
    spec = importlib.util.find_spec("Robotic_Arm.rm_robot_interface")
    origin = spec.origin if spec and spec.origin else "未找到"
    try:
        version = importlib.metadata.version("Robotic_Arm")
    except importlib.metadata.PackageNotFoundError:
        version = "未知"
    source_type = "pip/site-packages" if "site-packages" in origin else "本地路径/源码"
    return f"Robotic_Arm={version}, 来源={source_type}, 路径={origin}"


def read_arm_pose(arm: RM75BInterface) -> np.ndarray:
    """读取当前机械臂末端位姿 [x, y, z, rx, ry, rz]。"""
    ret, state = arm.arm.rm_get_current_arm_state()
    if ret != 0:
        raise RuntimeError(f"读取机械臂状态失败，ret={ret}")
    pose = np.asarray(state["pose"], dtype=float)
    if pose.shape[0] < 6 or not np.all(np.isfinite(pose[:6])):
        raise RuntimeError(f"机械臂位姿无效: {state.get('pose')}")
    return pose[:6].copy()


def wrap_angle(angle: np.ndarray | float) -> np.ndarray | float:
    """将角度归一到 [-pi, pi]，避免跨 pi 时产生大跳变。"""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def parse_rpy_axis_map(value: str) -> tuple[int, int, int]:
    """解析 rpy 轴映射，例如 1,0,2 表示交换 roll/pitch，yaw 保持不变。"""
    try:
        axis_map = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("rpy 轴映射必须是逗号分隔的整数，例如 1,0,2") from exc

    if len(axis_map) != 3 or sorted(axis_map) != [0, 1, 2]:
        raise argparse.ArgumentTypeError("rpy 轴映射必须包含且只包含 0,1,2，例如 1,0,2")
    return axis_map


def parse_rpy_axis_sign(value: str) -> tuple[float, float, float]:
    """解析 rpy 轴方向，例如 1,-1,1 表示反转第二个旋转通道。"""
    try:
        signs = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("rpy 轴方向必须是逗号分隔数字，例如 1,-1,1") from exc

    if len(signs) != 3 or any(sign not in (-1.0, 1.0) for sign in signs):
        raise argparse.ArgumentTypeError("rpy 轴方向只能由 1 或 -1 组成，例如 1,-1,1")
    return signs


def parse_scale_option(value: str) -> float:
    """解析离散 scale 档位。"""
    try:
        scale = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"scale 必须是数字，可选: {format_scale_options()}"
        ) from exc

    if scale not in SCALE_OPTIONS:
        raise argparse.ArgumentTypeError(f"scale 只能选择: {format_scale_options()}")
    return scale


def parse_reset_arm_pose(value: str) -> tuple[float, ...]:
    """解析 B 键复位目标末端位姿 [x,y,z,rx,ry,rz]。"""
    try:
        pose = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "复位末端位姿必须是逗号分隔数字，例如 0.43261,0.028079,0.026739,2.479,1.491,2.482"
        ) from exc

    if len(pose) != 6:
        raise argparse.ArgumentTypeError("复位末端位姿需要 6 个值: x,y,z,rx,ry,rz")
    return pose


def format_scale_options() -> str:
    """格式化 scale 档位，便于日志和 argparse 错误展示。"""
    return ", ".join(str(item) for item in SCALE_OPTIONS)


def quaternion_xyzw_to_rotation_matrix(quat_xyzw: np.ndarray, source_name: str = "位姿") -> np.ndarray:
    """将 SDK 返回的 xyzw 四元数转换成 3x3 旋转矩阵。"""
    quat = np.asarray(quat_xyzw, dtype=float)
    norm = np.linalg.norm(quat)
    if norm <= 1.0e-9 or not np.all(np.isfinite(quat)):
        raise RuntimeError(f"{source_name} 四元数无效: {quat_xyzw}")

    x, y, z, w = quat / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
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


def parse_pose_7d(value: Any, source_name: str) -> np.ndarray:
    """解析 `[x,y,z,qx,qy,qz,qw]`，兼容数组和逗号分隔字符串。"""
    if isinstance(value, str):
        pose = np.asarray([float(item.strip()) for item in value.split(",")], dtype=float)
    else:
        pose = np.asarray(value, dtype=float).reshape(-1)

    if pose.shape[0] < 7 or not np.all(np.isfinite(pose[:7])):
        raise RuntimeError(f"{source_name} 位姿无效: {value}")
    return pose[:7].copy()


def iter_motion_tracker_pose_values(tracker_data: dict) -> list[tuple[str, np.ndarray]]:
    """遍历所有 SN 下可用的 tracker `p/pose`，不做 SN 过滤。"""
    candidates: list[tuple[str, np.ndarray]] = []

    if "joints" in tracker_data:
        sn = str(tracker_data.get("sn", "motion_tracker"))
        for index, joint in enumerate(tracker_data.get("joints", [])):
            if isinstance(joint, dict) and "p" in joint:
                candidates.append((f"{sn}#{index}", parse_pose_7d(joint["p"], f"tracker {sn}#{index}")))
        return candidates

    for serial in sorted(tracker_data.keys(), key=str):
        metrics = tracker_data[serial]
        serial_text = str(serial)

        if isinstance(metrics, dict) and "joints" in metrics:
            for index, joint in enumerate(metrics.get("joints", [])):
                if isinstance(joint, dict) and "p" in joint:
                    candidates.append(
                        (f"{serial_text}#{index}", parse_pose_7d(joint["p"], f"tracker {serial_text}#{index}"))
                    )
            continue

        if isinstance(metrics, dict) and "p" in metrics:
            candidates.append((serial_text, parse_pose_7d(metrics["p"], f"tracker {serial_text}")))
            continue

        if isinstance(metrics, dict) and "pose" in metrics:
            candidates.append((serial_text, parse_pose_7d(metrics["pose"], f"tracker {serial_text}")))
            continue

        candidates.append((serial_text, parse_pose_7d(metrics, f"tracker {serial_text}")))

    return candidates


def read_motion_tracker_pose(
    xr_client: XrClient,
    preferred_tracker_id: str | None = None,
) -> tuple[str, np.ndarray, np.ndarray]:
    """读取任意可用 Motion Tracker pose，并转换到项目现有控制坐标系的 xyz+rpy。"""
    tracker_data = xr_client.get_motion_tracker_data()
    if not tracker_data:
        raise RuntimeError("未读取到 Motion Tracker 数据")

    errors: list[str] = []
    seen_tracker_ids: list[str] = []
    found_preferred_tracker = False
    for tracker_id, pose in iter_motion_tracker_pose_values(tracker_data):
        seen_tracker_ids.append(tracker_id)
        if preferred_tracker_id is not None and tracker_id != preferred_tracker_id:
            continue
        found_preferred_tracker = True

        try:
            tracker_xyz = R_HEADSET_TO_WORLD @ pose[:3]
            tracker_rotation = quaternion_xyzw_to_rotation_matrix(pose[3:7], f"tracker {tracker_id}")
            tracker_rotation = R_HEADSET_TO_WORLD @ tracker_rotation @ R_HEADSET_TO_WORLD.T
            tracker_rpy = rotation_matrix_to_rpy(tracker_rotation)
            return tracker_id, tracker_xyz, tracker_rpy
        except RuntimeError as exc:
            errors.append(str(exc))

    if preferred_tracker_id is not None and not found_preferred_tracker:
        raise RuntimeError(
            f"已绑定的 Motion Tracker 不可用: {preferred_tracker_id}, 当前可用: {seen_tracker_ids}"
        )

    detail = "；".join(errors) if errors else "所有 tracker 数据均缺少 p/pose"
    raise RuntimeError(f"未读取到有效 Motion Tracker 位姿: {detail}")


def clip_delta(delta_xyz: np.ndarray, max_delta_m: float) -> np.ndarray:
    """限制单次按住期间最大相对位移，避免手柄异常跳变直接传给机械臂。"""
    if max_delta_m <= 0:
        return delta_xyz
    return np.clip(delta_xyz, -max_delta_m, max_delta_m)


class RealmanMotionTrackerTeleop:
    """A 键切换启停，用 Motion Tracker 相对位姿驱动 RM75-B 末端 xyz+rpy。"""

    def __init__(self, config: TeleopConfig):
        self.config = config
        self.xr_client: XrClient | None = None
        self.arm: RM75BInterface | None = None
        self.rclpy: Any | None = None
        self.ros_node: Any | None = None
        self.float32_multi_array_cls: Any | None = None
        self.target_pub: Any | None = None
        self.actual_pub: Any | None = None

        self.tracker_origin_xyz: np.ndarray | None = None
        self.tracker_origin_rpy: np.ndarray | None = None
        self.arm_origin_pose: np.ndarray | None = None
        self.target_pose: np.ndarray | None = None
        self.teleop_enabled = False
        self.was_active = False
        self.last_a_pressed = False
        self.last_b_pressed = False
        self.last_scale_direction = 0
        self.active_tracker_id: str | None = None
        self.xyz_scale_factor = self.config.xyz_scale_factor
        self.a_start_event: dict[str, Any] | None = None
        self.last_warn_time = 0.0

        if self.config.enable_ros_publish:
            self.init_ros_publishers()
        self.init_hardware()
        self.log_info(
            "初始化完成：按 A 键启动 Motion Tracker 遥操，再按 A 键停止并清空追踪原点。"
        )
        self.log_info(f"当前 tracker xyz scale={self.xyz_scale_factor:g}，可选档位: {format_scale_options()}")
        self.log_info(f"B 键复位目标末端位姿 pose={list(self.config.reset_arm_pose)}")

    def log_info(self, message: str):
        if self.ros_node is not None:
            self.ros_node.get_logger().info(message)
        else:
            print(f"[INFO] {message}")

    def log_warning(self, message: str):
        if self.ros_node is not None:
            self.ros_node.get_logger().warning(message)
        else:
            print(f"[WARN] {message}")

    def init_ros_publishers(self):
        """按需启用 ROS2 发布，避免非 ROS Python 环境导入 rclpy 失败。"""
        try:
            import rclpy
            from std_msgs.msg import Float32MultiArray
        except Exception as exc:
            raise RuntimeError(
                "启用 --enable-ros-publish 需要当前 Python 环境能导入 rclpy。"
                "ROS Humble 默认匹配 Python 3.10，不能直接在 Python 3.13 conda 环境中使用。"
            ) from exc

        rclpy.init(args=None)
        self.rclpy = rclpy
        self.float32_multi_array_cls = Float32MultiArray
        self.ros_node = rclpy.create_node("realman_motion_tracker_teleop")
        self.target_pub = self.ros_node.create_publisher(Float32MultiArray, "/action", 10)
        self.actual_pub = self.ros_node.create_publisher(Float32MultiArray, "/state", 10)

    def init_hardware(self):
        self.log_info(describe_robotic_arm_package())
        self.xr_client = XrClient()
        self.arm = RM75BInterface(self.config.arm_ip, self.config.arm_port, enable_gripper=False)
        self.target_pose = read_arm_pose(self.arm)
        self.log_info(f"机械臂连接成功，当前末端位姿: {self.target_pose.tolist()}")

    def is_control_active(self) -> bool:
        assert self.xr_client is not None
        a_pressed = bool(self.xr_client.get_button_state_by_name("A"))
        if a_pressed and not self.last_a_pressed:
            self.teleop_enabled = not self.teleop_enabled
            state_text = "启动" if self.teleop_enabled else "停止"
            self.log_info(f"A 键触发：{state_text} Motion Tracker 遥操。")
        self.last_a_pressed = a_pressed
        return self.teleop_enabled

    def update_scale_from_joystick(self):
        """用右手柄摇杆左右方向离散调整 tracker xyz scale。"""
        assert self.xr_client is not None
        joystick = self.xr_client.get_joystick_state("right")
        if len(joystick) < 1:
            return

        axis_x = float(joystick[0])
        if axis_x <= -JOYSTICK_SCALE_THRESHOLD:
            direction = -1
        elif axis_x >= JOYSTICK_SCALE_THRESHOLD:
            direction = 1
        else:
            self.last_scale_direction = 0
            return

        if direction == self.last_scale_direction:
            return

        current_index = SCALE_OPTIONS.index(self.xyz_scale_factor)
        next_index = max(0, min(len(SCALE_OPTIONS) - 1, current_index + direction))
        self.last_scale_direction = direction
        if next_index == current_index:
            self.log_info(f"tracker xyz scale 已在边界: {self.xyz_scale_factor:g}")
            return

        self.xyz_scale_factor = SCALE_OPTIONS[next_index]
        self.log_info(f"tracker xyz scale 调整为 {self.xyz_scale_factor:g}")

    def reset_arm_to_initial_pose(self):
        """B 键复位：停止遥操并移动到固定初始关节姿态。"""
        assert self.arm is not None
        self.teleop_enabled = False
        self.tracker_origin_xyz = None
        self.tracker_origin_rpy = None
        self.arm_origin_pose = None
        self.active_tracker_id = None
        self.a_start_event = None
        self.was_active = False
        self.target_pose = None
        ret = self.arm.arm.rm_set_arm_slow_stop()
        if ret != 0:
            self.log_warning(f"B 键复位前缓停指令返回异常: ret={ret}")
        time.sleep(0.2)
        reset_pose = np.asarray(self.config.reset_arm_pose, dtype=float)
        self.target_pose = reset_pose.copy()
        self.log_info(f"B 键复位：移动到固定末端位姿 pose={reset_pose.tolist()}")
        period = 1.0 / float(self.config.control_rate_hz)
        send_count = max(1, int(DEFAULT_RESET_STREAM_DURATION_S * self.config.control_rate_hz))
        for _ in range(send_count):
            ret = self.arm.arm.rm_movep_canfd(reset_pose.tolist(), follow=False)
            if ret != 0:
                self.log_warning(f"B 键复位 rm_movep_canfd 返回异常: ret={ret}, target={reset_pose.tolist()}")
                break
            time.sleep(period)
        self.log_info("B 键复位指令已结束。")

    def handle_reset_button(self) -> bool:
        """处理 B 键上升沿；触发复位时返回 True，让本轮控制循环提前结束。"""
        assert self.xr_client is not None
        b_pressed = bool(self.xr_client.get_button_state_by_name("B"))
        should_reset = b_pressed and not self.last_b_pressed
        self.last_b_pressed = b_pressed
        if not should_reset:
            return False

        self.reset_arm_to_initial_pose()
        self.publish_state()
        return True

    def record_a_start_event(
        self,
        tracker_id: str,
        tracker_xyz: np.ndarray,
        arm_pose: np.ndarray,
    ):
        """记录 A 键启动遥操瞬间的 tracker 和机械臂 xyz。"""
        self.a_start_event = {
            "tracker_id": tracker_id,
            "tracker_xyz": tracker_xyz.copy(),
            "arm_pose": arm_pose.copy(),
            "xyz_scale_factor": self.xyz_scale_factor,
        }
        self.log_info(
            "A 键启动事件记录: "
            f"tracker_id={tracker_id}, "
            f"tracker_xyz={tracker_xyz.tolist()}, "
            f"arm_xyz={arm_pose[:3].tolist()}, "
            f"xyz_scale={self.xyz_scale_factor:g}"
        )

    def log_a_stop_event_comparison(
        self,
        stop_tracker_id: str | None,
        stop_tracker_xyz: np.ndarray | None,
        stop_arm_pose: np.ndarray | None,
    ):
        """记录 A 键停止瞬间数据，并比较 tracker xyz 与机械臂 xyz 偏移。"""
        self.log_info(
            "A 键停止事件记录: "
            f"tracker_id={stop_tracker_id}, "
            f"tracker_xyz={None if stop_tracker_xyz is None else stop_tracker_xyz.tolist()}, "
            f"arm_xyz={None if stop_arm_pose is None else stop_arm_pose[:3].tolist()}"
        )

        if self.a_start_event is None:
            self.log_warning("无法对比 A 键事件偏移：缺少启动事件记录。")
            return
        if stop_tracker_xyz is None or stop_arm_pose is None:
            self.log_warning("无法对比 A 键事件偏移：停止事件的 tracker 或机械臂位姿读取失败。")
            return

        start_tracker_id = str(self.a_start_event["tracker_id"])
        start_tracker_xyz = self.a_start_event["tracker_xyz"]
        start_arm_pose = self.a_start_event["arm_pose"]
        start_scale = float(self.a_start_event["xyz_scale_factor"])
        comparison_scale = self.xyz_scale_factor

        tracker_delta_xyz = stop_tracker_xyz - start_tracker_xyz
        expected_arm_delta_xyz = tracker_delta_xyz * comparison_scale
        expected_arm_delta_xyz = clip_delta(expected_arm_delta_xyz, self.config.max_delta_m)
        actual_arm_delta_xyz = stop_arm_pose[:3] - start_arm_pose[:3]
        xyz_error = actual_arm_delta_xyz - expected_arm_delta_xyz

        tracker_id_note = "无"
        if stop_tracker_id is not None and stop_tracker_id != start_tracker_id:
            tracker_id_note = f"{start_tracker_id}->{stop_tracker_id}"

        comparison_rows = "\n".join(
            (
                f"  {axis:<4}"
                f"{float(tracker_delta_xyz[index]):>18.6f}"
                f"{float(expected_arm_delta_xyz[index]):>24.6f}"
                f"{float(actual_arm_delta_xyz[index]):>22.6f}"
                f"{float(xyz_error[index]):>16.6f}"
            )
            for index, axis in enumerate(("x", "y", "z"))
        )

        self.log_info(
            "A 键事件 xyz 偏移对比:\n"
            f"  tracker_id: start={start_tracker_id}, stop={stop_tracker_id}, changed={tracker_id_note}\n"
            f"  计算公式: expected_arm_delta = clip(tracker_delta * xyz_scale, +/-max_delta_m)\n"
            f"  xyz_scale: current={comparison_scale:g}, start={start_scale:g}, "
            f"max_delta_m={self.config.max_delta_m}\n"
            "  轴       tracker位移(m)      期望机械臂位移(m)     实际机械臂位移(m)        误差(m)\n"
            f"{comparison_rows}"
        )

    def activate_control(self, tracker_id: str, tracker_xyz: np.ndarray, tracker_rpy: np.ndarray):
        assert self.arm is not None
        self.active_tracker_id = tracker_id
        self.tracker_origin_xyz = tracker_xyz.copy()
        self.tracker_origin_rpy = tracker_rpy.copy()
        self.arm_origin_pose = read_arm_pose(self.arm)
        self.target_pose = self.arm_origin_pose.copy()
        self.was_active = True
        self.record_a_start_event(tracker_id, tracker_xyz, self.arm_origin_pose)
        self.log_info(
            "Motion Tracker 遥操已激活，记录控制原点: "
            f"tracker_id={self.active_tracker_id}, "
            f"tracker_xyz={self.tracker_origin_xyz.tolist()}, "
            f"tracker_rpy={self.tracker_origin_rpy.tolist()}, "
            f"arm_pose={self.arm_origin_pose.tolist()}"
        )

    def deactivate_control(self):
        assert self.arm is not None
        stop_tracker_id = None
        stop_tracker_xyz = None
        stop_arm_pose = None

        if self.xr_client is not None:
            try:
                stop_tracker_id, stop_tracker_xyz, _ = read_motion_tracker_pose(
                    self.xr_client,
                    self.active_tracker_id,
                )
            except RuntimeError as exc:
                self.log_warning(f"A 键停止时读取 Motion Tracker 位姿失败: {exc}")

        try:
            stop_arm_pose = read_arm_pose(self.arm)
        except RuntimeError as exc:
            self.log_warning(str(exc))

        self.log_a_stop_event_comparison(stop_tracker_id, stop_tracker_xyz, stop_arm_pose)

        if self.config.slow_stop_on_release:
            ret = self.arm.arm.rm_set_arm_slow_stop()
            if ret != 0:
                self.log_warning(f"停止 tracker 遥操后缓停指令返回异常: ret={ret}")

        self.tracker_origin_xyz = None
        self.tracker_origin_rpy = None
        self.arm_origin_pose = None
        self.active_tracker_id = None
        self.target_pose = None
        self.a_start_event = None
        self.was_active = False
        self.log_info("Motion Tracker 遥操已停止，停止发送位姿透传并清空追踪原点。")

    def send_target_pose(self, tracker_xyz: np.ndarray, tracker_rpy: np.ndarray):
        assert self.arm is not None
        assert self.tracker_origin_xyz is not None
        assert self.tracker_origin_rpy is not None
        assert self.arm_origin_pose is not None

        delta_xyz = (tracker_xyz - self.tracker_origin_xyz) * self.xyz_scale_factor
        delta_xyz = clip_delta(delta_xyz, self.config.max_delta_m)
        raw_delta_rpy = wrap_angle(tracker_rpy - self.tracker_origin_rpy)
        delta_rpy = raw_delta_rpy[list(self.config.rpy_axis_map)]
        delta_rpy = delta_rpy * np.asarray(self.config.rpy_axis_sign) * self.config.rpy_scale_factor

        target_pose = self.arm_origin_pose.copy()
        target_pose[:3] = self.arm_origin_pose[:3] + delta_xyz
        target_pose[3:6] = wrap_angle(self.arm_origin_pose[3:6] + delta_rpy)
        self.target_pose = target_pose

        ret = self.arm.arm.rm_movep_canfd(target_pose.tolist(), follow=self.config.high_follow)
        if ret != 0:
            now = time.time()
            if now - self.last_warn_time > 1.0:
                self.log_warning(f"rm_movep_canfd 返回异常: ret={ret}, target={target_pose.tolist()}")
                self.last_warn_time = now

    def publish_state(self):
        # 关闭遥操循环内的状态发布，避免高频 rm_get_current_arm_state()
        # 与 rm_movep_canfd() 抢占同一 RealMan 通信链路。
        return

    def control_loop(self):
        assert self.xr_client is not None
        try:
            if self.handle_reset_button():
                return
            self.update_scale_from_joystick()
            active = self.is_control_active()

            # A 键切到停止时必须先清空控制引用；不要再依赖当前 tracker pose，
            # 否则停止瞬间定位异常会跳过 deactivate_control()，留下旧原点。
            if not active:
                if self.was_active:
                    self.deactivate_control()
                self.publish_state()
                return

            tracker_id, tracker_xyz, tracker_rpy = read_motion_tracker_pose(
                self.xr_client,
                self.active_tracker_id,
            )

            if not self.was_active:
                self.activate_control(tracker_id, tracker_xyz, tracker_rpy)
            self.send_target_pose(tracker_xyz, tracker_rpy)

            self.publish_state()
        except Exception as exc:
            now = time.time()
            if now - self.last_warn_time > 1.0:
                self.log_warning(f"控制循环异常: {exc}")
                self.last_warn_time = now

    def run(self):
        period = 1.0 / float(self.config.control_rate_hz)
        while True:
            start_time = time.time()
            self.control_loop()
            if self.rclpy is not None and self.ros_node is not None:
                self.rclpy.spin_once(self.ros_node, timeout_sec=0.0)

            sleep_time = period - (time.time() - start_time)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def cleanup(self):
        if self.arm is not None:
            try:
                if self.was_active and self.config.slow_stop_on_release:
                    self.arm.arm.rm_set_arm_slow_stop()
                self.arm.close()
            except Exception as exc:
                self.log_warning(f"关闭机械臂连接失败: {exc}")
            self.arm = None

        if self.xr_client is not None:
            try:
                self.xr_client.close()
            except Exception as exc:
                self.log_warning(f"关闭 XR SDK 失败: {exc}")
            self.xr_client = None

        if self.ros_node is not None:
            self.ros_node.destroy_node()
            self.ros_node = None

        if self.rclpy is not None:
            if self.rclpy.ok():
                self.rclpy.shutdown()
            self.rclpy = None


def parse_args(argv: list[str] | None = None) -> TeleopConfig:
    parser = argparse.ArgumentParser(description="Pico Motion Tracker 相对位姿控制 RealMan RM75-B xyz+rpy")
    parser.add_argument("--ip", default=DEFAULT_ARM_IP, help="RM75-B 控制器 IP")
    parser.add_argument("--port", type=int, default=DEFAULT_ARM_PORT, help="RM75-B 控制器端口")
    parser.add_argument("--rate", type=int, default=DEFAULT_CONTROL_RATE_HZ, help="控制频率 Hz")
    parser.add_argument(
        "--xyz-scale",
        type=parse_scale_option,
        default=DEFAULT_XYZ_SCALE_FACTOR,
        help=f"tracker xyz 位移到机械臂 xyz 位移的比例，可选: {format_scale_options()}",
    )
    parser.add_argument("--rpy-scale", type=float, default=DEFAULT_RPY_SCALE_FACTOR, help="tracker rpy 转动到机械臂 rpy 转动的比例")
    parser.add_argument("--rpy-axis-map", type=parse_rpy_axis_map, default=DEFAULT_RPY_AXIS_MAP, help="rpy 通道映射，默认 1,0,2 表示交换 roll/pitch 并保持 yaw")
    parser.add_argument("--rpy-axis-sign", type=parse_rpy_axis_sign, default=DEFAULT_RPY_AXIS_SIGN, help="rpy 通道方向，默认 1,1,1；如某轴方向反了可设为 -1")
    parser.add_argument(
        "--scale",
        type=parse_scale_option,
        default=None,
        help=f"兼容旧参数：等同于 --xyz-scale，可选: {format_scale_options()}",
    )
    parser.add_argument("--max-delta", type=float, default=DEFAULT_MAX_DELTA_M, help="单次按住允许的最大 xyz 相对位移，单位 m；<=0 表示不限制")
    parser.add_argument(
        "--reset-arm-pose",
        type=parse_reset_arm_pose,
        default=DEFAULT_RESET_ARM_POSE,
        help="B 键复位目标末端位姿 x,y,z,rx,ry,rz，默认 0.1166,0.0,0.7247,0.0,1.043,0.0",
    )
    parser.add_argument("--high-follow", action="store_true", help="启用 CANFD 高跟随；仅在控制周期稳定不超过 10ms 时使用")
    parser.add_argument("--no-slow-stop", action="store_true", help="停止 tracker 遥操时不发送 rm_set_arm_slow_stop")
    parser.add_argument("--enable-ros-publish", action="store_true", help="启用 ROS2 /action 和 /state 发布；默认关闭以兼容非 ROS Python 环境")
    args, _ = parser.parse_known_args(argv)

    return TeleopConfig(
        arm_ip=args.ip,
        arm_port=args.port,
        control_rate_hz=args.rate,
        xyz_scale_factor=args.xyz_scale if args.scale is None else args.scale,
        rpy_scale_factor=args.rpy_scale,
        rpy_axis_map=args.rpy_axis_map,
        rpy_axis_sign=args.rpy_axis_sign,
        max_delta_m=args.max_delta,
        reset_arm_pose=args.reset_arm_pose,
        high_follow=args.high_follow,
        slow_stop_on_release=not args.no_slow_stop,
        enable_ros_publish=args.enable_ros_publish,
    )


def main(args: list[str] | None = None):
    config = parse_args(args)
    teleop = RealmanMotionTrackerTeleop(config)

    try:
        teleop.run()
    except KeyboardInterrupt:
        teleop.log_info("检测到 Ctrl+C，正在退出。")
    finally:
        teleop.cleanup()


if __name__ == "__main__":
    main()
