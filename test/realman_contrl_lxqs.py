#!/usr/bin/env python3
"""使用 Pico 右手柄相对位姿控制 RealMan RM75-B 末端 xyz+rpy。"""

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

DEFAULT_ARM_IP = "192.168.5.200"
DEFAULT_ARM_PORT = 8080
DEFAULT_CONTROL_RATE_HZ = 50
DEFAULT_XYZ_SCALE_FACTOR = 1.0
DEFAULT_RPY_SCALE_FACTOR = 1.0
DEFAULT_RPY_AXIS_MAP = (1, 0, 2)
DEFAULT_RPY_AXIS_SIGN = (1.0, 1.0, 1.0)
DEFAULT_GRIP_THRESHOLD = 0.9
DEFAULT_MAX_DELTA_M = 0.25


@dataclass(frozen=True)
class TeleopConfig:
    arm_ip: str
    arm_port: int
    control_rate_hz: int
    xyz_scale_factor: float
    rpy_scale_factor: float
    rpy_axis_map: tuple[int, int, int]
    rpy_axis_sign: tuple[float, float, float]
    grip_threshold: float
    max_delta_m: float
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


def quaternion_xyzw_to_rotation_matrix(quat_xyzw: np.ndarray) -> np.ndarray:
    """将 SDK 返回的 xyzw 四元数转换成 3x3 旋转矩阵。"""
    quat = np.asarray(quat_xyzw, dtype=float)
    norm = np.linalg.norm(quat)
    if norm <= 1.0e-9 or not np.all(np.isfinite(quat)):
        raise RuntimeError(f"右手柄四元数无效: {quat_xyzw}")

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


def read_controller_pose(xr_client: XrClient) -> tuple[np.ndarray, np.ndarray]:
    """读取右手柄 SDK pose，并转换到项目现有控制坐标系的 xyz+rpy。"""
    pose = np.asarray(xr_client.get_pose_by_name("right_controller"), dtype=float)
    if pose.shape[0] < 7 or not np.all(np.isfinite(pose[:7])):
        raise RuntimeError(f"右手柄位姿无效: {pose}")

    controller_xyz = R_HEADSET_TO_WORLD @ pose[:3]
    controller_rotation = quaternion_xyzw_to_rotation_matrix(pose[3:7])
    controller_rotation = R_HEADSET_TO_WORLD @ controller_rotation @ R_HEADSET_TO_WORLD.T
    controller_rpy = rotation_matrix_to_rpy(controller_rotation)
    return controller_xyz, controller_rpy


def read_controller_xyz(xr_client: XrClient) -> np.ndarray:
    """兼容旧调用：只读取右手柄 xyz。"""
    controller_xyz, _ = read_controller_pose(xr_client)
    return controller_xyz


def clip_delta(delta_xyz: np.ndarray, max_delta_m: float) -> np.ndarray:
    """限制单次按住期间最大相对位移，避免手柄异常跳变直接传给机械臂。"""
    if max_delta_m <= 0:
        return delta_xyz
    return np.clip(delta_xyz, -max_delta_m, max_delta_m)


class RealmanXrIncrementalTeleop:
    """右手 grip 按住时，用手柄相对位姿驱动 RM75-B 末端 xyz+rpy。"""

    def __init__(self, config: TeleopConfig):
        self.config = config
        self.xr_client: XrClient | None = None
        self.arm: RM75BInterface | None = None
        self.rclpy: Any | None = None
        self.ros_node: Any | None = None
        self.float32_multi_array_cls: Any | None = None
        self.target_pub: Any | None = None
        self.actual_pub: Any | None = None

        self.controller_origin_xyz: np.ndarray | None = None
        self.controller_origin_rpy: np.ndarray | None = None
        self.arm_origin_pose: np.ndarray | None = None
        self.target_pose: np.ndarray | None = None
        self.grip_press_event: dict[str, np.ndarray] | None = None
        self.was_active = False
        self.last_warn_time = 0.0

        if self.config.enable_ros_publish:
            self.init_ros_publishers()
        self.init_hardware()
        self.log_info(
            "初始化完成：按住右手 grip 控制机械臂 xyz+rpy，松开停止并清空追踪原点。"
        )

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
        self.ros_node = rclpy.create_node("realman_xr_incremental_teleop")
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
        grip_value = float(self.xr_client.get_key_value_by_name("right_grip"))
        return grip_value >= self.config.grip_threshold

    def record_grip_press_event(
        self,
        controller_xyz: np.ndarray,
        controller_rpy: np.ndarray,
        arm_pose: np.ndarray,
    ):
        """记录 grip 按下瞬间的手柄和机械臂位姿，用于松开时做偏移对比。"""
        self.grip_press_event = {
            "controller_xyz": controller_xyz.copy(),
            "controller_rpy": controller_rpy.copy(),
            "arm_pose": arm_pose.copy(),
        }
        self.log_info(
            "grip 按下事件记录: "
            f"controller_xyz={controller_xyz.tolist()}, "
            f"controller_rpy={controller_rpy.tolist()}, "
            f"arm_pose={arm_pose.tolist()}"
        )

    def log_grip_release_comparison(
        self,
        release_controller_xyz: np.ndarray | None,
        release_controller_rpy: np.ndarray | None,
        release_arm_pose: np.ndarray | None,
    ):
        """记录 grip 松开瞬间位姿，并打印手柄偏移与机械臂实际偏移的对比。"""
        self.log_info(
            "grip 松开事件记录: "
            f"controller_xyz={None if release_controller_xyz is None else release_controller_xyz.tolist()}, "
            f"controller_rpy={None if release_controller_rpy is None else release_controller_rpy.tolist()}, "
            f"arm_pose={None if release_arm_pose is None else release_arm_pose.tolist()}"
        )

        if self.grip_press_event is None:
            self.log_warning("无法对比 grip 偏移：缺少按下事件记录。")
            return
        if release_controller_xyz is None or release_arm_pose is None:
            self.log_warning("无法对比 grip 偏移：松开事件的手柄或机械臂位姿读取失败。")
            return

        press_controller_xyz = self.grip_press_event["controller_xyz"]
        press_arm_pose = self.grip_press_event["arm_pose"]

        controller_delta_xyz = release_controller_xyz - press_controller_xyz
        expected_arm_delta_xyz = controller_delta_xyz * self.config.xyz_scale_factor
        expected_arm_delta_xyz = clip_delta(expected_arm_delta_xyz, self.config.max_delta_m)
        actual_arm_delta_xyz = release_arm_pose[:3] - press_arm_pose[:3]
        xyz_error = actual_arm_delta_xyz - expected_arm_delta_xyz

        comparison_rows = "\n".join(
            (
                f"  {axis:<4}"
                f"{float(controller_delta_xyz[index]):>18.6f}"
                f"{float(expected_arm_delta_xyz[index]):>24.6f}"
                f"{float(actual_arm_delta_xyz[index]):>22.6f}"
                f"{float(xyz_error[index]):>16.6f}"
            )
            for index, axis in enumerate(("x", "y", "z"))
        )

        self.log_info(
            "grip 事件 xyz 偏移对比:\n"
            f"  计算公式: expected_arm_delta = clip(controller_delta * xyz_scale, +/-max_delta_m)\n"
            f"  xyz_scale={self.config.xyz_scale_factor:g}, max_delta_m={self.config.max_delta_m}\n"
            "  轴     controller位移(m)    期望机械臂位移(m)     实际机械臂位移(m)        误差(m)\n"
            f"{comparison_rows}"
        )

    def activate_control(self, controller_xyz: np.ndarray, controller_rpy: np.ndarray):
        assert self.arm is not None
        self.controller_origin_xyz = controller_xyz.copy()
        self.controller_origin_rpy = controller_rpy.copy()
        self.arm_origin_pose = read_arm_pose(self.arm)
        self.target_pose = self.arm_origin_pose.copy()
        self.was_active = True
        self.record_grip_press_event(controller_xyz, controller_rpy, self.arm_origin_pose)
        self.log_info(
            "右手 grip 已按下，记录控制原点: "
            f"controller_xyz={self.controller_origin_xyz.tolist()}, "
            f"controller_rpy={self.controller_origin_rpy.tolist()}, "
            f"arm_pose={self.arm_origin_pose.tolist()}"
        )

    def deactivate_control(self):
        assert self.arm is not None
        release_controller_xyz = None
        release_controller_rpy = None
        release_arm_pose = None

        if self.xr_client is not None:
            try:
                release_controller_xyz, release_controller_rpy = read_controller_pose(self.xr_client)
            except RuntimeError as exc:
                self.log_warning(f"松开 grip 时读取右手柄位姿失败: {exc}")

        try:
            release_arm_pose = read_arm_pose(self.arm)
            self.target_pose = release_arm_pose.copy()
        except RuntimeError as exc:
            self.log_warning(str(exc))

        self.log_grip_release_comparison(
            release_controller_xyz,
            release_controller_rpy,
            release_arm_pose,
        )

        if self.config.slow_stop_on_release:
            ret = self.arm.arm.rm_set_arm_slow_stop()
            if ret != 0:
                self.log_warning(f"松开 grip 后缓停指令返回异常: ret={ret}")

        self.controller_origin_xyz = None
        self.controller_origin_rpy = None
        self.arm_origin_pose = None
        self.grip_press_event = None
        self.was_active = False
        self.log_info("右手 grip 已松开，停止发送位姿透传并清空追踪原点。")

    def send_target_pose(self, controller_xyz: np.ndarray, controller_rpy: np.ndarray):
        assert self.arm is not None
        assert self.controller_origin_xyz is not None
        assert self.controller_origin_rpy is not None
        assert self.arm_origin_pose is not None

        delta_xyz = (controller_xyz - self.controller_origin_xyz) * self.config.xyz_scale_factor
        delta_xyz = clip_delta(delta_xyz, self.config.max_delta_m)
        raw_delta_rpy = wrap_angle(controller_rpy - self.controller_origin_rpy)
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
        if self.target_pub is None or self.actual_pub is None or self.float32_multi_array_cls is None:
            return

        assert self.arm is not None
        try:
            actual_pose = read_arm_pose(self.arm)
        except RuntimeError as exc:
            self.log_warning(str(exc))
            return

        if self.target_pose is not None:
            action_msg = self.float32_multi_array_cls()
            action_msg.data = self.target_pose[:6].astype(float).tolist() + [1.0 if self.was_active else 0.0]
            self.target_pub.publish(action_msg)

        state_msg = self.float32_multi_array_cls()
        state_msg.data = actual_pose[:6].astype(float).tolist() + [1.0 if self.was_active else 0.0]
        self.actual_pub.publish(state_msg)

    def control_loop(self):
        assert self.xr_client is not None
        try:
            active = self.is_control_active()

            # 松开时必须先清空控制引用；不要再依赖当前手柄 pose，
            # 否则松开瞬间定位异常会跳过 deactivate_control()，留下旧原点。
            if not active:
                if self.was_active:
                    self.deactivate_control()
                self.publish_state()
                return

            controller_xyz, controller_rpy = read_controller_pose(self.xr_client)

            if not self.was_active:
                self.activate_control(controller_xyz, controller_rpy)
            self.send_target_pose(controller_xyz, controller_rpy)

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
    parser = argparse.ArgumentParser(description="Pico 右手柄相对位姿控制 RealMan RM75-B xyz+rpy")
    parser.add_argument("--ip", default=DEFAULT_ARM_IP, help="RM75-B 控制器 IP")
    parser.add_argument("--port", type=int, default=DEFAULT_ARM_PORT, help="RM75-B 控制器端口")
    parser.add_argument("--rate", type=int, default=DEFAULT_CONTROL_RATE_HZ, help="控制频率 Hz")
    parser.add_argument("--xyz-scale", type=float, default=DEFAULT_XYZ_SCALE_FACTOR, help="手柄 xyz 位移到机械臂 xyz 位移的比例")
    parser.add_argument("--rpy-scale", type=float, default=DEFAULT_RPY_SCALE_FACTOR, help="手柄 rpy 转动到机械臂 rpy 转动的比例")
    parser.add_argument("--rpy-axis-map", type=parse_rpy_axis_map, default=DEFAULT_RPY_AXIS_MAP, help="rpy 通道映射，默认 1,0,2 表示交换 roll/pitch 并保持 yaw")
    parser.add_argument("--rpy-axis-sign", type=parse_rpy_axis_sign, default=DEFAULT_RPY_AXIS_SIGN, help="rpy 通道方向，默认 1,1,1；如某轴方向反了可设为 -1")
    parser.add_argument("--scale", type=float, default=None, help="兼容旧参数：等同于 --xyz-scale")
    parser.add_argument("--grip-threshold", type=float, default=DEFAULT_GRIP_THRESHOLD, help="右手 grip 激活阈值")
    parser.add_argument("--max-delta", type=float, default=DEFAULT_MAX_DELTA_M, help="单次按住允许的最大 xyz 相对位移，单位 m；<=0 表示不限制")
    parser.add_argument("--high-follow", action="store_true", help="启用 CANFD 高跟随；仅在控制周期稳定不超过 10ms 时使用")
    parser.add_argument("--no-slow-stop", action="store_true", help="松开 grip 时不发送 rm_set_arm_slow_stop")
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
        grip_threshold=args.grip_threshold,
        max_delta_m=args.max_delta,
        high_follow=args.high_follow,
        slow_stop_on_release=not args.no_slow_stop,
        enable_ros_publish=args.enable_ros_publish,
    )


def main(args: list[str] | None = None):
    config = parse_args(args)
    teleop = RealmanXrIncrementalTeleop(config)

    try:
        teleop.run()
    except KeyboardInterrupt:
        teleop.log_info("检测到 Ctrl+C，正在退出。")
    finally:
        teleop.cleanup()


if __name__ == "__main__":
    main()
