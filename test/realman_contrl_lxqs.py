#!/usr/bin/env python3
"""使用 Pico 右手柄相对位移控制 RealMan RM75-B 末端 xyz。"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from xrobotoolkit_teleop.common.xr_client import XrClient
from xrobotoolkit_teleop.hardware.interface.rm75b import RM75BInterface
from xrobotoolkit_teleop.utils.geometry import R_HEADSET_TO_WORLD

DEFAULT_ARM_IP = "192.168.5.73"
DEFAULT_ARM_PORT = 8080
DEFAULT_CONTROL_RATE_HZ = 50
DEFAULT_SCALE_FACTOR = 1.0
DEFAULT_GRIP_THRESHOLD = 0.9
DEFAULT_MAX_DELTA_M = 0.25


@dataclass(frozen=True)
class TeleopConfig:
    arm_ip: str
    arm_port: int
    control_rate_hz: int
    scale_factor: float
    grip_threshold: float
    max_delta_m: float
    high_follow: bool
    slow_stop_on_release: bool


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


def read_controller_xyz(xr_client: XrClient) -> np.ndarray:
    """读取右手柄 SDK xyz，并转换到项目现有控制坐标系。"""
    pose = np.asarray(xr_client.get_pose_by_name("right_controller"), dtype=float)
    if pose.shape[0] < 3 or not np.all(np.isfinite(pose[:3])):
        raise RuntimeError(f"右手柄位姿无效: {pose}")
    return R_HEADSET_TO_WORLD @ pose[:3]


def clip_delta(delta_xyz: np.ndarray, max_delta_m: float) -> np.ndarray:
    """限制单次按住期间最大相对位移，避免手柄异常跳变直接传给机械臂。"""
    if max_delta_m <= 0:
        return delta_xyz
    return np.clip(delta_xyz, -max_delta_m, max_delta_m)


class RealmanXrIncrementalTeleop(Node):
    """右手 grip 按住时，用手柄相对位移驱动 RM75-B 末端 xyz。"""

    def __init__(self, config: TeleopConfig):
        super().__init__("realman_xr_incremental_teleop")
        self.config = config
        self.xr_client: XrClient | None = None
        self.arm: RM75BInterface | None = None

        self.controller_origin_xyz: np.ndarray | None = None
        self.arm_origin_pose: np.ndarray | None = None
        self.target_pose: np.ndarray | None = None
        self.was_active = False
        self.last_warn_time = 0.0

        self.target_pub = self.create_publisher(Float32MultiArray, "/action", 10)
        self.actual_pub = self.create_publisher(Float32MultiArray, "/state", 10)

        self.init_hardware()
        timer_period = 1.0 / float(self.config.control_rate_hz)
        self.timer = self.create_timer(timer_period, self.control_loop)
        self.get_logger().info("节点初始化完成：按住右手 grip 控制机械臂，松开停止并清空追踪原点。")

    def init_hardware(self):
        self.get_logger().info(describe_robotic_arm_package())
        self.xr_client = XrClient()
        self.arm = RM75BInterface(self.config.arm_ip, self.config.arm_port, enable_gripper=False)
        self.target_pose = read_arm_pose(self.arm)
        self.get_logger().info(f"机械臂连接成功，当前末端位姿: {self.target_pose.tolist()}")

    def is_control_active(self) -> bool:
        assert self.xr_client is not None
        grip_value = float(self.xr_client.get_key_value_by_name("right_grip"))
        return grip_value >= self.config.grip_threshold

    def activate_control(self, controller_xyz: np.ndarray):
        assert self.arm is not None
        self.controller_origin_xyz = controller_xyz.copy()
        self.arm_origin_pose = read_arm_pose(self.arm)
        self.target_pose = self.arm_origin_pose.copy()
        self.was_active = True
        self.get_logger().info(
            "右手 grip 已按下，记录控制原点: "
            f"controller_xyz={self.controller_origin_xyz.tolist()}, "
            f"arm_pose={self.arm_origin_pose.tolist()}"
        )

    def deactivate_control(self):
        assert self.arm is not None
        if self.config.slow_stop_on_release:
            ret = self.arm.arm.rm_set_arm_slow_stop()
            if ret != 0:
                self.get_logger().warning(f"松开 grip 后缓停指令返回异常: ret={ret}")

        try:
            self.target_pose = read_arm_pose(self.arm)
        except RuntimeError as exc:
            self.get_logger().warning(str(exc))

        self.controller_origin_xyz = None
        self.arm_origin_pose = None
        self.was_active = False
        self.get_logger().info("右手 grip 已松开，停止发送位姿透传并清空追踪原点。")

    def send_target_pose(self, controller_xyz: np.ndarray):
        assert self.arm is not None
        assert self.controller_origin_xyz is not None
        assert self.arm_origin_pose is not None

        delta_xyz = (controller_xyz - self.controller_origin_xyz) * self.config.scale_factor
        delta_xyz = clip_delta(delta_xyz, self.config.max_delta_m)

        target_pose = self.arm_origin_pose.copy()
        target_pose[:3] = self.arm_origin_pose[:3] + delta_xyz
        self.target_pose = target_pose

        ret = self.arm.arm.rm_movep_canfd(target_pose.tolist(), follow=self.config.high_follow)
        if ret != 0:
            now = time.time()
            if now - self.last_warn_time > 1.0:
                self.get_logger().warning(f"rm_movep_canfd 返回异常: ret={ret}, target={target_pose.tolist()}")
                self.last_warn_time = now

    def publish_state(self):
        assert self.arm is not None
        try:
            actual_pose = read_arm_pose(self.arm)
        except RuntimeError as exc:
            self.get_logger().warning(str(exc))
            return

        if self.target_pose is not None:
            action_msg = Float32MultiArray()
            action_msg.data = self.target_pose[:3].astype(float).tolist() + [1.0 if self.was_active else 0.0]
            self.target_pub.publish(action_msg)

        state_msg = Float32MultiArray()
        state_msg.data = actual_pose[:3].astype(float).tolist() + [1.0 if self.was_active else 0.0]
        self.actual_pub.publish(state_msg)

    def control_loop(self):
        assert self.xr_client is not None
        try:
            active = self.is_control_active()
            controller_xyz = read_controller_xyz(self.xr_client)

            if active:
                if not self.was_active:
                    self.activate_control(controller_xyz)
                self.send_target_pose(controller_xyz)
            elif self.was_active:
                self.deactivate_control()

            self.publish_state()
        except Exception as exc:
            now = time.time()
            if now - self.last_warn_time > 1.0:
                self.get_logger().warning(f"控制循环异常: {exc}")
                self.last_warn_time = now

    def cleanup(self):
        if self.arm is not None:
            try:
                if self.was_active and self.config.slow_stop_on_release:
                    self.arm.arm.rm_set_arm_slow_stop()
                self.arm.close()
            except Exception as exc:
                self.get_logger().warning(f"关闭机械臂连接失败: {exc}")
            self.arm = None

        if self.xr_client is not None:
            try:
                self.xr_client.close()
            except Exception as exc:
                self.get_logger().warning(f"关闭 XR SDK 失败: {exc}")
            self.xr_client = None


def parse_args(argv: list[str] | None = None) -> TeleopConfig:
    parser = argparse.ArgumentParser(description="Pico 右手柄相对位移控制 RealMan RM75-B xyz")
    parser.add_argument("--ip", default=DEFAULT_ARM_IP, help="RM75-B 控制器 IP")
    parser.add_argument("--port", type=int, default=DEFAULT_ARM_PORT, help="RM75-B 控制器端口")
    parser.add_argument("--rate", type=int, default=DEFAULT_CONTROL_RATE_HZ, help="控制频率 Hz")
    parser.add_argument("--scale", type=float, default=DEFAULT_SCALE_FACTOR, help="手柄位移到机械臂位移的比例")
    parser.add_argument("--grip-threshold", type=float, default=DEFAULT_GRIP_THRESHOLD, help="右手 grip 激活阈值")
    parser.add_argument("--max-delta", type=float, default=DEFAULT_MAX_DELTA_M, help="单次按住允许的最大 xyz 相对位移，单位 m；<=0 表示不限制")
    parser.add_argument("--high-follow", action="store_true", help="启用 CANFD 高跟随；仅在控制周期稳定不超过 10ms 时使用")
    parser.add_argument("--no-slow-stop", action="store_true", help="松开 grip 时不发送 rm_set_arm_slow_stop")
    args, _ = parser.parse_known_args(argv)

    return TeleopConfig(
        arm_ip=args.ip,
        arm_port=args.port,
        control_rate_hz=args.rate,
        scale_factor=args.scale,
        grip_threshold=args.grip_threshold,
        max_delta_m=args.max_delta,
        high_follow=args.high_follow,
        slow_stop_on_release=not args.no_slow_stop,
    )


def main(args: list[str] | None = None):
    config = parse_args(args)
    rclpy.init(args=args)
    teleop_node = RealmanXrIncrementalTeleop(config)

    try:
        rclpy.spin(teleop_node)
    except KeyboardInterrupt:
        teleop_node.get_logger().info("检测到 Ctrl+C，正在退出。")
    finally:
        teleop_node.cleanup()
        teleop_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
