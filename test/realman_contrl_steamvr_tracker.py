#!/usr/bin/env python3
"""使用 SteamVR/OpenVR 单个 Tracker 相对位姿控制 RealMan RM75-B 末端 xyz+rpy。"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
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

# 与 xrobotoolkit_teleop.utils.geometry.R_HEADSET_TO_WORLD 保持一致，
# 但避免当前 Python 环境因缺少 meshcat 而在导入 geometry.py 时失败。
R_HEADSET_TO_WORLD = np.array(
    [
        [0, 0, -1],
        [-1, 0, 0],
        [0, 1, 0],
    ],
    dtype=float,
)

DEFAULT_ARM_IP = "192.168.5.154"
DEFAULT_ARM_PORT = 8080
DEFAULT_CONTROL_RATE_HZ = 50
DEFAULT_XYZ_SCALE_FACTOR = 1.0
DEFAULT_RPY_SCALE_FACTOR = 1.0
DEFAULT_XYZ_AXIS_MAP = (0, 1, 2)
DEFAULT_XYZ_AXIS_SIGN = (1.0, 1.0, 1.0)
DEFAULT_RPY_AXIS_MAP = (1, 0, 2)
DEFAULT_RPY_AXIS_SIGN = (1.0, 1.0, 1.0)
DEFAULT_MAX_DELTA_M = 0.25
DEFAULT_TRACKING_UNIVERSE = "standing"
DEFAULT_COORDINATE_MODE = "project_world"
DEFAULT_HOLD_KEY = "space"
DEFAULT_HOLD_SOURCE = "auto"
DEFAULT_HOLD_RELEASE_TIMEOUT_S = 0.7


@dataclass(frozen=True)
class TeleopConfig:
    arm_ip: str
    arm_port: int
    control_rate_hz: int
    tracker_serial: str | None
    tracking_universe: str
    coordinate_mode: str
    xyz_scale_factor: float
    rpy_scale_factor: float
    xyz_axis_map: tuple[int, int, int]
    xyz_axis_sign: tuple[float, float, float]
    rpy_axis_map: tuple[int, int, int]
    rpy_axis_sign: tuple[float, float, float]
    hold_key: str
    hold_source: str
    hold_release_timeout_s: float
    max_delta_m: float
    high_follow: bool
    slow_stop_on_release: bool
    list_trackers: bool


@dataclass(frozen=True)
class TrackerInventoryItem:
    index: int
    serial: str
    model: str
    connected: bool
    pose_valid: bool


@dataclass(frozen=True)
class TrackerPoseSample:
    index: int
    serial: str
    raw_xyz: np.ndarray
    raw_rpy: np.ndarray
    control_xyz: np.ndarray
    control_rpy: np.ndarray


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


def parse_axis_map(value: str) -> tuple[int, int, int]:
    """解析 xyz/rpy 轴映射，例如 1,0,2。"""
    try:
        axis_map = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("轴映射必须是逗号分隔的整数，例如 1,0,2") from exc

    if len(axis_map) != 3 or sorted(axis_map) != [0, 1, 2]:
        raise argparse.ArgumentTypeError("轴映射必须包含且只包含 0,1,2，例如 1,0,2")
    return axis_map


def parse_axis_sign(value: str) -> tuple[float, float, float]:
    """解析 xyz/rpy 轴方向，例如 1,-1,1。"""
    try:
        signs = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("轴方向必须是逗号分隔数字，例如 1,-1,1") from exc

    if len(signs) != 3 or any(sign not in (-1.0, 1.0) for sign in signs):
        raise argparse.ArgumentTypeError("轴方向只能由 1 或 -1 组成，例如 1,-1,1")
    return signs


def tracking_universe_to_openvr(name: str) -> int:
    """将命令行字符串映射到 OpenVR TrackingUniverse。"""
    mapping = {
        "standing": openvr.TrackingUniverseStanding,
        "raw": openvr.TrackingUniverseRawAndUncalibrated,
        "seated": openvr.TrackingUniverseSeated,
    }
    return mapping[name]


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


def convert_openvr_pose(
    transform: np.ndarray,
    coordinate_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """将 OpenVR pose 转为原始坐标和项目控制坐标下的 xyz+rpy。"""
    raw_xyz = transform[:3, 3].astype(float)
    raw_rotation = transform[:3, :3].astype(float)
    raw_rpy = rotation_matrix_to_rpy(raw_rotation)

    if coordinate_mode == "raw_openvr":
        return raw_xyz, raw_rpy, raw_xyz.copy(), raw_rpy.copy()

    # OpenVR standing/raw universe 默认也是 x右/y上/z后，和项目已有 XR 输入轴语义一致，
    # 因此默认沿用 R_HEADSET_TO_WORLD 把设备位姿统一到机械臂控制世界系。
    control_xyz = R_HEADSET_TO_WORLD @ raw_xyz
    control_rotation = R_HEADSET_TO_WORLD @ raw_rotation @ R_HEADSET_TO_WORLD.T
    control_rpy = rotation_matrix_to_rpy(control_rotation)
    return raw_xyz, raw_rpy, control_xyz, control_rpy


def clip_delta(delta_xyz: np.ndarray, max_delta_m: float) -> np.ndarray:
    """限制单次按住期间最大相对位移，避免 tracker 异常跳变直接传给机械臂。"""
    if max_delta_m <= 0:
        return delta_xyz
    return np.clip(delta_xyz, -max_delta_m, max_delta_m)


class SteamVrTrackerReader:
    """封装 SteamVR/OpenVR tracker 枚举与位姿读取。"""

    def __init__(self, tracking_universe: str, coordinate_mode: str):
        self.tracking_universe = tracking_universe_to_openvr(tracking_universe)
        self.coordinate_mode = coordinate_mode
        openvr.init(openvr.VRApplication_Other)
        self.vrsystem = openvr.VRSystem()

    def _get_string_property(self, index: int, prop: int) -> str | None:
        try:
            return self.vrsystem.getStringTrackedDeviceProperty(index, prop)
        except Exception:
            return None

    def list_trackers(self) -> list[TrackerInventoryItem]:
        trackers: list[TrackerInventoryItem] = []
        poses = self.vrsystem.getDeviceToAbsoluteTrackingPose(
            self.tracking_universe,
            0.0,
            openvr.k_unMaxTrackedDeviceCount,
        )

        for index in range(openvr.k_unMaxTrackedDeviceCount):
            device_class = self.vrsystem.getTrackedDeviceClass(index)
            if device_class != openvr.TrackedDeviceClass_GenericTracker:
                continue

            serial = self._get_string_property(index, openvr.Prop_SerialNumber_String) or f"tracker_{index}"
            model = self._get_string_property(index, openvr.Prop_ModelNumber_String) or "未知型号"
            pose = poses[index]
            trackers.append(
                TrackerInventoryItem(
                    index=index,
                    serial=serial,
                    model=model,
                    connected=bool(pose.bDeviceIsConnected),
                    pose_valid=bool(pose.bPoseIsValid),
                )
            )

        return trackers

    def _find_tracker_index(self, tracker_serial: str | None) -> tuple[int, str]:
        trackers = self.list_trackers()
        if not trackers:
            raise RuntimeError("SteamVR 中未发现任何 GenericTracker，请先确认 tracker 已在设备列表中在线。")

        if tracker_serial is not None:
            for item in trackers:
                if item.serial == tracker_serial:
                    if not item.connected:
                        raise RuntimeError(f"指定 tracker 已识别但未连接: {tracker_serial}")
                    return item.index, item.serial
            known_serials = [item.serial for item in trackers]
            raise RuntimeError(f"指定 tracker 不存在: {tracker_serial}, 当前可见: {known_serials}")

        valid_items = [item for item in trackers if item.connected and item.pose_valid]
        if valid_items:
            item = valid_items[0]
            return item.index, item.serial

        connected_items = [item for item in trackers if item.connected]
        if connected_items:
            raise RuntimeError(
                "SteamVR 已识别 tracker，但当前没有有效 pose。"
                f" 当前设备: {[(item.serial, item.pose_valid) for item in connected_items]}"
            )

        raise RuntimeError("SteamVR 中存在 tracker，但都处于未连接状态。")

    def read_tracker_pose(self, tracker_serial: str | None) -> TrackerPoseSample:
        index, resolved_serial = self._find_tracker_index(tracker_serial)
        poses = self.vrsystem.getDeviceToAbsoluteTrackingPose(
            self.tracking_universe,
            0.0,
            openvr.k_unMaxTrackedDeviceCount,
        )
        pose = poses[index]
        if not pose.bDeviceIsConnected:
            raise RuntimeError(f"tracker 已断开连接: {resolved_serial}")
        if not pose.bPoseIsValid:
            raise RuntimeError(f"tracker 当前 pose 无效: {resolved_serial}")

        transform = mat34_to_matrix(pose.mDeviceToAbsoluteTracking)
        raw_xyz, raw_rpy, control_xyz, control_rpy = convert_openvr_pose(transform, self.coordinate_mode)
        return TrackerPoseSample(
            index=index,
            serial=resolved_serial,
            raw_xyz=raw_xyz,
            raw_rpy=raw_rpy,
            control_xyz=control_xyz,
            control_rpy=control_rpy,
        )

    def close(self):
        openvr.shutdown()


class HoldKeyWindow:
    """用一个小窗口捕获按下/松开事件，实现“按住才控制”。"""

    def __init__(self, hold_key: str):
        try:
            import tkinter as tk
        except Exception as exc:
            raise RuntimeError(
                "当前 Python 环境不可用 tkinter，无法监听按键按下/松开。"
                "请安装 tkinter，或改成带图形界面的 Python 环境运行。"
            ) from exc

        self.tk = tk
        self.hold_key = hold_key
        self.display_key = "Space" if hold_key.lower() == "space" else hold_key
        self.is_pressed = False
        self.should_exit = False
        self.has_focus = False
        self.runtime_text = ""

        self.root = tk.Tk()
        self.root.title("RM75 SteamVR Tracker Teleop")
        self.root.geometry("520x220")
        self.root.resizable(False, False)
        self.root.attributes("-topmost", True)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        title = tk.Label(
            self.root,
            text="按住指定按键才发送机械臂跟随",
            font=("Arial", 15, "bold"),
        )
        title.pack(pady=(16, 8))

        desc = tk.Label(
            self.root,
            text=(
                f"1. 点击当前窗口，让它获得键盘焦点\n"
                f"2. 按住 {self.display_key} 开始控制\n"
                f"3. 松开 {self.display_key} 停止跟随并缓停机械臂"
            ),
            justify="left",
            font=("Arial", 12),
        )
        desc.pack(pady=4)

        self.status_var = tk.StringVar()
        status = tk.Label(
            self.root,
            textvariable=self.status_var,
            justify="left",
            font=("Arial", 11),
            fg="#1E3A8A",
        )
        status.pack(pady=(12, 8))

        self.root.bind("<KeyPress>", self._on_key_press)
        self.root.bind("<KeyRelease>", self._on_key_release)
        self.root.bind("<FocusIn>", self._on_focus_in)
        self.root.bind("<FocusOut>", self._on_focus_out)

        self._update_status()
        self.root.after(100, self._focus_window)

    def _focus_window(self):
        if self.should_exit:
            return
        try:
            self.root.lift()
            self.root.focus_force()
        except Exception:
            return

    def _normalize_key(self, key_name: str) -> str:
        return key_name.strip().lower()

    def _is_hold_key_event(self, event: object) -> bool:
        keysym = self._normalize_key(getattr(event, "keysym", ""))
        char = self._normalize_key(getattr(event, "char", ""))
        target = self._normalize_key(self.hold_key)
        if target == "space":
            return keysym == "space"
        return keysym == target or char == target

    def _on_key_press(self, event: object):
        if self._is_hold_key_event(event):
            self.is_pressed = True
            self._update_status()

    def _on_key_release(self, event: object):
        if self._is_hold_key_event(event):
            self.is_pressed = False
            self._update_status()

    def _on_focus_in(self, _event: object):
        self.has_focus = True
        self._update_status()

    def _on_focus_out(self, _event: object):
        self.has_focus = False
        self.is_pressed = False
        self._update_status()

    def _on_close(self):
        self.should_exit = True

    def _update_status(self):
        focus_text = "已获得焦点" if self.has_focus else "未获得焦点"
        hold_text = "按键按住中" if self.is_pressed else "等待按下"
        runtime_text = self.runtime_text if self.runtime_text else "tracker: 未绑定"
        self.status_var.set(
            f"窗口状态: {focus_text}\n"
            f"控制状态: {hold_text}\n"
            f"{runtime_text}"
        )

    def set_runtime_text(self, message: str):
        self.runtime_text = message
        self._update_status()

    def poll(self):
        if self.should_exit:
            return
        try:
            self.root.update_idletasks()
            self.root.update()
        except Exception:
            self.should_exit = True

    def close(self):
        self.should_exit = True
        try:
            self.root.destroy()
        except Exception:
            pass


class RealmanSteamVrTrackerTeleop:
    """按住键时，用 SteamVR 单个 tracker 相对位姿驱动 RM75-B 末端 xyz+rpy。"""

    def __init__(self, config: TeleopConfig):
        self.config = config
        self.arm: RM75BInterface | None = None
        self.tracker_reader: SteamVrTrackerReader | None = None
        self.key_monitor: object | None = None

        self.active_tracker_serial: str | None = self.config.tracker_serial
        self.tracker_origin_xyz: np.ndarray | None = None
        self.tracker_origin_rpy: np.ndarray | None = None
        self.arm_origin_pose: np.ndarray | None = None
        self.target_pose: np.ndarray | None = None
        self.was_active = False
        self.last_warn_time = 0.0

        self.init_hardware()
        self.log_info(
            "初始化完成：按住键盘窗口中的指定按键时，使用单个 SteamVR tracker 做增量跟随控制。"
        )
        self.log_info(
            f"当前 hold_key={self.config.hold_key}, coordinate_mode={self.config.coordinate_mode}, "
            f"tracking_universe={self.config.tracking_universe}, hold_source={self.config.hold_source}"
        )

    def log_info(self, message: str):
        print(f"[INFO] {message}")

    def log_warning(self, message: str):
        print(f"[WARN] {message}")

    def init_hardware(self):
        self.log_info(describe_robotic_arm_package())
        self.tracker_reader = SteamVrTrackerReader(
            tracking_universe=self.config.tracking_universe,
            coordinate_mode=self.config.coordinate_mode,
        )
        self.key_monitor = create_hold_key_monitor(
            hold_key=self.config.hold_key,
            hold_source=self.config.hold_source,
            hold_release_timeout_s=self.config.hold_release_timeout_s,
        )
        self.arm = RM75BInterface(self.config.arm_ip, self.config.arm_port, enable_gripper=False)
        self.target_pose = read_arm_pose(self.arm)
        self.log_info(f"机械臂连接成功，当前末端位姿: {self.target_pose.tolist()}")

    def update_window_status(self, tracker_sample: TrackerPoseSample | None = None):
        if self.key_monitor is None:
            return
        if tracker_sample is None:
            tracker_serial = self.active_tracker_serial or "未绑定"
            self.key_monitor.set_runtime_text(f"tracker: {tracker_serial}")
            return
        self.key_monitor.set_runtime_text(
            f"tracker: {tracker_sample.serial}\n"
            f"control_xyz={np.round(tracker_sample.control_xyz, 4).tolist()}"
        )

    def is_control_active(self) -> bool:
        assert self.key_monitor is not None
        self.key_monitor.poll()
        if self.key_monitor.should_exit:
            raise KeyboardInterrupt
        return self.key_monitor.is_pressed

    def read_tracker_pose(self) -> TrackerPoseSample:
        assert self.tracker_reader is not None
        preferred_serial = self.active_tracker_serial
        sample = self.tracker_reader.read_tracker_pose(preferred_serial)
        self.active_tracker_serial = sample.serial
        self.update_window_status(sample)
        return sample

    def activate_control(self, tracker_sample: TrackerPoseSample):
        assert self.arm is not None
        self.tracker_origin_xyz = tracker_sample.control_xyz.copy()
        self.tracker_origin_rpy = tracker_sample.control_rpy.copy()
        self.arm_origin_pose = read_arm_pose(self.arm)
        self.target_pose = self.arm_origin_pose.copy()
        self.was_active = True
        self.log_info(
            "tracker 控制已激活，记录控制原点: "
            f"serial={tracker_sample.serial}, "
            f"raw_xyz={tracker_sample.raw_xyz.tolist()}, "
            f"control_xyz={tracker_sample.control_xyz.tolist()}, "
            f"raw_rpy={tracker_sample.raw_rpy.tolist()}, "
            f"control_rpy={tracker_sample.control_rpy.tolist()}, "
            f"arm_pose={self.arm_origin_pose.tolist()}"
        )

    def clear_control_state(self):
        self.tracker_origin_xyz = None
        self.tracker_origin_rpy = None
        self.arm_origin_pose = None
        self.was_active = False
        if self.config.tracker_serial is None:
            self.active_tracker_serial = None
        self.update_window_status(None)

    def deactivate_control(self):
        assert self.arm is not None
        if self.config.slow_stop_on_release:
            ret = self.arm.arm.rm_set_arm_slow_stop()
            if ret != 0:
                self.log_warning(f"松开按键后缓停指令返回异常: ret={ret}")

        try:
            self.target_pose = read_arm_pose(self.arm)
        except RuntimeError as exc:
            self.log_warning(str(exc))

        self.clear_control_state()
        self.log_info("按键已松开，停止发送位姿透传并清空 tracker 原点。")

    def fail_safe_stop(self, reason: str):
        assert self.arm is not None
        self.log_warning(f"控制异常，执行缓停并清空控制状态: {reason}")
        if self.was_active and self.config.slow_stop_on_release:
            ret = self.arm.arm.rm_set_arm_slow_stop()
            if ret != 0:
                self.log_warning(f"异常缓停指令返回异常: ret={ret}")
        self.clear_control_state()

    def send_target_pose(self, tracker_sample: TrackerPoseSample):
        assert self.arm is not None
        assert self.tracker_origin_xyz is not None
        assert self.tracker_origin_rpy is not None
        assert self.arm_origin_pose is not None

        raw_delta_xyz = tracker_sample.control_xyz - self.tracker_origin_xyz
        delta_xyz = raw_delta_xyz[list(self.config.xyz_axis_map)]
        delta_xyz = delta_xyz * np.asarray(self.config.xyz_axis_sign) * self.config.xyz_scale_factor
        delta_xyz = clip_delta(delta_xyz, self.config.max_delta_m)

        raw_delta_rpy = wrap_angle(tracker_sample.control_rpy - self.tracker_origin_rpy)
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

    def control_loop(self):
        active = self.is_control_active()
        if not active:
            if self.was_active:
                self.deactivate_control()
            return

        tracker_sample = self.read_tracker_pose()
        if not self.was_active:
            self.activate_control(tracker_sample)
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
                if self.was_active and self.config.slow_stop_on_release:
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


def print_tracker_inventory(config: TeleopConfig):
    """仅列出当前 SteamVR 可见 tracker，不连接机械臂。"""
    reader = SteamVrTrackerReader(
        tracking_universe=config.tracking_universe,
        coordinate_mode=config.coordinate_mode,
    )
    try:
        trackers = reader.list_trackers()
        if not trackers:
            print("[WARN] SteamVR 中未发现任何 GenericTracker。")
            return

        print(
            f"[INFO] 当前可见 tracker 列表，tracking_universe={config.tracking_universe}, "
            f"coordinate_mode={config.coordinate_mode}"
        )
        for item in trackers:
            status_text = f"connected={item.connected}, pose_valid={item.pose_valid}"
            print(
                f"  - index={item.index}, serial={item.serial}, "
                f"model={item.model}, {status_text}"
            )
            if item.connected and item.pose_valid:
                sample = reader.read_tracker_pose(item.serial)
                print(
                    f"    raw_xyz={sample.raw_xyz.tolist()}, "
                    f"control_xyz={sample.control_xyz.tolist()}, "
                    f"control_rpy={sample.control_rpy.tolist()}"
                )
    finally:
        reader.close()


class TerminalHoldKeyMonitor:
    """终端按键保持器。

    终端没有原生 key release 事件，这里依赖系统按键重复：
    在 release_timeout 时间内持续收到同一个键的重复输入时认为“仍按住”，
    超时则认为已经松开。
    """

    def __init__(self, hold_key: str, release_timeout_s: float):
        if not sys.stdin.isatty():
            raise RuntimeError("当前标准输入不是 TTY，无法在终端模式下监听按键。")

        self.hold_key = hold_key
        self.display_key = "Space" if hold_key.lower() == "space" else hold_key
        self.release_timeout_s = release_timeout_s
        self.is_pressed = False
        self.should_exit = False
        self.runtime_text = ""
        self.fd = sys.stdin.fileno()
        self.last_key_time = 0.0
        self.original_termios = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)

        print(
            "[INFO] 当前使用终端按键模式："
            f"请保持终端焦点并按住 {self.display_key}。"
            "终端模式依赖系统按键重复，松开后停止可能有轻微延迟。"
        )

    def _normalize_key(self, value: str) -> str:
        return value.strip().lower()

    def _matches(self, char: str) -> bool:
        target = self._normalize_key(self.hold_key)
        if target == "space":
            return char == " "
        if target in {"enter", "return"}:
            return char in {"\n", "\r"}
        return self._normalize_key(char) == target

    def set_runtime_text(self, _message: str):
        return

    def poll(self):
        while True:
            ready, _, _ = select.select([sys.stdin], [], [], 0.0)
            if not ready:
                break

            char = sys.stdin.read(1)
            if char in {"\x03", "\x04"}:
                self.should_exit = True
                return
            if self._matches(char):
                self.is_pressed = True
                self.last_key_time = time.monotonic()

        if self.is_pressed and (time.monotonic() - self.last_key_time) > self.release_timeout_s:
            self.is_pressed = False

    def close(self):
        try:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.original_termios)
        except Exception:
            pass


def create_hold_key_monitor(
    hold_key: str,
    hold_source: str,
    hold_release_timeout_s: float,
) -> HoldKeyWindow | TerminalHoldKeyMonitor:
    """根据当前环境创建按键保持器。"""
    if hold_source == "window":
        return HoldKeyWindow(hold_key)
    if hold_source == "terminal":
        return TerminalHoldKeyMonitor(hold_key, hold_release_timeout_s)

    display = os.environ.get("DISPLAY", "").strip()
    if display:
        try:
            return HoldKeyWindow(hold_key)
        except RuntimeError:
            pass

    return TerminalHoldKeyMonitor(hold_key, hold_release_timeout_s)


def parse_args(argv: list[str] | None = None) -> TeleopConfig:
    parser = argparse.ArgumentParser(description="SteamVR 单 tracker 相对位姿控制 RealMan RM75-B xyz+rpy")
    parser.add_argument("--ip", default=DEFAULT_ARM_IP, help="RM75-B 控制器 IP")
    parser.add_argument("--port", type=int, default=DEFAULT_ARM_PORT, help="RM75-B 控制器端口")
    parser.add_argument("--rate", type=int, default=DEFAULT_CONTROL_RATE_HZ, help="控制频率 Hz")
    parser.add_argument("--tracker-serial", default=None, help="指定 SteamVR tracker 序列号；默认自动选择第一个 connected 且 pose_valid 的 tracker")
    parser.add_argument("--tracking-universe", choices=("standing", "raw", "seated"), default=DEFAULT_TRACKING_UNIVERSE, help="OpenVR TrackingUniverse，默认 standing")
    parser.add_argument("--coordinate-mode", choices=("project_world", "raw_openvr"), default=DEFAULT_COORDINATE_MODE, help="tracker 坐标转换模式，默认 project_world")
    parser.add_argument("--xyz-scale", type=float, default=DEFAULT_XYZ_SCALE_FACTOR, help="tracker xyz 位移到机械臂 xyz 位移的比例")
    parser.add_argument("--rpy-scale", type=float, default=DEFAULT_RPY_SCALE_FACTOR, help="tracker rpy 转动到机械臂 rpy 转动的比例")
    parser.add_argument("--xyz-axis-map", type=parse_axis_map, default=DEFAULT_XYZ_AXIS_MAP, help="xyz 通道映射，默认 0,1,2")
    parser.add_argument("--xyz-axis-sign", type=parse_axis_sign, default=DEFAULT_XYZ_AXIS_SIGN, help="xyz 通道方向，默认 1,1,1；如某轴方向反了可设为 -1")
    parser.add_argument("--rpy-axis-map", type=parse_axis_map, default=DEFAULT_RPY_AXIS_MAP, help="rpy 通道映射，默认 1,0,2 表示交换 roll/pitch 并保持 yaw")
    parser.add_argument("--rpy-axis-sign", type=parse_axis_sign, default=DEFAULT_RPY_AXIS_SIGN, help="rpy 通道方向，默认 1,1,1；如某轴方向反了可设为 -1")
    parser.add_argument("--scale", type=float, default=None, help="兼容旧参数：等同于 --xyz-scale")
    parser.add_argument("--hold-key", default=DEFAULT_HOLD_KEY, help="按住该键才发送控制，默认 space")
    parser.add_argument("--hold-source", choices=("auto", "window", "terminal"), default=DEFAULT_HOLD_SOURCE, help="按键监听方式：auto 优先窗口、否则终端；window 需要 DISPLAY；terminal 依赖终端按键重复")
    parser.add_argument("--hold-release-timeout", type=float, default=DEFAULT_HOLD_RELEASE_TIMEOUT_S, help="终端按键模式下判定“已松开”的超时时间，默认 0.7 秒")
    parser.add_argument("--max-delta", type=float, default=DEFAULT_MAX_DELTA_M, help="单次按住允许的最大 xyz 相对位移，单位 m；<=0 表示不限制")
    parser.add_argument("--high-follow", action="store_true", help="启用 CANFD 高跟随；仅在控制周期稳定不超过 10ms 时使用")
    parser.add_argument("--no-slow-stop", action="store_true", help="松开按键时不发送 rm_set_arm_slow_stop")
    parser.add_argument("--list-trackers", action="store_true", help="仅打印当前 SteamVR 可见 tracker 列表与位姿，不连接机械臂")
    args, _ = parser.parse_known_args(argv)

    return TeleopConfig(
        arm_ip=args.ip,
        arm_port=args.port,
        control_rate_hz=args.rate,
        tracker_serial=args.tracker_serial,
        tracking_universe=args.tracking_universe,
        coordinate_mode=args.coordinate_mode,
        xyz_scale_factor=args.xyz_scale if args.scale is None else args.scale,
        rpy_scale_factor=args.rpy_scale,
        xyz_axis_map=args.xyz_axis_map,
        xyz_axis_sign=args.xyz_axis_sign,
        rpy_axis_map=args.rpy_axis_map,
        rpy_axis_sign=args.rpy_axis_sign,
        hold_key=args.hold_key,
        hold_source=args.hold_source,
        hold_release_timeout_s=args.hold_release_timeout,
        max_delta_m=args.max_delta,
        high_follow=args.high_follow,
        slow_stop_on_release=not args.no_slow_stop,
        list_trackers=args.list_trackers,
    )


def main(args: list[str] | None = None):
    config = parse_args(args)
    if config.list_trackers:
        print_tracker_inventory(config)
        return

    teleop = RealmanSteamVrTrackerTeleop(config)
    try:
        teleop.run()
    except KeyboardInterrupt:
        teleop.log_info("检测到退出信号，正在关闭。")
    finally:
        teleop.cleanup()


if __name__ == "__main__":
    main()
