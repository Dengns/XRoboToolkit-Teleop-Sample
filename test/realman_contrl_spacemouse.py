#!/usr/bin/env python3
"""使用 SpaceMouse 积分控制 RealMan RM75-B 末端 xyz+rpy。"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import os
import select
import sys
import termios
import threading
import time
import tty
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPACE_MOUSE_ROOT = PROJECT_ROOT / "spacemouse2rm75b"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from xrobotoolkit_teleop.hardware.interface.rm75b import RM75BInterface

DEFAULT_ARM_IP = "192.168.5.200"
DEFAULT_ARM_PORT = 8080
DEFAULT_CONTROL_RATE_HZ = 50
SCALE_OPTIONS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
DEFAULT_AXIS_ENABLE = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
DEFAULT_CONSECUTIVE_FAIL_LIMIT = 10
DEFAULT_DEADZONE = 40
DEFAULT_TRANSLATION_SCALE = 0.0004 / 350.0
DEFAULT_ROTATION_SCALE = 0.004 / 350.0
DEFAULT_MAX_TRANSLATION_PER_CYCLE = 0.001
DEFAULT_MAX_ROTATION_PER_CYCLE = 0.01
DEFAULT_EMA_ALPHA = 1.0
DEFAULT_AXIS_SIGNS = (1.0, -1.0, 1.0, -1.0, 1.0, 1.0)
DEFAULT_AXIS_MAP = (2, 0, 1, 5, 3, 4)
DEFAULT_ROTATION_AXIS_MAP = (1, 0, 2)
DEFAULT_WORKSPACE_MIN = (-0.5, -0.5, 0.0)
DEFAULT_WORKSPACE_MAX = (0.5, 0.5, 0.7)
SPNAV_EVENT_MOTION = 1
SPNAV_EVENT_BUTTON = 2


@dataclass(frozen=True)
class TeleopConfig:
    arm_ip: str
    arm_port: int
    control_rate_hz: int
    initial_scale_factor: float
    follow: bool


class MotionEvent(ctypes.Structure):
    """与 spnav.h 对齐的 motion event 结构。"""

    _fields_ = [
        ("type", ctypes.c_int),
        ("x", ctypes.c_int),
        ("y", ctypes.c_int),
        ("z", ctypes.c_int),
        ("rx", ctypes.c_int),
        ("ry", ctypes.c_int),
        ("rz", ctypes.c_int),
        ("period", ctypes.c_uint),
        ("data", ctypes.c_void_p),
    ]


class ButtonEvent(ctypes.Structure):
    """与 spnav.h 对齐的 button event 结构。"""

    _fields_ = [
        ("type", ctypes.c_int),
        ("press", ctypes.c_int),
        ("bnum", ctypes.c_int),
    ]


class SpnavEvent(ctypes.Union):
    """与 spnav.h 对齐的联合体。"""

    _fields_ = [
        ("type", ctypes.c_int),
        ("motion", MotionEvent),
        ("button", ButtonEvent),
    ]


def resolve_libspnav_path() -> str:
    """解析 libspnav 路径，优先使用本仓库提供的库文件。"""
    candidates = [
        os.environ.get("SPNAV_LIB_PATH"),
        str(SPACE_MOUSE_ROOT / "libspnav.so.0.4"),
        ctypes.util.find_library("spnav"),
        "/lib/libspnav.so.0",
        "/usr/lib/libspnav.so.0",
        "/usr/lib/x86_64-linux-gnu/libspnav.so.0",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        if os.path.isabs(candidate) and not os.path.exists(candidate):
            continue
        return candidate
    raise RuntimeError(
        "未找到可用的 libspnav。"
        "请确认已安装 spacenavd/libspnav，或把库文件放到 spacemouse2rm75b/libspnav.so.0.4，"
        "或通过环境变量 SPNAV_LIB_PATH 指定路径。"
    )


class SpaceMouseReader:
    """非阻塞 SpaceMouse 读取器，逻辑沿用 spacemouse2rm75b/spacemouse_input.py。"""

    def __init__(self, lib_path: str | None = None, deadzone: int = DEFAULT_DEADZONE):
        self.lib_path = lib_path or resolve_libspnav_path()
        self._lib = ctypes.CDLL(self.lib_path)
        self._lib.spnav_poll_event.argtypes = [ctypes.POINTER(SpnavEvent)]
        self._lib.spnav_poll_event.restype = ctypes.c_int
        self._lib.spnav_open.restype = ctypes.c_int
        self._lib.spnav_close.restype = None
        self._lib.spnav_dev_name.argtypes = [ctypes.c_char_p, ctypes.c_int]
        self._lib.spnav_dev_name.restype = None
        self._lib.spnav_dev_axes.restype = ctypes.c_int
        self._lib.spnav_dev_buttons.restype = ctypes.c_int

        self._deadzone = int(deadzone)
        self._lock = threading.Lock()
        self._axes = [0, 0, 0, 0, 0, 0]
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._is_open = False

    def open(self):
        """连接 spacenavd，并打印当前设备信息。"""
        if self._lib.spnav_open() == -1:
            raise RuntimeError("无法连接 spacenavd，请先确认守护进程已启动。")

        buf = ctypes.create_string_buffer(256)
        self._lib.spnav_dev_name(buf, 256)
        name = buf.value.decode()
        axes = self._lib.spnav_dev_axes()
        buttons = self._lib.spnav_dev_buttons()

        self._is_open = True
        print(
            f"[INFO] SpaceMouse 连接成功: {name} "
            f"(axes={axes}, buttons={buttons}, lib={self.lib_path})"
        )

    def start(self):
        """启动后台轮询线程。"""
        if not self._is_open:
            raise RuntimeError("SpaceMouseReader.start() 前必须先成功调用 open()")

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """停止轮询并断开设备连接。"""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._is_open:
            self._lib.spnav_close()
            self._is_open = False
            print("[INFO] SpaceMouse 已断开。")

    def get_axes(self) -> list[int]:
        """返回最新六轴值 [x, y, z, rx, ry, rz]。"""
        with self._lock:
            return list(self._axes)

    def _apply_deadzone(self, value: int) -> int:
        if abs(value) < self._deadzone:
            return 0
        return value - self._deadzone

    def _poll_loop(self):
        event = SpnavEvent()
        while not self._stop_event.is_set():
            ret = self._lib.spnav_poll_event(ctypes.byref(event))
            if ret == 0:
                time.sleep(0.001)
                continue

            if event.type != SPNAV_EVENT_MOTION:
                continue

            motion = event.motion
            axes = [
                self._apply_deadzone(motion.x),
                self._apply_deadzone(motion.y),
                self._apply_deadzone(motion.z),
                self._apply_deadzone(motion.rx),
                self._apply_deadzone(motion.ry),
                self._apply_deadzone(motion.rz),
            ]
            with self._lock:
                self._axes = axes


def format_scale_options() -> str:
    """格式化档位列表，便于日志展示。"""
    return ",".join(f"{scale:g}" for scale in SCALE_OPTIONS)


def parse_scale_option(value: str) -> float:
    """解析启动时的统一增量比例档位。"""
    try:
        scale = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"scale 必须是数字，可选档位: {format_scale_options()}"
        ) from exc

    if scale not in SCALE_OPTIONS:
        raise argparse.ArgumentTypeError(
            f"scale 只能选择以下档位: {format_scale_options()}"
        )
    return scale


def map_scale_digit_to_factor(digit: int) -> float:
    """将数字键映射到离散 scale 档位。"""
    if digit == 0:
        return SCALE_OPTIONS[-1]
    if 1 <= digit <= 9:
        return SCALE_OPTIONS[digit - 1]
    raise ValueError(f"不支持的数字键: {digit}")


def wrap_angle(angle: np.ndarray | float) -> np.ndarray | float:
    """将角度归一到 [-pi, pi]，避免 RPY 长时间积分后无限增长。"""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def read_arm_pose(arm: RM75BInterface) -> np.ndarray:
    """读取当前机械臂末端位姿 [x, y, z, rx, ry, rz]。"""
    ret, state = arm.arm.rm_get_current_arm_state()
    if ret != 0:
        raise RuntimeError(f"读取机械臂状态失败，ret={ret}")

    pose = np.asarray(state["pose"], dtype=float)
    if pose.shape[0] < 6 or not np.all(np.isfinite(pose[:6])):
        raise RuntimeError(f"机械臂位姿无效: {state.get('pose')}")
    return pose[:6].copy()


class TerminalDigitScaleMonitor:
    """终端即时数字键监听器。

    支持：
    - 1~9: 切换到 0.1~0.9
    - 0: 切换到 1.0
    - Ctrl+C / Ctrl+D: 退出
    """

    def __init__(self):
        if not sys.stdin.isatty():
            raise RuntimeError("当前标准输入不是 TTY，无法在终端模式下监听数字键。")

        self.should_exit = False
        self.fd = sys.stdin.fileno()
        self.original_termios = termios.tcgetattr(self.fd)
        self.buffer = bytearray()
        self.pending_digits: list[int] = []
        tty.setcbreak(self.fd)

        print(
            "[INFO] 当前使用终端数字键即时调参："
            "数字键 1-9 对应 scale 0.1-0.9，数字键 0 对应 1.0。"
            "按键后无需回车，且从当前控制帧立即生效。"
        )

    def _drain_buffer(self):
        while self.buffer:
            first = self.buffer[0]
            if first in (0x03, 0x04):
                del self.buffer[:1]
                self.should_exit = True
                return

            char = chr(first)
            del self.buffer[:1]
            if "0" <= char <= "9":
                self.pending_digits.append(int(char))

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

    def get_digits(self) -> list[int]:
        digits = self.pending_digits[:]
        self.pending_digits.clear()
        return digits

    def close(self):
        try:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.original_termios)
        except Exception:
            pass


class RealmanSpacemouseTeleop:
    """启动即控制，用 SpaceMouse 六轴积分驱动 RM75-B。"""

    def __init__(self, config: TeleopConfig):
        self.config = config
        self.mouse: SpaceMouseReader | None = None
        self.arm: RM75BInterface | None = None
        self.key_monitor: TerminalDigitScaleMonitor | None = None

        self.target_pose: np.ndarray | None = None
        self.runtime_scale_factor = self.config.initial_scale_factor
        self.axis_enable = np.asarray(DEFAULT_AXIS_ENABLE, dtype=float)
        self.smoothed_axes = np.zeros(6, dtype=float)
        self.consecutive_fail_count = 0

        self.init_hardware()
        self.log_info(
            "初始化完成：程序启动后立即进入控制，无需空格启动。"
            f"当前统一增量比例 scale={self.runtime_scale_factor:g}，可用档位: {format_scale_options()}"
        )
        self.log_info(
            "控制说明：平移 xyz 与姿态 rpy 都走积分增量，"
            "每一帧都会先处理数字键，再按当前 scale 计算本帧增量。"
        )

    def log_info(self, message: str):
        print(f"[INFO] {message}")

    def log_warning(self, message: str):
        print(f"[WARN] {message}")

    def init_hardware(self):
        self.key_monitor = TerminalDigitScaleMonitor()
        self.mouse = SpaceMouseReader()
        self.mouse.open()
        self.mouse.start()

        self.arm = RM75BInterface(self.config.arm_ip, self.config.arm_port, enable_gripper=False)
        self.target_pose = read_arm_pose(self.arm)

        self.log_info(f"机械臂连接成功，当前末端位姿: {np.round(self.target_pose, 4).tolist()}")
        self.log_info(
            "当前 SpaceMouse 映射参数沿用 spacemouse2rm75b/config.py 的既有设置："
            f"AXIS_MAP={list(DEFAULT_AXIS_MAP)}, "
            f"AXIS_SIGNS={list(DEFAULT_AXIS_SIGNS)}, "
            f"ROTATION_AXIS_MAP={list(DEFAULT_ROTATION_AXIS_MAP)}, "
            f"TRANSLATION_SCALE={DEFAULT_TRANSLATION_SCALE:g}, "
            f"ROTATION_SCALE={DEFAULT_ROTATION_SCALE:g}, "
            f"EMA_ALPHA={DEFAULT_EMA_ALPHA:g}, "
            f"DEADZONE={DEFAULT_DEADZONE}"
        )
        self.log_info(
            "与原 spacemouse2rm75b 默认配置相比，本脚本强制启用全部 6 轴，"
            f"当前 axis_enable={self.axis_enable.tolist()}。"
        )

    def handle_runtime_scale_commands(self):
        assert self.key_monitor is not None

        self.key_monitor.poll()
        if self.key_monitor.should_exit:
            raise KeyboardInterrupt

        for digit in self.key_monitor.get_digits():
            self.runtime_scale_factor = map_scale_digit_to_factor(digit)
            self.log_info(
                f"数字键 {digit}：统一增量比例切换为 {self.runtime_scale_factor:g}。"
                "新比例从当前这一帧开始生效。"
            )

    def compute_delta(self, raw_axes: list[int]) -> np.ndarray:
        """按当前 scale 计算本帧六轴增量。"""
        mapped = np.array(
            [
                raw_axes[DEFAULT_AXIS_MAP[index]] * DEFAULT_AXIS_SIGNS[index]
                for index in range(6)
            ],
            dtype=float,
        )

        alpha = float(DEFAULT_EMA_ALPHA)
        self.smoothed_axes = alpha * mapped + (1.0 - alpha) * self.smoothed_axes

        delta = np.zeros(6, dtype=float)
        delta[:3] = self.smoothed_axes[:3] * float(DEFAULT_TRANSLATION_SCALE)
        delta[3:] = self.smoothed_axes[3:] * float(DEFAULT_ROTATION_SCALE)
        delta[3:] = delta[3:][list(DEFAULT_ROTATION_AXIS_MAP)]

        delta *= self.runtime_scale_factor

        delta[:3] = np.clip(
            delta[:3],
            -float(DEFAULT_MAX_TRANSLATION_PER_CYCLE),
            float(DEFAULT_MAX_TRANSLATION_PER_CYCLE),
        )
        delta[3:] = np.clip(
            delta[3:],
            -float(DEFAULT_MAX_ROTATION_PER_CYCLE),
            float(DEFAULT_MAX_ROTATION_PER_CYCLE),
        )

        delta *= self.axis_enable
        return delta

    def clamp_target_pose(self):
        assert self.target_pose is not None
        self.target_pose[:3] = np.clip(
            self.target_pose[:3],
            np.asarray(DEFAULT_WORKSPACE_MIN, dtype=float),
            np.asarray(DEFAULT_WORKSPACE_MAX, dtype=float),
        )
        self.target_pose[3:] = wrap_angle(self.target_pose[3:])

    def send_target_pose(self):
        assert self.arm is not None
        assert self.target_pose is not None

        ret = self.arm.arm.rm_movep_canfd(
            self.target_pose.tolist(),
            follow=self.config.follow,
        )
        if ret != 0:
            self.consecutive_fail_count += 1
            self.log_warning(
                f"rm_movep_canfd 返回异常: ret={ret}, "
                f"连续失败次数={self.consecutive_fail_count}, "
                f"target={np.round(self.target_pose, 4).tolist()}"
            )
            if self.consecutive_fail_count >= DEFAULT_CONSECUTIVE_FAIL_LIMIT:
                raise RuntimeError(
                    f"rm_movep_canfd 已连续失败 {self.consecutive_fail_count} 次，触发停止保护。"
                )
            return

        self.consecutive_fail_count = 0

    def control_loop(self):
        assert self.mouse is not None
        assert self.target_pose is not None

        # 先处理数字键，保证新的 scale 从当前控制帧就参与本帧积分。
        self.handle_runtime_scale_commands()

        raw_axes = self.mouse.get_axes()
        delta = self.compute_delta(raw_axes)
        self.target_pose += delta
        self.clamp_target_pose()
        self.send_target_pose()

    def run(self):
        period = 1.0 / float(self.config.control_rate_hz)
        while True:
            start_time = time.monotonic()
            self.control_loop()
            sleep_time = period - (time.monotonic() - start_time)
            if sleep_time > 0.0:
                time.sleep(sleep_time)

    def cleanup(self):
        if self.arm is not None:
            try:
                self.arm.arm.rm_set_arm_slow_stop()
            except Exception as exc:
                self.log_warning(f"机械臂缓停失败: {exc}")
            try:
                self.arm.close()
            except Exception as exc:
                self.log_warning(f"关闭机械臂连接失败: {exc}")
            self.arm = None

        if self.mouse is not None:
            try:
                self.mouse.stop()
            except Exception as exc:
                self.log_warning(f"关闭 SpaceMouse 失败: {exc}")
            self.mouse = None

        if self.key_monitor is not None:
            self.key_monitor.close()
            self.key_monitor = None


def parse_args(argv: list[str] | None = None) -> TeleopConfig:
    parser = argparse.ArgumentParser(description="SpaceMouse 六轴积分控制 RealMan RM75-B")
    parser.add_argument("--ip", default=DEFAULT_ARM_IP, help="RM75-B 控制器 IP")
    parser.add_argument("--port", type=int, default=DEFAULT_ARM_PORT, help="RM75-B 控制器端口")
    parser.add_argument(
        "--rate",
        type=int,
        default=DEFAULT_CONTROL_RATE_HZ,
        help="控制频率 Hz，默认沿用原 spacemouse2rm75b 配置的 50Hz",
    )
    parser.add_argument(
        "--scale",
        type=parse_scale_option,
        default=SCALE_OPTIONS[-1],
        help="启动时的统一增量比例档位；运行中仍可用数字键 1-9/0 即时切换",
    )
    parser.add_argument(
        "--follow",
        action="store_true",
        help="改为高跟随模式发送 rm_movep_canfd；默认 False，沿用低跟随流式控制",
    )
    args = parser.parse_args(argv)
    return TeleopConfig(
        arm_ip=args.ip,
        arm_port=args.port,
        control_rate_hz=args.rate,
        initial_scale_factor=args.scale,
        follow=args.follow,
    )


def main(argv: list[str] | None = None):
    config = parse_args(argv)
    teleop = RealmanSpacemouseTeleop(config)
    try:
        teleop.run()
    except KeyboardInterrupt:
        teleop.log_info("检测到退出信号，正在停止控制。")
    finally:
        teleop.cleanup()


if __name__ == "__main__":
    main()
