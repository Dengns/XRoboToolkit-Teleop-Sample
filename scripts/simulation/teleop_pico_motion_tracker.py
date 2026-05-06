#!/usr/bin/env python3
"""Pico Motion Tracker 位置可视化工具。

默认直接使用 Pico SDK 返回的原始坐标系，不做项目 world 坐标转换。
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from xrobotoolkit_teleop.common.xr_client import XrClient
from xrobotoolkit_teleop.utils.geometry import R_HEADSET_TO_WORLD

DEFAULT_UPDATE_RATE_HZ = 90.0
DEFAULT_FPS_UPDATE_INTERVAL_S = 0.5
DEFAULT_PRINT_INTERVAL_S = 1.0


@dataclass(frozen=True)
class VisualizerConfig:
    update_rate_hz: float
    fps_update_interval_s: float
    print_interval_s: float
    coordinate_mode: str
    open_browser: bool


@dataclass
class TrackerState:
    xyz: np.ndarray
    rotation: np.ndarray


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


def convert_pose_to_visual_coordinate(pose: np.ndarray, coordinate_mode: str) -> tuple[np.ndarray, np.ndarray]:
    """按显示坐标系转换 tracker pose。"""
    xyz = pose[:3].copy()
    rotation = quaternion_xyzw_to_rotation_matrix(pose[3:7], "Motion Tracker")

    if coordinate_mode == "raw_pico":
        return xyz, rotation
    if coordinate_mode == "project_world":
        return R_HEADSET_TO_WORLD @ xyz, R_HEADSET_TO_WORLD @ rotation @ R_HEADSET_TO_WORLD.T

    raise ValueError(f"未知坐标模式: {coordinate_mode}")


def build_transform(xyz: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """构建 MeshCat 4x4 位姿矩阵。"""
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = xyz
    return transform


class MotionTrackerVisualizer:
    """读取 Pico Motion Tracker 数据，并在 MeshCat 中实时显示位置和读取帧率。"""

    def __init__(self, config: VisualizerConfig):
        self.config = config
        self.xr_client: XrClient | None = None
        self.viewer: meshcat.Visualizer | None = None
        self.tracker_states: dict[str, TrackerState] = {}
        self.read_count = 0
        self.read_fps = 0.0
        self.last_fps_update = time.monotonic()
        self.last_status_update = 0.0

    def init_xr_client(self):
        """初始化 XR SDK 客户端。"""
        self.xr_client = XrClient()
        print("[INFO] XRoboToolkit SDK 初始化完成")

    def init_visualizer(self):
        """初始化 MeshCat 可视化环境。"""
        self.viewer = meshcat.Visualizer()
        if self.config.open_browser:
            self.viewer.open()

        print(f"[INFO] MeshCat 可视化器启动，访问: {self.viewer.url()}")
        print(f"[INFO] 当前显示坐标系: {self.config.coordinate_mode}")
        self._create_coordinate_system()
        self._update_status_text(force=True)

    def _create_coordinate_system(self):
        """创建与当前显示模式一致的坐标系。"""
        assert self.viewer is not None

        axis_length = 1.0
        self.viewer["pico_coordinate/origin"].set_object(
            g.Sphere(radius=0.04),
            g.MeshPhongMaterial(color=0x111111),
        )

        self.viewer["pico_coordinate/x_axis"].set_object(
            g.Cylinder(height=axis_length, radius=0.01),
            g.MeshPhongMaterial(color=0xFF3333),
        )
        self.viewer["pico_coordinate/x_axis"].set_transform(
            tf.translation_matrix([axis_length / 2.0, 0.0, 0.0])
            @ tf.rotation_matrix(np.pi / 2.0, [0, 1, 0])
        )

        self.viewer["pico_coordinate/y_axis"].set_object(
            g.Cylinder(height=axis_length, radius=0.01),
            g.MeshPhongMaterial(color=0x33DD33),
        )
        self.viewer["pico_coordinate/y_axis"].set_transform(
            tf.translation_matrix([0.0, axis_length / 2.0, 0.0])
            @ tf.rotation_matrix(-np.pi / 2.0, [1, 0, 0])
        )

        self.viewer["pico_coordinate/z_axis"].set_object(
            g.Cylinder(height=axis_length, radius=0.01),
            g.MeshPhongMaterial(color=0x3366FF),
        )
        self.viewer["pico_coordinate/z_axis"].set_transform(
            tf.translation_matrix([0.0, 0.0, axis_length / 2.0])
        )

        self.viewer["pico_coordinate/x_label"].set_object(
            g.TextGeometry("Pico X", height=0.08, font_size=60),
            g.MeshPhongMaterial(color=0xFF3333),
        )
        self.viewer["pico_coordinate/x_label"].set_transform(
            tf.translation_matrix([axis_length + 0.12, 0.0, 0.0])
        )

        self.viewer["pico_coordinate/y_label"].set_object(
            g.TextGeometry("Pico Y", height=0.08, font_size=60),
            g.MeshPhongMaterial(color=0x33DD33),
        )
        self.viewer["pico_coordinate/y_label"].set_transform(
            tf.translation_matrix([0.0, axis_length + 0.12, 0.0])
        )

        self.viewer["pico_coordinate/z_label"].set_object(
            g.TextGeometry("Pico Z", height=0.08, font_size=60),
            g.MeshPhongMaterial(color=0x3366FF),
        )
        self.viewer["pico_coordinate/z_label"].set_transform(
            tf.translation_matrix([0.0, 0.0, axis_length + 0.12])
        )

        self.viewer["pico_coordinate/grid"].set_object(
            g.Grid(4.0, 40, 40),
            g.MeshPhongMaterial(color=0x888888, transparent=True, opacity=0.35),
        )
        self.viewer["pico_coordinate/grid"].set_transform(tf.translation_matrix([0.0, 0.0, -0.001]))

    def _update_status_text(self, force: bool = False):
        """在可视化界面中更新读取帧率和 tracker 数量。"""
        assert self.viewer is not None
        now = time.monotonic()
        if not force and now - self.last_status_update < self.config.fps_update_interval_s:
            return

        text = (
            f"Read FPS: {self.read_fps:.1f} | "
            f"Trackers: {len(self.tracker_states)} | "
            f"Coord: {self.config.coordinate_mode}"
        )
        self.viewer["status/read_fps"].set_object(
            g.TextGeometry(text, height=0.08, font_size=70),
            g.MeshPhongMaterial(color=0x00FFAA),
        )
        self.viewer["status/read_fps"].set_transform(tf.translation_matrix([-1.8, -1.8, 1.6]))
        self.last_status_update = now

    def _record_read_sample(self):
        """记录一次 Pico motion tracker 数据读取，用于计算读取 FPS。"""
        self.read_count += 1
        now = time.monotonic()
        elapsed = now - self.last_fps_update
        if elapsed >= self.config.fps_update_interval_s:
            self.read_fps = self.read_count / elapsed
            self.read_count = 0
            self.last_fps_update = now

    def _create_tracker_marker(self, tracker_id: str):
        """创建 tracker 标记对象。"""
        assert self.viewer is not None
        node = self.viewer[f"trackers/{tracker_id}"]

        node["body"].set_object(
            g.Sphere(radius=0.07),
            g.MeshPhongMaterial(color=0xFF8800, transparent=True, opacity=0.85),
        )

        axis_length = 0.28
        node["x_axis"].set_object(
            g.Cylinder(height=axis_length, radius=0.006),
            g.MeshPhongMaterial(color=0xFF3333),
        )
        node["x_axis"].set_transform(
            tf.translation_matrix([axis_length / 2.0, 0.0, 0.0])
            @ tf.rotation_matrix(np.pi / 2.0, [0, 1, 0])
        )

        node["y_axis"].set_object(
            g.Cylinder(height=axis_length, radius=0.006),
            g.MeshPhongMaterial(color=0x33DD33),
        )
        node["y_axis"].set_transform(
            tf.translation_matrix([0.0, axis_length / 2.0, 0.0])
            @ tf.rotation_matrix(-np.pi / 2.0, [1, 0, 0])
        )

        node["z_axis"].set_object(
            g.Cylinder(height=axis_length, radius=0.006),
            g.MeshPhongMaterial(color=0x3366FF),
        )
        node["z_axis"].set_transform(tf.translation_matrix([0.0, 0.0, axis_length / 2.0]))

        node["label"].set_object(
            g.TextGeometry(tracker_id, height=0.04, font_size=40),
            g.MeshPhongMaterial(color=0xFFFFFF),
        )
        node["label"].set_transform(tf.translation_matrix([0.0, 0.0, 0.14]))

    def _update_tracker_marker(self, tracker_id: str, state: TrackerState):
        """更新 tracker 的位姿显示。"""
        assert self.viewer is not None
        if tracker_id not in self.tracker_states:
            self._create_tracker_marker(tracker_id)

        self.viewer[f"trackers/{tracker_id}"].set_transform(build_transform(state.xyz, state.rotation))

    def _remove_tracker_marker(self, tracker_id: str):
        """移除不再存在的 tracker 标记。"""
        assert self.viewer is not None
        try:
            self.viewer[f"trackers/{tracker_id}"].delete()
        except Exception as exc:
            print(f"[WARN] 移除 tracker {tracker_id} 可视化节点失败: {exc}")

    def _update_all_trackers(self, tracker_data: dict):
        """解析并更新所有 Motion Tracker 标记。"""
        current_tracker_ids: set[str] = set()

        for tracker_id, pose in iter_motion_tracker_pose_values(tracker_data):
            current_tracker_ids.add(tracker_id)
            try:
                xyz, rotation = convert_pose_to_visual_coordinate(pose, self.config.coordinate_mode)
            except RuntimeError as exc:
                print(f"[WARN] tracker {tracker_id} 位姿无效: {exc}")
                continue

            state = TrackerState(xyz=xyz, rotation=rotation)
            self._update_tracker_marker(tracker_id, state)
            self.tracker_states[tracker_id] = state

        for tracker_id in list(self.tracker_states.keys()):
            if tracker_id not in current_tracker_ids:
                self._remove_tracker_marker(tracker_id)
                del self.tracker_states[tracker_id]

    def _print_status(self):
        """定期在终端输出状态，便于无浏览器时排查。"""
        print(f"[INFO] read_fps={self.read_fps:.1f}, trackers={len(self.tracker_states)}")
        for tracker_id, state in self.tracker_states.items():
            xyz_text = ", ".join(f"{value:.4f}" for value in state.xyz)
            print(f"       {tracker_id}: xyz=[{xyz_text}]")

    def run(self):
        """主运行循环。"""
        self.init_xr_client()
        self.init_visualizer()

        assert self.xr_client is not None
        assert self.viewer is not None

        print("[INFO] 开始 Pico Motion Tracker 可视化，按 Ctrl+C 退出。")
        period = 1.0 / self.config.update_rate_hz if self.config.update_rate_hz > 0 else 0.0
        last_print_time = time.monotonic()

        while True:
            loop_start = time.monotonic()
            try:
                tracker_data = self.xr_client.get_motion_tracker_data()
                self._record_read_sample()

                if tracker_data:
                    self._update_all_trackers(tracker_data)
                else:
                    for tracker_id in list(self.tracker_states.keys()):
                        self._remove_tracker_marker(tracker_id)
                        del self.tracker_states[tracker_id]

                self._update_status_text()

                now = time.monotonic()
                if now - last_print_time >= self.config.print_interval_s:
                    self._print_status()
                    last_print_time = now

                sleep_time = period - (time.monotonic() - loop_start)
                if sleep_time > 0:
                    time.sleep(sleep_time)

            except KeyboardInterrupt:
                print("\n[INFO] 检测到 Ctrl+C，正在退出。")
                break
            except Exception as exc:
                print(f"[WARN] 可视化循环异常: {exc}")
                time.sleep(0.2)

    def cleanup(self):
        """清理 XR SDK 和可视化资源。"""
        if self.xr_client is not None:
            try:
                self.xr_client.close()
                print("[INFO] XR Client 已关闭")
            except Exception as exc:
                print(f"[WARN] 关闭 XR Client 失败: {exc}")
            self.xr_client = None

        if self.viewer is not None:
            try:
                self.viewer.close()
                print("[INFO] MeshCat 可视化器已关闭")
            except Exception as exc:
                print(f"[WARN] 关闭 MeshCat 失败: {exc}")
            self.viewer = None


def parse_args(argv: list[str] | None = None) -> VisualizerConfig:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Pico Motion Tracker 位置和读取帧率可视化")
    parser.add_argument("--rate", type=float, default=DEFAULT_UPDATE_RATE_HZ, help="读取和刷新频率 Hz")
    parser.add_argument(
        "--fps-window",
        type=float,
        default=DEFAULT_FPS_UPDATE_INTERVAL_S,
        help="读取 FPS 统计刷新周期，单位秒",
    )
    parser.add_argument(
        "--print-interval",
        type=float,
        default=DEFAULT_PRINT_INTERVAL_S,
        help="终端状态打印周期，单位秒",
    )
    parser.add_argument(
        "--coordinate-mode",
        choices=("raw_pico", "project_world"),
        default="raw_pico",
        help="显示坐标系：raw_pico 为 Pico SDK 原始坐标；project_world 为项目 world 坐标",
    )
    parser.add_argument("--no-open", action="store_true", help="启动 MeshCat 但不自动打开浏览器")
    args, _ = parser.parse_known_args(argv)

    return VisualizerConfig(
        update_rate_hz=args.rate,
        fps_update_interval_s=args.fps_window,
        print_interval_s=args.print_interval,
        coordinate_mode=args.coordinate_mode,
        open_browser=not args.no_open,
    )


def main(args: list[str] | None = None):
    """主函数。"""
    config = parse_args(args)
    visualizer = MotionTrackerVisualizer(config)

    try:
        visualizer.run()
    finally:
        visualizer.cleanup()


if __name__ == "__main__":
    main()
