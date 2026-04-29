#!/usr/bin/env python3
"""Pico 头显与双手柄位姿精度测试脚本。

功能:
1) 零飘统计 (drift 模式)
2) 位移测量精度 (move 模式，可输入游标卡尺实测值)

运行示例:
python test/test_pose_precision.py drift --duration 30 --rate 50
python test/test_pose_precision.py move --axis x --expected-mm 100 --rate 60
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List

import numpy as np

from xrobotoolkit_teleop.common.xr_client import XrClient


TRACKED_NAMES = ["headset", "left_controller", "right_controller"]


@dataclass
class PoseSample:
    timestamp: float
    name: str
    position_world: np.ndarray
    quat_world: np.ndarray
    position_head_rel: np.ndarray


def format_vec_mm(vec: np.ndarray) -> str:
    return f"[{vec[0] * 1000.0:8.2f}, {vec[1] * 1000.0:8.2f}, {vec[2] * 1000.0:8.2f}] mm"


def format_quat(quat: np.ndarray) -> str:
    return f"[{quat[0]: .4f}, {quat[1]: .4f}, {quat[2]: .4f}, {quat[3]: .4f}]"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    qx, qy, qz, qw = q
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n < 1e-12:
        return np.eye(3)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n

    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ]
    )


def get_pose(xr: XrClient, name: str) -> np.ndarray:
    pose = xr.get_pose_by_name(name)
    if pose is None or len(pose) < 7:
        raise RuntimeError(f"设备 {name} 位姿数据无效: {pose}")
    return np.asarray(pose, dtype=float)


def collect_frame(xr: XrClient, timestamp: float) -> List[PoseSample]:
    poses: Dict[str, np.ndarray] = {name: get_pose(xr, name) for name in TRACKED_NAMES}

    head_p = poses["headset"][:3]
    head_q = poses["headset"][3:7]
    r_wh = quat_to_rotmat(head_q)
    r_hw = r_wh.T

    out: List[PoseSample] = []
    for name in TRACKED_NAMES:
        p = poses[name][:3]
        q = poses[name][3:7]
        p_rel = r_hw @ (p - head_p)
        out.append(
            PoseSample(
                timestamp=timestamp,
                name=name,
                position_world=p,
                quat_world=q,
                position_head_rel=p_rel,
            )
        )
    return out


def write_samples_csv(path: str, samples: List[PoseSample]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "timestamp",
                "name",
                "wx",
                "wy",
                "wz",
                "qx",
                "qy",
                "qz",
                "qw",
                "hrx",
                "hry",
                "hrz",
            ]
        )
        for s in samples:
            writer.writerow(
                [
                    f"{s.timestamp:.6f}",
                    s.name,
                    f"{s.position_world[0]:.9f}",
                    f"{s.position_world[1]:.9f}",
                    f"{s.position_world[2]:.9f}",
                    f"{s.quat_world[0]:.9f}",
                    f"{s.quat_world[1]:.9f}",
                    f"{s.quat_world[2]:.9f}",
                    f"{s.quat_world[3]:.9f}",
                    f"{s.position_head_rel[0]:.9f}",
                    f"{s.position_head_rel[1]:.9f}",
                    f"{s.position_head_rel[2]:.9f}",
                ]
            )


def run_monitor_test(xr: XrClient, rate: float, duration: float | None, clear_screen: bool) -> None:
    interval = 1.0 / rate
    start_time = time.time()
    first_frame = collect_frame(xr, start_time)
    world_origin = {sample.name: sample.position_world.copy() for sample in first_frame}
    head_rel_origin = {sample.name: sample.position_head_rel.copy() for sample in first_frame}

    print("[monitor] 已记录初始帧作为零点，按 Ctrl+C 退出。")

    next_t = time.time()
    while True:
        now = time.time()
        if duration is not None and now - start_time >= duration:
            print("\n[monitor] 已达到设定时长，结束监视。")
            return

        frame = collect_frame(xr, now)
        sample_map = {sample.name: sample for sample in frame}

        if clear_screen:
            print("\033[2J\033[H", end="")

        print("Pico 实时定位监视")
        print(f"运行时长: {now - start_time:7.2f} s | 刷新频率: {rate:.1f} Hz")
        print("说明: [SDK直接] 为 SDK 原始返回值, [脚本计算] 为基于原始值做的坐标变换或差分")
        print("")

        for name in TRACKED_NAMES:
            sample = sample_map[name]
            world_delta = sample.position_world - world_origin[name]
            head_rel_delta = sample.position_head_rel - head_rel_origin[name]

            print(f"[{name}]")
            print(f"  [SDK直接]  world_xyz       {format_vec_mm(sample.position_world)}")
            print(f"  [SDK直接]  quat_xyzw       {format_quat(sample.quat_world)}")
            print(f"  [脚本计算] world_delta     {format_vec_mm(world_delta)}")
            print(f"  [脚本计算] head_relative   {format_vec_mm(sample.position_head_rel)}")
            print(f"  [脚本计算] headrel_delta   {format_vec_mm(head_rel_delta)}")
            print("")

        next_t += interval
        time.sleep(max(0.0, next_t - time.time()))


def calc_drift_stats(samples: List[PoseSample], frame: str) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for name in TRACKED_NAMES:
        arr = []
        for s in samples:
            if s.name != name:
                continue
            arr.append(s.position_world if frame == "world" else s.position_head_rel)

        data = np.asarray(arr, dtype=float)
        if len(data) < 2:
            continue

        base = data[0]
        delta = data - base
        dist = np.linalg.norm(delta, axis=1)
        axis_std = np.std(data, axis=0)

        stats[name] = {
            "sample_count": int(len(data)),
            "max_drift_mm": float(np.max(dist) * 1000.0),
            "mean_drift_mm": float(np.mean(dist) * 1000.0),
            "std_x_mm": float(axis_std[0] * 1000.0),
            "std_y_mm": float(axis_std[1] * 1000.0),
            "std_z_mm": float(axis_std[2] * 1000.0),
            "peak_to_peak_mm": float(np.linalg.norm(np.max(data, axis=0) - np.min(data, axis=0)) * 1000.0),
        }
    return stats


def run_drift_test(xr: XrClient, duration: float, rate: float, out_dir: str) -> None:
    interval = 1.0 / rate
    end_time = time.time() + duration
    samples: List[PoseSample] = []

    print(f"[drift] 开始采样: 持续 {duration:.1f}s, 频率 {rate:.1f}Hz")
    print("[drift] 请尽量保持头显和手柄静止。")

    next_t = time.time()
    while time.time() < end_time:
        now = time.time()
        samples.extend(collect_frame(xr, now))
        next_t += interval
        sleep_t = max(0.0, next_t - time.time())
        time.sleep(sleep_t)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(out_dir, f"drift_samples_{stamp}.csv")
    write_samples_csv(csv_path, samples)

    summary = {
        "mode": "drift",
        "duration_s": duration,
        "rate_hz": rate,
        "world": calc_drift_stats(samples, "world"),
        "head_relative": calc_drift_stats(samples, "head_rel"),
        "csv": csv_path,
    }

    summary_path = os.path.join(out_dir, f"drift_summary_{stamp}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n[drift] 测试完成")
    print(f"[drift] 原始数据: {csv_path}")
    print(f"[drift] 汇总结果: {summary_path}")
    print("[drift] 建议重点看 head_relative 下左右手柄指标，可剔除头显微动影响。")


def read_axis(axis: str) -> np.ndarray:
    mapping = {
        "x": np.array([1.0, 0.0, 0.0]),
        "y": np.array([0.0, 1.0, 0.0]),
        "z": np.array([0.0, 0.0, 1.0]),
    }
    if axis not in mapping:
        raise ValueError("axis 仅支持 x/y/z")
    return mapping[axis]


def collect_window_mean(xr: XrClient, sec: float, rate: float, target: str, frame: str) -> np.ndarray:
    interval = 1.0 / rate
    end_time = time.time() + sec
    arr = []
    next_t = time.time()
    while time.time() < end_time:
        frame_samples = collect_frame(xr, time.time())
        for s in frame_samples:
            if s.name == target:
                arr.append(s.position_world if frame == "world" else s.position_head_rel)
                break
        next_t += interval
        time.sleep(max(0.0, next_t - time.time()))

    data = np.asarray(arr, dtype=float)
    if len(data) == 0:
        raise RuntimeError("窗口内未采到数据")
    return np.mean(data, axis=0)


def run_move_test(
    xr: XrClient,
    target: str,
    axis: str,
    expected_mm: float,
    caliper_mm: float,
    baseline_sec: float,
    settle_sec: float,
    rate: float,
    out_dir: str,
) -> None:
    axis_vec = read_axis(axis)
    expected_m = expected_mm / 1000.0
    caliper_m = caliper_mm / 1000.0

    print(f"[move] 目标设备: {target}, 轴向: {axis}")
    print("[move] 步骤1: 保持初始姿态，按回车开始基线采样")
    input()

    base_w = collect_window_mean(xr, baseline_sec, rate, target, "world")
    base_h = collect_window_mean(xr, baseline_sec, rate, target, "head_rel")

    print("[move] 步骤2: 沿指定轴移动手柄到目标位置，稳定后按回车")
    input()

    end_w = collect_window_mean(xr, settle_sec, rate, target, "world")
    end_h = collect_window_mean(xr, settle_sec, rate, target, "head_rel")

    delta_w = end_w - base_w
    delta_h = end_h - base_h

    proj_w = float(np.dot(delta_w, axis_vec))
    proj_h = float(np.dot(delta_h, axis_vec))
    norm_w = float(np.linalg.norm(delta_w))
    norm_h = float(np.linalg.norm(delta_h))

    summary = {
        "mode": "move",
        "target": target,
        "axis": axis,
        "expected_mm": expected_mm,
        "caliper_mm": caliper_mm,
        "baseline_sec": baseline_sec,
        "settle_sec": settle_sec,
        "rate_hz": rate,
        "world": {
            "delta_xyz_mm": (delta_w * 1000.0).tolist(),
            "delta_norm_mm": norm_w * 1000.0,
            "delta_axis_mm": proj_w * 1000.0,
            "error_vs_expected_mm": (proj_w - expected_m) * 1000.0,
            "error_vs_caliper_mm": (proj_w - caliper_m) * 1000.0,
        },
        "head_relative": {
            "delta_xyz_mm": (delta_h * 1000.0).tolist(),
            "delta_norm_mm": norm_h * 1000.0,
            "delta_axis_mm": proj_h * 1000.0,
            "error_vs_expected_mm": (proj_h - expected_m) * 1000.0,
            "error_vs_caliper_mm": (proj_h - caliper_m) * 1000.0,
        },
    }

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(out_dir, f"move_summary_{target}_{stamp}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n[move] 测试完成")
    print(f"[move] 结果文件: {summary_path}")
    print(f"[move] world 轴向位移: {proj_w * 1000.0:.2f} mm")
    print(f"[move] head_relative 轴向位移: {proj_h * 1000.0:.2f} mm")
    print("[move] 如果戴头显有晃动，优先使用 head_relative 结果与卡尺值对比。")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pico 头显/手柄位姿精度测试")
    sub = parser.add_subparsers(dest="mode", required=True)

    p_monitor = sub.add_parser("monitor", help="实时刷新查看定位数据")
    p_monitor.add_argument("--rate", type=float, default=10.0, help="刷新频率(Hz)")
    p_monitor.add_argument("--duration", type=float, default=None, help="监视时长(秒)，默认持续运行")
    p_monitor.add_argument("--no-clear", action="store_true", help="不要清屏，保留历史输出")

    p_drift = sub.add_parser("drift", help="零飘测试")
    p_drift.add_argument("--duration", type=float, default=30.0, help="采样时长(秒)")
    p_drift.add_argument("--rate", type=float, default=50.0, help="采样频率(Hz)")
    p_drift.add_argument("--out-dir", default="logs/precision_tests", help="输出目录")

    p_move = sub.add_parser("move", help="位移精度测试")
    p_move.add_argument("--target", default="right_controller", choices=["left_controller", "right_controller"], help="测试手柄")
    p_move.add_argument("--axis", default="x", choices=["x", "y", "z"], help="位移主轴")
    p_move.add_argument("--expected-mm", type=float, required=True, help="目标位移(mm)")
    p_move.add_argument("--caliper-mm", type=float, required=True, help="游标卡尺实测位移(mm)")
    p_move.add_argument("--baseline-sec", type=float, default=1.0, help="起点均值窗口(秒)")
    p_move.add_argument("--settle-sec", type=float, default=1.0, help="终点均值窗口(秒)")
    p_move.add_argument("--rate", type=float, default=60.0, help="采样频率(Hz)")
    p_move.add_argument("--out-dir", default="logs/precision_tests", help="输出目录")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if hasattr(args, "out_dir"):
        ensure_dir(args.out_dir)

    xr = XrClient()
    try:
        if args.mode == "monitor":
            run_monitor_test(
                xr,
                rate=args.rate,
                duration=args.duration,
                clear_screen=not args.no_clear,
            )
        elif args.mode == "drift":
            run_drift_test(xr, duration=args.duration, rate=args.rate, out_dir=args.out_dir)
        elif args.mode == "move":
            run_move_test(
                xr,
                target=args.target,
                axis=args.axis,
                expected_mm=args.expected_mm,
                caliper_mm=args.caliper_mm,
                baseline_sec=args.baseline_sec,
                settle_sec=args.settle_sec,
                rate=args.rate,
                out_dir=args.out_dir,
            )
        else:
            raise ValueError(f"不支持的模式: {args.mode}")
    finally:
        xr.close()


if __name__ == "__main__":
    main()
