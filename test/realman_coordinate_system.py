#!/usr/bin/env python3
"""按固定末端位姿验证 RM75-B/RM75-6F Base 坐标系 xyz 方向。"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from xrobotoolkit_teleop.hardware.interface.rm75b import RM75BInterface

DEFAULT_ARM_IP = "192.168.5.154"
DEFAULT_ARM_PORT = 8080
DEFAULT_CONTROL_RATE_HZ = 50
DEFAULT_ORIGIN_POSE = (0.1166, 0.0, 0.7247, 0.0, 1.043, 0.0)
DEFAULT_STEP_M = 0.10
DEFAULT_MOVE_DURATION_S = 2.0
DEFAULT_SETTLE_S = 1.0


def parse_pose6(value: str) -> tuple[float, ...]:
    """解析末端位姿 [x,y,z,rx,ry,rz]。"""
    try:
        pose = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "末端位姿必须是逗号分隔数字，例如 0.1166,0,0.7247,0,1.043,0"
        ) from exc

    if len(pose) != 6:
        raise argparse.ArgumentTypeError("末端位姿必须包含 6 个值: x,y,z,rx,ry,rz")
    return pose


def build_axis_target(origin_pose: np.ndarray, axis_index: int, delta_m: float) -> np.ndarray:
    """在 origin_pose 基础上只改变一个 xyz 轴。"""
    target_pose = origin_pose.copy()
    target_pose[axis_index] += delta_m
    return target_pose


def stream_pose(
    arm: RM75BInterface,
    pose: np.ndarray,
    label: str,
    rate_hz: int,
    duration_s: float,
) -> bool:
    """用低跟随 movep_canfd 重复发送目标位姿。"""
    print(f"[INFO] {label}: target={pose.tolist()}")
    period = 1.0 / float(rate_hz)
    send_count = max(1, int(duration_s * rate_hz))

    for _ in range(send_count):
        ret = arm.arm.rm_movep_canfd(pose.astype(float).tolist(), follow=False)
        if ret != 0:
            print(f"[WARN] {label}: rm_movep_canfd 返回异常 ret={ret}")
            return False
        time.sleep(period)

    return True


def run_coordinate_test(
    arm: RM75BInterface,
    origin_pose: np.ndarray,
    step_m: float,
    rate_hz: int,
    move_duration_s: float,
    settle_s: float,
):
    """按 +X/-X/+Y/-Y/+Z/-Z 顺序验证坐标系方向。"""
    steps = [
        ("X 正向 +10cm", 0, step_m),
        ("X 负向 -10cm", 0, -step_m),
        ("Y 正向 +10cm", 1, step_m),
        ("Y 负向 -10cm", 1, -step_m),
        ("Z 正向 +10cm", 2, step_m),
        ("Z 负向 -10cm", 2, -step_m),
    ]

    print("[INFO] 初始化原点状态")
    if not stream_pose(arm, origin_pose, "回到原点", rate_hz, move_duration_s):
        return
    time.sleep(settle_s)

    for label, axis_index, delta_m in steps:
        target_pose = build_axis_target(origin_pose, axis_index, delta_m)
        if not stream_pose(arm, target_pose, label, rate_hz, move_duration_s):
            return
        time.sleep(settle_s)

        if not stream_pose(arm, origin_pose, "回到原点", rate_hz, move_duration_s):
            return
        time.sleep(settle_s)

    print("[INFO] 坐标系方向验证流程结束。")


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="验证 RM75-B/RM75-6F Base 坐标系 xyz 方向")
    parser.add_argument("--ip", default=DEFAULT_ARM_IP, help="RM75 控制器 IP")
    parser.add_argument("--port", type=int, default=DEFAULT_ARM_PORT, help="RM75 控制器端口")
    parser.add_argument(
        "--origin-pose",
        type=parse_pose6,
        default=DEFAULT_ORIGIN_POSE,
        help="测试原点末端位姿 x,y,z,rx,ry,rz，默认 0.1166,0,0.7247,0,1.043,0",
    )
    parser.add_argument("--step", type=float, default=DEFAULT_STEP_M, help="每个方向移动距离，单位 m，默认 0.10")
    parser.add_argument("--rate", type=int, default=DEFAULT_CONTROL_RATE_HZ, help="movep_canfd 发送频率 Hz")
    parser.add_argument("--duration", type=float, default=DEFAULT_MOVE_DURATION_S, help="每个目标位姿持续发送秒数")
    parser.add_argument("--settle", type=float, default=DEFAULT_SETTLE_S, help="每步之间等待秒数")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    origin_pose = np.asarray(args.origin_pose, dtype=float)

    print(f"[INFO] 连接 RM75: {args.ip}:{args.port}")
    arm = RM75BInterface(args.ip, args.port, enable_gripper=False)
    try:
        run_coordinate_test(
            arm=arm,
            origin_pose=origin_pose,
            step_m=args.step,
            rate_hz=args.rate,
            move_duration_s=args.duration,
            settle_s=args.settle,
        )
    finally:
        try:
            arm.arm.rm_set_arm_slow_stop()
        except Exception as exc:
            print(f"[WARN] 退出前缓停失败: {exc}")
        arm.close()


if __name__ == "__main__":
    main()
