#!/usr/bin/env python3
"""
Wuji Hand 数字菜单姿态控制。

行为：
1. 连接 Wuji Hand。
2. 读取 5x4 关节上下限。
3. 使能所有关节。
4. 通过数字选择握拳、自然张开、伸直侧展等预设姿态。
5. 无论是否异常，最终失能所有关节。

注意：
- 默认 `--ratio 1.0` 会使用每个关节正向上限作为握拳目标。
- 如果首次实机测试，建议先用 `--ratio 0.7` 或 `--ratio 0.8`。
- 依据 JOINT_LIMIT.txt 和现有脚本约定：0.0 rad 是自然张开，正向上限是握拳。
- 伸直侧展只调整第 1 列侧摆轴，不让弯曲轴进入反向限位。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import traceback
from typing import Optional, Tuple

import numpy as np


OPEN_POSE = np.zeros((5, 4), dtype=np.float64)
RECORDED_LOWER_LIMIT = np.array(
    [
        [-0.070, -0.300, -0.532, -0.549],
        [-0.251, -0.444, -0.562, -0.522],
        [-0.251, -0.422, -0.586, -0.550],
        [-0.264, -0.436, -0.554, -0.558],
        [-0.279, -0.426, -0.565, -0.545],
    ],
    dtype=np.float64,
)
RECORDED_UPPER_LIMIT = np.array(
    [
        [1.667, 0.957, 1.689, 1.661],
        [1.660, 0.374, 1.656, 1.708],
        [1.640, 0.374, 1.631, 1.684],
        [1.646, 0.380, 1.664, 1.693],
        [1.625, 0.374, 1.652, 1.678],
    ],
    dtype=np.float64,
)
FINGER_NAMES = ["拇指", "食指", "中指", "无名指", "小指"]
JOINT_NAMES = ["J0", "J1", "J2", "J3"]
POSE_MENU = {
    "1": "握拳",
    "2": "自然张开（四指并拢，拇指回到自然张开位）",
    "3": "伸直侧展（中指不动，其余手指向两侧张开）",
}


def _start_watchdog(timeout: float) -> None:
    if timeout <= 0 or os.name != "nt":
        return

    pid = os.getpid()
    seconds = max(1.0, float(timeout))
    script = (
        f"Start-Sleep -Seconds {seconds}; "
        f"taskkill /PID {pid} /T /F 2>$null | Out-Null"
    )
    subprocess.Popen(
        ["powershell", "-NoProfile", "-WindowStyle", "Hidden", "-Command", script],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )


def _matrix_text(matrix: np.ndarray) -> str:
    rows = []
    for i, row in enumerate(matrix):
        values = ", ".join(f"{float(v):+0.3f}" for v in row)
        rows.append(f"  {FINGER_NAMES[i]}: [{values}]")
    return "\n".join(rows)


def _read_limits(hand, live: bool) -> tuple[np.ndarray, np.ndarray]:
    if not live:
        return RECORDED_LOWER_LIMIT.copy(), RECORDED_UPPER_LIMIT.copy()

    lower = np.asarray(hand.read_joint_lower_limit(), dtype=np.float64)
    upper = np.asarray(hand.read_joint_upper_limit(), dtype=np.float64)
    if lower.shape != (5, 4) or upper.shape != (5, 4):
        raise RuntimeError(
            f"关节限位形状异常: lower={lower.shape}, upper={upper.shape}, 预期 (5, 4)"
        )
    return lower, upper


def _safe_read_actual(hand, fallback: np.ndarray) -> np.ndarray:
    try:
        actual = np.asarray(hand.read_joint_actual_position(), dtype=np.float64)
        if actual.shape == (5, 4):
            return actual
    except Exception as exc:
        print(f"[警告] 读取当前姿态失败，改用回退姿态: {exc}")
    return fallback.copy()


def _build_fist_pose(lower: np.ndarray, upper: np.ndarray, ratio: float) -> np.ndarray:
    target = upper * ratio
    return np.clip(target, lower, upper).astype(np.float64)


def _build_natural_open_pose(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return np.clip(OPEN_POSE, lower, upper).astype(np.float64)


def _build_spread_open_pose(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    target = _build_natural_open_pose(lower, upper)
    # 第 1 列是侧摆轴：拇指 CM yaw，其余四指 MP yaw。
    # 其余列保持 0，避免伸直张开时触发弯曲轴反向限位。
    target[0, 1] = lower[0, 1]  # 拇指向外侧展开
    target[1, 1] = lower[1, 1]  # 食指向外侧展开
    target[2, 1] = 0.0          # 中指作为中心参考保持不动
    target[3, 1] = upper[3, 1]  # 无名指向外侧展开
    target[4, 1] = upper[4, 1]  # 小指向外侧展开
    return np.clip(target, lower, upper).astype(np.float64)


def _ramp_pose(hand, start: np.ndarray, goal: np.ndarray, seconds: float, steps: int, label: str) -> None:
    steps = max(1, int(steps))
    period = max(0.0, seconds) / steps
    for step in range(1, steps + 1):
        alpha = step / steps
        target = start + (goal - start) * alpha
        hand.write_joint_target_position_unchecked(target.astype(np.float64))
        if period > 0.0:
            time.sleep(period)
    print(f"[完成] {label}")


def _write_enabled_fast(hand, enabled: bool, label: str, timeout: float = 2.0) -> None:
    if not enabled:
        hand.write_joint_enabled(False, timeout)
        print(label)
        return

    try:
        hand.write_joint_enabled_unchecked(enabled)
    except Exception:
        hand.write_joint_enabled(enabled)
    print(label)


def _connect_hand():
    import wujihandpy

    # 用底层绑定避开 Python 包装层的后台升级检查；当前 Windows 现场动作脚本
    # 只需要同步打开设备并发送控制帧。
    hand = wujihandpy._core.Hand()
    hand.disable_thread_safe_check()
    return hand


def _print_menu() -> None:
    print("\n请选择动作：")
    for key, label in POSE_MENU.items():
        print(f"  {key}. {label}")
    print("  q. 退出并失能")


def _select_pose(
    choice: str,
    lower: np.ndarray,
    upper: np.ndarray,
    ratio: float,
) -> Optional[Tuple[str, np.ndarray]]:
    if choice == "1":
        return POSE_MENU[choice], _build_fist_pose(lower, upper, ratio)
    if choice == "2":
        return POSE_MENU[choice], _build_natural_open_pose(lower, upper)
    if choice == "3":
        return POSE_MENU[choice], _build_spread_open_pose(lower, upper)
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wuji Hand 数字菜单姿态控制")
    parser.add_argument(
        "--ratio",
        type=float,
        default=1.0,
        help="握拳使用的正向上限比例，1.0=完整上限，0.7=70%% 握拳。默认 1.0",
    )
    parser.add_argument(
        "--ramp-time",
        type=float,
        default=2.0,
        help="每段平滑移动时间，单位秒。默认 2.0",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=80,
        help="每段平滑移动分步数。默认 80",
    )
    parser.add_argument(
        "--hold",
        type=float,
        default=1.0,
        help="握拳后保持时间，单位秒。默认 1.0",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印张开/握拳目标矩阵，不连接硬件",
    )
    parser.add_argument(
        "--live-limits",
        action="store_true",
        help="连接后实时读取关节限位。默认使用 JOINT_LIMIT.txt 中已记录的实机限位，避免 Windows 现场阻塞读。",
    )
    parser.add_argument(
        "--open-only",
        action="store_true",
        help="只发送全 0 自然张开姿态并失能，不进入数字菜单。",
    )
    parser.add_argument(
        "--max-open-only",
        action="store_true",
        help="只发送伸直侧展姿态并失能，不进入数字菜单。",
    )
    parser.add_argument(
        "--once",
        choices=sorted(POSE_MENU.keys()),
        help="只执行一次指定数字动作并退出：1=握拳，2=自然张开，3=伸直侧展。",
    )
    parser.add_argument(
        "--watchdog-timeout",
        type=float,
        default=None,
        help=(
            "Windows 现场总超时保护，超过该秒数仍未退出则强制结束脚本进程；"
            "0=关闭。默认：菜单交互模式关闭，--once/--open-only/--max-open-only 为 8.0。"
        ),
    )
    return parser.parse_args()


def _resolve_watchdog_timeout(args: argparse.Namespace) -> float:
    if args.watchdog_timeout is not None:
        return float(args.watchdog_timeout)
    if args.once or args.open_only or args.max_open_only:
        return 8.0
    return 0.0


def main() -> None:
    args = parse_args()
    _start_watchdog(_resolve_watchdog_timeout(args))
    if not 0.0 <= args.ratio <= 1.0:
        raise SystemExit("--ratio 必须在 0.0 到 1.0 之间")

    if args.dry_run:
        print("[dry-run] 完全张开目标:")
        print(_matrix_text(OPEN_POSE))
        print("[dry-run] 需要连接硬件后才能读取真实 upper limit 并计算握拳目标。")
        return

    try:
        import wujihandpy  # noqa: F401
    except ImportError:
        raise SystemExit("错误: 未安装 wujihandpy，请先执行: pip install wujihandpy")

    hand = None
    try:
        print("[连接] 正在连接 Wuji Hand...")
        hand = _connect_hand()
        lower, upper = _read_limits(hand, args.live_limits)
        open_pose = np.clip(OPEN_POSE, lower, upper)
        fist_pose = _build_fist_pose(lower, upper, args.ratio)

        print("[信息] 关节下限:")
        print(_matrix_text(lower))
        print("[信息] 关节上限:")
        print(_matrix_text(upper))
        print(f"[信息] 握拳目标 ratio={args.ratio:.2f}:")
        print(_matrix_text(fist_pose))
        print("[信息] 张开目标:")
        print(_matrix_text(open_pose))
        print("[信息] 伸直侧展目标:")
        print(_matrix_text(_build_spread_open_pose(lower, upper)))

        _write_enabled_fast(hand, True, "[使能] 所有关节使能")
        time.sleep(0.2)

        if args.open_only:
            _ramp_pose(hand, open_pose, open_pose, args.ramp_time, args.steps, "已发送自然张开姿态")
            return

        if args.max_open_only:
            spread_open_pose = _build_spread_open_pose(lower, upper)
            _ramp_pose(hand, open_pose, spread_open_pose, args.ramp_time, args.steps, "已发送伸直侧展姿态")
            return

        current_pose = open_pose
        if args.once:
            selected = _select_pose(args.once, lower, upper, args.ratio)
            if selected is None:
                raise SystemExit(f"未知动作编号: {args.once}")
            label, target_pose = selected
            _ramp_pose(hand, current_pose, target_pose, args.ramp_time, args.steps, f"已到达{label}")
            if args.hold > 0.0:
                print(f"[保持] {label}保持 {args.hold:.2f}s")
                time.sleep(args.hold)
            return

        while True:
            _print_menu()
            choice = input("输入数字后回车: ").strip().lower()
            if choice in {"q", "quit", "exit"}:
                break

            selected = _select_pose(choice, lower, upper, args.ratio)
            if selected is None:
                print(f"[提示] 未知动作编号: {choice}")
                continue

            label, target_pose = selected
            print(f"[执行] {label}")
            print(_matrix_text(target_pose))
            _ramp_pose(hand, current_pose, target_pose, args.ramp_time, args.steps, f"已到达{label}")
            current_pose = target_pose
            if args.hold > 0.0:
                print(f"[保持] {label}保持 {args.hold:.2f}s")
                time.sleep(args.hold)

    except KeyboardInterrupt:
        print("\n[中断] 收到 Ctrl+C，尝试回到自然张开姿态")
        if hand is not None:
            try:
                lower, upper = _read_limits(hand, args.live_limits)
                open_pose = np.clip(OPEN_POSE, lower, upper)
                current_pose = open_pose
                _ramp_pose(hand, current_pose, open_pose, 1.0, 40, "中断后已张开")
            except Exception as exc:
                print(f"[警告] 中断后张开失败: {exc}")
    finally:
        if hand is not None:
            try:
                time.sleep(0.5)
                _write_enabled_fast(hand, False, "[失能] 所有关节已失能")
            except Exception as exc:
                print(f"[警告] 失能失败，请手动检查设备状态: {exc}")


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except SystemExit as exc:
        code = exc.code
        if isinstance(code, int):
            exit_code = code
        elif code:
            print(code, file=sys.stderr)
            exit_code = 1
    except BaseException:
        traceback.print_exc()
        exit_code = 1
    finally:
        # wujihandpy 1.6.0 在当前 Windows 现场可能会在解释器退出阶段等待后台线程。
        # 这里不再 flush，避免 Windows 控制台/子进程包装在收尾阶段继续阻塞。
        os._exit(exit_code)
