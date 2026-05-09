#!/usr/bin/env python3
"""
Wuji Hand 拇指四关节滑块调试界面。

矩阵证据：
- Wuji 姿态是 5x4 rad 矩阵。
- 第 0 行是 Thumb。
- 第 0~3 列分别是 CM pitch、CM yaw、MP pitch、IP pitch。

本脚本只修改第 0 行四个拇指关节，其余手指保持 0.0 rad 自然张开。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
import tkinter as tk
from dataclasses import dataclass
from tkinter import ttk
from typing import Optional

import numpy as np

from wuji_fist_open import (
    OPEN_POSE,
    RECORDED_LOWER_LIMIT,
    RECORDED_UPPER_LIMIT,
    _connect_hand,
    _read_limits,
    _write_enabled_fast,
)


THUMB_ROW = 0


@dataclass(frozen=True)
class ThumbJointInfo:
    column: int
    data_name: str
    joint_name: str
    cn_name: str
    motion_hint: str


THUMB_JOINTS = (
    ThumbJointInfo(
        0,
        "target[0, 0]",
        "CM pitch",
        "拇指根部俯仰/弯曲",
        "0.0 为自然张开，正值通常让拇指根部向掌心弯曲。",
    ),
    ThumbJointInfo(
        1,
        "target[0, 1]",
        "CM yaw",
        "拇指根部侧摆",
        "控制拇指横向内外摆；正负方向以实机滑动观察为准。",
    ),
    ThumbJointInfo(
        2,
        "target[0, 2]",
        "MP pitch",
        "拇指掌指关节弯曲",
        "0.0 为自然张开，正值通常让掌指关节继续弯曲。",
    ),
    ThumbJointInfo(
        3,
        "target[0, 3]",
        "IP pitch",
        "拇指指间关节弯曲",
        "0.0 为自然张开，正值通常让指尖关节弯曲。",
    ),
)


def _matrix_text(matrix: np.ndarray) -> str:
    return np.array2string(matrix, precision=3, suppress_small=True)


def _thumb_row_text(matrix: np.ndarray) -> str:
    values = ", ".join(f"{float(matrix[THUMB_ROW, i]):+0.3f}" for i in range(4))
    return f"[{values}]"


def _print_joint_table(lower: np.ndarray, upper: np.ndarray) -> None:
    print("Wuji 拇指四个数据对应关系（单位 rad）：")
    for info in THUMB_JOINTS:
        lo = float(lower[THUMB_ROW, info.column])
        hi = float(upper[THUMB_ROW, info.column])
        print(
            f"  数据{info.column + 1}: {info.data_name} = {info.joint_name} "
            f"({info.cn_name}), 范围 [{lo:+0.3f}, {hi:+0.3f}]"
        )
        print(f"      {info.motion_hint}")


class ThumbSliderApp:
    def __init__(
        self,
        root: tk.Tk,
        hand,
        lower: np.ndarray,
        upper: np.ndarray,
        dry_run: bool,
        send_interval_ms: int,
    ) -> None:
        self.root = root
        self.hand = hand
        self.lower = lower
        self.upper = upper
        self.dry_run = dry_run
        self.send_interval_ms = max(10, int(send_interval_ms))
        self.target = np.clip(OPEN_POSE.copy(), lower, upper).astype(np.float64)
        self.slider_vars: dict[int, tk.DoubleVar] = {}
        self.entry_vars: dict[int, tk.StringVar] = {}
        self.status_var = tk.StringVar(value="准备就绪：拇指目标为全 0 自然张开")
        self.auto_send_var = tk.BooleanVar(value=True)
        self._pending_send: Optional[str] = None
        self._syncing_widgets = False
        self._closed = False
        self._disabled = False

        self._build_ui()
        self._set_all_thumb_values(np.zeros(4, dtype=np.float64), schedule=False)

    @property
    def closed(self) -> bool:
        return self._closed

    def _build_ui(self) -> None:
        self.root.title("Wuji 拇指四关节滑块调试")
        self.root.geometry("860x520")
        self.root.minsize(760, 460)

        main = ttk.Frame(self.root, padding=14)
        main.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main.columnconfigure(0, weight=1)

        title = ttk.Label(
            main,
            text="Wuji Thumb Slider - 第 0 行拇指四关节",
            font=("Microsoft YaHei UI", 14, "bold"),
        )
        title.grid(row=0, column=0, sticky="w")

        mapping = ttk.Label(
            main,
            text=(
                "数据对应：target[0,0]=CM pitch，target[0,1]=CM yaw，"
                "target[0,2]=MP pitch，target[0,3]=IP pitch；单位 rad。"
            ),
            foreground="#3a3a3a",
        )
        mapping.grid(row=1, column=0, sticky="w", pady=(4, 10))

        sliders = ttk.LabelFrame(main, text="拇指四关节")
        sliders.grid(row=2, column=0, sticky="nsew")
        sliders.columnconfigure(1, weight=1)

        for row, info in enumerate(THUMB_JOINTS):
            self._build_joint_row(sliders, row, info)

        controls = ttk.Frame(main)
        controls.grid(row=3, column=0, sticky="ew", pady=(12, 8))
        controls.columnconfigure(8, weight=1)

        ttk.Button(controls, text="发送当前", command=self.send_now).grid(
            row=0, column=0, padx=(0, 8)
        )
        ttk.Button(controls, text="回零张开", command=self.reset_to_open).grid(
            row=0, column=1, padx=(0, 8)
        )
        ttk.Button(controls, text="读取实际拇指", command=self.read_actual_thumb).grid(
            row=0, column=2, padx=(0, 8)
        )
        ttk.Checkbutton(
            controls,
            text="自动发送",
            variable=self.auto_send_var,
        ).grid(row=0, column=3, padx=(0, 8))
        ttk.Button(controls, text="失能退出", command=self.close).grid(
            row=0, column=4, padx=(0, 8)
        )

        status = ttk.Label(main, textvariable=self.status_var, anchor="w")
        status.grid(row=4, column=0, sticky="ew", pady=(6, 0))

        table = ttk.LabelFrame(main, text="当前下发目标")
        table.grid(row=5, column=0, sticky="nsew", pady=(10, 0))
        table.columnconfigure(0, weight=1)
        self.target_text = tk.Text(table, height=7, wrap="none", font=("Consolas", 10))
        self.target_text.grid(row=0, column=0, sticky="nsew")
        self.target_text.configure(state="disabled")
        self._refresh_target_text()

    def _build_joint_row(
        self,
        parent: ttk.LabelFrame,
        row: int,
        info: ThumbJointInfo,
    ) -> None:
        lo = float(self.lower[THUMB_ROW, info.column])
        hi = float(self.upper[THUMB_ROW, info.column])
        label_text = (
            f"数据{info.column + 1}  {info.data_name}\n"
            f"{info.joint_name} - {info.cn_name}"
        )
        ttk.Label(parent, text=label_text, width=30).grid(
            row=row, column=0, sticky="w", padx=(10, 8), pady=8
        )

        var = tk.DoubleVar(value=0.0)
        self.slider_vars[info.column] = var
        scale = tk.Scale(
            parent,
            from_=lo,
            to=hi,
            resolution=0.001,
            orient="horizontal",
            showvalue=False,
            variable=var,
            command=lambda raw, col=info.column: self._on_slider(col, raw),
        )
        scale.grid(row=row, column=1, sticky="ew", pady=8)

        value_var = tk.StringVar(value="0.000")
        self.entry_vars[info.column] = value_var
        entry = ttk.Entry(parent, textvariable=value_var, width=9, justify="right")
        entry.grid(row=row, column=2, sticky="e", padx=(8, 4), pady=8)
        entry.bind("<Return>", lambda _evt, item=info: self._apply_entry(item))
        entry.bind("<FocusOut>", lambda _evt, item=info: self._apply_entry(item))

        range_text = f"{lo:+0.3f} .. {hi:+0.3f}"
        ttk.Label(parent, text=range_text, width=18).grid(
            row=row, column=3, sticky="w", padx=(4, 10), pady=8
        )
        ttk.Label(parent, text=info.motion_hint, foreground="#555555").grid(
            row=row, column=4, sticky="w", padx=(0, 10), pady=8
        )

    def _on_slider(self, column: int, raw: str) -> None:
        if self._syncing_widgets:
            return
        self._set_joint_value(column, float(raw), schedule=True)

    def _apply_entry(self, info: ThumbJointInfo) -> None:
        text = self.entry_vars[info.column].get().strip()
        try:
            value = float(text)
        except ValueError:
            self._sync_widget_value(info.column)
            self.status_var.set(f"输入无效：{info.data_name} 需要数字")
            return
        self._set_joint_value(info.column, value, schedule=True)

    def _set_all_thumb_values(self, values: np.ndarray, schedule: bool) -> None:
        for info in THUMB_JOINTS:
            self._set_joint_value(info.column, float(values[info.column]), schedule=False)
        if schedule:
            self._schedule_send()

    def _set_joint_value(self, column: int, value: float, schedule: bool) -> None:
        lo = float(self.lower[THUMB_ROW, column])
        hi = float(self.upper[THUMB_ROW, column])
        clipped = min(max(float(value), lo), hi)
        self.target[THUMB_ROW, column] = clipped
        self._sync_widget_value(column)
        self._refresh_target_text()
        if schedule and self.auto_send_var.get():
            self._schedule_send()

    def _sync_widget_value(self, column: int) -> None:
        value = float(self.target[THUMB_ROW, column])
        self._syncing_widgets = True
        try:
            self.slider_vars[column].set(value)
            self.entry_vars[column].set(f"{value:+0.3f}")
        finally:
            self._syncing_widgets = False

    def _refresh_target_text(self) -> None:
        lines = [
            "5x4 目标矩阵，单位 rad；只改变第 0 行 Thumb：",
            _matrix_text(self.target),
            "",
            f"Thumb 行: {_thumb_row_text(self.target)}",
        ]
        self.target_text.configure(state="normal")
        self.target_text.delete("1.0", "end")
        self.target_text.insert("1.0", "\n".join(lines))
        self.target_text.configure(state="disabled")

    def _schedule_send(self) -> None:
        if self._closed:
            return
        if self._pending_send is not None:
            self.root.after_cancel(self._pending_send)
        self._pending_send = self.root.after(self.send_interval_ms, self.send_now)

    def send_now(self) -> None:
        if self._closed:
            return
        if self._pending_send is not None:
            self.root.after_cancel(self._pending_send)
            self._pending_send = None
        payload = self.target.astype(np.float64)
        if self.dry_run:
            self.status_var.set(f"[dry-run] 当前 Thumb 行：{_thumb_row_text(payload)}")
            return
        try:
            self.hand.write_joint_target_position_unchecked(payload)
            self.status_var.set(f"已发送 Thumb 行：{_thumb_row_text(payload)}")
        except Exception as exc:
            self.status_var.set(f"发送失败：{exc}")

    def reset_to_open(self) -> None:
        self._set_all_thumb_values(np.zeros(4, dtype=np.float64), schedule=False)
        self.send_now()
        self.status_var.set("已回到拇指全 0 自然张开目标")

    def read_actual_thumb(self) -> None:
        if self.dry_run:
            self.status_var.set("[dry-run] 未连接硬件，无法读取实际拇指位置")
            return
        try:
            actual = np.asarray(self.hand.read_joint_actual_position(), dtype=np.float64)
            if actual.shape != (5, 4):
                raise RuntimeError(f"实际位置形状异常: {actual.shape}, 预期 (5, 4)")
            self._set_all_thumb_values(actual[THUMB_ROW].astype(np.float64), schedule=False)
            self.status_var.set(f"已读取实际 Thumb 行：{_thumb_row_text(actual)}")
        except Exception as exc:
            self.status_var.set(f"读取实际位置失败：{exc}")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._pending_send is not None:
            self.root.after_cancel(self._pending_send)
            self._pending_send = None
        self.shutdown_hand()
        self.root.destroy()

    def shutdown_hand(self) -> None:
        if self._disabled or self.dry_run or self.hand is None:
            return
        self._disabled = True
        try:
            time.sleep(0.2)
            _write_enabled_fast(self.hand, False, "[失能] 所有关节已失能")
        except Exception as exc:
            print(f"[警告] 失能失败，请手动检查设备状态: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wuji Hand 拇指四关节滑块调试界面")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打开界面和打印目标，不连接硬件、不下发。",
    )
    parser.add_argument(
        "--live-limits",
        action="store_true",
        help="连接后实时读取关节限位；默认使用 wuji_fist_open.py 中已记录限位。",
    )
    parser.add_argument(
        "--send-interval-ms",
        type=int,
        default=50,
        help="拖动滑块时自动发送的防抖间隔，单位毫秒。默认 50。",
    )
    parser.add_argument(
        "--print-joints",
        action="store_true",
        help="只打印四个拇指数据的对应关系和记录限位后退出。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lower = RECORDED_LOWER_LIMIT.copy()
    upper = RECORDED_UPPER_LIMIT.copy()

    if args.print_joints:
        _print_joint_table(lower, upper)
        return

    hand = None
    app: Optional[ThumbSliderApp] = None
    try:
        if args.dry_run:
            print("[dry-run] 不连接硬件，只打开滑块界面")
        else:
            try:
                import wujihandpy  # noqa: F401
            except ImportError:
                raise SystemExit("错误: 未安装 wujihandpy，请先执行: pip install wujihandpy")

            print("[连接] 正在连接 Wuji Hand...")
            hand = _connect_hand()
            lower, upper = _read_limits(hand, args.live_limits)
            _print_joint_table(lower, upper)
            _write_enabled_fast(hand, True, "[使能] 所有关节使能")
            time.sleep(0.2)

        root = tk.Tk()
        app = ThumbSliderApp(
            root=root,
            hand=hand,
            lower=lower,
            upper=upper,
            dry_run=args.dry_run,
            send_interval_ms=args.send_interval_ms,
        )
        root.protocol("WM_DELETE_WINDOW", app.close)
        root.mainloop()
    finally:
        if app is not None:
            app.shutdown_hand()
        elif hand is not None:
            try:
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
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)
