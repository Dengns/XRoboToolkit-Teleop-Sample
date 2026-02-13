#!/usr/bin/env python3
"""测试Piper夹爪的位置和力反馈"""

import time
from piper_sdk import C_PiperInterface_V2

can_port = "can0"
piper = C_PiperInterface_V2(can_port)
piper.ConnectPort()
time.sleep(0.1)

piper.MotionCtrl_2(0x01, 0x01, 100, 0x00)
piper.EnableArm(7)
time.sleep(1)

# 使能夹爪
piper.GripperCtrl(0, 1000, 0x01, 0)
time.sleep(0.5)

print("读取夹爪反馈中... 用手捏夹爪试试 (Ctrl+C退出)")
while True:
    gripper = piper.GetArmGripperMsgs()
    angle_mm = gripper.gripper_state.grippers_angle / 1000.0  # → mm
    effort_nm = gripper.gripper_state.grippers_effort / 1000.0  # → N·m
    print(f"开度={angle_mm:.1f}mm  力矩={effort_nm:.3f}N·m")
    time.sleep(0.2)
