#!/usr/bin/env python3
"""Test ZhiXing 90D gripper with correct 32-bit position encoding.

Register map (from modbus协议.pdf):
  256 (0x0100): Motor enable (1=enable)
  257 (0x0101): Position command type (0=absolute, 1=relative, 2=speed, 3=torque, 4=force)
  258 (0x0102): Motor position HIGH word  }  32-bit signed
  259 (0x0103): Motor position LOW word   }
  260 (0x0104): Speed (0~100%)
  261 (0x0105): Torque (0~100%)
  264 (0x0108): Motor action (0=none, 1=trigger motion, 3=decel stop)
  1044-1045:   Real-time position feedback (high/low)
"""

import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from xrobotoolkit_teleop.hardware.interface.rm75b import (
    RM75BInterface,
    rm_peripheral_read_write_params_t,
)

IP = "192.168.5.46"
PORT = 8080
DEV = 1  # Modbus slave address

arm = RM75BInterface(ip=IP, port=PORT, enable_gripper=True)
time.sleep(1)


def write_reg(addr, value):
    """Write single 16-bit register."""
    params = rm_peripheral_read_write_params_t(DEV, addr, 1)
    return arm.arm.rm_write_single_register(params, value)


def read_reg(addr):
    """Read single 16-bit register."""
    params = rm_peripheral_read_write_params_t(DEV, addr, 1)
    ret, val = arm.arm.rm_read_holding_registers(params)
    return ret, val


def set_position_and_go(pos, speed=100):
    """Set 32-bit position via two separate single-register writes, then trigger."""
    high_word = (pos >> 16) & 0xFFFF
    low_word = pos & 0xFFFF
    write_reg(258, high_word)   # position HIGH
    write_reg(259, low_word)    # position LOW
    write_reg(260, speed)       # speed %
    write_reg(264, 1)           # trigger motion


def read_position():
    """Read current 32-bit position feedback from reg 1044-1045."""
    _, hi = read_reg(1044)
    _, lo = read_reg(1045)
    pos = (hi << 16) | (lo & 0xFFFF)
    # Handle signed 32-bit
    if pos >= 0x80000000:
        pos -= 0x100000000
    return pos


# --- Step 1: Read current position ---
print(f"\n=== Current position feedback: {read_position()} ===\n")

# --- Step 2: Go to 0 (closed), read position ---
print("Going to position 0 (close)...")
set_position_and_go(0, speed=50)
time.sleep(2)
pos_closed = read_position()
print(f"  Closed position feedback: {pos_closed}")

# --- Step 3: Go to 1000 (open), read position ---
print("Going to position 1000 (open)...")
set_position_and_go(1000, speed=50)
time.sleep(2)
pos_open = read_position()
print(f"  Open position feedback: {pos_open}")

print(f"\n=== Range: closed={pos_closed}, open={pos_open} ===\n")

# --- Step 4: Test intermediate positions ---
test_positions = [0, 200, 400, 600, 800, 1000]
print("=== Testing intermediate positions ===")
for pos in test_positions:
    set_position_and_go(pos, speed=50)
    time.sleep(1.5)
    fb = read_position()
    input(f"  cmd={pos:4d}  feedback={fb:6d}  →  观察开度，Enter 继续...")

print("\n=== Done ===")
arm.close()
