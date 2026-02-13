#!/usr/bin/env python3
"""测试Pico右手trigger和grip的值"""

import time
from xrobotoolkit_teleop.common.xr_client import XrClient

xr = XrClient()
print("按右手各个键试试... (Ctrl+C退出)")
while True:
    rt = xr.get_key_value_by_name("right_trigger")
    rg = xr.get_key_value_by_name("right_grip")
    print(f"right_trigger={rt:.2f}  right_grip={rg:.2f}")
    time.sleep(0.3)
