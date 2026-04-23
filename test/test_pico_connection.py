#!/usr/bin/env python3
"""简单的Pico连接测试脚本"""

import time
from xrobotoolkit_teleop.common.xr_client import XrClient


def main():
    print("正在连接Pico...")
    try:
        xr_client = XrClient()
        print("✓ 连接成功！\n")

        # 测试读取数据
        print("读取Pico数据 (按Ctrl+C退出)...")
        while True:
            # 读取右控制器位姿
            right_pose = xr_client.get_pose_by_name("right_controller")
            print(f"右控制器: 位置={right_pose[:3]}, 四元数={right_pose[3:]}")

            # 读取握力
            right_grip = xr_client.get_key_value_by_name("right_grip")
            print(f"右握力: {right_grip:.2f}")

            # 读取按键
            button_a = xr_client.get_button_state_by_name("A")
            print(f"A按键: {button_a}")

            # 读取体感传感器
            data = xr_client.get_motion_tracker_data()
            for serial, metrics in data.items():
                print(f"设备序列号: {serial}")
                print(f"  - 位姿 (Pose): {metrics['pose']}")
                print(f"  - 速度 (Velocity): {metrics['velocity']}")
                print(f"  - 加速度 (Acceleration): {metrics['acceleration']}")
                print("-" * 20)
                
            print("---")
            time.sleep(0.5)  # 每2秒读一次

    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"✗ 连接失败: {e}")
        print("\n可能的原因:")
        print("1. Pico设备未连接")
        print("2. XRoboToolkit SDK未正确安装")
        print("3. SDK版本不匹配")
    finally:
        xr_client.close()
        print("已断开连接")


if __name__ == "__main__":
    main()
