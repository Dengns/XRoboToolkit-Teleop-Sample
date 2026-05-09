import time
import math
import asyncio
import inspect
import traceback
import numpy as np

try:
    import wujihandpy
except ImportError:
    print("错误: 未安装 wujihandpy，请执行: pip install wujihandpy")
    exit(1)

HOME_POS = 0.0
    
def fist(hand):
    time.sleep(0.2)

    # 所有手指弯曲（这里用各关节上限的 80% 作为握拳位置）
    upper = hand.read_joint_upper_limit()
    fist = upper * 0.7
    hand.write_joint_target_position(fist.astype(np.float64))
    time.sleep(1.0)

    # 松开
    hand.write_joint_target_position(np.zeros((5, 4), dtype=np.float64))
    time.sleep(0.5)

def OK(hand):
    time.sleep(0.2)

    upper = hand.read_joint_upper_limit()
    targets = np.zeros((5, 4), dtype=np.float64)

    # 拇指和食指捏合
    targets[0] = upper[0] * 0.4   # 拇指弯曲
    targets[1] = upper[1] * 0.7   # 食指弯曲
    # 中指、无名指、小指保持张开 (0.0)

    hand.write_joint_target_position(targets)
    time.sleep(1.0)

    hand.write_joint_target_position(np.zeros((5, 4), dtype=np.float64))
    time.sleep(0.5)

def wave(hand):
    time.sleep(0.2)
    upper = hand.read_joint_upper_limit()
    for fi in range(5):  # 从拇指到小指依次弯曲
        target = upper[fi] * 0.6
        hand.finger(fi).write_joint_target_position(target.astype(np.float64))
        time.sleep(0.3)
    time.sleep(0.5)
    # 全部张开
    hand.write_joint_target_position(np.zeros((5, 4), dtype=np.float64))
    time.sleep(0.5)

def go_home(hand):
    """批量写入零点位置，让整手回到自然张开姿势"""
    home = np.full((5, 4), HOME_POS, dtype=np.float64)
    hand.write_joint_target_position(home)
    time.sleep(0.5)

def main():
    try:
        hand = wujihandpy.Hand()
    except RuntimeError as e:
        fail(f"连接失败: {e}")
        print("\n请检查:")
        print("  1. 灵巧手是否已通过 USB 连接")
        print("  2. 状态指示灯是否为绿色")
        print("  3. Linux 下是否已配置 udev 规则")
        return

    try:
        hand.write_joint_enabled(True)
        # ===== 在这里写你的控制逻辑 =====
        fist(hand)
        time.sleep(1.5)
        go_home(hand)
        OK(hand)
        time.sleep(1.5)
        go_home(hand)
        wave(hand)
        time.sleep(1.5)
        go_home(hand)
    except KeyboardInterrupt:
        print("\n\n  ⏹️  用户中断测试")
    finally:
        hand.write_joint_enabled(False)  # 无论如何都失能


if __name__ == "__main__":
    main()