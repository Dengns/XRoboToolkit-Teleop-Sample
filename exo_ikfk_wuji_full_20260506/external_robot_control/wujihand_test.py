#!/usr/bin/env python3
"""
Wuji Hand SDK 综合功能测试脚本
==============================
基于 wujihandpy SDK，在安全范围内依次测试以下功能：
  1. 设备连接与基本信息读取
  2. 关节限位读取
  3. 单关节/批量关节位置读取
  4. 关节温度与错误码读取
  5. 关节使能与失能
  6. 单关节位置写入（安全小幅运动）
  7. 批量位置写入（逐指测试）
  8. 异步读写测试
  9. Unchecked 读写测试
 10. get 缓存值测试
 11. 实时控制接口测试（平滑正弦运动）

使用前请确保：
  - 已安装 wujihandpy: pip install wujihandpy
  - 灵巧手已通过 USB 连接并状态指示灯为绿色
  - Linux 下已配置 udev 规则

安全说明：
  - 零点 (0.0 rad) 为自然张开姿势，所有复位均回到零点
  - 运动目标 = 从零点出发，向正方向偏移行程的 SAFE_RATIO
  - 脚本结束时必定失能所有关节（通过 finally 保证）
  - 可通过 Ctrl+C 随时中断，中断后自动失能
"""

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

# ─── 全局配置 ───────────────────────────────────────────────
HOME_POS = 0.0            # 零点 = 自然张开姿势
SAFE_RATIO = 0.20         # 安全运动范围占总行程（从零点向正方向）的比例
MOTION_SLEEP = 0.6        # 运动后等待时间 (秒)
REALTIME_DURATION = 3.0   # 实时控制正弦运动持续时间 (秒)
REALTIME_FREQ = 0.5       # 正弦运动频率 (Hz)
REALTIME_HZ = 200         # 实时控制发送频率 (Hz)
TEST_FINGER = 1           # 默认测试手指 (1=食指)
TEST_JOINT = 0            # 默认测试关节 (0=MCP)

# 手指名称映射
FINGER_NAMES = {0: "拇指", 1: "食指", 2: "中指", 3: "无名指", 4: "小指"}
JOINT_NAMES = {0: "J0", 1: "J1", 2: "J2", 3: "J3"}


# ─── 工具函数 ───────────────────────────────────────────────
def section(title: str):
    """打印测试段落标题"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def ok(msg: str):
    print(f"  ✅ {msg}")


def info(msg: str):
    print(f"  ℹ️  {msg}")


def warn(msg: str):
    print(f"  ⚠️  {msg}")


def fail(msg: str):
    print(f"  ❌ {msg}")


def safe_target(lower: float, upper: float, ratio: float = SAFE_RATIO) -> float:
    """
    计算安全目标位置：从零点出发，向正方向偏移 ratio 比例的正向行程。
    结果 clamp 到 [lower, upper]。
    """
    positive_range = upper - HOME_POS
    target = HOME_POS + positive_range * ratio
    return max(lower, min(upper, target))


def go_home(hand, desc="回到零点 (自然张开)"):
    """批量写入零点位置，让整手回到自然张开姿势"""
    info(desc)
    home = np.full((5, 4), HOME_POS, dtype=np.float64)
    hand.write_joint_target_position(home)
    time.sleep(MOTION_SLEEP)


# ═══════════════════════════════════════════════════════════
#  测试 1: 设备连接与基本信息
# ═══════════════════════════════════════════════════════════
def test_device_info(hand: wujihandpy.Hand):
    section("测试 1: 设备连接与基本信息读取")

    try:
        sys_time = hand.read_system_time()
        ok(f"系统运行时间: {sys_time} μs ({sys_time / 1e6:.2f} s)")
    except Exception as e:
        warn(f"读取系统时间失败: {e}")

    try:
        handedness = hand.read_handedness()
        ok(f"手型 (handedness): {handedness}")
    except Exception as e:
        warn(f"读取手型失败: {e}")

    try:
        fw_version = hand.read_firmware_version()
        ok(f"固件版本: {fw_version}")
    except Exception as e:
        warn(f"读取固件版本失败: {e}")


# ═══════════════════════════════════════════════════════════
#  测试 2: 关节限位读取
# ═══════════════════════════════════════════════════════════
def test_joint_limits(hand: wujihandpy.Hand):
    section("测试 2: 关节限位读取")

    try:
        upper_limits = hand.read_joint_upper_limit()
        lower_limits = hand.read_joint_lower_limit()
        ok("成功读取所有关节限位 (5×4 矩阵, 单位: rad)")
        print()
        for i in range(5):
            for j in range(4):
                lo = lower_limits[i][j]
                hi = upper_limits[i][j]
                rng = hi - lo
                print(f"    {FINGER_NAMES[i]} {JOINT_NAMES[j]}: "
                      f"[{lo:+.3f}, {hi:+.3f}]  行程={math.degrees(rng):.1f}°")
        return lower_limits, upper_limits
    except Exception as e:
        fail(f"读取关节限位失败: {e}")
        return None, None


# ═══════════════════════════════════════════════════════════
#  测试 3: 关节位置读取（单关节 + 批量）
# ═══════════════════════════════════════════════════════════
def test_read_positions(hand: wujihandpy.Hand):
    section("测试 3: 关节位置读取")

    # 单关节读取
    try:
        pos = hand.finger(TEST_FINGER).joint(TEST_JOINT).read_joint_actual_position()
        ok(f"单关节读取 - {FINGER_NAMES[TEST_FINGER]} {JOINT_NAMES[TEST_JOINT]}: "
           f"{pos:.4f} rad ({math.degrees(pos):.2f}°)")
    except Exception as e:
        fail(f"单关节读取失败: {e}")

    # 批量读取
    try:
        positions = hand.read_joint_actual_position()
        ok(f"批量读取 - 所有 {positions.size} 个关节位置:")
        for i in range(5):
            vals = [f"{positions[i][j]:+.3f}" for j in range(4)]
            print(f"    {FINGER_NAMES[i]}: [{', '.join(vals)}] rad")
    except Exception as e:
        fail(f"批量位置读取失败: {e}")


# ═══════════════════════════════════════════════════════════
#  测试 4: 温度与错误码读取
# ═══════════════════════════════════════════════════════════
def test_read_status(hand: wujihandpy.Hand):
    section("测试 4: 关节温度与错误码")

    try:
        temps = hand.read_joint_temperature()
        ok("关节温度 (5×4):")
        for i in range(5):
            vals = [f"{temps[i][j]:.1f}" for j in range(4)]
            print(f"    {FINGER_NAMES[i]}: [{', '.join(vals)}] °C")
    except Exception as e:
        warn(f"读取温度失败 (部分固件版本可能不支持): {e}")

    try:
        errors = hand.read_joint_error_code()
        has_error = np.any(errors != 0)
        if has_error:
            warn(f"存在关节错误码:\n{errors}")
        else:
            ok("所有关节错误码均为 0 (正常)")
    except Exception as e:
        warn(f"读取错误码失败: {e}")


# ═══════════════════════════════════════════════════════════
#  测试 5: 关节使能与失能
# ═══════════════════════════════════════════════════════════
def test_enable_disable(hand: wujihandpy.Hand):
    section("测试 5: 关节使能/失能")

    info("使能所有关节...")
    hand.write_joint_enabled(True)
    time.sleep(0.3)

    try:
        enabled = hand.read_joint_enabled()
        if np.all(enabled):
            ok("所有关节已使能")
        else:
            warn(f"部分关节未成功使能: {enabled}")
    except Exception as e:
        warn(f"读取使能状态失败: {e}")

    info("失能所有关节...")
    hand.write_joint_enabled(False)
    time.sleep(0.3)

    try:
        enabled = hand.read_joint_enabled()
        if not np.any(enabled):
            ok("所有关节已失能")
        else:
            warn(f"部分关节未成功失能: {enabled}")
    except Exception as e:
        warn(f"读取使能状态失败: {e}")


# ═══════════════════════════════════════════════════════════
#  测试 6: 单关节位置写入（安全小幅运动）
# ═══════════════════════════════════════════════════════════
def test_single_joint_write(hand: wujihandpy.Hand, lower_limits, upper_limits):
    section("测试 6: 单关节位置写入")

    fi, ji = TEST_FINGER, TEST_JOINT
    lo = lower_limits[fi][ji]
    hi = upper_limits[fi][ji]
    target = safe_target(lo, hi, SAFE_RATIO)

    info(f"目标: {FINGER_NAMES[fi]} {JOINT_NAMES[ji]}")
    info(f"限位: [{lo:+.3f}, {hi:+.3f}] rad")
    info(f"零点: {HOME_POS:.3f} rad, 运动目标: {target:+.3f} rad")

    hand.write_joint_enabled(True)
    time.sleep(0.2)

    # 先回零点（自然张开）
    info("回到零点 (自然张开)...")
    hand.finger(fi).joint(ji).write_joint_target_position(HOME_POS)
    time.sleep(MOTION_SLEEP)
    pos_home = hand.finger(fi).joint(ji).read_joint_actual_position()
    ok(f"零点实际位置: {pos_home:+.4f} rad (误差: {abs(pos_home - HOME_POS):.4f})")

    # 移动到目标（小幅弯曲）
    info(f"移动到目标 {target:+.3f} rad (小幅弯曲)...")
    hand.finger(fi).joint(ji).write_joint_target_position(target)
    time.sleep(MOTION_SLEEP)
    pos_target = hand.finger(fi).joint(ji).read_joint_actual_position()
    ok(f"目标实际位置: {pos_target:+.4f} rad (误差: {abs(pos_target - target):.4f})")

    # 回零点
    info("回到零点...")
    hand.finger(fi).joint(ji).write_joint_target_position(HOME_POS)
    time.sleep(MOTION_SLEEP)

    hand.write_joint_enabled(False)
    time.sleep(0.2)
    ok("单关节写入测试完成")


# ═══════════════════════════════════════════════════════════
#  测试 7: 批量位置写入（逐指测试）
# ═══════════════════════════════════════════════════════════
def test_bulk_write(hand: wujihandpy.Hand, lower_limits, upper_limits):
    section("测试 7: 批量位置写入 (逐指)")

    hand.write_joint_enabled(True)
    time.sleep(0.2)

    # 先回零点
    go_home(hand, "所有关节回到零点 (自然张开)...")

    # 逐指做小幅弯曲
    for fi in range(5):
        targets = np.array(
            [safe_target(lower_limits[fi][j], upper_limits[fi][j], SAFE_RATIO)
             for j in range(4)],
            dtype=np.float64,
        )
        info(f"{FINGER_NAMES[fi]} 弯曲...")
        hand.finger(fi).write_joint_target_position(targets)
        time.sleep(MOTION_SLEEP)

        # 读取实际位置
        actual = hand.finger(fi).read_joint_actual_position()
        errors = np.abs(actual - targets)
        ok(f"{FINGER_NAMES[fi]} 最大跟踪误差: {errors.max():.4f} rad")

        # 回零点
        hand.finger(fi).write_joint_target_position(
            np.full(4, HOME_POS, dtype=np.float64)
        )
        time.sleep(MOTION_SLEEP * 0.5)

    hand.write_joint_enabled(False)
    time.sleep(0.2)
    ok("批量写入测试完成")


# ═══════════════════════════════════════════════════════════
#  测试 8: 异步读写
# ═══════════════════════════════════════════════════════════
async def _test_async_rw(hand: wujihandpy.Hand, lower_limits, upper_limits):
    section("测试 8: 异步读写 (async)")

    # 异步读取
    try:
        positions = await hand.read_joint_actual_position_async()
        ok(f"异步批量读取成功, shape={positions.shape}")
    except Exception as e:
        fail(f"异步读取失败: {e}")
        return

    # 异步写入 - 写到零点
    fi, ji = TEST_FINGER, TEST_JOINT

    hand.write_joint_enabled(True)
    await asyncio.sleep(0.2)

    try:
        await hand.finger(fi).joint(ji).write_joint_target_position_async(HOME_POS)
        await asyncio.sleep(MOTION_SLEEP)
        pos = await hand.finger(fi).joint(ji).read_joint_actual_position_async()
        ok(f"异步写入+读取: 目标={HOME_POS:.3f}, 实际={pos:+.4f}")
    except Exception as e:
        fail(f"异步写入失败: {e}")

    hand.write_joint_enabled(False)
    await asyncio.sleep(0.2)
    ok("异步读写测试完成")


def test_async_rw(hand, lower_limits, upper_limits):
    asyncio.run(_test_async_rw(hand, lower_limits, upper_limits))


# ═══════════════════════════════════════════════════════════
#  测试 9: Unchecked 读写
# ═══════════════════════════════════════════════════════════
def test_unchecked_rw(hand: wujihandpy.Hand, lower_limits, upper_limits):
    section("测试 9: Unchecked 读写")

    fi, ji = TEST_FINGER, TEST_JOINT

    hand.write_joint_enabled(True)
    time.sleep(0.2)

    # unchecked 读（返回 None，不阻塞）
    try:
        result = hand.finger(fi).joint(ji).read_joint_actual_position_unchecked()
        ok(f"unchecked 读取返回: {result} (预期 None，表示请求已发送)")
    except Exception as e:
        fail(f"unchecked 读取失败: {e}")

    # unchecked 写到零点（不阻塞）
    try:
        hand.finger(fi).joint(ji).write_joint_target_position_unchecked(HOME_POS)
        ok(f"unchecked 写入已发送 (目标={HOME_POS:.3f})")
        time.sleep(MOTION_SLEEP)
    except Exception as e:
        fail(f"unchecked 写入失败: {e}")

    hand.write_joint_enabled(False)
    time.sleep(0.2)
    ok("Unchecked 读写测试完成")


# ═══════════════════════════════════════════════════════════
#  测试 10: get 缓存值
# ═══════════════════════════════════════════════════════════
def test_get_cached(hand: wujihandpy.Hand):
    section("测试 10: get 缓存值读取")

    # 先做一次正常读取填充缓存
    try:
        _ = hand.read_joint_actual_position()
    except Exception:
        pass

    try:
        cached = hand.get_joint_actual_position()
        ok(f"缓存读取成功, shape={cached.shape}")
        info("(get 返回的是上一次 read 的缓存值，不发起新的通信)")
    except Exception as e:
        warn(f"缓存读取失败: {e}")

    try:
        cached_single = hand.finger(TEST_FINGER).joint(TEST_JOINT).get_joint_actual_position()
        ok(f"单关节缓存读取: {cached_single:+.4f} rad")
    except Exception as e:
        warn(f"单关节缓存读取失败: {e}")

    ok("缓存值测试完成")


# ═══════════════════════════════════════════════════════════
#  辅助: 自动发现 IFilter 实现类
# ═══════════════════════════════════════════════════════════
def discover_filter():
    """
    运行时自省 wujihandpy 模块结构，自动发现可用的 IFilter 子类并实例化。
    返回 (filter_instance, filter_name) 或 (None, None)。
    """
    import importlib

    filter_mod = None

    # 尝试多种可能的模块路径
    for mod_path in [
        "wujihandpy._core.filter",
        "wujihandpy.filter",
        "wujihandpy._core",
    ]:
        try:
            filter_mod = importlib.import_module(mod_path)
            break
        except ImportError:
            continue

    if filter_mod is None:
        # 从 wujihandpy 或 _core 的属性中找 filter
        for base in [wujihandpy, getattr(wujihandpy, '_core', None)]:
            if base is None:
                continue
            if hasattr(base, 'filter'):
                filter_mod = base.filter
                break

    if filter_mod is None:
        return None, None

    info(f"发现 filter 模块: {filter_mod}")
    public_names = [x for x in dir(filter_mod) if not x.startswith('_')]
    info(f"模块内容: {public_names}")

    # 查找 IFilter 基类
    ifilter_cls = getattr(filter_mod, 'IFilter', None)

    # 收集所有可能的 filter 实现类
    candidates = []
    for name in dir(filter_mod):
        obj = getattr(filter_mod, name)
        if not inspect.isclass(obj):
            continue
        if name == 'IFilter':
            continue
        # 是 IFilter 的子类，或名字含 Filter
        if (ifilter_cls and issubclass(obj, ifilter_cls)) or 'Filter' in name:
            candidates.append((name, obj))

    info(f"发现 filter 候选类: {[c[0] for c in candidates]}")

    # 尝试实例化
    for name, cls in candidates:
        # 先尝试无参数
        try:
            instance = cls()
            ok(f"成功创建 filter: {name}()")
            return instance, name
        except TypeError:
            pass
        # 再尝试常见参数（平滑系数、截止频率等）
        for args in [(0.5,), (0.1,), (0.8,), (200,), (100, 0.5)]:
            try:
                instance = cls(*args)
                ok(f"成功创建 filter: {name}{args}")
                return instance, name
            except (TypeError, Exception):
                continue

    return None, None


# ═══════════════════════════════════════════════════════════
#  测试 11: 实时控制接口（正弦运动）
# ═══════════════════════════════════════════════════════════
def test_realtime_control(hand: wujihandpy.Hand, lower_limits, upper_limits):
    section("测试 11: 实时控制接口 (正弦运动)")

    fi, ji = TEST_FINGER, TEST_JOINT
    lo = lower_limits[fi][ji]
    hi = upper_limits[fi][ji]
    # 正弦运动围绕零点，振幅为正向行程的一小部分
    amplitude = (hi - HOME_POS) * SAFE_RATIO * 0.5

    info(f"目标: {FINGER_NAMES[fi]} {JOINT_NAMES[ji]}")
    info(f"中心: {HOME_POS:.3f} rad (零点), 振幅: {amplitude:.3f} rad")
    info(f"频率: {REALTIME_FREQ} Hz, 持续: {REALTIME_DURATION} s")

    hand.write_joint_enabled(True)
    time.sleep(0.2)

    # 先回零点
    hand.finger(fi).joint(ji).write_joint_target_position(HOME_POS)
    time.sleep(MOTION_SLEEP)

    controller = None
    try:
        # ── 自动发现 filter ──
        info("正在自省 filter 模块...")
        filter_instance, filter_name = discover_filter()

        if filter_instance is None:
            warn("未能自动发现 IFilter 实现")
            # 打印签名帮助调试
            try:
                doc = hand.realtime_controller.__doc__
                if doc:
                    info(f"realtime_controller 文档:\n{doc}")
            except Exception:
                pass
            warn("回退到 unchecked 高频写入模拟实时控制")
            _realtime_fallback(hand, fi, ji, amplitude)
            return

        # ── 创建 realtime_controller ──
        # 签名: hand.realtime_controller(enable_upstream: bool, filter: IFilter)
        #   enable_upstream=True  → 正常 SDO 通道保持可用
        #   filter                → 平滑滤波器
                # ── 创建 realtime_controller ──
        info(f"创建 realtime_controller(enable_upstream=True, filter={filter_name})")
        controller = hand.realtime_controller(True, filter_instance)
        info("realtime_controller 已创建，开始正弦运动...")

        # 自动判断 IController 是否具备写入方法
        # 如果 controller 对象不具备写入接口，说明它只维持后台实时线程，仍需使用 hand 写入
        if hasattr(controller, 'write_joint_target_position') or hasattr(controller, 'finger'):
            _run_sine_loop(controller, fi, ji, amplitude)
        else:
            info("检测到 controller 仅为句柄，使用 hand 对象发送高频数据...")
            _run_sine_loop(hand, fi, ji, amplitude)



    except TypeError as e:
        warn(f"realtime_controller 参数不匹配: {e}")
        info("请查阅最新 API 文档确认 IFilter 的正确构造方式")
        info("回退到 unchecked 高频写入...")
        _realtime_fallback(hand, fi, ji, amplitude)
    except Exception as e:
        warn(f"实时控制测试异常: {e}")
        traceback.print_exc()
    finally:
        # 释放 controller
        if controller is not None:
            for method in ['close', 'stop', 'destroy']:
                if hasattr(controller, method):
                    try:
                        getattr(controller, method)()
                        break
                    except Exception:
                        continue
            # 如果支持 del
            controller = None

        # 回零点并失能
        try:
            hand.finger(fi).joint(ji).write_joint_target_position(HOME_POS)
            time.sleep(MOTION_SLEEP)
        except Exception:
            pass
        hand.write_joint_enabled(False)
        time.sleep(0.2)
        ok("实时控制测试完成")


def _run_sine_loop(writer, fi, ji, amplitude):
    """
    用给定的 writer (IController 或 Hand) 执行正弦运动循环。
    """
    dt = 1.0 / REALTIME_HZ
    t_start = time.monotonic()
    count = 0
    error_printed = False

    while True:
        elapsed = time.monotonic() - t_start
        if elapsed >= REALTIME_DURATION:
            break

        target = HOME_POS + amplitude * math.sin(
            2 * math.pi * REALTIME_FREQ * elapsed
        )

        success = False

        # 1. 尝试直接单关节写入
        try:
            writer.finger(fi).joint(ji).write_joint_target_position(target)
            success = True
        except Exception as e1:
            # 2. 尝试整手数组 float64
            try:
                pos_array_64 = np.full((5, 4), HOME_POS, dtype=np.float64)
                pos_array_64[fi][ji] = target
                writer.write_joint_target_position(pos_array_64)
                success = True
            except Exception as e2:
                # 3. 尝试整手数组 float32 
                try:
                    pos_array_32 = np.full((5, 4), HOME_POS, dtype=np.float32)
                    pos_array_32[fi][ji] = target
                    writer.write_joint_target_position(pos_array_32)
                    success = True
                except Exception as e3:
                    if not error_printed:
                        warn("实时控制写入失败，捕获到以下异常：")
                        print(f"    -> 单关节尝试失败: {e1}")
                        print(f"    -> 整手(float64)尝试失败: {e2}")
                        print(f"    -> 整手(float32)尝试失败: {e3}")
                        error_printed = True
                    break

        if not success:
            break

        count += 1
        t_next = t_start + count * dt
        sleep_time = t_next - time.monotonic()
        if sleep_time > 0:
            time.sleep(sleep_time)

    if count > 0:
        ok(f"正弦运动完成, 共发送 {count} 帧 "
           f"(实际频率 ~{count / REALTIME_DURATION:.0f} Hz)")
    else:
        fail("正弦运动发送了 0 帧，因为写入异常提前退出。请查看上方的报错信息。")



def _realtime_fallback(hand, fi, ji, amplitude):
    """
    当 realtime_controller 不可用时，用 unchecked 高频写入模拟实时控制。
    unchecked 不阻塞，适合延迟敏感场景。
    """
    info("使用 write_joint_target_position_unchecked 高频写入替代...")
    dt = 1.0 / REALTIME_HZ
    t_start = time.monotonic()
    count = 0

    while True:
        elapsed = time.monotonic() - t_start
        if elapsed >= REALTIME_DURATION:
            break

        target = HOME_POS + amplitude * math.sin(
            2 * math.pi * REALTIME_FREQ * elapsed
        )
        hand.finger(fi).joint(ji).write_joint_target_position_unchecked(target)

        count += 1
        t_next = t_start + count * dt
        sleep_time = t_next - time.monotonic()
        if sleep_time > 0:
            time.sleep(sleep_time)

    ok(f"fallback 正弦运动完成, 共发送 {count} 帧 "
       f"(实际频率 ~{count / REALTIME_DURATION:.0f} Hz)")


# ═══════════════════════════════════════════════════════════
#  主入口
# ═══════════════════════════════════════════════════════════
def main():
    print(r"""
    ╔══════════════════════════════════════════════════╗
    ║     Wuji Hand SDK 综合功能测试脚本              ║
    ║     零点复位 · 小幅运动 · 自动失能              ║
    ╚══════════════════════════════════════════════════╝
    """)

    # ── 连接设备 ──
    info("正在连接灵巧手...")
    try:
        hand = wujihandpy.Hand()
        ok("设备连接成功!")
    except RuntimeError as e:
        fail(f"连接失败: {e}")
        print("\n请检查:")
        print("  1. 灵巧手是否已通过 USB 连接")
        print("  2. 状态指示灯是否为绿色")
        print("  3. Linux 下是否已配置 udev 规则")
        return

    try:
        # ── 只读测试（不需要使能）──
        test_device_info(hand)
        lower_limits, upper_limits = test_joint_limits(hand)
        test_read_positions(hand)
        test_read_status(hand)

        if lower_limits is None or upper_limits is None:
            fail("无法获取关节限位，跳过运动类测试")
            return

        # ── 使能/失能测试 ──
        test_enable_disable(hand)

        # ── 运动类测试 ──
        print(f"\n{'─'*60}")
        print("  即将开始运动类测试，灵巧手将产生小幅运动")
        print("  零点 = 自然张开 🖐️，运动 = 小幅弯曲后回到张开")
        print("  按 Enter 继续，或 Ctrl+C 中断...")
        print(f"{'─'*60}")
        try:
            input()
        except EOFError:
            pass

        test_single_joint_write(hand, lower_limits, upper_limits)
        test_bulk_write(hand, lower_limits, upper_limits)
        test_async_rw(hand, lower_limits, upper_limits)
        test_unchecked_rw(hand, lower_limits, upper_limits)
        test_get_cached(hand)
        test_realtime_control(hand, lower_limits, upper_limits)

        # ── 完成 ──
        section("全部测试完成!")
        ok("所有功能测试已执行，请查看上方输出确认结果")

    except KeyboardInterrupt:
        print("\n\n  ⏹️  用户中断测试")
    except Exception as e:
        fail(f"未预期的异常: {e}")
        traceback.print_exc()
    finally:
        # 确保最终失能
        try:
            hand.write_joint_enabled(False)
            info("已安全失能所有关节")
        except Exception:
            warn("失能操作失败，请手动检查设备状态")


if __name__ == "__main__":
    main()
