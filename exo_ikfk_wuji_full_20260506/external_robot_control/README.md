# 舞肌灵巧手 (Wuji Hand) SDK 快速上手指南

> 基于 `wujihandpy` Python SDK，从零开始控制 20 自由度仿生灵巧手。

---

## 目录

- [1. 硬件概览](#1-硬件概览)
- [2. 环境准备](#2-环境准备)
- [3. 核心概念](#3-核心概念)
- [4. 30 秒快速入门](#4-30-秒快速入门)
- [5. 读取数据](#5-读取数据)
- [6. 写入控制](#6-写入控制)
- [7. 实时控制 (Realtime Control)](#7-实时控制-realtime-control)
- [8. 异步与 Unchecked 操作](#8-异步与-unchecked-操作)
- [9. 预设动作示例](#9-预设动作示例)
- [10. 安全须知与常见问题](#10-安全须知与常见问题)
- [11. API 速查表](#11-api-速查表)

---

## 1. 硬件概览

Wuji Hand 是一款 **20 自由度** 全驱仿生灵巧手：

- **5 根手指**，每根 **4 个独立关节**（MCP / PIP / DIP / 侧摆）
- 关节角度单位为 **弧度 (rad)**，零点 `0.0` 对应手指自然张开 
- 正方向为手指弯曲（握拳方向），负方向为反向伸展
- 超出限位的指令会被 SDK 自动 clamp 到上下限，不会损坏硬件
- 通信接口：**USB Type-C**，状态指示灯 🟢 绿色 = 可通信



## 2. 环境准备

### 2.1 安装 SDK

```bash
pip install wujihandpy
```

> 要求 Linux (glibc 2.28+)，Python 3.8–3.14。Windows 暂不支持。

### 2.2 配置 USB 权限 (Linux)

首次使用需配置 udev 规则，允许非 root 用户访问 USB 设备：

```bash
echo 'SUBSYSTEM=="usb", ATTR{idVendor}=="0483", MODE="0666"' | \
  sudo tee /etc/udev/rules.d/95-wujihand.rules && \
  sudo udevadm control --reload-rules && sudo udevadm trigger
```

> 如果在 Docker 容器内开发，此命令需在**宿主机**上执行。

### 2.3 验证连接

将灵巧手通过 USB 连接电脑，等待状态指示灯变为 🟢 绿色，然后：

```python
import wujihandpy
hand = wujihandpy.Hand()  # 成功即表示连接正常
```

若报 `RuntimeError: Failed to init`，请检查 USB 连接和 udev 规则。

---

## 3. 核心概念

### 3.1 三层设备模型

```
Hand (整手)
 ├── Finger (手指, 0-4)
 │    └── Joint (关节, 0-3)
```

- **`Hand`**：通过 `wujihandpy.Hand()` 创建，代表整个灵巧手
- **`Finger`**：通过 `hand.finger(i)` 获取（`i` = 0~4），轻量引用，不持有独立状态
- **`Joint`**：通过 `finger.joint(j)` 获取（`j` = 0~3），同样是轻量引用

### 3.2 数据层级

SDK 的可读写数据分为三层：

| 层级 | 说明 | 示例字段 |
|------|------|----------|
| Hand 层 | 整手唯一的数据 | `handedness`, `firmware_version`, `system_time` |
| Finger 层 | 每根手指独立的数据 | （当前版本无数据） |
| Joint 层 | 每个关节独立的数据 | `joint_enabled`, `joint_actual_position`, `joint_target_position`, `joint_temperature`, `joint_error_code` |

> Joint 层的数据名统一以 `joint_` 前缀命名。

### 3.3 关节角度约定

- 单位：**弧度 (rad)**，类型 `np.float64`
- 零点和正方向遵循 [URDF 文件](https://github.com/wuji-technology/wuji-hand-description) 定义
- **`0.0` rad = 手指自然张开**（平伸状态）
- **正值 = 手指弯曲**（向掌心方向）
- 每个关节的限位范围可通过 `read_joint_upper_limit()` / `read_joint_lower_limit()` 读取

### 3.4 使能 (Enable) 与失能 (Disable)

灵巧手上电后关节默认处于 **失能** 状态（电机不通电），必须先 **使能** 才能控制运动：

```python
hand.write_joint_enabled(True)   # 使能所有关节（电机通电，进入位置控制）
# ... 执行运动 ...
hand.write_joint_enabled(False)  # 失能所有关节（电机断电，手指可自由活动）
```

>  **操作完毕务必失能**，避免电机长时间带载发热。

---

## 4. 30 秒快速入门

```python
import time
import wujihandpy

hand = wujihandpy.Hand()

try:
    # 1. 使能所有关节
    hand.write_joint_enabled(True)

    # 2. 食指 MCP 关节弯曲到 1.0 rad（约 57°）
    hand.finger(1).joint(0).write_joint_target_position(1.0)
    time.sleep(0.5)

    # 3. 回到零点（自然张开）
    hand.finger(1).joint(0).write_joint_target_position(0.0)
    time.sleep(0.5)

finally:
    # 4. 无论如何都失能（放在 finally 中保证执行）
    hand.write_joint_enabled(False)
```

这段代码会让食指做一个 "弯曲→伸直" 的动作。

---

## 5. 读取数据

### 5.1 读取基本信息

```python
# 系统运行时间（微秒）
uptime_us = hand.read_system_time()

# 手型（左手/右手）
handedness = hand.read_handedness()

# 固件版本
version = hand.read_firmware_version()
```

### 5.2 读取单个关节位置

```python
# 读取食指 MCP 关节的当前位置
pos = hand.finger(1).joint(0).read_joint_actual_position()
print(f"食指 MCP: {pos:.4f} rad")
```

### 5.3 批量读取所有关节位置

```python
import numpy as np

# 返回 5×4 的 ndarray，行=手指，列=关节
positions = hand.read_joint_actual_position()
print(positions)
# [[ 0.002  0.001  0.003 -0.001]   ← 拇指
#  [ 0.005  0.002  0.001 -0.002]   ← 食指
#  [ ...                        ]   ← 中指
#  [ ...                        ]   ← 无名指
#  [ ...                        ]]  ← 小指
```

### 5.4 读取关节限位

```python
upper = hand.read_joint_upper_limit()  # 5×4, 各关节上限 (rad)
lower = hand.read_joint_lower_limit()  # 5×4, 各关节下限 (rad)

# 单个关节的限位
hi = hand.finger(1).joint(0).read_joint_upper_limit()
lo = hand.finger(1).joint(0).read_joint_lower_limit()
print(f"食指 MCP 限位: [{lo:.3f}, {hi:.3f}] rad")
```

### 5.5 读取温度与错误码

```python
temps  = hand.read_joint_temperature()   # 5×4, 单位 ℃
errors = hand.read_joint_error_code()    # 5×4, 0=正常
```

---

## 6. 写入控制

> 所有写入操作（`write_*`）均为**阻塞式**，返回时保证指令已成功下发。
> 写入的角度若超出限位会被自动 clamp，不会报错。

### 6.1 单关节控制

```python
# 食指 MCP 弯曲到 0.8 rad
hand.finger(1).joint(0).write_joint_target_position(0.8)
```

### 6.2 单指批量控制

```python
import numpy as np

# 食指 4 个关节同时运动
hand.finger(1).write_joint_target_position(
    np.array([0.8, 0.5, 0.3, 0.0], dtype=np.float64)
    #         MCP   PIP   DIP  侧摆
)
```

也可以给同一根手指的所有关节写入相同值：

```python
hand.finger(1).write_joint_target_position(0.5)  # 4 个关节都去 0.5 rad
```

### 6.3 全手批量控制

```python
# 5×4 数组，一次性控制全部 20 个关节
targets = np.zeros((5, 4), dtype=np.float64)
targets[1, 0] = 0.8  # 食指 MCP
targets[2, 0] = 0.6  # 中指 MCP
hand.write_joint_target_position(targets)
```

### 6.4 回到零点（自然张开）

```python
hand.write_joint_target_position(np.zeros((5, 4), dtype=np.float64))
```

---

## 7. 实时控制 (Realtime Control)

普通的 `read` / `write` 通过缓冲池传输，**最高约 100 Hz**。如果需要更高频率的平滑位置控制（如遥操作、轨迹跟踪），需要使用 **realtime_controller**。

### 7.1 创建实时控制器

```python
from wujihandpy._core.filter import IFilter  # 导入滤波器基类

# 自动发现可用的滤波器（运行时自省）
import wujihandpy._core.filter as fmod
print(dir(fmod))  # 查看可用的 Filter 类名

# 创建 realtime_controller
# 参数 1: enable_upstream (bool) — True 保持 SDO 通道可用（推荐）
# 参数 2: filter (IFilter)       — 平滑滤波器实例
controller = hand.realtime_controller(True, filter_instance)
```

> `realtime_controller` 返回一个 `IController` 对象。`IController` 自身具备 `write_joint_target_position` 方法，无需通过 `hand` 对象发送数据。

### 7.2 实时正弦运动示例

```python
import time, math
import numpy as np

hand.write_joint_enabled(True)

# 创建滤波器和控制器（以实际可用的 Filter 类为准）
controller = hand.realtime_controller(True, filter_instance)

hz = 200          # 控制频率
duration = 3.0    # 持续时间
freq = 0.5        # 正弦频率
amplitude = 0.3   # 振幅 (rad)

dt = 1.0 / hz
t_start = time.monotonic()
count = 0

try:
    while True:
        elapsed = time.monotonic() - t_start
        if elapsed >= duration:
            break

        target = amplitude * math.sin(2 * math.pi * freq * elapsed)

        hand.finger(1).joint(0).write_joint_target_position(target)

        count += 1
        t_next = t_start + count * dt
        sleep_time = t_next - time.monotonic()
        if sleep_time > 0:
            time.sleep(sleep_time)

    print(f"完成: {count} 帧, ~{count/duration:.0f} Hz")
finally:
    hand.finger(1).joint(0).write_joint_target_position(0.0)
    time.sleep(0.5)
    hand.write_joint_enabled(False)
```

---

## 8. 异步与 Unchecked 操作

### 8.1 异步 (Async)

所有读写函数都有 `_async` 后缀版本，适用于异步事件循环：

```python
import asyncio

async def main():
    hand = wujihandpy.Hand()

    # 异步读取
    positions = await hand.read_joint_actual_position_async()

    # 异步写入
    hand.write_joint_enabled(True)
    await hand.finger(1).joint(0).write_joint_target_position_async(0.5)
    await asyncio.sleep(0.5)
    hand.write_joint_enabled(False)

asyncio.run(main())
```

> 异步版本不阻塞当前事件循环，但返回时同样保证操作已成功。

### 8.2 Unchecked（不校验）

`_unchecked` 后缀版本**立即返回，不阻塞，不保证成功**，适合延迟敏感场景：

```python
# 发送写入请求后立即返回
hand.finger(1).joint(0).write_joint_target_position_unchecked(0.5)

# 发送读取请求后立即返回 None（数据稍后通过 get 获取）
hand.finger(1).joint(0).read_joint_actual_position_unchecked()
```

### 8.3 get 缓存值

`get_*` 系列函数从本地缓存读取上一次成功的数据，**永不阻塞，永不通信**：

```python
# 先发起一次读取
hand.read_joint_actual_position()

# 之后可以多次获取缓存值（零开销）
cached = hand.get_joint_actual_position()
single = hand.finger(1).joint(0).get_joint_actual_position()
```

> 如果从未读取过数据，`get` 返回值未定义（通常为 0）。

---

## 9. 预设动作示例

### 9.1 握拳

```python
import numpy as np, time

hand.write_joint_enabled(True)
time.sleep(0.2)

# 所有手指弯曲（这里用各关节上限的 80% 作为握拳位置）
upper = hand.read_joint_upper_limit()
fist = upper * 0.8
hand.write_joint_target_position(fist.astype(np.float64))
time.sleep(1.0)

# 松开
hand.write_joint_target_position(np.zeros((5, 4), dtype=np.float64))
time.sleep(0.5)
hand.write_joint_enabled(False)
```

### 9.2 比 "OK" 手势

```python
hand.write_joint_enabled(True)
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
hand.write_joint_enabled(False)
```

### 9.3 逐指波浪

```python
hand.write_joint_enabled(True)
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
hand.write_joint_enabled(False)
```

### 9.4 使能保护模板

所有控制代码建议统一使用 `try / finally` 模板：

```python
import wujihandpy

hand = wujihandpy.Hand()

try:
    hand.write_joint_enabled(True)
    # ===== 在这里写你的控制逻辑 =====
    pass
finally:
    hand.write_joint_enabled(False)  # 无论如何都失能
```

---

## 10. 安全须知与常见问题

### 安全须知

1. **务必在 `finally` 中失能**：脚本异常退出时若关节仍使能，电机会持续带载发热
2. **先小幅测试**：首次控制时建议用限位的 10%–20% 幅度试运行，确认方向正确
3. **关注温度**：可通过 `read_joint_temperature()` 监控，长时间高负载运行前建议设置温度阈值
4. **Ctrl+C 安全中断**：在 `try / except KeyboardInterrupt / finally` 中妥善处理

### 常见问题

| 问题 | 解决方案 |
|------|----------|
| `RuntimeError: Failed to init` | 检查 USB 连接、指示灯是否绿色、udev 规则是否配置 |
| `ERROR_ACCESS` | Linux 下未配置 udev 规则，或 Docker 中未在宿主机执行配置 |
| `Could not find a version` (安装失败) | `python3 -m pip install --upgrade pip` 后重试 |
| `externally-managed-environment` | 使用 `python3 -m venv` 创建虚拟环境后安装 |
| 写入后关节不动 | 确认已执行 `write_joint_enabled(True)` |
| 运动方向反了 | 零点 `0.0` = 张开，正值 = 弯曲（握拳方向） |
| realtime_controller 报参数错误 | 需要传入 `(enable_upstream: bool, filter: IFilter)` 两个参数 |

---

## 11. API 速查表

### 设备连接

```python
hand = wujihandpy.Hand()           # 连接设备
finger = hand.finger(i)            # 获取手指引用 (i=0~4)
joint = finger.joint(j)            # 获取关节引用 (j=0~3)
```

### 读取 (同步/阻塞)

在 `Hand` / `Finger` / `Joint` 上均可调用，区别在于返回单值还是数组。

| 方法 | Hand 返回 | Joint 返回 | 说明 |
|------|-----------|------------|------|
| `read_joint_actual_position()` | `ndarray[5,4]` | `float64` | 当前实际位置 |
| `read_joint_target_position()` | `ndarray[5,4]` | `float64` | 当前目标位置 |
| `read_joint_upper_limit()` | `ndarray[5,4]` | `float64` | 关节上限 |
| `read_joint_lower_limit()` | `ndarray[5,4]` | `float64` | 关节下限 |
| `read_joint_enabled()` | `ndarray[5,4]` | `bool` | 使能状态 |
| `read_joint_temperature()` | `ndarray[5,4]` | `float64` | 温度 (℃) |
| `read_joint_error_code()` | `ndarray[5,4]` | `int` | 错误码 (0=正常) |
| `read_system_time()` | `int` | — | 运行时间 (μs) |
| `read_handedness()` | — | — | 左手/右手 |
| `read_firmware_version()` | — | — | 固件版本 |

### 写入 (同步/阻塞)

| 方法 | 参数 | 说明 |
|------|------|------|
| `write_joint_enabled(bool)` | `True` / `False` | 使能/失能 |
| `write_joint_target_position(val)` | `float64` 或 `ndarray` | 设置目标位置 |

### 异步版本 (后缀 `_async`)

```python
await hand.read_joint_actual_position_async()
await hand.finger(1).joint(0).write_joint_target_position_async(0.5)
```

### Unchecked 版本 (后缀 `_unchecked`)

```python
hand.read_joint_actual_position_unchecked()      # 返回 None
hand.finger(1).joint(0).write_joint_target_position_unchecked(0.5)
```

### 缓存读取 (前缀 `get_`)

```python
hand.get_joint_actual_position()                  # 返回上次 read 的缓存
```

### 实时控制

```python
controller = hand.realtime_controller(enable_upstream, filter)
# enable_upstream: bool — 是否保持 SDO 通道
# filter: IFilter       — 平滑滤波器实例
```

---

## 参考链接

- **官方文档**：https://docs.wuji.tech/docs/zh/wuji-hand/latest/
- **SDK 教程**：https://docs.wuji.tech/docs/zh/wuji-hand/latest/sdk-user-guide/introduction/
- **API 参考**：https://docs.wuji.tech/docs/zh/wuji-hand/latest/sdk-user-guide/api-reference/
- **GitHub (wujihandpy)**：https://github.com/wuji-technology/wujihandpy
- **URDF 模型**：https://github.com/wuji-technology/wuji-hand-description
- **ROS2 驱动**：https://github.com/wuji-technology/wujihandros2
