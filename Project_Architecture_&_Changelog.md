# 项目核心架构与维护日志

## 1. 系统架构简述
- 该仓库是 `XRoboToolkit-Teleop-Sample-Python`，定位为 Pico/XR 输入到仿真或真机控制的 Python 侧遥操作样例。
- 输入层通过 `xrobotoolkit_sdk`（由 `XRoboToolkit-PC-Service-Pybind` 提供）读取头显、手柄、按键、手势、体感追踪数据。
- 控制层由通用控制基类 + 仿真控制器 + 硬件控制器组成，统一处理目标位姿映射、夹爪逻辑、控制循环。
- 执行层覆盖 MuJoCo/Placo 仿真和多类真机接口（UR、ARX R5、Piper、RM75-B、Galaxea R1 Lite 等）。
- 数据层包含日志记录、图像压缩与日志分析脚本，用于遥操作过程数据回放与检查。

## 2. 核心函数索引字典
| 模块/所在文件 | 函数名称 | 核心功能简述 | 依赖关系/备注 |
|---|---|---|---|
| `xrobotoolkit_teleop/common/xr_client.py` | `XrClient.__init__` | 初始化 XR SDK 连接 | 依赖 `xrobotoolkit_sdk.init()` |
| `xrobotoolkit_teleop/common/xr_client.py` | `XrClient.get_pose_by_name` | 读取左右手柄/头显位姿 | 返回 `[x,y,z,qx,qy,qz,qw]` |
| `xrobotoolkit_teleop/common/xr_client.py` | `XrClient.get_key_value_by_name` | 读取 trigger/grip 模拟量 | 左右手触发键/握把 |
| `xrobotoolkit_teleop/common/xr_client.py` | `XrClient.get_button_state_by_name` | 读取 A/B/X/Y/menu/摇杆按下状态 | 返回布尔值 |
| `xrobotoolkit_teleop/common/xr_client.py` | `XrClient.get_motion_tracker_data` | 读取体感追踪器 pose/velocity/acceleration | 返回按序列号聚合字典 |
| `xrobotoolkit_teleop/common/xr_client.py` | `XrClient.get_body_tracking_data` | 读取全身关节追踪数据 | 不可用时返回 `None` |
| `xrobotoolkit_teleop/common/base_teleop_controller.py` | `BaseTeleopController.run` | 遥操作主循环框架 | 子类实现具体 backend |
| `xrobotoolkit_teleop/common/base_hardware_teleop_controller.py` | `HardwareTeleopController.run` | 真机控制循环（含相机/日志） | 扩展自 `BaseTeleopController` |
| `xrobotoolkit_teleop/simulation/mujoco_teleop_controller.py` | `MujocoTeleopController` 系列方法 | MuJoCo 遥操作控制与可视化 | 依赖 MuJoCo 资产与几何工具 |
| `xrobotoolkit_teleop/simulation/placo_teleop_controller.py` | `PlacoTeleopController` 系列方法 | Placo IK 侧遥操作控制 | 支持浏览器可视化 |
| `test/test_pico_connection.py` | `main` | Pico 连接与数据读取快速联调脚本 | 输出手柄、按键、体感数据 |
| `test/test_trigger.py` | 顶层循环脚本 | 快速验证右手 trigger/grip 变化 | 最小化输入链路验证 |
| `test/test_pose_precision.py` | `run_monitor_test` / `run_drift_test` / `run_move_test` | 实时定位观察、零飘统计与位移精度评估 | 同时输出 world 与 head-relative 两套结果 |
| `test/realman_contrl_lxqs.py` | `RealmanXrIncrementalTeleop` / `read_controller_xyz` / `read_arm_pose` | 使用 Pico 右手柄按住 grip 后的相对 xyz 位移直连控制 RM75-B 末端 | 依赖 `XrClient`、`RM75BInterface`、RealMan `rm_movep_canfd`；ROS2 发布为可选 |

## 3. 变更与 Bug 追踪日志 (Changelog)
### [2026-04-24]
- **更新类型**: Docs / Initialization
- **修改目的/Bug现象**:
  - 按项目规则检查发现根目录缺失 `Project_Architecture_&_Changelog.md`，导致“先查文档后改代码”的协作链路无法执行。
- **具体修改内容**:
  - 新建本文件。
  - 基于当前代码结构补充系统架构概览与核心函数索引。
  - 记录当前会话初始化动作，作为后续维护基线。

### [2026-04-24]
- **更新类型**: Feature / Test
- **修改目的/Bug现象**:
  - 需要对 Pico 头显与左右手柄做量化精度测试，覆盖零飘和按卡尺实测位移的误差评估。
  - 用户反馈戴头显时可能出现微动，需提供能剔除头显抖动影响的统计口径。
- **具体修改内容**:
  - 新增 `test/test_pose_precision.py`：
    - `drift` 模式：定频采样头显/左右手柄位姿，统计零飘（max/mean/std/峰峰值）。
    - `move` 模式：支持输入目标位移与卡尺实测值，计算 world 与 head-relative 两套位移误差。
    - 输出 `csv + json` 到 `logs/precision_tests/`，便于复盘与批量分析。
  - 在函数索引字典中新增该测试脚本入口说明。

### [2026-04-24]
- **更新类型**: Feature / Test
- **修改目的/Bug现象**:
  - 需要实时观察 `headset / left_controller / right_controller` 的定位刷新结果，直接验证 SDK 返回值与相对头显变换后的表现。
- **具体修改内容**:
  - 为 `test/test_pose_precision.py` 新增 `monitor` 模式。
  - 终端实时刷新显示每个设备的：
    - `world_xyz`
    - `world_delta`（相对启动瞬间）
    - 四元数
    - `head_relative`
    - `headrel_delta`（相对启动瞬间）
  - 支持 `--rate`、`--duration` 和 `--no-clear` 参数，方便做连续观察或保留历史输出。

### [2026-04-24]
- **更新类型**: Feature / Test
- **修改目的/Bug现象**:
  - 用户需要在终端明确区分哪些定位值是 SDK 直接返回，哪些是脚本二次计算，避免观察时误读数据来源。
- **具体修改内容**:
  - 调整 `test/test_pose_precision.py` 的 `monitor` 输出格式。
  - 为 `world_xyz`、`quat_xyzw` 标注 `[SDK直接]`。
  - 为 `world_delta`、`head_relative`、`headrel_delta` 标注 `[脚本计算]`。

### [2026-04-24]
- **更新类型**: Feature / Test
- **修改目的/Bug现象**:
  - `test/realman_contrl_lxqs.py` 原脚本使用 SpaceMouse 轴值按周期累加位移，属于力度积分控制；Pico 右手柄 SDK 返回的是空间坐标，应按按下瞬间作为原点计算相对位移。
  - 原脚本依赖 `/home/rimbot/...` 下的本地 SpaceMouse/O6 模块，当前仓库内不存在这些模块，不适合作为 Pico 右手柄直连 RM75-B 的测试入口。
- **具体修改内容**:
  - 将 `test/realman_contrl_lxqs.py` 改为 `RealmanXrIncrementalTeleop` ROS2 节点。
  - 通过 `XrClient.get_pose_by_name("right_controller")` 读取右手柄 xyz，并使用项目既有 `R_HEADSET_TO_WORLD` 转换到控制坐标系。
  - 使用 `right_grip` 作为控制激活键：按下时记录右手柄原点与当前机械臂末端位姿，按住期间用相对位移更新目标 xyz，松开时清空原点并调用 `rm_set_arm_slow_stop()`。
  - 使用 RealMan `rm_movep_canfd()` 发送末端位姿透传，默认低跟随；保留 `--high-follow` 作为显式选项。
  - 新增 RealMan Python 包来源输出，便于区分 pip 安装和本地源码导入。

### [2026-04-24]
- **更新类型**: Bugfix / Test
- **修改目的/Bug现象**:
  - 用户在 `ts_pico_teleop` conda 环境中直接运行 `python test/realman_contrl_lxqs.py`，顶层导入 `rclpy` 时报错 `ModuleNotFoundError: No module named 'rclpy._rclpy_pybind11'`。
  - 报错栈显示当前 Python 为 3.13，而 ROS Humble 的 `rclpy` 路径来自 `/opt/ros/humble/.../python3.10`，二进制扩展 ABI 不匹配。
- **具体修改内容**:
  - 移除 `test/realman_contrl_lxqs.py` 顶层 `rclpy` / `std_msgs` 导入。
  - 将 `RealmanXrIncrementalTeleop` 从 ROS2 `Node` 改为普通 Python 控制类，默认使用 `time.sleep()` 定频循环。
  - 新增 `--enable-ros-publish` 参数，仅在显式启用时按需导入 `rclpy` 并发布 `/action`、`/state`。
  - 增加纯 Python 日志输出，保证没有 ROS2 时仍可运行核心 RealMan 末端增量控制逻辑。
