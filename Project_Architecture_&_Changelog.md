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
| `test/realman_contrl_lxqs.py` | `RealmanXrIncrementalTeleop` / `read_controller_pose` / `read_arm_pose` / `record_grip_press_event` / `log_grip_release_comparison` | 使用 Pico 右手柄按住 grip 后的相对 xyz+rpy 位姿直连控制 RM75-B 末端，并在 grip 按下/松开时记录手柄与机械臂位姿做 scale 感知偏移对比 | 依赖 `XrClient`、`RM75BInterface`、RealMan `rm_movep_canfd`；ROS2 发布为可选 |
| `test/realman_contrl_motion_tracker.py` | `RealmanMotionTrackerTeleop` / `read_motion_tracker_pose` / `iter_motion_tracker_pose_values` / `parse_pose_7d` / `record_a_start_event` / `log_a_stop_event_comparison` | 使用 Pico Motion Tracker 的相对 xyz+rpy 位姿直连控制 RM75-B 末端，A 键切换启停并记录启停事件 xyz 偏移对比，B 键复位，右摇杆调 scale，支持固定复位末端位姿 | 不按 SN 过滤，兼容 `pose` 和原始 `joints[*].p`；依赖 RealMan `rm_movep_canfd` |
| `test/realman_contrl_steamvr_tracker.py` | `RealmanSteamVrTrackerTeleop` / `SteamVrTrackerReader` / `HoldKeyWindow` / `TerminalHoldKeyMonitor` / `convert_openvr_pose` | 使用 SteamVR/OpenVR 单个 Vive Tracker 的相对 xyz+rpy 位姿增量控制 RM75-B，按住指定键时跟随、松开即缓停；支持自动选 tracker、列表查看、xyz/rpy 比例与轴映射调节 | 默认将 OpenVR 位姿通过 `R_HEADSET_TO_WORLD` 统一到项目控制坐标；无 DISPLAY 时回退到终端按键保持模式 |
| `test/realman_coordinate_system.py` | `run_coordinate_test` / `stream_pose` / `build_axis_target` | 通过固定原点和 xyz 正负 10cm 往返移动验证 RM75 Base 坐标系方向 | 依赖 `RM75BInterface`、RealMan `rm_movep_canfd` |

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

### [2026-04-24]
- **更新类型**: Bugfix / Test
- **修改目的/Bug现象**:
  - 用户反馈一次控制移动到某位置后，松开按键并移动遥控器，再次按下时机械臂不响应，必须把遥控器移回上一次松开附近才会重新开始。
  - 代码排查发现 `control_loop()` 在判断 `active` 后无论是否松开都会先读取 `controller_xyz`；如果松开后手柄 pose 短暂无效或读取异常，会在执行 `deactivate_control()` 前进入异常分支，导致上一轮 `controller_origin_xyz` / `arm_origin_pose` 未清空。
- **具体修改内容**:
  - 调整 `test/realman_contrl_lxqs.py` 的控制循环顺序。
  - 当 `right_grip` 未激活时，优先执行 `deactivate_control()`、发布状态并立即返回，不再依赖当前手柄位姿。
  - 仅在确认处于按住控制状态后读取右手柄 xyz 并发送 `rm_movep_canfd()`，保证下一次按下必定以新的手柄位置作为原点。

### [2026-04-24]
- **更新类型**: Feature / Test
- **修改目的/Bug现象**:
  - 用户希望 `test/realman_contrl_lxqs.py` 不只透传 xyz，也尝试透传 rpy 姿态。
  - 用户要求新增用于计算“透传数据和实际数据比例”的控制比例，且 xyz 与 rpy 分开配置，默认值均为 1。
- **具体修改内容**:
  - 新增 `DEFAULT_XYZ_SCALE_FACTOR = 1.0` 和 `DEFAULT_RPY_SCALE_FACTOR = 1.0`。
  - 将配置字段从单一 `scale_factor` 拆分为 `xyz_scale_factor` 与 `rpy_scale_factor`。
  - 新增 `read_controller_pose()`：读取右手柄 `[x,y,z,qx,qy,qz,qw]`，将位置和四元数转换到项目控制坐标系，并输出 `xyz+rpy`。
  - 控制激活时同时记录右手柄 `xyz+rpy` 原点；按住期间分别计算 `delta_xyz` 和 `delta_rpy`，拼成完整 `[x,y,z,rx,ry,rz]` 后通过 `rm_movep_canfd()` 透传。
  - 新增命令行参数 `--xyz-scale`、`--rpy-scale`，保留旧 `--scale` 作为 `--xyz-scale` 兼容参数。
  - ROS2 可选发布数据从 `xyz+active` 扩展为 `xyz+rpy+active`。

### [2026-04-24]
- **更新类型**: Bugfix / Test
- **修改目的/Bug现象**:
  - 用户实机反馈当前以 xy 为水平面的坐标系中，水平面旋转映射正确，但另外两个旋转通道（绕轴心旋转与上下旋转）对应关系反了。
  - 代码排查发现 `send_target_pose()` 将 `raw_delta_rpy=[roll,pitch,yaw]` 直接加到机械臂 `[rx,ry,rz]`，没有提供通道重排。
- **具体修改内容**:
  - 新增 `DEFAULT_RPY_AXIS_MAP = (1, 0, 2)`，默认交换 roll/pitch 两个通道，保持 yaw 不变。
  - 新增 `DEFAULT_RPY_AXIS_SIGN = (1.0, 1.0, 1.0)` 和命令行参数 `--rpy-axis-sign`，用于后续单独反转某个旋转通道方向。
  - 新增 `--rpy-axis-map` 参数，允许实机继续调整 rpy 到机械臂 rx/ry/rz 的映射关系。
  - 在 `send_target_pose()` 中先按 `rpy_axis_map` 重排 `raw_delta_rpy`，再按 `rpy_axis_sign` 和 `rpy_scale_factor` 生成透传姿态增量。

### [2026-04-29]
- **更新类型**: Feature / Test
- **修改目的/Bug现象**:
  - 需要将 `test/realman_contrl_motion_tracker.py` 从右手柄拷贝脚本改为 Pico Motion Tracker 遥操入口。
  - 用户要求用手柄 A 键作为 tracker 遥操的触发和停止条件：第一次按下启动，再次按下停止。
  - 用户要求不按 SN 过滤 tracker，所有 SN 均可作为输入源；原始输入可能包含 `joints[*].p` 字符串形式的 7 维 pose。
- **具体修改内容**:
  - 将控制类改为 `RealmanMotionTrackerTeleop`，A 键上升沿切换 `teleop_enabled`，停止时优先缓停并清空 tracker 原点。
  - 新增 `parse_pose_7d()` 和 `iter_motion_tracker_pose_values()`，兼容 SDK 封装后的 `pose` 数组、直接 `p` 字段和原始 `joints[*].p` 字段。
  - 新增 `read_motion_tracker_pose()`，遍历所有 SN 下的候选 tracker 位姿，不做 SN 白名单过滤；启动后绑定首次选中的 tracker id，避免多 tracker 在线时控制源跳变。
  - 保留原机械臂末端位姿透传逻辑：按启动瞬间 tracker 位姿和机械臂末端位姿建立原点，按相对 `xyz+rpy` 增量生成 `[x,y,z,rx,ry,rz]` 后调用 `rm_movep_canfd()`。

### [2026-04-29]
- **更新类型**: Feature / Test
- **修改目的/Bug现象**:
  - 用户要求 `test/realman_contrl_motion_tracker.py` 支持 B 键复位：按下后机械臂回到指定固定初始位置。
  - 用户要求 tracker 的 `scale` 可运行时调整，默认值为 1，可选档位为 `0.125,0.25,0.5,1.0,2.0,4.0,8.0`，右摇杆向左减小、向右增大，并在调整时打印当前值。
- **具体修改内容**:
  - 新增 `SCALE_OPTIONS`、`parse_scale_option()` 和运行时 `xyz_scale_factor`，将 `DEFAULT_XYZ_SCALE_FACTOR` 改为 `1.0`。
  - 新增 `update_scale_from_joystick()`，读取右手柄摇杆 x 轴，按离散档位调整 tracker xyz scale，并输出日志。
  - 新增 `DEFAULT_RESET_JOINTS_DEG = (0,-30,0,60,0,30,0)`，作为 RM75-6F 推荐待机复位关节角；新增 `--reset-joints-deg` 支持现场覆盖。
  - 新增 `handle_reset_button()` 和 `reset_arm_to_initial_pose()`，B 键上升沿停止 tracker 遥操、清空原点，并通过 `RM75BInterface.go_home()` / RealMan `rm_movej` 阻塞移动到复位关节角。

### [2026-04-29]
- **更新类型**: Bugfix / Test
- **修改目的/Bug现象**:
  - 用户实机运行 `test/realman_contrl_motion_tracker.py` 时高频出现 RealMan SDK 报错：`[rm_get_current_arm_state] get_current_arm_state send error` / `send err: -2`。
  - 代码排查发现 `publish_state()` 在启用 ROS 发布时会在控制循环内调用 `read_arm_pose()`，从而高频触发 `rm_get_current_arm_state()`，可能与 `rm_movep_canfd()` 透传抢占同一 RealMan 通信链路。
- **具体修改内容**:
  - 将 `RealmanMotionTrackerTeleop.publish_state()` 改为 no-op，保留函数和调用点但不再读取机械臂状态、不再发布 `/action` / `/state`。
  - 该调整避免遥操循环内周期性调用 `rm_get_current_arm_state()`，降低 RealMan TCP/JSON 发送冲突风险。

### [2026-04-29]
- **更新类型**: Bugfix / Test
- **修改目的/Bug现象**:
  - 用户按 B 键复位时仍出现 RealMan SDK 发送错误：复位前后分别触发 `rm_get_current_arm_state` 和 `rm_movej` 的 `send err: -2`。
  - 代码排查发现 `reset_arm_to_initial_pose()` 在 `go_home()` 后立即调用 `read_arm_pose()`，仍会触发 `rm_get_current_arm_state()`；同时复位前没有明确停止上一轮 `rm_movep_canfd()` 透传。
- **具体修改内容**:
  - 修改 B 键复位流程：先清空 tracker 控制状态，再强制调用 `rm_set_arm_slow_stop()`，等待 0.2 秒后再调用 `go_home()` / `rm_movej`。
  - 移除 B 键复位完成后的 `read_arm_pose()`，避免复位路径再次触发 `rm_get_current_arm_state()`。
  - 移除停止 tracker 遥操 `deactivate_control()` 中的 `read_arm_pose()`，停止时仅缓停并清空控制原点。

### [2026-04-29]
- **更新类型**: Bugfix / Test
- **修改目的/Bug现象**:
  - 用户希望 `test/realman_contrl_motion_tracker.py` 使用固定机械臂初始末端位姿 `arm_pose=[0.43261, 0.028079, 0.026739, 2.479, 1.491, 2.482]`，避免 A 键反复启停时每次从 `rm_get_current_arm_state()` 读取到不同的 `arm_pose` 作为控制原点。
  - 代码排查发现 `activate_control()` 每次 A 键启动都会调用 `read_arm_pose()`，导致机械臂控制原点绑定到实机当前末端位姿。
- **具体修改内容**:
  - 新增 `DEFAULT_INITIAL_ARM_POSE = (0.43261, 0.028079, 0.026739, 2.479, 1.491, 2.482)`。
  - 新增 `--initial-arm-pose` 参数和 `parse_initial_arm_pose()`，支持按 `x,y,z,rx,ry,rz` 覆盖固定初始末端位姿。
  - 修改 `init_hardware()` 和 `activate_control()`，不再读取当前机械臂状态作为 `target_pose` / `arm_origin_pose`，而是统一使用固定初始末端位姿。

### [2026-04-29]
- **更新类型**: Bugfix / Test
- **修改目的/Bug现象**:
  - 用户澄清上一条需求表达不准确：A 键启动时的初始化位置仍应使用 `read_arm_pose()` 读取当前机械臂末端位姿；给出的 `arm_pose=[0.43261, 0.028079, 0.026739, 2.479, 1.491, 2.482]` 应作为 B 键复位目标位姿。
- **具体修改内容**:
  - 移除固定初始末端位姿逻辑，`init_hardware()` 和 `activate_control()` 恢复使用 `read_arm_pose()` 获取当前机械臂末端位姿。
  - 将固定 pose 改为 `DEFAULT_RESET_ARM_POSE = (0.43261, 0.028079, 0.026739, 2.479, 1.491, 2.482)`。
  - 将 B 键复位从 7 维关节角 `go_home()` 改为 6 维末端位姿 `rm_movep_canfd()` 短时透传；新增 `--reset-arm-pose` 参数支持覆盖。

### [2026-04-29]
- **更新类型**: Feature / Test
- **修改目的/Bug现象**:
  - 用户需要一个独立脚本验证 RM75-B/RM75-6F 机械臂 Base 坐标系中 x/y/z 正负方向的实际运动方向。
- **具体修改内容**:
  - 新增 `test/realman_coordinate_system.py`。
  - 默认测试原点设置为 `[0.1166, 0.0, 0.7247, 0.0, 1.043, 0.0]`，支持 `--origin-pose` 覆盖。
  - 测试流程按顺序执行：回原点、X+10cm、回原点、X-10cm、回原点、Y+10cm、回原点、Y-10cm、回原点、Z+10cm、回原点、Z-10cm、回原点。
  - 通过 `rm_movep_canfd()` 低跟随重复发送每个目标位姿，支持 `--step`、`--rate`、`--duration`、`--settle` 参数调节。

### [2026-04-29]
- **更新类型**: Feature / Test
- **修改目的/Bug现象**:
  - 用户需要在 `test/realman_contrl_lxqs.py` 中记录右手 grip 按下和松开两个事件点的手柄/机械臂位姿。
  - 需要打印手柄位移经过 `xyz_scale_factor`、`max_delta_m`、`rpy_axis_map`、`rpy_axis_sign`、`rpy_scale_factor` 后对应的期望机械臂偏移，并与机械臂实际偏移比较。
- **具体修改内容**:
  - 新增 `grip_press_event` 缓存，`activate_control()` 在读取机械臂当前末端位姿后调用 `record_grip_press_event()` 记录按下事件。
  - 新增 `log_grip_release_comparison()`，在松开 grip 时打印手柄 `xyz/rpy` 原始偏移、考虑 scale 和映射后的期望机械臂 `xyz/rpy` 偏移、机械臂实际偏移以及误差。
  - 调整 `deactivate_control()`，松开时先尽力读取当前手柄和机械臂位姿用于事件记录；读取失败只报警，不阻塞缓停和清空控制原点。

### [2026-04-29]
- **更新类型**: Feature / Test
- **修改目的/Bug现象**:
  - 用户需要在 `test/realman_contrl_motion_tracker.py` 中记录 A 键启动和再次按下停止两个事件点的输入源/机械臂位置。
  - 需要打印 Motion Tracker 的 xyz 偏移经过当前 `xyz_scale_factor` 和 `max_delta_m` 后对应的期望机械臂 xyz 偏移，并与机械臂实际 xyz 偏移比较；旋转角不参与比较。
- **具体修改内容**:
  - 新增 `a_start_event` 缓存，`activate_control()` 在读取机械臂当前末端位姿后调用 `record_a_start_event()` 记录 A 键启动事件。
  - 新增 `log_a_stop_event_comparison()`，A 键停止时打印 tracker xyz、机械臂 xyz、期望机械臂 xyz 偏移、实际机械臂 xyz 偏移和误差。
  - 调整 `deactivate_control()`，停止遥操时先尽力读取绑定 tracker 和机械臂当前位姿用于事件对比；读取失败只报警，不阻塞缓停和清空控制原点。
  - B 键复位时同步清空未完成的 A 键事件记录，避免复位后误用上一轮启动事件。

### [2026-04-29]
- **更新类型**: Refactor / Test
- **修改目的/Bug现象**:
  - 用户反馈 `test/realman_contrl_motion_tracker.py` 中 A 键事件 xyz 偏移对比日志以一长串 list 输出，不便于现场直观比较 x/y/z 三个方向的误差。
- **具体修改内容**:
  - 将 `log_a_stop_event_comparison()` 的 xyz 偏移对比日志改为多行表格。
  - 日志新增计算公式说明、启动/停止 tracker id、当前/启动时 scale 和 `max_delta_m`。
  - 每个轴单独展示 tracker 位移、期望机械臂位移、实际机械臂位移和误差；不改变原有控制和误差计算逻辑。

### [2026-05-03]
- **更新类型**: Feature / Test
- **修改目的/Bug现象**:
  - 用户需要一个基于单个 SteamVR/Vive Tracker 的 RM75-B 增量控制脚本，控制逻辑要求与 `test/realman_contrl_lxqs.py` 一致：按住才控制、松开即停止，并以开始控制瞬间作为相对原点。
  - 现有 `test/test(1).py` 仅负责 OpenVR tracker 枚举与相对位姿验证，不能直接接入 RM75-B 控制链路，也没有按住/松开激活逻辑。
  - 当前 Python 环境缺少 `meshcat`，如果直接导入 `xrobotoolkit_teleop.utils.geometry` 会在读取坐标变换常量前先失败，需要避免把可视化依赖带入真机控制脚本。
- **具体修改内容**:
  - 新增 `test/realman_contrl_steamvr_tracker.py`：
    - 复用 `RM75BInterface` + `rm_movep_canfd()` 的末端位姿增量透传框架，使用单个 SteamVR `GenericTracker` 做 `xyz+rpy` 相对原点控制。
    - 新增 `SteamVrTrackerReader`，从 OpenVR `TrackingUniverse` 读取 tracker 绝对位姿，支持 `--tracker-serial` 指定序列号和 `--list-trackers` 仅查看在线设备与当前位姿。
    - 基于 OpenVR 绝对位姿默认采用 `project_world` 模式，将 `x右/y上/z后` 的设备坐标通过与项目一致的 `R_HEADSET_TO_WORLD` 映射为机械臂控制世界系；同时开放 `--coordinate-mode`、`--xyz-axis-map/sign`、`--rpy-axis-map/sign` 便于现场继续微调。
    - 新增按住键激活控制逻辑：有 `DISPLAY` 时使用 `HoldKeyWindow` 捕获真实按下/松开；无 `DISPLAY` 时回退到 `TerminalHoldKeyMonitor`，在终端中按住指定键维持控制。

### [2026-05-03]
- **更新类型**: Bugfix / Test
- **修改目的/Bug现象**:
  - 用户通过 SSH 终端实测 `test/realman_contrl_steamvr_tracker.py` 时，SteamVR 中 tracker `connected=True` 但 `pose_valid=False`，脚本持续打印 `tracker 当前 pose 无效` 并反复进入 fail-safe，现场日志刷屏且机械臂重复缓停。
  - 代码排查发现 `read_tracker_pose()` 在 `pose_valid=False` 时会直接抛异常，而 `run()` 对所有异常统一调用 `fail_safe_stop()`，导致每个控制周期都重复清空状态和报警。
- **具体修改内容**:
  - 为 `test/realman_contrl_steamvr_tracker.py` 新增 `waiting_for_tracker_pose` 和 `last_tracker_pose_error` 状态。
  - 新增 `is_tracker_pose_runtime_error()`，将 `pose 无效 / 已断开 / 当前没有有效 pose` 识别为可恢复的 tracker 可用性异常。
  - 新增 `pause_for_tracker_pose_loss()`：tracker 丢追踪时仅在首次异常缓停一次并清空控制状态，随后进入“等待 tracker 恢复有效 pose”状态，不再每周期重复触发 fail-safe。
  - 新增 `on_tracker_pose_recovered()`：tracker 恢复有效 pose 后打印恢复日志；如果用户仍按住激活键，则以恢复瞬间重新建立控制原点继续增量控制。

### [2026-05-06]
- **更新类型**: Docs / Test
- **修改目的/Bug现象**:
  - 用户需要一份面向低代码或无代码操作者的 `test/realman_contrl_steamvr_tracker.py` 超简使用文档。
  - 文档范围只保留 conda 环境、命令行启动方式、按住/松开时的基础控制逻辑，以及最简单的运行检查项。
- **具体修改内容**:
  - 新增 `test/realman_contrl_steamvr_tracker_usage.md`。
  - 文档明确使用 `ts_pico_teleop` conda 环境。
  - 根据脚本实际日志和参数行为，整理出最小启动命令、`Space` 按住跟随/松开停止、`Ctrl+C` 退出，以及启动成功/常见异常的最简判断方式。
