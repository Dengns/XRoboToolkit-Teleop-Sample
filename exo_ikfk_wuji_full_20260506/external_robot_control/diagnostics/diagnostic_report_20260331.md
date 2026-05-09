# 外骨骼到 O6 诊断原因报告

生成日期：2026-03-31

## 1. 报告范围

本报告基于以下已完成的诊断结果与代码证据整理：

- `diagnostics/thumb_audit_20260331_112822.json`
- `diagnostics/latency_pipeline_20260331_113650.json`
- `diagnostics/latency_pipeline_20260331_113650.csv`
- `diagnostics/stability_20260331_114116.json`
- `diagnostics/latency_actuator_20260331_114845.json`
- `exo_to_o6_bridge.py`
- `exo_mapping_wizard.py`
- `joint_map_output.py`
- `linkerhand-python-sdk/LinkerHand/core/rs485/linker_hand_o6_rs485.py`

本轮链路背景按现场实际配置理解为：

1. 传感器侧产生 `/mocap/skeleton_data`
2. 数据经 `10.42.0.2` 路由发送到 `10.42.0.3`
3. `10.42.0.3` 上 ROS / rosbridge 对外开放 `ws://10.42.0.3:9090`
4. 本地 Python 通过 `roslibpy` 订阅后做解析、映射、NN 推理
5. 再通过 RS485 下发 O6

## 2. 结论摘要

### 2.1 P0 启动后前 1 分钟抖动明显

结论：现象属实，且主要集中在拇指链路，不是 O6 软件推理耗时导致。

证据：

- `stability_20260331_114116.json` 中，`thumb_cmc_pitch.predicted_pose.std` 从 `16.722` 下降到 `11.068`，下降约 `33%`
- `thumb_cmc_yaw.predicted_pose.std` 从 `10.642` 下降到 `7.044`，下降约 `34%`
- `thumb_cmc_pitch.sensor_delta.std` 从 `0.2187` 下降到 `0.1470`，下降约 `32.8%`
- 其余四指 `sensor_delta.std` 基本为 `0.0`，说明明显抖动并不在四指主关节输入上

判断：

- 抖动源首先出现在拇指输入特征，而不是在 O6 执行器端凭空产生
- 预测姿态的波动幅度明显大于原始 `delta`，说明当前映射/模型会放大拇指输入扰动
- “前 60 秒更明显，后续有所好转”与现场观察一致

### 2.2 P1 延时大的主因不是本机推理，而是节流和执行器响应

结论：本机解析与 NN 推理很快，真正明显的延时来自三部分：

1. ROS 消息突发到达与缓存抖动
2. 双重 20Hz 节流导致大量帧被丢弃
3. RS485 写入与 O6 本体动作响应明显慢于软件推理

证据：

- `latency_pipeline_20260331_113650.json`
  - `message_count = 2396`
  - `accepted_count = 271`
  - `dropped_count = 2125`
  - `drop_rate = 88.69%`
  - `callback_ready_ms.mean = 0.331`
  - `predict_ms.mean = 0.318`
- `latency_pipeline_20260331_113650.csv`
  - `message_interval_ms` 中 `0ms` 记录很多，`<1ms` 也很多，说明本地接收不是均匀到达，而是有明显 burst
  - 由旧版报告反推的 `source_to_arrival_normalized_ms`：
    - mean 约 `28.518ms`
    - p95 约 `86.334ms`
    - max 约 `220.983ms`
- `latency_actuator_20260331_114845.json`
  - `cmd_write_ms.mean = 41.020`
  - `reach50_ms.mean = 327.599`
  - `reach90_ms.mean = 480.139`
  - 拇指两轴 `reach90_ms` 平均约 `259.187`
  - 四指 `reach90_ms` 平均约 `590.615`

代码证据：

- `exo_to_o6_bridge.py:74` 定义 `SEND_INTERVAL = 1.0 / SEND_HZ`
- `exo_to_o6_bridge.py:292-294` 在 ROS 回调内先做一次 `< 50ms` 的丢帧节流
- `exo_to_o6_bridge.py:324` 发送线程再 `time.sleep(SEND_INTERVAL)` 一次
- `linkerhand-python-sdk/LinkerHand/core/rs485/linker_hand_o6_rs485.py:91` 定义 `FRAME_GAP = 0.030`

判断：

- 从“本机收到 ROS 消息”到“姿态算完”平均只要 `0.331ms`，不是主要瓶颈
- 但桥接回调已经先把大部分消息丢掉，后面发送线程又固定 20Hz，再加一次等待
- RS485 单次写入本身就约 `41ms`
- O6 执行到 90% 目标位平均还要约 `480ms`
- 因此用户感受到的“大延时”主要不是算法慢，而是“节流 + 总线 + 执行器响应”的叠加

### 2.3 P2 拇指弯曲和开闭映射不理想是结构性问题

结论：问题属实，而且是当前映射结构直接决定的，不是偶然训练噪声。

证据：

- `joint_map_output.py:5` 与 `joint_map_output.py:7`
  - `thumb_cmc_pitch` 和 `thumb_cmc_yaw` 都映射到 `joint_RightSkeletonThumbBase`
- `exo_calibration.json:4-5` 中也记录为同一主关节
- `thumb_audit_20260331_112822.json`
  - `same_primary_joint = true`
  - `shared_feature_ratio = 1.0`
  - 两个拇指槽位使用完全相同的 thumb feature 集合

补充说明：

- 当前训练数据是“时间插值标注”还是“主关节自标注”，这一点在 `exo_training_data.json` 中没有落盘字段直接记录
- 但代码上 `exo_mapping_wizard.py:219-220` 明确支持两种模式
- 用户现场补充说明“当前版本采集使用的是时间插值而不是主关节”，因此本次关于拇指“候选主轴排序”的结论只能视为相关性参考，不能直接当作物理真值

## 3. 当前链路各步骤延时

下面先给出“本轮已经测到的延时”，再给出“当前还缺失、必须补采的步骤”。

### 3.1 已有数据能够确认的步骤

| 链路步骤 | 当前可得数值 | 说明 |
|---|---:|---|
| 源消息时间戳 -> 本机 `roslibpy` 接收 `raw` | mean `-1917.086ms`，range `220.983ms` | 原始值为负，说明两端时钟未对齐，不能当真实网络绝对延时 |
| 源消息时间戳 -> 本机接收 `normalized` | mean `28.518ms`，p95 `86.334ms`，max `220.983ms` | 仅能看作链路抖动 / 缓存波动，不是绝对时延 |
| 本机收到消息 -> 解析完成 | 当前旧报告未单独持久化 | 从总耗时看属于极小量，需新版脚本补采精确值 |
| 解析 -> NN/映射完成 | `predict_ms.mean = 0.318ms`，`p95 = 1.009ms` | 软件推理很快 |
| 本机收到消息 -> 姿态准备完成 | `callback_ready_ms.mean = 0.331ms`，`p95 = 1.010ms` | 本机计算不是瓶颈 |
| 回调层丢帧率 | `88.69%` | 主要是 20Hz 节流导致 |
| RS485 单次写入 | `cmd_write_ms.mean = 41.020ms` | 已经显著高于本机推理 |
| O6 达到 50% 目标位 | `reach50_ms.mean = 327.599ms` | 执行器本体响应明显 |
| O6 达到 90% 目标位 | `reach90_ms.mean = 480.139ms` | 用户主观感知延时主要来源之一 |

### 3.2 结合代码可推断的额外等待

以下是“代码明确存在，但本轮旧报告没有逐项量化”的等待项。

1. ROS 回调节流等待
   - 代码位置：`exo_to_o6_bridge.py:292-294`
   - 机制：小于 `50ms` 的新消息直接丢弃
   - 影响：不是把所有消息排队，而是直接舍弃，因此会带来明显的跟手性下降

2. 发送线程固定 20Hz 等待
   - 代码位置：`exo_to_o6_bridge.py:324`
   - 机制：发送线程每次都 `sleep(50ms)`
   - 影响：即使上游已经选好姿态，真正下发还会受第二层 20Hz 周期限制

3. RS485 帧间隔等待
   - 代码位置：`linker_hand_o6_rs485.py:91` 与 `_bus_free()`
   - 机制：相邻读写之间强制留出 `30ms`
   - 影响：任何“写命令 + 读状态”的组合都会被总线仲裁继续拉长

### 3.3 以“用户最终感知延时”为目标的粗略分段

以下为基于当前证据的粗分段，不是最终精确闭环测量值：

1. 传感器源时间戳到本机接收
   - 当前只能用 `normalized` 看波动：常见额外抖动约 `0-86ms`，最坏约 `221ms`
   - `raw` 因时钟未对齐不可直接解释

2. 本机解析与推理
   - 约 `0.3ms`

3. 回调/发送调度
   - 双重 20Hz 机制使大量帧被跳过
   - 单看发送周期，理论上还会引入 `0-50ms` 的额外排队等待
   - 这里的 `0-50ms` 是基于代码节拍的推断值，不是本轮直接测量值

4. RS485 写命令
   - 约 `41ms`

5. O6 执行器到明显动作
   - `first_motion_ms = 0` 不能解释成“零延时”
   - 因诊断轮询周期是 `50ms`，它只说明“首次运动发生在第一次读状态之前或之时”

6. O6 达到大部分目标位
   - 平均到 `90%` 约 `480ms`
   - 四指慢于拇指，四指平均约 `591ms`

## 4. 时钟不对齐问题

### 4.1 已确认现象

`latency_pipeline_20260331_113650.json` 中：

- `source_timestamp_present = true`
- `source_to_arrival_ms.mean = -1917.086`

这说明：

1. 消息里大概率确实存在某个可解析的时间戳字段
2. 但该时间戳和本机 `time.time_ns()` 不在同一时间基准下
3. 因此当前的 `source -> arrival raw` 不是“真实网络延时”

### 4.2 当前能确认什么

- 可以确认“源端时间戳存在的概率很高”
- 可以确认“跨设备时钟未对齐”
- 可以确认“链路存在明显的 burst / cache 抖动”

### 4.3 当前还不能确认什么

- 不能确认旧报告里时间戳的具体字段路径，因为旧版 `latency-pipeline` 还没有把 `source_stamp_path` 落盘
- 不能把 `-1917ms` 解释成真实传输快了 1.9 秒，这是时钟基准不一致造成的
- 不能把 `10.42.0.2 -> 10.42.0.3`、ROS 队列、rosbridge、websocket、本机回调这几段再继续拆开，因为中间没有逐点埋时间戳

### 4.4 建议的解决方案

建议按优先级分两层做：

1. 先做“同一时钟基准”
   - 让传感器侧主机与控制主机做 NTP / PTP 同步
   - 若无法做系统级同步，则在消息中同时带“传感器单调时钟”和“ROS 主机写入时的本机时钟”

2. 再做“逐节点打点”
   - 传感器采样完成：`sensor_capture_ns`
   - 传感器打包发送前：`sensor_publish_ns`
   - `10.42.0.3` ROS 收到后：`ros_host_rx_ns`
   - rosbridge 发 websocket 前：`rosbridge_tx_ns`
   - 本机 roslibpy 收到：`client_rx_ns`
   - 本机 parse 完成：`parse_done_ns`
   - 本机 pose 完成：`pose_done_ns`
   - 发送线程实际写串口开始：`cmd_write_start_ns`
   - 写串口完成：`cmd_write_end_ns`
   - 第一次状态变化：`first_motion_ns`
   - 达到 50% / 90%：`reach50_ns` / `reach90_ns`

只有以上时间戳落到同一份记录里，才能把全链路彻底拆开。

## 5. 传感器侧是否存在可用时间戳

当前判断：有较大概率存在，但旧报告未记录具体字段路径，暂时不能写成“已明确是某字段”。

依据：

1. `latency_pipeline_20260331_113650.json` 的 `source_timestamp_present = true`
2. 旧版 `latency_pipeline_20260331_113650.csv` 已成功计算出每帧 `source_to_arrival_ms`
3. `test_sensor_connection.py` 和 `sensor_visualizer.py` 当前只读取了 `segment.header.frame_id`，没有把 `stamp` 打印出来，因此现场工具本身也没有把字段路径保留下来

因此当前最稳妥的结论是：

- “旧消息里至少有一个字段被诊断脚本识别成了时间戳”
- “该字段路径需要重新跑增强后的 `latency-pipeline` 才能确定并落盘”

## 6. 对三个问题的原因归因

### 6.1 P0 抖动

主因排序：

1. 拇指输入特征在启动阶段本身更漂
2. 当前拇指映射/模型会把输入扰动放大到输出姿态
3. 启动早期链路 burst 也会放大观感上的抖动

不是主因的项：

- 本机解析 / NN 推理耗时

### 6.2 P1 延时

主因排序：

1. 双重 20Hz 节流
2. RS485 写入和帧间隔
3. O6 执行器物理到位时间
4. 上游链路 burst / 缓存抖动

不是主因的项：

- 本机 parse / NN 推理速度

### 6.3 P2 拇指映射

主因排序：

1. `thumb_cmc_pitch` 与 `thumb_cmc_yaw` 共用同一主关节
2. 两个拇指槽位使用完全相同的特征集合
3. 当前训练标签来自时间插值时，候选主轴相关性只能作参考，不能直接当物理真值

## 7. 下一步建议

建议下一轮按以下顺序继续：

1. 重新运行增强后的 `latency-pipeline`
   - 目标：把 `source_stamp_path`、`arrival_to_parse_ms`、`parse_to_pose_ms`、`source_to_arrival_normalized_ms` 直接落盘

2. 如果可以改上游消息格式
   - 在传感器采样端、ROS 主机端各补一个时间戳
   - 这样能真正拆出 `10.42.0.2 -> 10.42.0.3` 与 rosbridge 的占比

3. 等全链路记录补齐后，再决定是否改控制策略
   - 对 P0：死区、启动稳定期、输入滤波
   - 对 P1：去掉双重节流中的一层，或改成“最新值覆盖 + 单发送线程”
   - 对 P2：重新做拇指主轴映射与拇指专用标签/模型

