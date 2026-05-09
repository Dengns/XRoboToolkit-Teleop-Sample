# 外骨骼 → O6 完整测试综合报告

- 生成时间：2026-03-31 15:18
- 对应测试批次：`run_all_20260331_151258`
- 本轮关联文件：
  - `thumb_audit_20260331_151258.json`
  - `latency_pipeline_20260331_151300.json`
  - `stability_20260331_151322.json`
  - `latency_fullchain_20260331_151455.json`
  - `ros_host_bridge_probe_20260331_151454.json`
  - `fullchain_alignment_20260331_151528.json`
  - `latency_actuator_20260331_151528.json`

## 1. 结论摘要

- 本轮最明确的主瓶颈不是本机解析或 NN 推理，而是“跨机网络段 + 本机发送等待 + RS485 写入”三段累加。
- 从 `source stamp` 到 `COM10` 写入完成的全链路近似延时为：
  - `p50 = 138.733 ms`
  - `p95 = 182.989 ms`
  - `max = 199.811 ms`
- 本轮 `stability` 没有复现“启动前 1 分钟抖动更明显”的现象。90 秒内所有主关节 `sensor_delta` 都是 0，预测 pose 也完全恒定。
- 拇指映射问题仍然明确存在：
  - `thumb_cmc_pitch`
  - `thumb_cmc_yaw`
  - 当前都共用 `joint_RightSkeletonThumbBase`
  - 两个槽位使用的 thumb feature 集合完全相同
- `latency-fullchain` 实际发送到手的频率明显低于 20Hz 标称值。
  - 20 秒内 `sent_count = 218`
  - 实际约 `10.9 Hz`
  - 这与 `cmd_write_ms ≈ 41 ms` 和发送线程等待叠加相符

## 2. 各段延时结果

以 [fullchain_alignment_20260331_151528.json](/c:/Users/Administrator/Desktop/工作学习/rimbot/机器人操控/diagnostics/fullchain_alignment_20260331_151528.json) 为准：

- `source -> 10.42.0.3 ROS2 回调`
  - `p50 = 36.049 ms`
  - `p95 = 41.034 ms`
  - `max = 47.129 ms`
- `10.42.0.3 ROS2 回调 -> 10.42.0.3 本地 rosbridge 客户端`
  - `p50 = 4.009 ms`
  - `p95 = 13.330 ms`
  - `max = 24.548 ms`
- `10.42.0.3 rosbridge -> 本机 roslibpy 到达`
  - `p50 = 20.271 ms`
  - `p95 = 41.746 ms`
  - `max = 84.606 ms`
- `本机到达 -> parse`
  - `p50 = 0.000 ms`
  - `p95 = 0.000 ms`
  - `max = 1.504 ms`
- `parse -> pose/NN`
  - `p50 = 0.000 ms`
  - `p95 = 1.009 ms`
  - `max = 2.000 ms`
- `本机到达 -> command_selected`
  - `p50 = 0.844 ms`
  - `p95 = 1.034 ms`
  - `max = 2.000 ms`
- `command_selected -> 发送线程取走`
  - `p50 = 28.588 ms`
  - `p95 = 66.237 ms`
  - `max = 89.024 ms`
- `串口写入耗时`
  - `p50 = 41.039 ms`
  - `p95 = 41.768 ms`
  - `max = 43.062 ms`
- `source -> COM10 写入完成`
  - `p50 = 138.733 ms`
  - `p95 = 182.989 ms`
  - `max = 199.811 ms`

## 3. 对延时问题的解释

- `latency-pipeline` 显示本机软件计算并不慢：
  - `parse_ms p95 = 0.000 ms`
  - `predict_ms p95 = 1.006 ms`
  - 说明本机 CPU 推理不是主要瓶颈
- 真正占时间的是后半段：
  - 跨机 `rosbridge -> 本机` 网络段中位数约 `20.271 ms`
  - 本机发送等待中位数约 `28.588 ms`
  - RS485 写入中位数约 `41.039 ms`
- 这三段合计已经约 `89.9 ms`
- 再加上源时间戳到远端主机回调约 `36.049 ms`
- 总体就落在 `138.733 ms` 中位量级

## 4. 关于“20Hz 控制”与实际发送频率

以 [latency_fullchain_20260331_151455.json](/c:/Users/Administrator/Desktop/工作学习/rimbot/机器人操控/diagnostics/latency_fullchain_20260331_151455.json) 为准：

- 20 秒共收到 `2402` 条消息
- 仅 `333` 条进入 accepted
- 最终真正写入手的只有 `218` 条
- `drop_rate = 86.14%`
- `replaced_before_send_count = 115`

这说明两件事：

- 上游消息频率远高于当前可发送频率，绝大多数帧在本机已经被节流掉
- 已被 accepted 的命令里，还有相当一部分在真正写串口之前又被后来的命令覆盖掉了

因此当前链路的“有效手部更新率”不是 20Hz，而是更接近 11Hz。

## 5. 稳定性结果

以 [stability_20260331_151322.json](/c:/Users/Administrator/Desktop/工作学习/rimbot/机器人操控/diagnostics/stability_20260331_151322.json) 为准：

- `early` 段 7126 帧
- `late` 段 3589 帧
- 六个槽位的 `sensor_delta.std = 0`
- 六个槽位的 `sensor_delta.range = 0`
- 六个槽位的 `predicted_pose.std = 0`

这意味着本轮静止测试期间：

- 映射主关节没有观测到漂移
- NN/映射输出没有放大漂移
- “前 1 分钟更抖”的现象在本轮没有复现

所以本轮不能用来证明 p0 现象仍然存在，只能说明：

- 在这次测试条件下，静止漂移没有出现
- 若现场仍有抖动，更可能与“实时控制运行初期、串口发送节奏、实际手部运动附近的小扰动”有关，而不是这次静态 90 秒里的传感器慢漂

## 6. 拇指映射审计

以 [thumb_audit_20260331_151258.json](/c:/Users/Administrator/Desktop/工作学习/rimbot/机器人操控/diagnostics/thumb_audit_20260331_151258.json) 为准：

- `thumb_cmc_pitch` 主关节：`joint_RightSkeletonThumbBase`
- `thumb_cmc_yaw` 主关节：`joint_RightSkeletonThumbBase`
- `shared_feature_ratio = 100%`
- `range_overlap_corr = -0.7865`

候选主轴排序显示：

- 对 `thumb_cmc_pitch` 更像主轴的是：
  - `joint_RightSkeletonThumb4`
  - `joint_RightSkeletonThumbBase`
  - `joint_RightSkeletonThumb2`
- 对 `thumb_cmc_yaw` 更像主轴的是：
  - `joint_RightSkeletonThumb3`
  - `joint_RightSkeletonThumbBase`
  - `joint_RightSkeletonThumb1`

这说明当前拇指 pitch / yaw 没有被拆成两个相对独立的驱动来源，p2 问题与这一点高度一致。

## 7. 执行器响应

以 [latency_actuator_20260331_151528.json](/c:/Users/Administrator/Desktop/工作学习/rimbot/机器人操控/diagnostics/latency_actuator_20260331_151528.json) 为准：

- `cmd_write_ms`
  - `mean = 41.122 ms`
  - `p50 = 41.113 ms`
  - `p95 = 41.470 ms`
- `reach50_ms`
  - `mean = 297.033 ms`
  - `p50 = 304.686 ms`
  - `p95 = 396.742 ms`
- `reach90_ms`
  - `mean = 479.838 ms`
  - `p50 = 578.741 ms`
  - `p95 = 579.519 ms`

最慢的一组主要集中在：

- `pinky_mcp_pitch open_to_mid = 579.894 ms`
- `middle_mcp_pitch mid_to_close = 579.212 ms`
- `index_mcp_pitch open_to_mid = 579.091 ms`
- `ring_mcp_pitch open_to_mid = 579.044 ms`

拇指侧摆明显更快：

- `thumb_cmc_yaw open_to_mid reach90 = 213.126 ms`
- `thumb_cmc_yaw mid_to_close reach90 = 122.027 ms`

说明：

- 不同关节机械响应差异明显
- 四指主弯曲槽位普遍慢于拇指 yaw
- `first_motion_ms = 0` 在本轮没有区分度，更像是采样粒度/读状态时序导致的观测结果，不建议把它作为本轮主要判断依据

## 8. 本轮最可信的诊断结论

- p1 延时问题已经量化：
  - 全链路中位延时约 `139 ms`
  - 主要由跨机网络、本机发送等待、RS485 写入三段组成
- p2 拇指映射问题已经量化：
  - 两个拇指槽位共用主关节
  - 候选主轴已经显示出 pitch / yaw 分离方向
- p0 启动抖动问题本轮未复现：
  - 本轮静态 90 秒数据不能证明存在启动漂移
  - 需要在“实际启控初期”重新采一次专门的 stability/fullchain 联合数据

## 9. 建议下一步

- 第一优先级：处理发送节奏
  - 当前真正写手频率只有约 `10.9 Hz`
  - 应优先检查发送线程调度策略和串口写入阻塞是否叠加导致有效频率减半
- 第二优先级：拇指重映射
  - `thumb_cmc_pitch` 优先评估 `joint_RightSkeletonThumb4`
  - `thumb_cmc_yaw` 优先评估 `joint_RightSkeletonThumb3`
- 第三优先级：复测 p0
  - 在“刚开始操控”的前 60 秒再跑一次联合诊断
  - 最好同时记录 `drive-hand=true` 的稳定性数据和 fullchain 数据

