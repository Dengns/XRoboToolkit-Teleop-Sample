# 全链路时间对齐报告

- 生成时间: 2026-03-31 15:04:44
- 本地报告: latency_fullchain_20260331_150412.json
- 远程报告: ros_host_bridge_probe_20260331_150410.json
- 估计时钟偏移: -222.696 ms
- 时钟采样最小 RTT: 54.267 ms
- 对齐到的样本数: 2385
- 成功写入 O6 的对齐样本数: 217

## 分段摘要

- source_to_ros_host_callback_ms: count=2385, mean=37.591 ms, p50=36.817 ms, p95=42.417 ms, max=67.989 ms
- ros_host_callback_to_remote_rosbridge_client_ms: count=2385, mean=5.577 ms, p50=4.019 ms, p95=13.588 ms, max=18.925 ms
- remote_rosbridge_to_local_arrival_ms: count=2385, mean=24.044 ms, p50=20.742 ms, p95=46.534 ms, max=101.597 ms
- local_arrival_to_parse_ms: count=2385, mean=0.012 ms, p50=0.000 ms, p95=0.000 ms, max=1.503 ms
- parse_to_pose_ms: count=2385, mean=0.404 ms, p50=0.000 ms, p95=1.197 ms, max=1.999 ms
- local_arrival_to_command_selected_ms: count=333, mean=0.514 ms, p50=0.503 ms, p95=1.290 ms, max=1.999 ms
- command_selected_to_send_thread_pick_ms: count=217, mean=31.083 ms, p50=31.432 ms, p95=62.983 ms, max=85.818 ms
- command_selected_to_write_start_ms: count=217, mean=31.083 ms, p50=31.432 ms, p95=62.983 ms, max=85.818 ms
- cmd_write_ms: count=217, mean=41.031 ms, p50=40.980 ms, p95=41.765 ms, max=42.742 ms
- source_to_cmd_write_end_aligned_ms: count=217, mean=144.822 ms, p50=141.746 ms, p95=191.338 ms, max=261.519 ms

## 备注

- 时钟偏移通过 SSH 往返采样估计，适合诊断量级，不等价于 PTP/NTP/Chrony 高精度同步。
- remote_rosbridge_to_local_arrival_ms 已应用偏移修正，表示 10.42.0.3 rosbridge 到本机 roslibpy 回调的近似单向耗时。
- source_to_cmd_write_end_aligned_ms 为从消息 source stamp 到 COM10 写入完成的近似全链路耗时。
