# ?????????

- ????: 2026-03-31 14:21:26
- ????: latency_fullchain_20260331_141808.json
- ????: ros_host_bridge_probe_20260331_141745.json
- ??????: -165.388 ms (min RTT=58.173 ms)
- ????: 0??????? O6: 0

## ??????

- source_to_ros_host_callback_ms: count=0, mean=0.000 ms, p50=0.000 ms, p95=0.000 ms, max=0.000 ms
- ros_host_callback_to_remote_rosbridge_client_ms: count=0, mean=0.000 ms, p50=0.000 ms, p95=0.000 ms, max=0.000 ms
- remote_rosbridge_to_local_arrival_ms: count=0, mean=0.000 ms, p50=0.000 ms, p95=0.000 ms, max=0.000 ms
- local_arrival_to_parse_ms: count=0, mean=0.000 ms, p50=0.000 ms, p95=0.000 ms, max=0.000 ms
- parse_to_pose_ms: count=0, mean=0.000 ms, p50=0.000 ms, p95=0.000 ms, max=0.000 ms
- local_arrival_to_command_selected_ms: count=0, mean=0.000 ms, p50=0.000 ms, p95=0.000 ms, max=0.000 ms
- command_selected_to_send_thread_pick_ms: count=0, mean=0.000 ms, p50=0.000 ms, p95=0.000 ms, max=0.000 ms
- command_selected_to_write_start_ms: count=0, mean=0.000 ms, p50=0.000 ms, p95=0.000 ms, max=0.000 ms
- cmd_write_ms: count=0, mean=0.000 ms, p50=0.000 ms, p95=0.000 ms, max=0.000 ms
- source_to_cmd_write_end_aligned_ms: count=0, mean=0.000 ms, p50=0.000 ms, p95=0.000 ms, max=0.000 ms

## ??

- remote_rosbridge_to_local_arrival_ms ??? SSH ????????????????????? SSH ???????
- source_to_cmd_write_end_aligned_ms ??? sent_to_hand=True ????
- ???????? 20Hz ????? 20Hz ?????fullchain ???????????