# ?????????

- ????: 2026-03-31 14:24:35
- ????: latency_fullchain_20260331_142210.json
- ????: ros_host_bridge_probe_20260331_142207.json
- ??????: -169.520 ms
- ?????? RTT: 53.730 ms
- ???????: 1427
- ???? O6 ??????: 131

## ????

- source_to_ros_host_callback_ms: count=1427, mean=39.562 ms, p50=38.860 ms, p95=43.979 ms, max=53.919 ms
- ros_host_callback_to_remote_rosbridge_client_ms: count=1425, mean=5.779 ms, p50=3.960 ms, p95=13.650 ms, max=25.938 ms
- remote_rosbridge_to_local_arrival_ms: count=1425, mean=19.211 ms, p50=16.252 ms, p95=39.984 ms, max=84.476 ms
- local_arrival_to_parse_ms: count=1427, mean=0.014 ms, p50=0.000 ms, p95=0.000 ms, max=1.216 ms
- parse_to_pose_ms: count=1427, mean=0.382 ms, p50=0.000 ms, p95=1.009 ms, max=2.562 ms
- local_arrival_to_command_selected_ms: count=199, mean=0.606 ms, p50=0.907 ms, p95=1.029 ms, max=2.562 ms
- command_selected_to_send_thread_pick_ms: count=131, mean=29.431 ms, p50=28.505 ms, p95=58.693 ms, max=69.839 ms
- command_selected_to_write_start_ms: count=131, mean=29.431 ms, p50=28.505 ms, p95=58.693 ms, max=69.839 ms
- cmd_write_ms: count=131, mean=41.141 ms, p50=41.049 ms, p95=42.080 ms, max=43.106 ms
- source_to_cmd_write_end_aligned_ms: count=131, mean=142.230 ms, p50=140.959 ms, p95=190.621 ms, max=235.670 ms

## ??

- ?????? SSH ?????????????????? PTP/NTP ??????
- remote_rosbridge_to_local_arrival_ms ?????????? 10.42.0.3 rosbridge ??? roslibpy ??????????
- source_to_cmd_write_end_aligned_ms ???? source stamp ? COM10 ?????????????