"""固定现场配置真源。

这里存的是当前现场已经验证过的 rosbridge 与外骨骼话题配置。
桥接、诊断、调试和测试脚本都应该从这里读，避免多份硬编码飘散。
"""

# 现场固定常量统一收口在这里，不额外开放配置覆盖。
ROSBRIDGE_HOST = '10.42.0.3'
ROSBRIDGE_PORT = 9090
SKELETON_TOPIC = '/mocap/skeleton_data'
SKELETON_TOPIC_TYPE = 'io_msgs2/SquashedSkeletonData'
