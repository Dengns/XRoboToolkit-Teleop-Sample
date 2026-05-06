# `realman_contrl_steamvr_tracker.py` 快速使用说明

这份说明只讲最基本的用法。

## 1. 用哪个 conda 环境

```bash
conda activate ts_pico_teleop
cd /home/ubuntu/Shawn_tang_workspace/Pico/XRoboToolkit-Teleop-Sample-Python
```

## 2. 怎么启动

如果机械臂 IP 就是脚本默认值 `192.168.5.200`，直接运行：

```bash
python test/realman_contrl_steamvr_tracker.py
```

如果机械臂 IP 不是 `192.168.5.200`，手动写上：

```bash
python test/realman_contrl_steamvr_tracker.py --ip 你的机械臂IP
```

## 3. 怎么跟随

1. 先确认 SteamVR 里 tracker 在线。
2. 让这个终端窗口保持在当前焦点。
3. 把 tracker 放到你想开始跟随的位置。
4. 按住键盘空格键，开始跟随。
5. 挪动或转动 tracker，机械臂会跟着动。
6. 松开空格键，机械臂停止跟随。
7. 结束时按 `Ctrl+C` 退出。

补充一句：每次重新按住空格，脚本都会把“当前 tracker 位置”当成新的起点重新开始。

## 4. 最简单检查

启动后如果看到下面这些信息，说明基本正常：

- 出现 `当前使用终端按键模式`：说明脚本已经在监听空格键。
- 出现 `机械臂连接成功`：说明机械臂已经连上。
- 按住空格后出现 `tracker 控制已激活`：说明 tracker 数据已经进入控制。
- 松开空格后出现 `空格已松开，停止发送控制并清空 tracker 原点。`：说明停止正常。

如果看到这些提示，可以这样理解：

- `SteamVR 中未发现任何 GenericTracker`：SteamVR 没有识别到 tracker。
- `tracker 当前 pose 无效`：tracker 当前没有有效定位，等它恢复后再按住空格。
- `当前标准输入不是 TTY`：这个脚本要直接在终端里运行，不能用没有终端输入的方式启动。
