# `realman_contrl_steamvr_tracker.py` 快速使用说明


## 1.  conda 环境

```bash
conda activate ts_pico_teleop
cd /home/ubuntu/Shawn_tang_workspace/Pico/XRoboToolkit-Teleop-Sample-Python
```

## 2. 启动

机械臂目前有线链接绑定 `192.168.5.200`，直接运行：

```bash
python test/realman_contrl_steamvr_tracker.py
```

如果机械臂 IP 不是 `192.168.5.200`，
网线链接机械臂，电脑ip改为手动分配，分配为192.168.5.xx然后进入`192.168.5.200`进行查看ip手动写上：

```bash
python test/realman_contrl_steamvr_tracker.py --ip 你的机械臂IP
```

## 3. 怎么跟随

1. 打开steam确认 SteamVR 里 tracker 在线。
2. 让这个终端窗口保持在当前焦点。
3. 把 tracker 放到你想开始跟随的位置。标签向前
4. 按一次键盘空格键，开始跟随。
5. 挪动或转动 tracker，机械臂会跟着动。
6. 再按一次空格键，机械臂停止跟随。
7. 跟随过程中按键盘 `↑` / `↓`，可以即时增大或减小 `xyz scale`，不需要回车。
8. 跟随过程中按数字键 `1-9`，可以直接切到预设 `xyz scale` 档位，不需要回车：
   - `1=0.125`
   - `2=0.25`
   - `3=0.5`
   - `4=0.75`
   - `5=1.0`
   - `6=1.5`
   - `7=2.0`
   - `8=4.0`
   - `9=8.0`
9. 结束时按 `Ctrl+C` 退出。


## 4. 最简单检查

启动后如果看到下面这些信息，说明基本正常：

- 出现 `当前使用终端按键模式`：说明脚本已经在监听空格、方向键和数字键。
- 出现 `机械臂连接成功`：说明机械臂已经连上。
- 按一次空格后出现 `空格触发：启动 SteamVR Tracker 遥操。`：说明已进入可跟随状态。
- 启动后出现 `tracker 控制已激活`：说明 tracker 数据已经进入控制。
- 再按一次空格后出现 `已停止发送控制并清空 tracker 原点。`：说明停止正常。
- 按 `↑` / `↓` 或 `1-9` 后出现 `tracker xyz scale 调整为 ...`：说明比例调整已经即时生效。

如果看到这些提示，可以这样理解：

- `SteamVR 中未发现任何 GenericTracker`：SteamVR 没有识别到 tracker。
- `tracker 当前 pose 无效`：tracker 当前没有有效定位，等它恢复后再重新开始或继续启用跟随。
- `当前标准输入不是 TTY`：这个脚本要直接在终端里运行，不能用没有终端输入的方式启动。
