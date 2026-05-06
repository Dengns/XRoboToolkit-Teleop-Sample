# SteamVR RM75 迁移包说明

## 1. 这个文件夹里有什么

这个目录是从主仓库里抽出来的最小可运行链路，只保留 `SteamVR/OpenVR -> Vive Tracker -> RealMan RM75-B` 这条控制链路需要的文件：

- `test/realman_contrl_steamvr_tracker.py`
- `test/realman_contrl_steamvr_tracker_usage.md`
- `xrobotoolkit_teleop/hardware/interface/rm75b.py`
- `xrobotoolkit_teleop/__init__.py`
- `xrobotoolkit_teleop/hardware/__init__.py`
- `xrobotoolkit_teleop/hardware/interface/__init__.py`
- `environment.yml`
- `requirements.txt`
- `install_conda_env.sh`

这个迁移包不包含 Pico SDK、MuJoCo、MeshCat、Realsense、R5 等其他链路依赖。

## 2. 新服务器需要提前具备什么

在开始安装 Python 环境前，请先确认：

1. 服务器已经安装 `conda`。
2. 服务器可以联网访问 `pip` / `conda` 软件源。
3. 服务器已经安装并能正常打开 `Steam` 与 `SteamVR`。
4. Vive Tracker 已在 `SteamVR` 里显示在线，并且 pose 有效。
5. 服务器网卡已经和机械臂控制器在同一网段，例如 `192.168.5.xx`。
6. 你会通过真实交互终端运行脚本，例如本地终端或 SSH 终端。

注意：这个脚本依赖终端按住 `Space` 键控制，所以不能用没有 TTY 输入的后台方式启动。

## 3. 复制到新服务器后的推荐目录

把整个 `steamvr_rm75_bundle` 文件夹原样复制到新服务器，例如：

```bash
~/steamvr_rm75_bundle
```

后续命令都假设你已经进入这个目录。

## 4. 创建 conda 环境

推荐直接运行：

```bash
bash install_conda_env.sh
```

默认会创建或更新名为 `ts_pico_teleop` 的 conda 环境。

如果你想手动安装，也可以使用：

```bash
conda env create -f environment.yml
conda activate ts_pico_teleop
pip install -r requirements.txt
```

如果环境已经存在，可以改用：

```bash
conda env update -f environment.yml --prune
conda activate ts_pico_teleop
pip install -r requirements.txt
```

## 5. 启动脚本

进入迁移包目录后：

```bash
conda activate ts_pico_teleop
python test/realman_contrl_steamvr_tracker.py
```

如果机械臂 IP 不是默认的 `192.168.5.200`：

```bash
python test/realman_contrl_steamvr_tracker.py --ip 你的机械臂IP
```

## 6. 运行时最小操作说明

1. 打开 `SteamVR`，确认 tracker 在线。
2. 保持当前终端窗口为焦点。
3. 把 tracker 放到你准备开始跟随的位置。
4. 按住键盘 `Space`，开始控制。
5. 移动或旋转 tracker，机械臂会跟随。
6. 松开 `Space`，机械臂缓停并清空本轮控制原点。
7. 按 `Ctrl+C` 退出脚本。

## 7. 启动成功时应该看到什么

如果链路正常，通常会看到这些日志：

- `当前使用终端按键模式`
- `机械臂连接成功`
- `tracker 控制已激活`
- `空格已松开，停止发送控制并清空 tracker 原点。`

## 8. 常见问题

### 8.1 `SteamVR 中未发现任何 GenericTracker`

说明 SteamVR 没识别到 tracker，先去 SteamVR 设备界面检查设备是否真的在线。

### 8.2 `tracker 当前 pose 无效`

说明 tracker 虽然在线，但当前没有有效定位。一般要先让基站视野恢复正常，再重新按住 `Space`。

### 8.3 `当前标准输入不是 TTY`

说明当前启动方式没有真实终端输入。请改用直接终端或 SSH 交互终端运行。

### 8.4 连得上 SteamVR，但连不上机械臂

先检查：

- 服务器 IP 是否与机械臂在同一网段
- 能否 `ping` 通机械臂 IP
- 启动命令里的 `--ip` 是否写对

## 9. 当前迁移包固定的关键版本

为了尽量复刻当前工作机，本迁移包使用下面这些版本：

- Python `3.13`
- `numpy==2.4.1`
- `openvr==2.12.1401`
- `Robotic_Arm==1.1.4`

如果后续你想严格与当前机器保持一致，优先不要改这些版本。
