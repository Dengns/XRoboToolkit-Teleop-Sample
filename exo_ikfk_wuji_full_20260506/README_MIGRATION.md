# 外骨骼 IK/FK 到 WuJi 左手迁移包

生成日期：2026-05-06

本目录是把“外骨骼 skeleton -> MuJoCo IK/FK -> WuJi 左手 5x4 rad 目标矩阵”链路迁移到另一台电脑的完整包。

## 1. 目录结构

```text
exo_ikfk_wuji_full_20260506/
├─ fangzhenyingshe/              # 当前 IK/FK 与 WuJi 实时遥操作主仓库
│  ├─ scripts/teleop_exo_to_wuji_left.py
│  ├─ scripts/compare_hand_3d.py
│  ├─ scripts/compare_hand_3d_live.py
│  ├─ scripts/delivery_core/
│  ├─ scripts/hand_mapping_params.py
│  ├─ third_party/io_mocap_description/
│  ├─ outputs/exo_wuji_open_baseline.json
│  └─ tests/test_teleop_wuji_target.py
├─ external_robot_control/       # 外骨骼实时数据桥接依赖
│  ├─ core/
│  ├─ runtime/calibration/exo_calibration.json
│  ├─ linkerhand-python-sdk/LinkerHand/
│  └─ wujihand_test.py 等诊断脚本
├─ requirements/
│  ├─ requirements-teleop.txt
│  └─ requirements-extra-linkerhand.txt
└─ docs/
```

## 2. 代码证据与运行闭包

- 主入口：`fangzhenyingshe/scripts/teleop_exo_to_wuji_left.py`
- 主入口静态导入：`compare_hand_3d.py`、`compare_hand_3d_live.py`、`delivery_core.sim_loader`、`delivery_core.thumb_mapping`
- MuJoCo 模型默认资产：`fangzhenyingshe/third_party/io_mocap_description/blender_human_skeleton_v4.urdf`
- 实时外骨骼动态导入：`roslibpy`、`core.mapping.exo_mapping`、`core.runtime.assets`、`core.runtime.site_config`
- 外骨骼 open_base 校准文件：`external_robot_control/runtime/calibration/exo_calibration.json`
- WuJi 实机 SDK：`wujihandpy`
- WuJi 张开基线：`fangzhenyingshe/outputs/exo_wuji_open_baseline.json`
- 默认外部仓库路径：`fangzhenyingshe/scripts/compare_hand_3d_live.py` 现已默认指向本迁移包内的 `../external_robot_control`，Linux/Windows 迁移后不再依赖原 Windows 现场机器路径。

## 3. 目标机器环境

推荐 Python 3.11。先在迁移包根目录创建虚拟环境：

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements\requirements-teleop.txt
```

Linux:

```bash
python3.11 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements/requirements-teleop.txt
```

如果还要运行 `external_robot_control/` 里的 LinkerHand/O6/O7 旧桥接或诊断脚本，再安装：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements\requirements-extra-linkerhand.txt
```

Linux 对应：

```bash
./.venv/bin/python -m pip install -r requirements/requirements-extra-linkerhand.txt
```

## 4. 硬件与网络前置条件

1. 目标机器能访问 rosbridge：默认 `10.42.0.3:9090`。
2. rosbridge 发布 topic：`/mocap/skeleton_data`，类型 `io_msgs2/SquashedSkeletonData`。
3. WuJi 手已连接，`wujihandpy.Hand()` 能正常初始化。
4. Windows 下确认 WuJi USB 驱动正常。
5. Linux 下确认当前用户有 WuJi USB 设备权限。若 SDK 使用串口或 USB 设备，通常需要把用户加入对应设备组，或临时调整设备权限。

外骨骼 rosbridge 配置在：

```text
external_robot_control/core/runtime/site_config.py
```

如现场 IP 或 topic 变化，修改此文件。

## 5. 验证步骤

在迁移包根目录执行。

Windows PowerShell:

```powershell
$PY = ".\.venv\Scripts\python.exe"
& $PY -X utf8 -m unittest discover -s fangzhenyingshe\tests
& $PY -X utf8 -c "import numpy, scipy, mujoco, matplotlib, roslibpy, wujihandpy; print('imports ok')"
```

Linux:

```bash
PY=./.venv/bin/python
$PY -X utf8 -m unittest discover -s fangzhenyingshe/tests
$PY -X utf8 -c "import numpy, scipy, mujoco, matplotlib, roslibpy, wujihandpy; print('imports ok')"
```

## 6. 干跑外骨骼到 IK/FK

干跑会连接外骨骼 skeleton，只计算并打印一帧 WuJi 目标，不连接 WuJi 实机。

Windows PowerShell:

```powershell
$PY = "..\.venv\Scripts\python.exe"
Set-Location .\fangzhenyingshe
& $PY -X utf8 scripts\teleop_exo_to_wuji_left.py --dry-run --profile
```

Linux:

```bash
cd fangzhenyingshe
../.venv/bin/python -X utf8 scripts/teleop_exo_to_wuji_left.py --dry-run --profile
```

如果你把 `external_robot_control/` 挪到了别的位置，再显式传：

```bash
python -X utf8 scripts/teleop_exo_to_wuji_left.py --dry-run --profile --external-repo <你的路径>
```

## 7. 重新标定 WuJi 张开基线

如果换了外骨骼佩戴姿态、WuJi 逻辑零位或现场手势基准，先把外骨骼和 WuJi 逻辑手张开，然后运行：

Windows PowerShell:

```powershell
$PY = "..\.venv\Scripts\python.exe"
Set-Location .\fangzhenyingshe
& $PY -X utf8 scripts\teleop_exo_to_wuji_left.py --calibrate
```

Linux:

```bash
cd fangzhenyingshe
../.venv/bin/python -X utf8 scripts/teleop_exo_to_wuji_left.py --calibrate
```

该命令会更新：

```text
fangzhenyingshe/outputs/exo_wuji_open_baseline.json
```

## 8. 正式运行

Windows PowerShell:

```powershell
$PY = "..\.venv\Scripts\python.exe"
Set-Location .\fangzhenyingshe
& $PY -X utf8 scripts\teleop_exo_to_wuji_left.py --send-mode unchecked --profile
```

Linux:

```bash
cd fangzhenyingshe
../.venv/bin/python -X utf8 scripts/teleop_exo_to_wuji_left.py --send-mode unchecked --profile
```

停止时按 `Ctrl+C`。脚本会尝试 WuJi 回零并失能。

## 9. 常见问题

### 找不到外部项目

默认情况下脚本会直接查找本迁移包内的 `external_robot_control`。如果你手动改过目录结构，报错类似“外部项目目录不存在”时，再检查 `--external-repo` 是否指向正确位置。

### 找不到 open_base

报错类似“open_base 为空”时，检查：

```text
external_robot_control/runtime/calibration/exo_calibration.json
```

如果现场需要重新做外骨骼校准，需要回到外部项目流程重新生成该文件。

### 导入 wujihandpy 失败

说明目标虚拟环境没有安装 WuJi SDK，或该 SDK 不支持当前操作系统/Python 版本。先运行：

```powershell
python -m pip install wujihandpy
```

再测试：

```powershell
python -X utf8 -c "import wujihandpy; print(wujihandpy)"
```

### 连接 rosbridge 失败

确认目标机器能访问 `10.42.0.3:9090`，且现场 rosbridge 已启动。若 IP 不同，修改：

```text
external_robot_control/core/runtime/site_config.py
```

### MuJoCo URDF 或 STL 加载失败

确认 `fangzhenyingshe/third_party/io_mocap_description/` 完整存在。该目录已随包复制，正常不需要额外下载。

## 10. 文件取舍说明

实时遥操作链路不依赖训练出的 MLP 模型，因此本包没有复制 `outputs/hand_mapping_models/`、`outputs/clean_hand_mapping/`、`outputs/live_hand_mapping_record.csv` 等采样/训练产物。

如果后续要在新机器上继续采样、清洗和训练，可从原仓库额外复制完整 `outputs/`，或重新运行采样训练流程。
