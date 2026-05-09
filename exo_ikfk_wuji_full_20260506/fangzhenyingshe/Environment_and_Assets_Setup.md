# 交付版环境与运行说明

更新时间：2026-04-30

## 1. 目标

当前仓库保留的交付链路只有 4 类能力：

- 全手 3D 预览与手动参数映射
- 实时数据采集
- 离线批量采样
- 数据清洗与训练集导出

已删除的单指调试、单部位验证、最小派生模型和 LEAP 旁支资产不再作为交付版内容。

## 2. 保留目录

- `scripts/compare_hand_3d.py`
- `scripts/compare_hand_3d_live.py`
- `scripts/collect_hand_mapping_dataset.py`
- `scripts/prepare_clean_hand_mapping_dataset.py`
- `scripts/hand_mapping_params.py`
- `scripts/core/`
- `third_party/io_mocap_description/`

## 3. Python 环境

当前项目既有可运行环境记录为：

```powershell
C:\Users\Administrator\Desktop\工作学习\rimbot\GEN3标准多维触觉传感器产品资料包_20251202\.venv\Scripts\python.exe
```

建议所有涉及 MuJoCo 的运行都使用：

```powershell
& 'C:\Users\Administrator\Desktop\工作学习\rimbot\GEN3标准多维触觉传感器产品资料包_20251202\.venv\Scripts\python.exe' -X utf8 <script>
```

## 4. 依赖说明

交付链路依赖以下本地能力：

- `mujoco`
- `numpy`
- `scipy`
- `matplotlib`
- `lxml`
- `PyYAML`

实时采集额外依赖外部项目：

```powershell
C:\Users\Administrator\Desktop\工作学习\rimbot\机器人操控
```

`compare_hand_3d_live.py` 会从该项目读取：

- rosbridge 连接信息
- `open_base` 校准基线
- skeleton 解析逻辑

## 5. 全手 3D 预览

交互预览：

```powershell
& 'C:\Users\Administrator\Desktop\工作学习\rimbot\GEN3标准多维触觉传感器产品资料包_20251202\.venv\Scripts\python.exe' -X utf8 scripts\compare_hand_3d.py
```

无界面摘要：

```powershell
& 'C:\Users\Administrator\Desktop\工作学习\rimbot\GEN3标准多维触觉传感器产品资料包_20251202\.venv\Scripts\python.exe' -X utf8 scripts\compare_hand_3d.py --headless
```

说明：

- 默认加载 `third_party/io_mocap_description/blender_human_skeleton_v4.urdf`
- 默认只保留右手可视化重点
- 四指和拇指默认参数统一来自 `scripts/hand_mapping_params.py`
- 交互界面支持切换当前手指、手动调整长度、根部偏移和末端偏移

## 6. 实时数据采集

启动实时预览与录制：

```powershell
& 'C:\Users\Administrator\Desktop\工作学习\rimbot\GEN3标准多维触觉传感器产品资料包_20251202\.venv\Scripts\python.exe' -X utf8 scripts\compare_hand_3d_live.py
```

仅查看参数说明：

```powershell
& 'C:\Users\Administrator\Desktop\工作学习\rimbot\GEN3标准多维触觉传感器产品资料包_20251202\.venv\Scripts\python.exe' -X utf8 scripts\compare_hand_3d_live.py --help
```

说明：

- 默认从外部 `机器人操控` 项目读取实时 skeleton
- 默认录制输出为 `outputs/live_hand_mapping_record.csv`
- 录制粒度为“单指样本”，不是整手整帧
- 支持普通样本录制和 calibration 均值样本录制

## 7. 离线批量采样

无界面快速采样：

```powershell
& 'C:\Users\Administrator\Desktop\工作学习\rimbot\GEN3标准多维触觉传感器产品资料包_20251202\.venv\Scripts\python.exe' -X utf8 scripts\collect_hand_mapping_dataset.py --headless --fingers all --valid-per-finger 1
```

正常交互采样：

```powershell
& 'C:\Users\Administrator\Desktop\工作学习\rimbot\GEN3标准多维触觉传感器产品资料包_20251202\.venv\Scripts\python.exe' -X utf8 scripts\collect_hand_mapping_dataset.py --fingers all --valid-per-finger 1000 --output-prefix outputs\hand_mapping_dataset
```

说明：

- 该脚本保留为交付版中的离线补充能力
- 输出文件为：
  - `outputs/hand_mapping_dataset_valid.csv`
  - `outputs/hand_mapping_dataset_rejected.jsonl`
- 若只需要真实链路交付，可不使用该脚本

## 8. 数据清洗

对实时录制结果做清洗：

```powershell
& 'C:\Users\Administrator\Desktop\工作学习\rimbot\GEN3标准多维触觉传感器产品资料包_20251202\.venv\Scripts\python.exe' -X utf8 scripts\prepare_clean_hand_mapping_dataset.py
```

输出目录默认是：

```powershell
outputs/clean_hand_mapping
```

关键输出包括：

- `clean_hand_mapping_dataset.csv`
- `baseline_summary.csv`
- `preprocess_report.json`
- `per_finger/index_training.csv`
- `per_finger/middle_training.csv`
- `per_finger/ring_training.csv`
- `per_finger/pinky_training.csv`
- `per_finger/thumb_training.csv`

## 9. 输出保留策略

`outputs/` 继续保持 git ignore。

交付版本地建议保留：

- `outputs/live_hand_mapping_record.csv`
- `outputs/clean_hand_mapping/**`

交付版本地建议删除的批量调试/中间产物：

- `outputs/hand_mapping_dataset_valid.csv`
- `outputs/hand_mapping_dataset_rejected.jsonl`
- 其他临时 smoke 数据

## 10. 验证建议

最小验证顺序：

1. `scripts/compare_hand_3d.py --headless`
2. `scripts/compare_hand_3d_live.py --help`
3. `scripts/collect_hand_mapping_dataset.py --headless --fingers all --valid-per-finger 1`
4. `scripts/prepare_clean_hand_mapping_dataset.py`
