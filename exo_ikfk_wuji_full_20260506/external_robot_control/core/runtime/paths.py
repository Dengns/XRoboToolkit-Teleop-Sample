from __future__ import annotations

import sys
from shutil import copy2
from pathlib import Path


# 运行产物统一收口到 runtime/，避免校准、训练和模型文件长期散落在仓库根目录。
REPO_ROOT = Path(__file__).resolve().parents[2]
SDK_ROOT = REPO_ROOT / 'linkerhand-python-sdk'
RUNTIME_ROOT = REPO_ROOT / 'runtime'
CALIBRATION_DIR = RUNTIME_ROOT / 'calibration'
TRAINING_DIR = RUNTIME_ROOT / 'training'
MODELS_DIR = RUNTIME_ROOT / 'models'
GENERATED_DIR = RUNTIME_ROOT / 'generated'
DIAGNOSTICS_DIR = REPO_ROOT / 'diagnostics'


def ensure_repo_on_path() -> None:
    """保证项目模块可以从仓库根目录稳定导入。"""
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def ensure_sdk_on_path() -> None:
    # 第三方 SDK 保留在仓库目录内，统一在这里补 sys.path。
    ensure_repo_on_path()
    sdk = str(SDK_ROOT)
    if sdk not in sys.path:
        sys.path.insert(0, sdk)


def ensure_runtime_dirs() -> None:
    """按需创建运行时目录，减少上层重复判断目录是否存在。"""
    for path in (CALIBRATION_DIR, TRAINING_DIR, MODELS_DIR, GENERATED_DIR, DIAGNOSTICS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def resolve_runtime_path(runtime_path: Path, legacy_path: Path) -> Path:
    """
    统一运行产物真源路径。

    若 runtime 中已存在目标文件，直接返回新路径；
    若仅根目录旧文件存在，则自动复制到 runtime 并返回新路径；
    若两边都不存在，也返回 runtime 目标路径，便于上层直接按新结构报错或写入。
    """
    ensure_runtime_dirs()
    if runtime_path.exists():
        return runtime_path
    if legacy_path.exists():
        copy2(legacy_path, runtime_path)
        return runtime_path
    return runtime_path


def calibration_write_path() -> Path:
    """校准文件唯一写入路径。"""
    ensure_runtime_dirs()
    return CALIBRATION_DIR / 'exo_calibration.json'


def calibration_read_path() -> Path:
    """校准文件读取路径，兼容旧根目录文件但始终收口到 runtime/。"""
    return resolve_runtime_path(calibration_write_path(), REPO_ROOT / 'exo_calibration.json')


def training_write_path() -> Path:
    """训练数据唯一写入路径。"""
    ensure_runtime_dirs()
    return TRAINING_DIR / 'exo_training_data.json'


def training_read_path() -> Path:
    """训练数据读取路径，必要时自动迁移旧文件。"""
    return resolve_runtime_path(training_write_path(), REPO_ROOT / 'exo_training_data.json')


def model_write_path() -> Path:
    """NN 模型唯一写入路径。"""
    ensure_runtime_dirs()
    return MODELS_DIR / 'exo_nn_model.pkl'


def model_read_path() -> Path:
    """NN 模型读取路径，必要时自动迁移旧文件。"""
    return resolve_runtime_path(model_write_path(), REPO_ROOT / 'exo_nn_model.pkl')


def generated_joint_map_write_path() -> Path:
    """joint_map 生成文件唯一写入路径。"""
    ensure_runtime_dirs()
    return GENERATED_DIR / 'joint_map_output.py'


def generated_joint_map_read_path() -> Path:
    """joint_map 生成文件读取路径，和其他 runtime 资产保持同一迁移策略。"""
    return resolve_runtime_path(generated_joint_map_write_path(), REPO_ROOT / 'joint_map_output.py')
