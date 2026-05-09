from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.profiles.hand_profiles import HandProfile
from core.runtime.paths import (
    calibration_read_path,
    calibration_write_path,
    generated_joint_map_read_path,
    generated_joint_map_write_path,
    model_read_path,
    model_write_path,
    training_read_path,
    training_write_path,
)


@dataclass
class BridgeAssets:
    """桥接启动阶段会一次性加载的运行资产。"""
    joint_map: dict[str, str]
    open_base: dict[str, float]
    use_nn: bool = False
    nn_models: dict[str, Any] = field(default_factory=dict)
    nn_finger_joints: dict[str, list[str]] = field(default_factory=dict)
    calibration_path: Path | None = None
    model_path: Path | None = None


def load_json(path: Path) -> dict[str, Any]:
    """统一按 UTF-8 读取 JSON 资产。"""
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def load_pickle(path: Path) -> dict[str, Any]:
    """统一读取训练阶段落盘的二进制模型。"""
    with path.open('rb') as f:
        return pickle.load(f)


def load_calibration_payload(required: bool = True) -> tuple[dict[str, Any], Path]:
    """加载校准文件，并返回最终生效的路径供上层打印或诊断。"""
    path = calibration_read_path()
    if not path.exists():
        if required:
            raise RuntimeError(f'找不到校准文件: {path}')
        return {}, path
    return load_json(path), path


def load_training_payload(required: bool = True) -> tuple[dict[str, Any], Path]:
    """加载采样训练数据。"""
    path = training_read_path()
    if not path.exists():
        if required:
            raise RuntimeError(f'找不到训练数据文件: {path}')
        return {}, path
    return load_json(path), path


def load_generated_joint_map_payload(required: bool = True) -> tuple[dict[str, Any], Path]:
    """加载向导生成的 joint_map Python 文件。"""
    path = generated_joint_map_read_path()
    if not path.exists():
        if required:
            raise RuntimeError(f'找不到 joint_map 输出文件: {path}')
        return {}, path
    namespace: dict[str, Any] = {}
    exec(path.read_text(encoding='utf-8'), namespace)
    return namespace.get('JOINT_MAP', {}), path


def load_bridge_assets(profile: HandProfile, load_model: bool = True) -> BridgeAssets:
    """
    聚合桥接运行依赖的校准、映射和可选 NN 模型。

    这里把“校准缺失”和“模型缺失”区别对待：
    校准缺失直接报错，模型缺失则允许桥接回退到线性模式继续运行。
    """
    calib, calib_path = load_calibration_payload(required=True)
    joint_map = calib.get('joint_map', {})
    open_base = calib.get('open_base', {})
    if not joint_map:
        raise RuntimeError('校准文件中 joint_map 为空，请重新运行 exo_mapping_wizard.py。')
    if not open_base:
        raise RuntimeError('校准文件中 open_base 为空，请重新运行 exo_mapping_wizard.py。')

    assets = BridgeAssets(
        joint_map=joint_map,
        open_base=open_base,
        calibration_path=calib_path,
    )

    if not load_model:
        return assets

    model_path = model_read_path()
    if not model_path.exists():
        # NN 只是增强项，不是桥接启动的硬前置条件。
        return assets

    data = load_pickle(model_path)
    assets.nn_models = data.get('models', {})
    assets.nn_finger_joints = data.get('finger_joints', {})
    model_base = data.get('open_base', {})
    if model_base:
        # 训练时若保存了更贴近模型输入分布的基线，这里覆盖到运行态。
        assets.open_base.update(model_base)
    assets.use_nn = bool(assets.nn_models)
    assets.model_path = model_path
    return assets


__all__ = [
    'BridgeAssets',
    'calibration_read_path',
    'calibration_write_path',
    'generated_joint_map_read_path',
    'generated_joint_map_write_path',
    'load_bridge_assets',
    'load_calibration_payload',
    'load_generated_joint_map_payload',
    'load_training_payload',
    'model_read_path',
    'model_write_path',
    'training_read_path',
    'training_write_path',
]
