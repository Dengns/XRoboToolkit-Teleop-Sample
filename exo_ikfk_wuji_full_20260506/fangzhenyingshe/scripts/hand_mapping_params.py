#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""统一保存手部映射默认参数。"""

from __future__ import annotations

from dataclasses import dataclass


FOUR_FINGER_ORDER = ("index", "middle", "ring", "pinky")


@dataclass(frozen=True)
class FourFingerMappingParams:
    """四指默认参数。

    说明：
    - `base_mm` 使用用户输入语义的原始 `(x, y)`。
    - 真正进入四指映射时，会再按 `(x, y) -> (y, -x)` 转成内部 `base_offset`。
    """

    label: str
    lengths_mm: tuple[float, float, float]
    base_mm: tuple[float, float]
    tip_mm: tuple[float, float]


@dataclass(frozen=True)
class ThumbMappingParams:
    """拇指默认参数。"""

    lengths_mm: tuple[float, float, float]
    base_mm: tuple[float, float, float]
    tip_mm: tuple[float, float]


FOUR_FINGER_MAPPING_PARAMS: dict[str, FourFingerMappingParams] = {
    "index": FourFingerMappingParams(
        label="食指",
        lengths_mm=(35.0, 23.0, 14.0),
        base_mm=(35.0, 18.0),
        tip_mm=(-10.0, 0.0),
    ),
    "middle": FourFingerMappingParams(
        label="中指",
        lengths_mm=(47.0, 25.0, 15.0),
        base_mm=(35.0, 20.0),
        tip_mm=(-10.0, 0.0),
    ),
    "ring": FourFingerMappingParams(
        label="无名指",
        lengths_mm=(45.0, 20.0, 15.0),
        base_mm=(41.0, 20.0),
        tip_mm=(-10.0, 0.0),
    ),
    "pinky": FourFingerMappingParams(
        label="小指",
        lengths_mm=(30.0, 17.0, 14.0),
        base_mm=(48.0, 12.0),
        tip_mm=(-10.0, 0.0),
    ),
}

INDEX_MAPPING_PARAMS = FOUR_FINGER_MAPPING_PARAMS["index"]
MIDDLE_MAPPING_PARAMS = FOUR_FINGER_MAPPING_PARAMS["middle"]
RING_MAPPING_PARAMS = FOUR_FINGER_MAPPING_PARAMS["ring"]
PINKY_MAPPING_PARAMS = FOUR_FINGER_MAPPING_PARAMS["pinky"]

THUMB_MAPPING_PARAMS = ThumbMappingParams(
    lengths_mm=(40.0, 30.0, 23.0),
    base_mm=(-15.0, -20.0, -35.0),
    tip_mm=(-10.0, 0.0),
)


def get_four_finger_params(finger_key: str) -> FourFingerMappingParams:
    """返回指定四指的统一默认参数。"""
    if finger_key not in FOUR_FINGER_MAPPING_PARAMS:
        raise KeyError(f"未知四指 key：{finger_key}")
    return FOUR_FINGER_MAPPING_PARAMS[finger_key]


def get_thumb_params() -> ThumbMappingParams:
    """返回拇指的统一默认参数。"""
    return THUMB_MAPPING_PARAMS


def default_baseline_summary_lines() -> tuple[str, ...]:
    """返回当前默认基线的摘要文本。"""
    lines: list[str] = []
    for finger_key in FOUR_FINGER_ORDER:
        params = get_four_finger_params(finger_key)
        l1, l2, l3 = params.lengths_mm
        base_x, base_y = params.base_mm
        lines.append(f"{params.label} {l1:.0f}/{l2:.0f}/{l3:.0f} base {base_x:.0f}/{base_y:.0f}")

    tip_dx, tip_dy = INDEX_MAPPING_PARAMS.tip_mm
    lines.append(f"tip offset 默认 ({tip_dx:.0f},{tip_dy:.0f}) mm")
    return tuple(lines)
