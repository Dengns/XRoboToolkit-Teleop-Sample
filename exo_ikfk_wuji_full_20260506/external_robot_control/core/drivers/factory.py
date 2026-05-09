from __future__ import annotations

"""手部驱动构造与释放的统一入口。"""

from core.profiles.hand_profiles import HandProfile
from core.runtime.paths import ensure_sdk_on_path

ensure_sdk_on_path()
from LinkerHand.linker_hand_api import LinkerHandApi  # type: ignore


def create_hand_driver(
    profile: HandProfile,
    *,
    modbus_port: str = 'None',
    can_channel: str = 'can0',
) -> LinkerHandApi:
    """按 hand profile 构造 SDK 驱动实例。"""
    # 调用方只关心“我要一只什么手”，串口 / CAN 细节在这里统一落到 SDK。
    return LinkerHandApi(
        hand_type=profile.hand_side,
        hand_joint=profile.hand_joint,
        modbus=modbus_port,
        can=can_channel,
    )


def close_hand_driver(api: object) -> None:
    """兼容释放 SDK 句柄，避免外层和底层驱动的关闭接口不一致。"""
    close_fn = getattr(api, 'close', None)
    if callable(close_fn):
        close_fn()
        return

    # LinkerHandApi 外层未必直接暴露 close，这里兜底探测其内部 hand 驱动。
    inner_driver = getattr(api, 'hand', None)
    inner_close = getattr(inner_driver, 'close', None)
    if callable(inner_close):
        inner_close()
