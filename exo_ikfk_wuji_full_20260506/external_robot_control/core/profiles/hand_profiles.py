from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HandSlotSpec:
    # finger_keyword / prefer_non_base 供向导、诊断和调试脚本复用。
    name: str
    label: str
    finger_keyword: str
    arc_max: float
    safe_open_value: int
    prefer_non_base: bool = True
    # 1 表示沿用默认 delta=base-cur；-1 表示反向使用 delta=cur-base。
    sensor_direction: int = 1


@dataclass(frozen=True)
class HandProfile:
    # 通过 profile 抽象不同手型的槽位顺序、物理弧度和安全开手位。
    hand_side: str
    hand_joint: str
    slot_specs: tuple[HandSlotSpec, ...]
    driver_kind: str
    expected_dof: int

    @property
    def slot_names(self) -> tuple[str, ...]:
        """驱动发送顺序对应的槽位名。"""
        return tuple(slot.name for slot in self.slot_specs)

    @property
    def slot_labels(self) -> dict[str, str]:
        """用于日志、UI 展示的中文槽位标签。"""
        return {slot.name: slot.label for slot in self.slot_specs}

    @property
    def slot_arc_max(self) -> tuple[float, ...]:
        """每个槽位的物理最大弧度限制。"""
        return tuple(slot.arc_max for slot in self.slot_specs)

    @property
    def safe_open_pose(self) -> tuple[int, ...]:
        """启动复位和断流保护共用的安全张开位。"""
        return tuple(slot.safe_open_value for slot in self.slot_specs)


O6_RIGHT_PROFILE = HandProfile(
    hand_side='right',
    hand_joint='O6',
    driver_kind='linkerhand_sdk',
    expected_dof=6,
    slot_specs=(
        HandSlotSpec('thumb_cmc_pitch', '拇指弯曲', 'Thumb', 0.58, 255, True),
        HandSlotSpec('thumb_cmc_yaw', '拇指侧摆', 'Thumb', 1.36, 70, True, -1),
        HandSlotSpec('index_mcp_pitch', '食指弯曲', 'Index', 1.6, 255, True),
        HandSlotSpec('middle_mcp_pitch', '中指弯曲', 'Middle', 1.6, 255, True),
        HandSlotSpec('ring_mcp_pitch', '无名指弯曲', 'Ring', 1.6, 255, True),
        HandSlotSpec('pinky_mcp_pitch', '小指弯曲', 'Pinky', 1.6, 255, True),
    ),
)

O6_LEFT_PROFILE = HandProfile(
    hand_side='left',
    hand_joint='O6',
    driver_kind='linkerhand_sdk',
    expected_dof=6,
    slot_specs=(
        HandSlotSpec('thumb_cmc_pitch', '拇指弯曲', 'Thumb', 0.58, 255, True),
        HandSlotSpec('thumb_cmc_yaw', '拇指侧摆', 'Thumb', 1.36, 179, False, -1),
        HandSlotSpec('index_mcp_pitch', '食指弯曲', 'Index', 1.6, 255, True),
        HandSlotSpec('middle_mcp_pitch', '中指弯曲', 'Middle', 1.6, 255, True),
        HandSlotSpec('ring_mcp_pitch', '无名指弯曲', 'Ring', 1.6, 255, True),
        HandSlotSpec('pinky_mcp_pitch', '小指弯曲', 'Pinky', 1.6, 255, True),
    ),
)

# O7 与 O6 的前 6 个驱动槽位保持一致，新增 thumb_cmc_roll。
# 当前安全位沿用 O6 的稳定张开姿态，并补一个来自 O7/L7 协议样例的拇指旋转张开位。
O7_RIGHT_PROFILE = HandProfile(
    hand_side='right',
    hand_joint='O7',
    driver_kind='linkerhand_sdk',
    expected_dof=7,
    slot_specs=(
        HandSlotSpec('thumb_cmc_pitch', '拇指弯曲', 'Thumb', 0.58, 255, True),
        # O7 现场验证显示拇指内外摆与当前映射相反，因此 yaw 槽位采用反向传感器符号。
        HandSlotSpec('thumb_cmc_yaw', '拇指侧摆', 'Thumb', 1.36, 70, False, -1),
        HandSlotSpec('index_mcp_pitch', '食指弯曲', 'Index', 1.6, 255, True),
        HandSlotSpec('middle_mcp_pitch', '中指弯曲', 'Middle', 1.6, 255, True),
        HandSlotSpec('ring_mcp_pitch', '无名指弯曲', 'Ring', 1.6, 255, True),
        HandSlotSpec('pinky_mcp_pitch', '小指弯曲', 'Pinky', 1.6, 255, True),
        HandSlotSpec('thumb_cmc_roll', '拇指旋转', 'Thumb', 1.54, 55, True),
    ),
)

O7_LEFT_PROFILE = HandProfile(
    hand_side='left',
    hand_joint='O7',
    driver_kind='linkerhand_sdk',
    expected_dof=7,
    slot_specs=(
        HandSlotSpec('thumb_cmc_pitch', '拇指弯曲', 'Thumb', 0.58, 255, True),
        HandSlotSpec('thumb_cmc_yaw', '拇指侧摆', 'Thumb', 1.36, 179, False, -1),
        HandSlotSpec('index_mcp_pitch', '食指弯曲', 'Index', 1.6, 255, True),
        HandSlotSpec('middle_mcp_pitch', '中指弯曲', 'Middle', 1.6, 255, True),
        HandSlotSpec('ring_mcp_pitch', '无名指弯曲', 'Ring', 1.6, 255, True),
        HandSlotSpec('pinky_mcp_pitch', '小指弯曲', 'Pinky', 1.6, 255, True),
        HandSlotSpec('thumb_cmc_roll', '拇指旋转', 'Thumb', 1.01, 55, True),
    ),
)

# 用注册表集中管理“手侧 + 型号 -> profile”，避免桥接、向导和诊断各自硬编码。
HAND_PROFILES: dict[tuple[str, str], HandProfile] = {
    (O6_RIGHT_PROFILE.hand_side, O6_RIGHT_PROFILE.hand_joint.upper()): O6_RIGHT_PROFILE,
    (O6_LEFT_PROFILE.hand_side, O6_LEFT_PROFILE.hand_joint.upper()): O6_LEFT_PROFILE,
    (O7_RIGHT_PROFILE.hand_side, O7_RIGHT_PROFILE.hand_joint.upper()): O7_RIGHT_PROFILE,
    (O7_LEFT_PROFILE.hand_side, O7_LEFT_PROFILE.hand_joint.upper()): O7_LEFT_PROFILE,
}

def get_hand_profile(hand_side: str = 'right', hand_joint: str = 'O6') -> HandProfile:
    # 所有入口统一从注册表取 profile，避免各处重复写死常量。
    key = (hand_side.lower(), hand_joint.upper())
    if key not in HAND_PROFILES:
        raise KeyError(f'未注册的 hand profile: side={hand_side} joint={hand_joint}')
    return HAND_PROFILES[key]
