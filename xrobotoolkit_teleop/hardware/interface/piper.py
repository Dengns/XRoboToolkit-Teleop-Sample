import time
from typing import List, Union

import numpy as np

from piper_sdk import C_PiperInterface_V2

# Conversion factor: radians → millidegs (0.001 degrees)
# 1000 * 180 / pi = 57295.7795
RAD_TO_MILLIDEGS = 57295.7795


class PiperInterface:
    """Hardware interface for the Piper 6-DOF robotic arm via CAN bus (piper_sdk).

    Reference: xbox_arm_controller_dangerous.py (verified working)
    """

    def __init__(self, can_port: str = "can0", auto_enable: bool = True):
        self.can_port = can_port
        self.piper = C_PiperInterface_V2(can_port)
        self.piper.ConnectPort()
        time.sleep(0.1)

        # Set control mode first (MOVE_J for joint control)
        self.piper.MotionCtrl_2(0x01, 0x01, 100, 0x00)
        time.sleep(0.1)

        if auto_enable:
            print(f"Enabling Piper arm on {can_port}...")
            self._enable_with_timeout(timeout=5.0)
            print("Piper arm enabled.")

        # Enable gripper
        self.piper.GripperCtrl(0, 1000, 0x01, 0)
        time.sleep(0.1)

    def _enable_with_timeout(self, timeout: float = 5.0):
        """Enable arm with timeout, matching the pattern from xbox controller."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            self.piper.EnableArm(7)
            low_spd_info = self.piper.GetArmLowSpdInfoMsgs()
            enable_flag = (
                low_spd_info.motor_1.foc_status.driver_enable_status
                and low_spd_info.motor_2.foc_status.driver_enable_status
                and low_spd_info.motor_3.foc_status.driver_enable_status
                and low_spd_info.motor_4.foc_status.driver_enable_status
                and low_spd_info.motor_5.foc_status.driver_enable_status
                and low_spd_info.motor_6.foc_status.driver_enable_status
            )
            if enable_flag:
                return True
            time.sleep(0.5)
        raise RuntimeError(f"Piper arm enable timeout on {self.can_port}")

    def get_joint_positions(self) -> np.ndarray:
        """Returns current joint positions in radians. Shape: (6,)"""
        joint = self.piper.GetArmJointMsgs()
        return np.array([
            joint.joint_state.joint_1 / RAD_TO_MILLIDEGS,
            joint.joint_state.joint_2 / RAD_TO_MILLIDEGS,
            joint.joint_state.joint_3 / RAD_TO_MILLIDEGS,
            joint.joint_state.joint_4 / RAD_TO_MILLIDEGS,
            joint.joint_state.joint_5 / RAD_TO_MILLIDEGS,
            joint.joint_state.joint_6 / RAD_TO_MILLIDEGS,
        ])

    def get_joint_velocities(self) -> np.ndarray:
        """Returns joint velocities. Shape: (6,)
        Note: piper_sdk basic API does not expose direct velocity feedback."""
        return np.zeros(6)

    def set_joint_positions(self, positions: Union[List[float], np.ndarray]):
        """Set joint positions in radians. Shape: (6,)"""
        joints = [round(p * RAD_TO_MILLIDEGS) for p in positions]
        self.piper.MotionCtrl_2(0x01, 0x01, 100, 0x00)
        self.piper.JointCtrl(*joints)

    def get_gripper_position(self) -> float:
        """Returns gripper stroke in meters."""
        gripper = self.piper.GetArmGripperMsgs()
        return gripper.gripper_state.grippers_angle / 1_000_000.0  # 0.001mm → m

    def set_gripper_position(self, position_m: float, effort: float = 1.0):
        """Set gripper position in meters (0~0.07) and effort in N (0.5~2.0)."""
        angle_units = abs(round(position_m * 1_000_000))  # m → 0.001mm
        effort_units = round(effort * 1000)  # N → 0.001N
        self.piper.GripperCtrl(angle_units, effort_units, 0x01, 0)

    def go_home(self):
        """Move all joints to zero position at 50% speed."""
        self.piper.MotionCtrl_2(0x01, 0x01, 50, 0x00)
        self.piper.JointCtrl(0, 0, 0, 0, 0, 0)

    def get_arm_status(self):
        """Returns arm status object."""
        return self.piper.GetArmStatus()

    def close(self):
        """Disable and disconnect."""
        self.piper.DisableArm(7)
        time.sleep(0.1)
        print(f"Piper arm on {self.can_port} disabled.")
