import os
import time
from typing import Dict

import numpy as np

from xrobotoolkit_teleop.common.base_hardware_teleop_controller import HardwareTeleopController
from xrobotoolkit_teleop.hardware.interface.piper import PiperInterface
from xrobotoolkit_teleop.utils.geometry import R_HEADSET_TO_WORLD
from xrobotoolkit_teleop.utils.path_utils import ASSET_PATH

DEFAULT_PIPER_URDF_PATH = os.path.join(ASSET_PATH, "piper/piper_description.urdf")

DEFAULT_CAN_PORT = "can0"
DEFAULT_SCALE_FACTOR = 1.5

DEFAULT_PIPER_MANIPULATOR_CONFIG = {
    "right_arm": {
        "link_name": "link6",
        "pose_source": "right_controller",
        "control_trigger": "right_grip",
        "control_mode": "pose",
        "gripper_config": {
            "type": "parallel",
            "gripper_trigger": "right_trigger",
            "joint_names": ["joint7"],
            "open_pos": [0.07],
            "close_pos": [0.0],
        },
    },
}

# Joint names in the Piper URDF (6-DOF arm)
PIPER_ARM_JOINT_NAMES = [f"joint{i}" for i in range(1, 7)]


class PiperTeleopController(HardwareTeleopController):
    def __init__(
        self,
        robot_urdf_path: str = DEFAULT_PIPER_URDF_PATH,
        manipulator_config: dict = DEFAULT_PIPER_MANIPULATOR_CONFIG,
        can_port: str = DEFAULT_CAN_PORT,
        R_headset_world: np.ndarray = R_HEADSET_TO_WORLD,
        scale_factor: float = DEFAULT_SCALE_FACTOR,
        visualize_placo: bool = False,
        control_rate_hz: int = 50,
        enable_log_data: bool = False,
        log_dir: str = "logs/piper",
        log_freq: float = 50,
        enable_camera: bool = False,
        camera_fps: int = 30,
    ):
        self.can_port = can_port
        super().__init__(
            robot_urdf_path=robot_urdf_path,
            manipulator_config=manipulator_config,
            R_headset_world=R_headset_world,
            floating_base=False,
            scale_factor=scale_factor,
            visualize_placo=visualize_placo,
            control_rate_hz=control_rate_hz,
            enable_log_data=enable_log_data,
            log_dir=log_dir,
            log_freq=log_freq,
            enable_camera=enable_camera,
            camera_fps=camera_fps,
        )

    def _placo_setup(self):
        super()._placo_setup()
        # Compute joint index slice in placo_robot.state.q for the 6 arm joints
        self.placo_arm_joint_slice = slice(
            self.placo_robot.get_joint_offset(PIPER_ARM_JOINT_NAMES[0]),
            self.placo_robot.get_joint_offset(PIPER_ARM_JOINT_NAMES[-1]) + 1,
        )

    def _robot_setup(self):
        """Initialize the Piper arm via CAN bus."""
        print(f"Setting up Piper arm on CAN port: {self.can_port}")
        self.arm = PiperInterface(can_port=self.can_port)

        print("Moving to home position...")
        self.arm.go_home()
        time.sleep(2)
        print("Piper arm ready.")

    def _initialize_camera(self):
        """Camera initialization - override if you add cameras later."""
        pass

    def _update_robot_state(self):
        """Read current joint positions from the arm and update Placo."""
        self.placo_robot.state.q[self.placo_arm_joint_slice] = self.arm.get_joint_positions()

    def _send_command(self):
        """Send the solved joint targets to the Piper arm."""
        if self.active.get("right_arm", False):
            q_des = self.placo_robot.state.q[self.placo_arm_joint_slice].copy()
            self.arm.set_joint_positions(q_des)

        # Gripper control
        if "gripper_config" in self.manipulator_config["right_arm"]:
            gripper_config = self.manipulator_config["right_arm"]["gripper_config"]
            joint_name = gripper_config["joint_names"][0]
            gripper_target = self.gripper_pos_target["right_arm"][joint_name]
            self.arm.set_gripper_position(gripper_target)

    def _get_robot_state_for_logging(self) -> Dict:
        """Return current robot state for data logging."""
        return {
            "qpos": self.arm.get_joint_positions(),
            "qvel": self.arm.get_joint_velocities(),
            "qpos_des": self.placo_robot.state.q[self.placo_arm_joint_slice].copy(),
            "gripper_qpos": self.arm.get_gripper_position(),
            "gripper_qpos_des": self.gripper_pos_target.get("right_arm", {}),
        }

    def _get_camera_frame_for_logging(self) -> Dict:
        """Return camera frames for logging - empty until cameras are configured."""
        return {}

    def _shutdown_robot(self):
        """Graceful shutdown: move home and disable."""
        print("Shutting down Piper arm...")
        self.arm.go_home()
        time.sleep(2)
        self.arm.close()
