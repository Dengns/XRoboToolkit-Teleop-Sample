import os
import time
from typing import Dict

import numpy as np

from xrobotoolkit_teleop.common.base_hardware_teleop_controller import HardwareTeleopController
from xrobotoolkit_teleop.hardware.interface.rm75b import RM75BInterface
from xrobotoolkit_teleop.utils.geometry import R_HEADSET_TO_WORLD
from xrobotoolkit_teleop.utils.path_utils import ASSET_PATH

DEFAULT_RM75B_URDF_PATH = os.path.join(ASSET_PATH, "realman/rm_75_kinematics.urdf")

DEFAULT_IP = "192.168.5.73"
DEFAULT_PORT = 8080
DEFAULT_SCALE_FACTOR = 1

DEFAULT_RM75B_MANIPULATOR_CONFIG = {
    "right_arm": {
        "link_name": "Link7",  # Note: capital L in URDF
        "pose_source": "right_controller",
        "control_trigger": "right_grip",
        "control_mode": "pose",
        "gripper_config": {
            "type": "parallel",
            "gripper_trigger": "right_trigger",
            "joint_names": ["zhixing_gripper"],
            "open_pos": [200.0],
            "close_pos": [0.0],
        },
    },
}

# 7-DOF arm joint names matching the URDF
RM75B_ARM_JOINT_NAMES = [f"joint{i}" for i in range(1, 8)]


class RM75BTeleopController(HardwareTeleopController):
    def __init__(
        self,
        robot_urdf_path: str = DEFAULT_RM75B_URDF_PATH,
        manipulator_config: dict = DEFAULT_RM75B_MANIPULATOR_CONFIG,
        ip: str = DEFAULT_IP,
        port: int = DEFAULT_PORT,
        R_headset_world: np.ndarray = R_HEADSET_TO_WORLD,
        scale_factor: float = DEFAULT_SCALE_FACTOR,
        visualize_placo: bool = False,
        control_rate_hz: int = 50,
        enable_log_data: bool = False,
        log_dir: str = "logs/rm75b",
        log_freq: float = 50,
        enable_camera: bool = False,
        camera_fps: int = 30,
    ):
        self.ip = ip
        self.port = port
        self.arm = None  # Will be created in _robot_setup

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
        # Compute joint index slice in placo_robot.state.q for the 7 arm joints
        self.placo_arm_joint_slice = slice(
            self.placo_robot.get_joint_offset(RM75B_ARM_JOINT_NAMES[0]),
            self.placo_robot.get_joint_offset(RM75B_ARM_JOINT_NAMES[-1]) + 1,
        )

    def _robot_setup(self):
        """Initialize the RM75-B arm via TCP/IP.
        Reads current joint positions as initial config (avoids singularity at all-zeros).
        Handles double-call: __init__ -> super().__init__() calls _robot_setup(),
        then run() calls _robot_setup() again. Close old connection first."""
        if self.arm is not None:
            print("Closing previous RM75-B connection before re-setup...")
            try:
                self.arm.close()
            except Exception:
                pass

        print(f"Setting up RM75-B arm at {self.ip}:{self.port}")
        self.arm = RM75BInterface(ip=self.ip, port=self.port, enable_gripper=False)

        # Read current arm position as initial configuration + home target
        startup_joints = self.arm.get_joint_positions()
        self._home_joints_deg = np.degrees(startup_joints).tolist()
        print(f"Current joint positions (deg): {[round(d, 1) for d in self._home_joints_deg]}")

        # First call (from __init__): set q_init so _placo_setup() picks it up
        self.q_init = startup_joints

        # Second call (from run()): placo already exists, sync state directly
        if hasattr(self, "placo_arm_joint_slice"):
            self.placo_robot.state.q[self.placo_arm_joint_slice] = startup_joints
            self.placo_robot.update_kinematics()
            self.sync_end_effector_poses_to_placo_tasks()

        print("RM75-B arm ready (using current position as initial state).")

    def _initialize_camera(self):
        pass

    def _update_robot_state(self):
        """Read current 7 joint positions from the arm and update Placo state."""
        self.placo_robot.state.q[self.placo_arm_joint_slice] = self.arm.get_joint_positions()

    def _send_command(self):
        """Send the solved joint targets to the RM75-B arm + gripper via Modbus."""
        if self.active.get("right_arm", False):
            q_des = self.placo_robot.state.q[self.placo_arm_joint_slice].copy()
            self.arm.set_joint_positions(q_des)

        # Gripper control
        if "gripper_config" in self.manipulator_config["right_arm"]:
            gripper_config = self.manipulator_config["right_arm"]["gripper_config"]
            joint_name = gripper_config["joint_names"][0]
            gripper_target = self.gripper_pos_target["right_arm"][joint_name]
            # DEBUG: uncomment to see trigger→position mapping
            # trigger_val = self.xr_client.get_key_value_by_name(gripper_config["gripper_trigger"])
            # print(f"[GRIPPER] trigger={trigger_val:.3f}  target={gripper_target:.0f}")
            self.arm.set_gripper_position(gripper_target)

    def _get_robot_state_for_logging(self) -> Dict:
        return {
            "qpos": self.arm.get_joint_positions(),
            "qvel": self.arm.get_joint_velocities(),
            "qpos_des": self.placo_robot.state.q[self.placo_arm_joint_slice].copy(),
            "gripper_qpos_des": self.gripper_pos_target.get("right_arm", {}),
        }

    def _get_camera_frame_for_logging(self) -> Dict:
        return {}

    def _shutdown_robot(self):
        print("Shutting down RM75-B arm...")
        try:
            self.arm.go_home(self._home_joints_deg)
            time.sleep(3)
        except Exception as e:
            print(f"[WARN] go_home during shutdown: {e}")
        self.arm.close()
