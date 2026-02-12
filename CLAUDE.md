# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

XR-based teleoperation framework for controlling robots through VR/AR devices (Pico headsets via XRoboToolkit SDK) in both MuJoCo simulation and real hardware. Python 3.10+, uses Placo for inverse kinematics.

## Setup and Commands

```sh
# Conda setup (Ubuntu 22.04/24.04)
./setup_conda.sh --conda xr-robotics
conda activate xr-robotics
./setup_conda.sh --install

# Alternative: system-wide
./setup.sh

# Format code (line length: 120)
black .

# Run simulation examples
python scripts/simulation/teleop_dual_ur5e_mujoco.py
python scripts/simulation/teleop_shadow_hand_mujoco.py

# Run hardware examples
python scripts/hardware/teleop_dual_ur5e_hardware.py --reset
```

There is no test suite. Validation is done via simulation/hardware demo scripts and `scripts/misc/test_data_log_analysis.py` for inspecting logged `.pkl` files.

## Architecture

### Controller Hierarchy

```
BaseTeleopController (abstract)
├── MujocoTeleopController          # MuJoCo simulation (single-threaded viewer loop)
├── PlacoTeleopController           # Placo IK visualization only (meshcat)
└── HardwareTeleopController (abstract)
    ├── DualArmURController         # Dual UR5e arms
    ├── ARXTeleopController         # ARX R5 arms (CAN bus)
    └── GalaxeaR1LiteTeleopController  # Galaxea R1 Lite humanoid
```

`BaseTeleopController` owns the Placo IK solver, XrClient, and the core `_update_ik()` loop. Subclasses implement `_robot_setup()`, `_update_robot_state()`, `_send_command()`, `_get_link_pose()`, and `run()`.

`HardwareTeleopController` adds multi-threaded execution: separate threads for IK solving (`_ik_thread`), robot command sending (`_control_thread`), data logging (`_data_logging_thread`), and camera streaming (`_camera_thread`).

### Control Flow

1. XR device input → `XrClient` (wraps `xrt` SDK)
2. Grip activation check (`control_trigger > 0.9`) → delta pose computation via `_process_xr_pose()`
3. Placo IK solver updates frame/position tasks → `solver.solve(True)`
4. `_send_command()` pushes joint targets to hardware/simulation

### manipulator_config Dict (Central Configuration)

Every controller is parameterized by a `manipulator_config` dict. This is the key structure for adding robot support:

```python
{
    "right_hand": {
        "link_name": "right_tool0",           # URDF link for IK target
        "pose_source": "right_controller",     # XrClient pose source
        "control_trigger": "right_grip",       # Grip button to activate
        "control_mode": "pose",                # "pose" (6DOF) or "position" (3DOF)
        "gripper_config": {                    # Optional
            "type": "parallel",
            "gripper_trigger": "right_trigger",
            "joint_names": ["right_finger_joint"],
            "open_pos": [0.0],
            "close_pos": [0.8],
        },
        "motion_tracker": {                    # Optional, for redundant arms
            "serial": "tracker_serial_number",
            "link_target": "elbow_link",
        },
    },
}
```

### XrClient API

`XrClient` wraps the XRoboToolkit SDK. Key methods and their valid arguments:

- `get_pose_by_name(name)` → `np.ndarray[7]` as `[x,y,z,qx,qy,qz,qw]`
  - Names: `"left_controller"`, `"right_controller"`, `"headset"`
- `get_key_value_by_name(name)` → `float` in `[0,1]`
  - Names: `"left_trigger"`, `"right_trigger"`, `"left_grip"`, `"right_grip"`
- `get_button_state_by_name(name)` → `bool`
  - Names: `"A"`, `"B"`, `"X"`, `"Y"`, `"left_menu_button"`, `"right_menu_button"`, `"left_axis_click"`, `"right_axis_click"`
- `get_hand_tracking_state(hand)` → `np.ndarray[27,7]` or `None` — `"left"` / `"right"`
- `get_joystick_state(controller)` → `list[2]` — `"left"` / `"right"`
- `get_motion_tracker_data()` → `dict` keyed by tracker serial
- `get_body_tracking_data()` → `dict` or `None`

**Quaternion convention:** XrClient returns `[qx,qy,qz,qw]`, but internally the codebase converts to `[w,x,y,z]` for `meshcat.transformations` (used by Placo). Watch for this when processing poses.

### Headset-to-World Transform

All XR poses are in headset-relative coordinates and must be transformed to the robot world frame:

```python
R_HEADSET_TO_WORLD = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
```

This is passed as `R_headset_world` to controller constructors. The transform is applied in `_process_xr_pose()` to both position (matrix multiply) and orientation (sandwich quaternion product).

### MuJoCo ↔ Placo Conversion

`xrobotoolkit_teleop/utils/mujoco_utils.py` handles bidirectional joint state conversion between MuJoCo's `qpos` format and Placo/Pinocchio's `q` format. These differ in joint ordering and floating-base representation. Use `calc_mujoco_qpos_from_placo_q()` and `calc_placo_q_from_mujoco_qpos()`.

### Hardware Interface Layer

`xrobotoolkit_teleop/hardware/interface/` contains low-level wrappers for each hardware type. Each interface encapsulates a single communication protocol (RTDE for UR robots, CAN for ARX, Dynamixel SDK for servos, librealsense for cameras). Camera interfaces extend `BaseCameraInterface` which defines the `start()`/`stop()`/`get_frames()`/`get_compressed_frames()` contract.

### Data Logging

Hardware controllers support data collection via the `DataLogger` class. Users toggle logging with the B button and discard a session with right-axis-click. Logs are saved as timestamped `.pkl` files in `logs/`. Each entry contains: `timestamp`, `qpos`, `qvel`, `qpos_des`, and optionally `image` (compressed JPG frames keyed by camera name).

## Coding Conventions

- **Formatter:** `black` with line length 120
- **Style:** PEP 8, `snake_case` for functions/variables, `PascalCase` for classes
- **Asset paths:** Always use absolute paths via `path_utils.ASSET_PATH`, never relative paths for URDF loading
- **Adding a new robot:** Extend `BaseTeleopController` (simulation) or `HardwareTeleopController` (hardware), implement the abstract methods, define a `manipulator_config` dict, and add a hardware interface in `hardware/interface/` if needed
