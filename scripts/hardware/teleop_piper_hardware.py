#!/usr/bin/env python3
"""Piper robotic arm teleoperation via Pico XR headset (direct piper_sdk, no ROS)."""

import tyro
from dataclasses import dataclass

from xrobotoolkit_teleop.hardware.piper_teleop_controller import PiperTeleopController


@dataclass
class Args:
    can_port: str = "can0"
    scale_factor: float = 1.5
    visualize_placo: bool = False
    enable_log_data: bool = False
    log_dir: str = "logs/piper"
    enable_camera: bool = False


def main(args: Args):
    controller = PiperTeleopController(
        can_port=args.can_port,
        scale_factor=args.scale_factor,
        visualize_placo=args.visualize_placo,
        enable_log_data=args.enable_log_data,
        log_dir=args.log_dir,
        enable_camera=args.enable_camera,
    )
    controller.run()


if __name__ == "__main__":
    main(tyro.cli(Args))
