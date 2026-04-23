#!/usr/bin/env python3
"""RealMan RM75-B + ZhiXing 90D gripper teleoperation via Pico XR headset."""

import tyro
from dataclasses import dataclass

from xrobotoolkit_teleop.hardware.rm75b_teleop_controller import RM75BTeleopController


@dataclass
class Args:
    ip: str = "192.168.5.73"
    port: int = 8080
    scale_factor: float = 1
    visualize_placo: bool = False
    enable_log_data: bool = False
    log_dir: str = "logs/rm75b"
    control_rate_hz: int = 50


def main(args: Args):
    controller = RM75BTeleopController(
        ip=args.ip,
        port=args.port,
        scale_factor=args.scale_factor,
        visualize_placo=args.visualize_placo,
        enable_log_data=args.enable_log_data,
        log_dir=args.log_dir,
        control_rate_hz=args.control_rate_hz,
    )
    controller.run()


if __name__ == "__main__":
    main(tyro.cli(Args))
