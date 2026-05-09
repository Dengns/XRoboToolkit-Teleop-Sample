#!/usr/bin/env python3
"""
UDEXREAL glove (LEFT) -> Wuji dexterous hand (LEFT) realtime teleoperation.

Pipeline
--------
UDEXREAL HandDriver streams JSON frames over UDP. Each frame contains a
"Parameter" list with entries like {"Name": "l0", "Value": -60.0}, ...
We:
  1. Receive frames in a background thread, keep only the latest one.
  2. Map the 20 left-hand parameters l0..l19 to the 5x4 Wuji target array.
  3. Convert deg -> rad and push to wujihandpy.realtime_controller at fixed Hz.

Key gotchas (per UDEXREAL HandDriver PDF + provided mapping):
  - Glove pitch values are NEGATIVE for finger-toward-palm flexion. The
    mapping table lists abs ranges, so we take abs() before linear mapping.
  - Glove yaw l3 (Thumb CM yaw) range [-30, 10] is signed (NOT abs).
  - Wuji takes RADIANS, never degrees.
"""

from __future__ import annotations

import argparse
import json
import socket
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

# wujihandpy is the vendor library; only needed when actually driving the robot.
# We import it lazily inside teleop() so dry-run / offline tests don't require it.


# =========================================================================
# Joint mapping
# =========================================================================
# Wuji 5x4 target layout (matches 3_realtime.py):
#   row 0 = F1 = Thumb   (col0=CM_pitch, col1=CM_yaw,  col2=MP_pitch,  col3=IP_pitch)
#   row 1 = F2 = Index   (col0=MP_pitch, col1=MP_yaw,  col2=PIP_pitch, col3=DIP_pitch)
#   row 2 = F3 = Middle  (col0=MP_pitch, col1=MP_yaw,  col2=PIP_pitch, col3=DIP_pitch)
#   row 3 = F4 = Ring    (col0=MP_pitch, col1=MP_yaw,  col2=PIP_pitch, col3=DIP_pitch)
#   row 4 = F5 = Pinky   (col0=MP_pitch, col1=MP_yaw,  col2=PIP_pitch, col3=DIP_pitch)


@dataclass
class JointMap:
    row: int          # 0..4
    col: int          # 0..3
    name: str         # human-readable, used for logs
    glove_key: str    # e.g. "l2" for left glove
    glove_lo: float   # source range low  (deg)
    glove_hi: float   # source range high (deg)
    wuji_lo: float    # destination value at glove_lo (deg)
    wuji_hi: float    # destination value at glove_hi (deg)
    use_abs: bool     # take abs(glove_val) before mapping
    piecewise: bool = False  # signed-through-zero mapping (0 -> 0, sides scaled separately)


def build_left_hand_mapping() -> list[JointMap]:
    """Mapping table updated to wuji's precise joint limits.

    Wuji limits (deg, converted from the rad limits provided):
        Thumb  J0 CM_pitch  [ -4.01,  +95.51]
        Thumb  J1 CM_yaw    [-17.19,  +54.83]
        Thumb  J2 MP_pitch  [-30.48,  +96.77]
        Thumb  J3 IP_pitch  [-31.46,  +95.17]
        Index  J0 MP_pitch  [-14.38,  +95.11]
        Index  J1 MP_yaw    [-25.44,  +21.43]
        Index  J2 PIP_pitch [-32.20,  +94.88]
        Index  J3 DIP_pitch [-29.91,  +97.86]
        Middle J0 MP_pitch  [-14.38,  +93.97]
        Middle J1 MP_yaw    [-24.18,  +21.43]
        Middle J2 PIP_pitch [-33.58,  +93.45]
        Middle J3 DIP_pitch [-31.51,  +96.49]
        Ring   J0 MP_pitch  [-15.13,  +94.31]
        Ring   J1 MP_yaw    [-24.98,  +21.77]
        Ring   J2 PIP_pitch [-31.74,  +95.34]
        Ring   J3 DIP_pitch [-31.97,  +97.00]
        Pinky  J0 MP_pitch  [-15.99,  +93.11]
        Pinky  J1 MP_yaw    [-24.41,  +21.43]
        Pinky  J2 PIP_pitch [-32.37,  +94.65]
        Pinky  J3 DIP_pitch [-31.23,  +96.14]

    Policy
    ------
    Per the user: glove only captures motion toward the palm (flexion), so
    we don't use any wuji negative range for pitch joints. Only the thumb
    CM yaw and index MP yaw need bidirectional behaviour; other yaws stay
    one-sided.

    Sign handling for yaw
    ---------------------
      * Thumb CM yaw  (l3): signed mapping [-30, +10] -> [-17, +55].
        Glove l3 negative = thumb adduction (toward palm, "拇指向手掌弯曲
        时偏航为负"), positive = abduction. Wuji negative = adduction here
        too (origin shift preserved from the original [-30, 10] -> [0, 80]).

      * Index MP yaw  (l7): signed mapping [-30, +30] -> [+21, -25].
        Note dst is REVERSED ([+21, -25] not [-25, +21]) because empirically
        the glove's index yaw sign is opposite to wuji's. So this mapping
        is effectively  wuji = -glove * (50 / 60). If the index direction
        is still wrong on hardware, swap dst back to [-25, +21].

      * Middle / Ring / Pinky yaw: one-sided abs mapping into [0, 21]ish.
    """
    return [
        # Thumb (row 0)
        JointMap(0, 0, "Thumb CM pitch",  "l2",   0.0,  75.0,   0.0,  95.51, True),
        JointMap(0, 1, "Thumb CM yaw",    "l3", -30.0,  10.0, -17.19, 54.83, False, piecewise=True),
        JointMap(0, 2, "Thumb MP pitch",  "l1",   0.0,  90.0,   0.0,  96.77, True),
        JointMap(0, 3, "Thumb IP pitch",  "l0",   0.0,  90.0,   0.0,  95.17, True),

        # Index (row 1) -- yaw uses piecewise sign-flipped mapping:
        #   glove l7 = 0   -> wuji 0  (clean neutral)
        #   glove l7 = -30 -> wuji +21.43 (toward middle finger / abduction)
        #   glove l7 = +30 -> wuji -25.44 (toward thumb / adduction)
        # The dst signs are SWAPPED relative to glove because empirically the
        # glove firmware's index yaw sign is opposite to wuji's. If hardware
        # observation says it's still wrong, swap dst lo/hi back.
        JointMap(1, 0, "Index MP pitch",  "l6",   0.0,  81.0,   0.0,  95.11, True),
        JointMap(1, 1, "Index MP yaw",    "l7", -30.0,  30.0, -25.44, +21.43, False),
        JointMap(1, 2, "Index PIP pitch", "l5",   0.0, 100.0,   0.0,  94.88, True),
        JointMap(1, 3, "Index DIP pitch", "l4",   0.0,  80.0,   0.0,  97.86, True),

        # Middle (row 2) -- one-sided yaw
        JointMap(2, 0, "Middle MP pitch", "l10",  0.0,  81.0,   0.0,  93.97, True),
        JointMap(2, 1, "Middle MP yaw",   "l11",  0.0,   5.0,   0.0,  21.43, True, piecewise=True),
        JointMap(2, 2, "Middle PIP pitch","l9",   0.0,  81.0,   0.0,  93.45, True),
        JointMap(2, 3, "Middle DIP pitch","l8",   0.0,  80.0,   0.0,  96.49, True),

        # Ring (row 3) -- one-sided yaw
        JointMap(3, 0, "Ring MP pitch",   "l14",  0.0,  81.0,   0.0,  94.31, True),
        JointMap(3, 1, "Ring MP yaw",     "l15",  0.0,  20.0,   0.0,  21.77, True),
        JointMap(3, 2, "Ring PIP pitch",  "l13",  0.0, 100.0,   0.0,  95.34, True),
        JointMap(3, 3, "Ring DIP pitch",  "l12",  0.0,  80.0,   0.0,  97.00, True),

        # Pinky (row 4) -- one-sided yaw
        JointMap(4, 0, "Pinky MP pitch",  "l18",  0.0, 100.0,   0.0,  93.11, True),
        JointMap(4, 1, "Pinky MP yaw",    "l19",  0.0,  30.0,   0.0,  21.43, True),
        JointMap(4, 2, "Pinky PIP pitch", "l17",  0.0, 100.0,   0.0,  94.65, True),
        JointMap(4, 3, "Pinky DIP pitch", "l16",  0.0,  80.0,   0.0,  96.14, True),
    ]


def map_value(val: float,
              src_lo: float, src_hi: float,
              dst_lo: float, dst_hi: float,
              use_abs: bool,
              piecewise: bool = False) -> float:
    """Remap glove deg -> wuji deg.

    Modes (in priority order):
      use_abs=True   -> output = lerp(|val|, [src_lo, src_hi] -> [dst_lo, dst_hi]),
                        then clipped to dst range.
      piecewise=True -> signed through-zero mapping. val=0 -> 0.
                        val>0 scaled to dst_hi by ratio val/src_hi.
                        val<0 scaled to dst_lo by ratio val/src_lo.
                        Sign of dst_lo/dst_hi controls direction (you can
                        swap them to invert).
      otherwise      -> standard linear lerp([src_lo, src_hi] -> [dst_lo, dst_hi]),
                        with clipping.
    """
    if use_abs:
        val = abs(val)
        if src_hi == src_lo:
            return dst_lo
        t = (val - src_lo) / (src_hi - src_lo)
        t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
        return dst_lo + t * (dst_hi - dst_lo)

    if piecewise:
        if val == 0.0:
            return 0.0
        if val > 0.0:
            # val in (0, src_hi]  ->  output in (0, dst_hi] (sign of dst_hi preserved)
            t = val / src_hi if src_hi != 0 else 0.0
            t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
            return t * dst_hi
        else:
            # val in [src_lo, 0)  ->  output in [dst_lo, 0) (sign of dst_lo preserved)
            t = val / src_lo if src_lo != 0 else 0.0
            t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
            return t * dst_lo

    # Standard linear mapping with clipping.
    if src_hi == src_lo:
        return dst_lo
    t = (val - src_lo) / (src_hi - src_lo)
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    return dst_lo + t * (dst_hi - dst_lo)


def build_target_deg(params: dict[str, float],
                     mapping: list[JointMap],
                     last_target_deg: np.ndarray) -> np.ndarray:
    """Build 5x4 target in degrees. Hold last value if a key is absent."""
    target = last_target_deg.copy()
    for m in mapping:
        v = params.get(m.glove_key)
        if v is None:
            continue
        target[m.row, m.col] = map_value(
            v, m.glove_lo, m.glove_hi, m.wuji_lo, m.wuji_hi, m.use_abs, m.piecewise
        )
    return target


# =========================================================================
# UDP receiver (latest-only)
# =========================================================================
class GloveReceiver:
    """Background UDP listener that always exposes only the most recent frame.

    Old packets are dropped to minimize end-to-end teleop latency.
    """

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self._latest: Optional[dict[str, float]] = None
        self._calib_status: Optional[int] = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._frame_count = 0

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def get_latest(self) -> Optional[dict[str, float]]:
        with self._lock:
            return None if self._latest is None else dict(self._latest)

    @property
    def calibration_status(self) -> Optional[int]:
        return self._calib_status

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def _loop(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(0.5)
        sock.bind((self.host, self.port))
        try:
            while not self._stop.is_set():
                try:
                    payload, _ = sock.recvfrom(65535)
                except socket.timeout:
                    continue
                params = self._parse(payload)
                if params is None:
                    continue
                with self._lock:
                    self._latest = params
                    self._frame_count += 1
                    if "L_CalibrationStatus" in params:
                        self._calib_status = int(params["L_CalibrationStatus"])
        finally:
            sock.close()

    @staticmethod
    def _parse(payload: bytes) -> Optional[dict[str, float]]:
        try:
            obj = json.loads(payload.decode("utf-8", errors="replace"))
        except json.JSONDecodeError:
            return None

        # HandDriver envelope: {" hand": {"Parameter": [{"Name":..., "Value":...}, ...]}}
        # Note the leading space in " hand" — current firmware ships it that way.
        body = obj.get("data", obj)
        hand = body.get(" hand") or body.get("hand")
        if not hand:
            return None
        params = hand.get("Parameter")
        if not params:
            return None

        out: dict[str, float] = {}
        for entry in params:
            name = entry.get("Name")
            value = entry.get("Value")
            if name is None or not isinstance(value, (int, float)):
                continue
            out[name] = float(value)
        return out


# =========================================================================
# Teleop loop
# =========================================================================
def _ramp_to_pose(ctl,
                  current_deg: np.ndarray,
                  goal_deg: np.ndarray,
                  rate_hz: float,
                  slew_deg_s: float,
                  settle_threshold_deg: float = 0.5,
                  hold_time_s: float = 0.3,
                  max_time_s: float = 4.0,
                  label: str = "ramp") -> np.ndarray:
    """Slew current_deg toward goal_deg, sending commands every tick.

    Returns the final commanded pose (deg). Bounded by max_time_s so we
    never block forever. Catches its own KeyboardInterrupt so a second
    Ctrl+C during the ramp aborts to disable, instead of hanging.
    """
    period = 1.0 / rate_hz
    max_delta = slew_deg_s * period
    deadline = time.time() + max_time_s
    try:
        while time.time() < deadline:
            delta = goal_deg - current_deg
            np.clip(delta, -max_delta, max_delta, out=delta)
            current_deg = current_deg + delta
            ctl.set_joint_target_position(np.deg2rad(current_deg))
            if float(np.max(np.abs(goal_deg - current_deg))) < settle_threshold_deg:
                break
            time.sleep(period)
        # Hold at goal so the LowPass filter has time to converge.
        hold_ticks = max(1, int(hold_time_s * rate_hz))
        for _ in range(hold_ticks):
            ctl.set_joint_target_position(np.deg2rad(goal_deg))
            time.sleep(period)
        current_deg = goal_deg.copy()
        print(f"[teleop] {label}: reached goal pose.")
    except KeyboardInterrupt:
        print(f"[teleop] {label} interrupted; disabling immediately.")
    return current_deg


def teleop(host: str = "0.0.0.0",
           port: int = 5555,
           rate_hz: float = 100.0,
           filter_cutoff_hz: float = 8.0,
           max_velocity_deg_s: float = 90.0,
           ramp_speed_deg_s: float = 60.0,
           print_period: float = 0.5,
           dry_run: bool = False) -> None:
    mapping = build_left_hand_mapping()
    receiver = GloveReceiver(host, port)
    receiver.start()

    print(f"[teleop] listening on udp://{host}:{port}")
    print("[teleop] waiting for first glove packet...")
    while receiver.get_latest() is None:
        time.sleep(0.05)
    print(f"[teleop] glove connected. L_CalibrationStatus={receiver.calibration_status}")
    if receiver.calibration_status != 3:
        print("[teleop] WARNING: calibration status != 3. Re-calibrate the glove "
              "in HandDriver before relying on the mapping.")

    if dry_run:
        _run_dry(receiver, mapping, rate_hz, print_period)
        receiver.stop()
        return

    # Lazy import — only needed when actually driving the hand.
    import wujihandpy

    hand = wujihandpy.Hand()
    np.set_printoptions(precision=2, suppress=True)

    period = 1.0 / rate_hz
    max_delta_per_tick = max_velocity_deg_s * period  # max deg change per control tick

    # Two state variables:
    #   target_raw_deg  -- desired pose from glove (latest, possibly fast-changing)
    #   target_cmd_deg  -- pose actually sent to wuji (slew-rate limited toward raw)
    target_raw_deg = np.zeros((5, 4), dtype=np.float64)
    target_cmd_deg = np.zeros((5, 4), dtype=np.float64)
    zero_pose = np.zeros((5, 4), dtype=np.float64)

    try:
        hand.write_joint_enabled(True)
        with hand.realtime_controller(
            enable_upstream=True,
            filter=wujihandpy.filter.LowPass(cutoff_freq=filter_cutoff_hz),
        ) as ctl:
            print(f"[teleop] running at {rate_hz:.0f} Hz, "
                  f"LowPass cutoff={filter_cutoff_hz:.1f} Hz, "
                  f"slew limit={max_velocity_deg_s:.0f} deg/s. Ctrl+C to stop.")
            try:
                last_print = 0.0
                while True:
                    t0 = time.time()

                    # 1) Pull latest glove frame and update raw target.
                    params = receiver.get_latest()
                    if params is not None:
                        target_raw_deg = build_target_deg(params, mapping, target_raw_deg)

                    # 2) Slew-rate limit: cap how fast cmd can chase raw.
                    delta = target_raw_deg - target_cmd_deg
                    np.clip(delta, -max_delta_per_tick, max_delta_per_tick, out=delta)
                    target_cmd_deg = target_cmd_deg + delta

                    # 3) Send (radians).
                    ctl.set_joint_target_position(np.deg2rad(target_cmd_deg))

                    # 4) Periodic status print.
                    if t0 - last_print >= print_period:
                        actual_rad = ctl.get_joint_actual_position()
                        err_deg = target_cmd_deg - np.rad2deg(actual_rad)
                        print(f"[teleop] frames_rx={receiver.frame_count}\n"
                              f"  raw(deg):\n{target_raw_deg}\n"
                              f"  cmd(deg):\n{target_cmd_deg}\n"
                              f"  err(deg):\n{err_deg}\n")
                        last_print = t0

                    # 5) Soft fixed-rate sleep.
                    dt = time.time() - t0
                    if dt < period:
                        time.sleep(period - dt)
            except KeyboardInterrupt:
                print("\n[teleop] Ctrl+C received — ramping hand to zero (open) pose...")
            except Exception as e:
                print(f"\n[teleop] error in main loop: {e!r} — attempting safe ramp to zero...")

            # 6) Smoothly drive the hand to zero pose BEFORE leaving the
            #    realtime controller / disabling joints. This still runs
            #    inside the `with` block so `ctl` is alive.
            target_cmd_deg = _ramp_to_pose(
                ctl,
                target_cmd_deg,
                zero_pose,
                rate_hz=rate_hz,
                slew_deg_s=ramp_speed_deg_s,
                label="open-on-exit",
            )
    finally:
        try:
            hand.write_joint_enabled(False)
        except Exception as e:
            print(f"[teleop] failed to disable hand: {e}")
        receiver.stop()
        print("[teleop] stopped.")


def _run_dry(receiver: GloveReceiver,
             mapping: list[JointMap],
             rate_hz: float,
             print_period: float) -> None:
    """No-hand mode: just print mapped targets so you can sanity-check the pipeline
    without the robot connected."""
    print("[teleop] DRY RUN — not opening wujihandpy.Hand()")
    np.set_printoptions(precision=2, suppress=True)
    last_deg = np.zeros((5, 4), dtype=np.float64)
    period = 1.0 / rate_hz
    last_print = 0.0
    try:
        while True:
            t0 = time.time()
            params = receiver.get_latest()
            if params is not None:
                last_deg = build_target_deg(params, mapping, last_deg)
            if t0 - last_print >= print_period:
                print(f"[dry] frames_rx={receiver.frame_count}  "
                      f"target(deg):\n{last_deg}\n")
                last_print = t0
            dt = time.time() - t0
            if dt < period:
                time.sleep(period - dt)
    except KeyboardInterrupt:
        print("\n[dry] stopped.")


# =========================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="UDEXREAL left glove -> Wuji left hand teleop.")
    p.add_argument("--host", default="0.0.0.0", help="UDP bind host (default 0.0.0.0)")
    p.add_argument("--port", type=int, default=5555, help="UDP bind port (default 5555)")
    p.add_argument("--rate", type=float, default=100.0, help="control rate Hz (default 100)")
    p.add_argument("--cutoff", type=float, default=5.0,
                   help="LowPass filter cutoff Hz (default 8). Increase for faster response.")
    p.add_argument("--max-velocity", type=float, default=90.0,
                   help="per-joint slew rate limit during teleop, deg/s (default 90). "
                        "Lower = slower / easier to stop mid-motion.")
    p.add_argument("--ramp-speed", type=float, default=60.0,
                   help="slew speed used to open the hand on exit, deg/s (default 60).")
    p.add_argument("--print-period", type=float, default=0.5,
                   help="seconds between status prints (default 0.5)")
    p.add_argument("--dry-run", action="store_true",
                   help="do not open wujihandpy.Hand(); just print mapped targets")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    teleop(
        host=args.host,
        port=args.port,
        rate_hz=args.rate,
        filter_cutoff_hz=args.cutoff,
        max_velocity_deg_s=args.max_velocity,
        ramp_speed_deg_s=args.ramp_speed,
        print_period=args.print_period,
        dry_run=args.dry_run,
    )
