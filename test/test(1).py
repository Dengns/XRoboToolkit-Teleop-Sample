import openvr
import numpy as np


class ViveTrackerSystem:
    def __init__(self, serial_list, tracking_universe="standing"):
        """
        serial_list: list[str]
            tracker 序列号列表
            默认 serial_list[0] 作为 base tracker
        tracking_universe: "standing" | "raw" | "seated"
        """
        if len(serial_list) < 1:
            raise ValueError("serial_list must contain at least one tracker")

        self.serial_list = serial_list
        self.base_serial = serial_list[0]
        self.tracker_indices = {}

        if tracking_universe == "standing":
            self.tracking_universe = openvr.TrackingUniverseStanding
        elif tracking_universe == "raw":
            self.tracking_universe = openvr.TrackingUniverseRawAndUncalibrated
        elif tracking_universe == "seated":
            self.tracking_universe = openvr.TrackingUniverseSeated
        else:
            raise ValueError("tracking_universe must be one of: standing, raw, seated")

        openvr.init(openvr.VRApplication_Other)
        self.vrsystem = openvr.VRSystem()

        self._init_trackers()

    def _get_string_property(self, index, prop):
        try:
            return self.vrsystem.getStringTrackedDeviceProperty(index, prop)
        except Exception:
            return None

    def _init_trackers(self):
        available = {}

        for i in range(openvr.k_unMaxTrackedDeviceCount):
            cls = self.vrsystem.getTrackedDeviceClass(i)
            if cls == openvr.TrackedDeviceClass_GenericTracker:
                serial = self._get_string_property(i, openvr.Prop_SerialNumber_String)
                if serial is not None:
                    available[serial] = i

        for s in self.serial_list:
            if s not in available:
                raise RuntimeError(f"Tracker {s} not found in SteamVR")
            self.tracker_indices[s] = available[s]

        print("Tracker mapping:")
        for s, idx in self.tracker_indices.items():
            role = " [BASE]" if s == self.base_serial else ""
            print(f"  {s} -> index {idx}{role}")

    @staticmethod
    def _mat34_to_matrix(mat):
        return np.array([
            [mat[0][0], mat[0][1], mat[0][2], mat[0][3]],
            [mat[1][0], mat[1][1], mat[1][2], mat[1][3]],
            [mat[2][0], mat[2][1], mat[2][2], mat[2][3]],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float64)

    @staticmethod
    def _rotation_to_quaternion(R):
        """
        旋转矩阵 -> 四元数 [x, y, z, w]
        """
        q = np.empty(4, dtype=np.float64)
        trace = np.trace(R)

        if trace > 0:
            s = np.sqrt(trace + 1.0) * 2.0
            q[3] = 0.25 * s
            q[0] = (R[2, 1] - R[1, 2]) / s
            q[1] = (R[0, 2] - R[2, 0]) / s
            q[2] = (R[1, 0] - R[0, 1]) / s
        else:
            if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
                q[3] = (R[2, 1] - R[1, 2]) / s
                q[0] = 0.25 * s
                q[1] = (R[0, 1] + R[1, 0]) / s
                q[2] = (R[0, 2] + R[2, 0]) / s
            elif R[1, 1] > R[2, 2]:
                s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
                q[3] = (R[0, 2] - R[2, 0]) / s
                q[0] = (R[0, 1] + R[1, 0]) / s
                q[1] = 0.25 * s
                q[2] = (R[1, 2] + R[2, 1]) / s
            else:
                s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
                q[3] = (R[1, 0] - R[0, 1]) / s
                q[0] = (R[0, 2] + R[2, 0]) / s
                q[1] = (R[1, 2] + R[2, 1]) / s
                q[2] = 0.25 * s

        q /= np.linalg.norm(q)
        return q

    def get_poses(self):
        """
        返回所有 tracker 在 world 下的绝对位姿
        {
          serial: {
            "pos": np.array([x, y, z]),
            "quat": np.array([x, y, z, w]),
            "T": np.array shape=(4,4),
            "valid": bool
          }
        }
        """
        result = {}

        poses = self.vrsystem.getDeviceToAbsoluteTrackingPose(
            self.tracking_universe,
            0.0,
            openvr.k_unMaxTrackedDeviceCount
        )

        for serial, idx in self.tracker_indices.items():
            pose = poses[idx]

            if not pose.bPoseIsValid:
                result[serial] = {
                    "pos": None,
                    "quat": None,
                    "T": None,
                    "valid": False
                }
                continue

            T = self._mat34_to_matrix(pose.mDeviceToAbsoluteTracking)
            pos = T[:3, 3]
            quat = self._rotation_to_quaternion(T[:3, :3])

            result[serial] = {
                "pos": pos,
                "quat": quat,
                "T": T,
                "valid": True
            }

        return result

    def get_relative_poses(self):
        """
        返回其他所有 tracker 相对 base tracker 的位姿
        base = serial_list[0]

        {
          other_serial: {
            "pos": np.array([x, y, z]),
            "quat": np.array([x, y, z, w]),
            "T": np.array shape=(4,4),
            "valid": bool
          }
        }
        """
        abs_poses = self.get_poses()

        if not abs_poses[self.base_serial]["valid"]:
            raise RuntimeError(f"Base tracker {self.base_serial} pose is invalid")

        T_world_base = abs_poses[self.base_serial]["T"]
        T_base_world = np.linalg.inv(T_world_base)

        relative_result = {}

        for serial in self.serial_list:
            if serial == self.base_serial:
                continue

            if not abs_poses[serial]["valid"]:
                relative_result[serial] = {
                    "pos": None,
                    "quat": None,
                    "T": None,
                    "valid": False
                }
                continue

            T_world_other = abs_poses[serial]["T"]
            T_base_other = T_base_world @ T_world_other

            pos = T_base_other[:3, 3]
            quat = self._rotation_to_quaternion(T_base_other[:3, :3])

            relative_result[serial] = {
                "pos": pos,
                "quat": quat,
                "T": T_base_other,
                "valid": True
            }

        return relative_result

    def get_relative_pose(self, serial):
        """
        返回某个 tracker 相对 base tracker 的位姿
        """
        if serial == self.base_serial:
            return {
                "pos": np.zeros(3),
                "quat": np.array([0.0, 0.0, 0.0, 1.0]),
                "T": np.eye(4),
                "valid": True
            }

        if serial not in self.tracker_indices:
            raise KeyError(f"{serial} not managed by this instance")

        return self.get_relative_poses()[serial]

    def shutdown(self):
        openvr.shutdown()

import time

if __name__ == "__main__":
    serials = [
        "LHR-FA64F3C2",   # 第一个作为 base
        "LHR-D3D7B75F",
    ]

    system = ViveTrackerSystem(serials)

    try:
        while True:
            rel_poses = system.get_relative_poses()

            print("\n==== relative to base ====")
            for serial, data in rel_poses.items():
                print(f"\ntracker: {serial}")
                print("valid:", data["valid"])
                if data["valid"]:
                    print("pos :", data["pos"])
                    print("quat:", data["quat"])

            time.sleep(0.01)

    except KeyboardInterrupt:
        pass

    finally:
        system.shutdown()