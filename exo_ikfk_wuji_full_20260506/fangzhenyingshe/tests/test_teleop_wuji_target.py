from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "teleop_exo_to_wuji_left.py"


def load_module_with_stubs():
    scripts_dir = str(REPO_ROOT / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

    mujoco_stub = types.ModuleType("mujoco")
    mujoco_stub.MjModel = object
    mujoco_stub.MjData = object

    compare_stub = types.ModuleType("compare_hand_3d")
    compare_stub.FINGER_CONFIGS = {}
    compare_stub.FINGER_ORDER = ("index", "middle", "ring", "pinky")
    compare_stub.HAND_ORDER = ("thumb", "index", "middle", "ring", "pinky")
    compare_stub.THUMB_KEY = "thumb"
    compare_stub.build_mapping_state = lambda *args, **kwargs: {}
    compare_stub.build_thumb_mapping_state = lambda *args, **kwargs: {}

    live_stub = types.ModuleType("compare_hand_3d_live")
    live_stub.DEFAULT_EXTERNAL_REPO = Path("external")
    live_stub.RealtimeSkeletonStream = object
    live_stub.apply_live_qpos = lambda *args, **kwargs: 0
    live_stub.build_default_finger_states = lambda: {}
    live_stub.build_joint_handles = lambda model: ({}, {}, {})
    live_stub.build_live_limit_overrides = lambda: {}
    live_stub.load_external_bridge_runtime = lambda repo: None

    sim_stub = types.ModuleType("delivery_core.sim_loader")
    sim_stub.DEFAULT_URDF = Path("dummy.urdf")
    sim_stub.load_model_from_urdf = lambda *args, **kwargs: (None, None, Path("bundle.urdf"))

    thumb_stub = types.ModuleType("delivery_core.thumb_mapping")
    thumb_stub.make_thumb_reference = lambda model: None

    old_modules = {
        name: sys.modules.get(name)
        for name in (
            "mujoco",
            "compare_hand_3d",
            "compare_hand_3d_live",
            "delivery_core.sim_loader",
            "delivery_core.thumb_mapping",
        )
    }
    sys.modules.update(
        {
            "mujoco": mujoco_stub,
            "compare_hand_3d": compare_stub,
            "compare_hand_3d_live": live_stub,
            "delivery_core.sim_loader": sim_stub,
            "delivery_core.thumb_mapping": thumb_stub,
        }
    )
    try:
        spec = importlib.util.spec_from_file_location("teleop_exo_to_wuji_left_under_test", SCRIPT_PATH)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.modules.pop("teleop_exo_to_wuji_left_under_test", None)
        for name, old_value in old_modules.items():
            if old_value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_value


class WuJiTargetMappingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_module_with_stubs()

    def make_states(self) -> dict[str, dict[str, object]]:
        return {
            "thumb": {"human_angles": (0.1, -0.2, -0.3, -0.4)},
            "index": {"human_angles": (-0.5, -0.6, -0.7)},
            "middle": {"human_angles": (-0.8, -0.9, -1.0)},
            "ring": {"human_angles": (-1.1, -1.2, -1.3)},
            "pinky": {"human_angles": (-1.4, -1.5, -1.6)},
        }

    def test_human_matrix_layout(self) -> None:
        matrix = self.module.mapping_states_to_human_matrix(self.make_states())
        expected = np.array(
            [
                [0.1, -0.2, -0.3, -0.4],
                [-0.5, 0.0, -0.6, -0.7],
                [-0.8, 0.0, -0.9, -1.0],
                [-1.1, 0.0, -1.2, -1.3],
                [-1.4, 0.0, -1.5, -1.6],
            ],
            dtype=np.float64,
        )
        np.testing.assert_allclose(matrix, expected)

    def test_thumb_cm_swapped_layout(self) -> None:
        matrix = self.module.mapping_states_to_human_matrix(
            self.make_states(),
            thumb_cm_order=self.module.THUMB_CM_ORDER_SWAPPED,
        )
        self.assertAlmostEqual(matrix[0, 0], -0.2)
        self.assertAlmostEqual(matrix[0, 1], 0.1)
        self.assertAlmostEqual(matrix[0, 2], -0.3)
        self.assertAlmostEqual(matrix[0, 3], -0.4)

    def test_baseline_minus_current_makes_flexion_positive(self) -> None:
        baseline = np.zeros((5, 4), dtype=np.float64)
        target = self.module.mapping_states_to_wuji_target(self.make_states(), baseline, gain=1.0)
        self.assertGreater(target[1, 0], 0.0)
        self.assertGreater(target[1, 2], 0.0)
        self.assertGreater(target[1, 3], 0.0)
        self.assertEqual(target[1, 1], 0.0)

    def test_thumb_cm_pitch_and_yaw_use_baseline_minus_current(self) -> None:
        states = {
            "thumb": {"human_angles": (0.4, 0.3, -0.2, -0.1)},
            "index": {"human_angles": (0.0, 0.0, 0.0)},
            "middle": {"human_angles": (0.0, 0.0, 0.0)},
            "ring": {"human_angles": (0.0, 0.0, 0.0)},
            "pinky": {"human_angles": (0.0, 0.0, 0.0)},
        }
        baseline = np.zeros((5, 4), dtype=np.float64)
        target = self.module.mapping_states_to_wuji_target(
            states,
            baseline,
            gain=1.0,
            thumb_cm_order=self.module.THUMB_CM_ORDER_DIRECT,
        )
        self.assertAlmostEqual(target[0, 0], -0.4)
        self.assertAlmostEqual(target[0, 1], -0.3)
        self.assertAlmostEqual(target[0, 2], 0.2)
        self.assertAlmostEqual(target[0, 3], 0.1)

    def test_thumb_cm_swapped_relative_direction(self) -> None:
        states = {
            "thumb": {"human_angles": (0.4, 0.3, -0.2, -0.1)},
            "index": {"human_angles": (0.0, 0.0, 0.0)},
            "middle": {"human_angles": (0.0, 0.0, 0.0)},
            "ring": {"human_angles": (0.0, 0.0, 0.0)},
            "pinky": {"human_angles": (0.0, 0.0, 0.0)},
        }
        baseline = np.zeros((5, 4), dtype=np.float64)
        target = self.module.mapping_states_to_wuji_target(
            states,
            baseline,
            gain=1.0,
            thumb_cm_order=self.module.THUMB_CM_ORDER_SWAPPED,
        )
        self.assertAlmostEqual(target[0, 0], -0.3)
        self.assertAlmostEqual(target[0, 1], -0.4)
        self.assertAlmostEqual(target[0, 2], 0.2)
        self.assertAlmostEqual(target[0, 3], 0.1)

    def test_gain_and_limits_clip_target(self) -> None:
        baseline = np.zeros((5, 4), dtype=np.float64)
        lower = np.full((5, 4), -0.25, dtype=np.float64)
        upper = np.full((5, 4), 0.75, dtype=np.float64)
        target = self.module.mapping_states_to_wuji_target(
            self.make_states(),
            baseline,
            gain=2.0,
            thumb_cm_order=self.module.THUMB_CM_ORDER_DIRECT,
            lower_limits=lower,
            upper_limits=upper,
        )
        self.assertAlmostEqual(target[1, 0], 0.75)
        self.assertAlmostEqual(target[0, 0], -0.2)
        self.assertEqual(target[1, 1], 0.0)

    def test_debug_target_keeps_unclipped_matrix(self) -> None:
        baseline = np.zeros((5, 4), dtype=np.float64)
        lower = np.full((5, 4), -0.25, dtype=np.float64)
        upper = np.full((5, 4), 0.75, dtype=np.float64)
        debug = self.module.compute_wuji_target_debug(
            self.make_states(),
            baseline,
            gain=2.0,
            lower_limits=lower,
            upper_limits=upper,
        )
        self.assertAlmostEqual(debug["unclipped"][1, 0], 1.0)
        self.assertAlmostEqual(debug["target"][1, 0], 0.75)
        self.assertEqual(debug["target"][1, 1], 0.0)

    def test_chinese_command_details_labels_thumb_and_yaw(self) -> None:
        baseline = np.zeros((5, 4), dtype=np.float64)
        current = self.module.mapping_states_to_human_matrix(self.make_states())
        debug = self.module.compute_wuji_target_debug(self.make_states(), baseline, gain=1.0)
        text = self.module.format_wuji_command_details(
            current_human_matrix=current,
            human_baseline_matrix=baseline,
            unclipped_target_matrix=debug["unclipped"],
            target_matrix=debug["target"],
        )
        self.assertIn("拇指翻折侧摆相对差值", text)
        self.assertIn("原始翻折侧摆", text)
        self.assertIn("发送翻折侧摆", text)
        self.assertIn("[", text)

    def test_seed_runtime_human_angles_from_baseline_matrix(self) -> None:
        runtime = types.SimpleNamespace(
            finger_states={
                "thumb": {"human_angles": (0.0, 0.0, 0.0, 0.0)},
                "index": {"human_angles": (0.0, 0.0, 0.0)},
                "middle": {"human_angles": (0.0, 0.0, 0.0)},
                "ring": {"human_angles": (0.0, 0.0, 0.0)},
                "pinky": {"human_angles": (0.0, 0.0, 0.0)},
            }
        )
        baseline = np.array(
            [
                [-1.57, -0.26, 0.26, 0.06],
                [0.10, 0.0, 0.20, 0.30],
                [0.40, 0.0, 0.50, 0.60],
                [0.70, 0.0, 0.80, 0.90],
                [1.00, 0.0, 1.10, 1.20],
            ],
            dtype=np.float64,
        )
        self.module.seed_runtime_human_angles(runtime, baseline)
        self.assertEqual(runtime.finger_states["thumb"]["human_angles"], (-1.57, -0.26, 0.26, 0.06))
        self.assertEqual(runtime.finger_states["index"]["human_angles"], (0.10, 0.20, 0.30))

    def test_save_and_load_baseline_json(self) -> None:
        baseline = np.arange(20, dtype=np.float64).reshape(5, 4) / 100.0
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "baseline.json"
            payload = self.module.save_human_baseline(
                path,
                baseline,
                metadata={"frame_count": 30, "thumb_cm_order": self.module.THUMB_CM_ORDER_SWAPPED},
            )
            loaded = self.module.load_human_baseline(path, thumb_cm_order=self.module.THUMB_CM_ORDER_SWAPPED)
        np.testing.assert_allclose(loaded, baseline)
        self.assertEqual(payload["side_policy"], "four_finger_yaw_zero")
        self.assertEqual(payload["metadata"]["frame_count"], 30)

    def test_load_legacy_direct_baseline_keeps_default_direct(self) -> None:
        baseline = np.arange(20, dtype=np.float64).reshape(5, 4) / 100.0
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "legacy_baseline.json"
            self.module.save_human_baseline(path, baseline, metadata={"frame_count": 30})
            loaded = self.module.load_human_baseline(path)
        np.testing.assert_allclose(loaded, baseline)


if __name__ == "__main__":
    unittest.main()
