"""
EA-AVS-MVP v7.0 纯 Python 单元测试套件 —— test_v70_pure_python.py
================================================================

无需 Habitat-Sim 物理引擎或 GPU，测试核心数据结构、动作播放逻辑、观测序列化与几何计算。
"""

import json
import math
import pickle
import shutil
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import numpy as np
import yaml

from ea_avs_mvp_v7.motion.motion_player import MotionPlayer
from ea_avs_mvp_v7.simulation.observation_generator import ObservationRecord
from ea_avs_mvp_v7.scripts.generate_observation_dataset import compute_viewpoints_around_human
from tools.motion_assets.data_paths import get_data_root, to_relative_data_path


class TestV70PurePython(unittest.TestCase):
    """v7.0 纯 Python 测试用例集。"""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    # TEST 1: Config Loading & Structure Validation
    # ------------------------------------------------------------------
    def test_01_config_structure(self):
        cfg_path = Path("ea_avs_mvp_v7/configs/v70_humanoid_sim.yaml")
        self.assertTrue(cfg_path.exists(), "Config file must exist")
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        self.assertIn("habitat", cfg)
        self.assertIn("camera", cfg)
        self.assertIn("humanoid", cfg)
        self.assertIn("motion", cfg)
        self.assertEqual(cfg["camera"]["width"], 640)
        self.assertEqual(cfg["camera"]["height"], 480)
        self.assertEqual(cfg["humanoid"]["avatar_name"], "neutral_0")

    # ------------------------------------------------------------------
    # TEST 2: Manifest Loading & Path Relativity
    # ------------------------------------------------------------------
    def test_02_manifest_paths_relativity(self):
        manifest_path = get_data_root() / "assets/motions/raw/motion_asset_manifest.json"
        if manifest_path.exists():
            with open(manifest_path, "r", encoding="utf-8") as f:
                items = json.load(f)
            self.assertGreaterEqual(len(items), 17)
            for m in items:
                lp = m.get("local_motion_path", "")
                self.assertFalse(lp.startswith("/"), f"Must be relative path: {lp}")
                self.assertFalse("/home/" in lp, f"Must not contain hardcoded machine paths: {lp}")
                self.assertIn(m.get("target_class"), ["standing", "sitting", "bending", "reaching", "fall_related"])

    # ------------------------------------------------------------------
    # TEST 3: Motion Player Playback & Stepping Logic
    # ------------------------------------------------------------------
    def test_03_motion_player_playback(self):
        num_frames = 20
        joints = np.zeros((num_frames, 216), dtype=np.float32)
        transforms = np.zeros((num_frames, 4, 4), dtype=np.float32)
        transforms[:, 0, 0] = 1.0
        transforms[:, 1, 1] = 1.0
        transforms[:, 2, 2] = 1.0
        transforms[:, 3, 3] = 1.0

        mock_pkl = Path(self.tmp_dir) / "mock_motion.pkl"
        with open(mock_pkl, "wb") as f:
            pickle.dump({
                "pose_motion": {
                    "joints_array": joints,
                    "transform_array": transforms,
                    "displacement": None,
                    "fps": 30.0,
                },
                "metadata": {
                    "target_class": "fall_related",
                    "proc_label": "fall to the ground",
                    "babel_sid": 794,
                }
            }, f)

        player = MotionPlayer(mock_pkl, playback_fps=30.0)
        self.assertEqual(player.total_frames, num_frames)
        self.assertEqual(player.action_class, "fall_related")
        self.assertEqual(player.action_label, "fall to the ground")
        self.assertAlmostEqual(player.duration_seconds, 20.0 / 30.0)

        # Step testing
        p0 = player.get_current_pose()
        self.assertEqual(p0["frame_index"], 0)
        self.assertFalse(player.is_finished())

        j, t = player.step(advance=True)
        self.assertEqual(j.shape, (216,))
        self.assertEqual(t.shape, (4, 4))
        self.assertEqual(player.current_frame, 1)

        # Seek testing
        player.seek(15)
        self.assertEqual(player.current_frame, 15)

        # Reset testing
        player.reset()
        self.assertEqual(player.current_frame, 0)

    # ------------------------------------------------------------------
    # TEST 4: Observation Record Serialization & JSON Compatibility
    # ------------------------------------------------------------------
    def test_04_observation_record_serialization(self):
        rec = ObservationRecord(
            frame_index=5,
            timestamp=5.0 / 30.0,
            action_class="fall_related",
            action_label="fall down",
            babel_sid=750,
            camera_position=[0.0, 1.2, 2.0],
            camera_yaw_deg=180.0,
            camera_pose_matrix=np.eye(4).tolist(),
            camera_intrinsics={"fx": 320.0, "fy": 320.0, "cx": 320.0, "cy": 240.0, "width": 640, "height": 480},
            human_base_position=[0.0, 0.0, 0.0],
            human_base_yaw=0.0,
            human_pose_gt_world={"pelvis": [0.0, 0.8, 0.0], "head": [0.0, 1.6, 0.0]},
            rgb_relative_path="runs/v70_obs/rgb/0005.png",
            depth_relative_path="runs/v70_obs/depth/0005.npy",
        )

        rec_dict = asdict(rec)
        json_str = json.dumps(rec_dict, indent=2)
        loaded = json.loads(json_str)

        self.assertEqual(loaded["frame_index"], 5)
        self.assertEqual(loaded["action_class"], "fall_related")
        self.assertEqual(loaded["human_pose_gt_world"]["head"], [0.0, 1.6, 0.0])
        self.assertEqual(loaded["camera_position"], [0.0, 1.2, 2.0])

    # ------------------------------------------------------------------
    # TEST 5: Surrounding Viewpoints Geometry Computation
    # ------------------------------------------------------------------
    def test_05_viewpoints_geometry(self):
        human_pos = np.array([1.0, 0.0, -1.0])
        radius = 2.5
        num_views = 8
        viewpoints = compute_viewpoints_around_human(
            human_pos=human_pos,
            radius=radius,
            num_views=num_views,
            camera_height=1.2,
        )

        self.assertEqual(len(viewpoints), 8)
        for vp in viewpoints:
            pos = vp["position"]
            dist_xz = math.sqrt((pos[0] - human_pos[0]) ** 2 + (pos[2] - human_pos[2]) ** 2)
            self.assertAlmostEqual(dist_xz, radius, places=4)
            self.assertIn("yaw_deg", vp)
            self.assertIn("view_id", vp)


if __name__ == "__main__":
    unittest.main()
