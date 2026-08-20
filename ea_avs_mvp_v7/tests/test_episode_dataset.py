"""
Episode Dataset 单元测试 —— test_episode_dataset.py
===================================================
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from ea_avs_mvp_v7.core.episode import Episode, EpisodeFrame
from ea_avs_mvp_v7.core.paths import to_relative_data_path
from ea_avs_mvp_v7.evaluation.basic_metrics import compute_episode_statistics
from ea_avs_mvp_v7.observation.recorder import ObservationRecorder


class TestEpisodeDataset(unittest.TestCase):
    """Episode 数据集生成与统计指标测试。"""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.rgb_p = Path(self.tmp_dir) / "frame_000000.png"
        self.depth_p = Path(self.tmp_dir) / "frame_000000.npy"

        Image.new("RGB", (640, 480)).save(self.rgb_p)
        np.save(self.depth_p, np.ones((480, 640), dtype=np.float32))

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_episode_metrics_computation(self):
        """测试 Episode 质量指标计算。"""
        frames = [
            EpisodeFrame(
                frame_index=0,
                timestamp=0.0,
                camera_position=[0.0, 1.2, 2.0],
                camera_yaw_deg=180.0,
                camera_pose_matrix=[[1, 0, 0, 0], [0, 1, 0, 1.2], [0, 0, 1, 2], [0, 0, 0, 1]],
                camera_intrinsics={"fx": 320.0, "fy": 320.0, "cx": 320.0, "cy": 240.0},
                human_base_position=[0.0, 0.0, 0.0],
                human_base_yaw=0.0,
                human_pose_gt_world={"pelvis": [0.0, 0.9, 0.0], "head": [0.0, 1.7, 0.0]},
                action_class="fall_related",
                action_label="fall down",
                babel_sid=3522,
                rgb_relative_path=to_relative_data_path(self.rgb_p),
                depth_relative_path=to_relative_data_path(self.depth_p),
            )
        ]

        ep = Episode(
            episode_id="ep_test",
            scene_id="apartment_1",
            motion_id="fall_related_3522",
            action_class="fall_related",
            action_label="fall down",
            num_frames=1,
            camera_view_id="view_0",
            camera_initial_position=[0.0, 1.2, 2.0],
            camera_initial_yaw_deg=180.0,
            human_initial_position=[0.0, 0.0, 0.0],
            human_initial_yaw_deg=0.0,
            frames=frames,
        )

        stats = compute_episode_statistics(ep)
        self.assertEqual(stats["total_frames"], 1)
        self.assertEqual(stats["rgb_valid_ratio"], 1.0)
        self.assertEqual(stats["depth_valid_ratio"], 1.0)
        self.assertEqual(stats["avg_gt_keypoints_per_frame"], 2.0)

    def test_recorder_save_metadata(self):
        """测试 ObservationRecorder 生成顶层 metadata.json。"""
        recorder = ObservationRecorder(self.tmp_dir)
        target_dir = Path(self.tmp_dir) / "test_ep_dir"
        meta_dict = {
            "scene_id": "apartment_1",
            "episode_id": "test_ep_dir",
            "motion_id": "fall_related_3522",
            "action_class": "fall_related",
            "action_label": "fall down",
            "robot_pose": [0.0, 0.1, 2.0, 180.0],
            "camera_pose": [[1, 0, 0, 0], [0, 1, 0, 1], [0, 0, 1, 2], [0, 0, 0, 1]],
            "human_pose_gt": {"pelvis": [0.0, 0.9, 0.0]},
            "frames": [],
        }

        meta_path = recorder.record_episode_metadata(target_dir, meta_dict)
        self.assertTrue(meta_path.exists())
        with open(meta_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        self.assertEqual(loaded["scene_id"], "apartment_1")


if __name__ == "__main__":
    unittest.main()
