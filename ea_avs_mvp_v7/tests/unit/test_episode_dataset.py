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
    """Episode 数据结构与统计指标单元测试。"""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.rgb_p = Path(self.tmp_dir) / "frame_000000.png"
        self.depth_p = Path(self.tmp_dir) / "frame_000000.npy"

        Image.new("RGB", (640, 480)).save(self.rgb_p)
        np.save(self.depth_p, np.ones((480, 640), dtype=np.float32))

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_episode_serialization_and_stats(self):
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

        d = ep.to_dict()
        self.assertEqual(d["episode_id"], "ep_test")
        self.assertEqual(len(d["frames"]), 1)

        stats = compute_episode_statistics(ep)
        self.assertEqual(stats["total_frames"], 1)
        self.assertEqual(stats["rgb_valid_ratio"], 1.0)
        self.assertEqual(stats["depth_valid_ratio"], 1.0)


if __name__ == "__main__":
    unittest.main()
