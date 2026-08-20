"""
Smoke Pipeline 测试 —— test_smoke_pipeline.py
=============================================
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ea_avs_mvp_v7.core.episode import Episode, EpisodeFrame
from ea_avs_mvp_v7.evaluation.basic_metrics import compute_episode_statistics
from ea_avs_mvp_v7.scripts.verify_episode_output import verify_episode_dir


class TestSmokePipeline(unittest.TestCase):
    """端到端感知管线与数据产物冒烟校验。"""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.ep_dir = Path(self.tmp_dir) / "smoke_ep"
        self.ep_dir.mkdir()
        (self.ep_dir / "rgb").mkdir()
        (self.ep_dir / "depth").mkdir()

        # 生成 1 帧虚拟数据
        from PIL import Image
        Image.new("RGB", (640, 480)).save(self.ep_dir / "rgb" / "frame_000000.png")
        np.save(self.ep_dir / "depth" / "frame_000000.npy", np.ones((480, 640), dtype=np.float32))

        meta = {
            "scene_id": "apartment_1",
            "episode_id": "smoke_ep",
            "motion_id": "fall_related_3522",
            "action_class": "fall_related",
            "action_label": "fall down",
            "robot_pose": [0.0, 0.1, 2.0, 180.0],
            "camera_pose": [[1, 0, 0, 0], [0, 1, 0, 1], [0, 0, 1, 2], [0, 0, 0, 1]],
            "human_pose_gt": {"pelvis": [0.0, 0.9, 0.0]},
            "frames": [
                {
                    "frame_id": 0,
                    "timestamp": 0.0,
                    "robot_pose": [0.0, 0.1, 2.0, 180.0],
                    "camera_pose": [[1, 0, 0, 0], [0, 1, 0, 1], [0, 0, 1, 2], [0, 0, 0, 1]],
                    "human_pose_gt": {"pelvis": [0.0, 0.9, 0.0]},
                    "rgb_path": str((self.ep_dir / "rgb" / "frame_000000.png").resolve()),
                    "depth_path": str((self.ep_dir / "depth" / "frame_000000.npy").resolve()),
                }
            ],
        }
        with open(self.ep_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_smoke_episode_verification(self):
        res = verify_episode_dir(self.ep_dir)
        self.assertEqual(res["episode_id"], "smoke_ep")
        self.assertEqual(res["num_frames"], 1)


if __name__ == "__main__":
    unittest.main()
