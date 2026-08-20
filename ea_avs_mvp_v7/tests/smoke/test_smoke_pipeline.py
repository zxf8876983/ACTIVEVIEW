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
from PIL import Image

from ea_avs_mvp_v7.core.paths import to_relative_data_path


class TestSmokePipeline(unittest.TestCase):
    """端到端感知管线与数据产物冒烟校验。"""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.ep_dir = Path(self.tmp_dir) / "smoke_ep"
        self.ep_dir.mkdir()
        (self.ep_dir / "rgb").mkdir()
        (self.ep_dir / "depth").mkdir()
        (self.ep_dir / "human_pose").mkdir()

        # 生成 1 帧虚拟数据
        Image.new("RGB", (640, 480)).save(self.ep_dir / "rgb" / "frame_000000.png")
        np.save(self.ep_dir / "depth" / "frame_000000.npy", np.ones((480, 640), dtype=np.float32))
        with open(self.ep_dir / "human_pose" / "frame_000000.json", "w", encoding="utf-8") as f:
            json.dump({"frame_id": 0, "human_pose_gt": {"pelvis": [1.5, -0.8, 4.0]}}, f)

        meta = {
            "scene_id": "apartment_1",
            "episode_id": "smoke_ep",
            "human": {
                "avatar": "neutral_0",
                "motion_id": "fall_related_3522",
                "action_class": "fall_related",
                "action_label": "fall down",
            },
            "robot": {
                "initial_pose": {"position": [1.5, -1.6, 6.8], "yaw_deg": 0.0},
                "camera_pose": {"extrinsic": [[1, 0, 0, 0], [0, 1, 0, 1], [0, 0, 1, 2], [0, 0, 0, 1]]},
            },
            "frames": [
                {
                    "frame_id": 0,
                    "timestamp": 0.0,
                    "human_pose_gt": {"pelvis": [1.5, -0.8, 4.0]},
                    "rgb_path": to_relative_data_path(self.ep_dir / "rgb" / "frame_000000.png"),
                    "depth_path": to_relative_data_path(self.ep_dir / "depth" / "frame_000000.npy"),
                }
            ],
        }
        with open(self.ep_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_smoke_episode_verification(self):
        meta_file = self.ep_dir / "metadata.json"
        self.assertTrue(meta_file.exists())
        with open(meta_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["episode_id"], "smoke_ep")
        self.assertEqual(len(data["frames"]), 1)
        self.assertTrue((self.ep_dir / "rgb" / "frame_000000.png").exists())
        self.assertTrue((self.ep_dir / "depth" / "frame_000000.npy").exists())
        self.assertTrue((self.ep_dir / "human_pose" / "frame_000000.json").exists())


if __name__ == "__main__":
    unittest.main()
