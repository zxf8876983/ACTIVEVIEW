"""
Episode Output 校验模块单元测试 —— test_episode_output.py
=========================================================
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from ea_avs_mvp_v7.scripts.verify_episode_output import verify_episode_dir


class TestEpisodeOutput(unittest.TestCase):
    """Episode 目录与 metadata.json 校验器测试。"""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.ep_dir = Path(self.tmp_dir) / "episode_test"
        self.ep_dir.mkdir()

        # 创建合规目录结构
        rgb_dir = self.ep_dir / "rgb"
        depth_dir = self.ep_dir / "depth"
        rgb_dir.mkdir()
        depth_dir.mkdir()

        # 创建 2 帧数据
        img = Image.new("RGB", (640, 480), (100, 100, 100))
        img.save(rgb_dir / "frame_000000.png")
        img.save(rgb_dir / "frame_000001.png")

        np.save(depth_dir / "frame_000000.npy", np.ones((480, 640), dtype=np.float32))
        np.save(depth_dir / "frame_000001.npy", np.ones((480, 640), dtype=np.float32))

        # 创建 metadata.json
        meta_content = {
            "scene_id": "apartment_1",
            "episode_id": "episode_test",
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
                    "rgb_path": str((rgb_dir / "frame_000000.png").resolve()),
                    "depth_path": str((depth_dir / "frame_000000.npy").resolve()),
                },
                {
                    "frame_id": 1,
                    "timestamp": 0.033,
                    "robot_pose": [0.0, 0.1, 2.0, 180.0],
                    "camera_pose": [[1, 0, 0, 0], [0, 1, 0, 1], [0, 0, 1, 2], [0, 0, 0, 1]],
                    "human_pose_gt": {"pelvis": [0.0, 0.85, 0.0]},
                    "rgb_path": str((rgb_dir / "frame_000001.png").resolve()),
                    "depth_path": str((depth_dir / "frame_000001.npy").resolve()),
                },
            ],
        }
        with open(self.ep_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta_content, f, indent=2)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_valid_episode_verification(self):
        """测试合法 Episode 目录校验通过。"""
        res = verify_episode_dir(self.ep_dir)
        self.assertEqual(res["episode_id"], "episode_test")
        self.assertEqual(res["num_frames"], 2)
        self.assertEqual(res["rgb_count"], 2)
        self.assertEqual(res["depth_count"], 2)

    def test_missing_metadata_raises_error(self):
        """测试缺少 metadata.json 时报错。"""
        (self.ep_dir / "metadata.json").unlink()
        with self.assertRaises(FileNotFoundError):
            verify_episode_dir(self.ep_dir)

    def test_missing_top_level_field_raises_error(self):
        """测试 metadata 缺失必要字段时报错。"""
        with open(self.ep_dir / "metadata.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        del meta["robot_pose"]
        with open(self.ep_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f)

        with self.assertRaises(KeyError):
            verify_episode_dir(self.ep_dir)


if __name__ == "__main__":
    unittest.main()
