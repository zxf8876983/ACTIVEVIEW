"""
Episode 模块单元测试 —— test_episode.py
=======================================
"""

import json
import unittest

from ea_avs_mvp_v7.core.episode import Episode, EpisodeFrame


class TestEpisode(unittest.TestCase):
    """Episode 数据结构与序列化测试。"""

    def test_episode_frame_serialization(self):
        f = EpisodeFrame(
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
            action_label="fall to the ground",
            babel_sid=3522,
            rgb_relative_path="runs/ep_01/rgb/0000.png",
            depth_relative_path="runs/ep_01/depth/0000.npy",
        )

        d = f.to_dict()
        self.assertEqual(d["frame_index"], 0)
        self.assertEqual(d["action_class"], "fall_related")

        json_str = json.dumps(d)
        loaded = json.loads(json_str)
        self.assertEqual(loaded["babel_sid"], 3522)

    def test_episode_serialization(self):
        ep = Episode(
            episode_id="ep_001",
            scene_id="apartment_1",
            motion_id="fall_related_3522",
            action_class="fall_related",
            action_label="fall down",
            num_frames=1,
            camera_view_id="view_00",
            camera_initial_position=[0.0, 1.2, 2.0],
            camera_initial_yaw_deg=180.0,
            human_initial_position=[0.0, 0.0, 0.0],
            human_initial_yaw_deg=0.0,
            frames=[],
        )

        d = ep.to_dict()
        self.assertEqual(d["episode_id"], "ep_001")
        self.assertEqual(d["scene_id"], "apartment_1")
        self.assertEqual(d["num_frames"], 1)


if __name__ == "__main__":
    unittest.main()
