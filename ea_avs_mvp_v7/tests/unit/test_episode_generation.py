"""
Unit Tests for Episode Generation Schema & Structures
"""

import unittest
from ea_avs_mvp_v7.core.episode import Episode, EpisodeFrame


class TestEpisodeGeneration(unittest.TestCase):
    def test_episode_frame_creation(self):
        frame = EpisodeFrame(
            frame_index=0,
            timestamp=0.0,
            camera_position=[1.5, -1.60, 6.8],
            camera_yaw_deg=0.0,
            camera_pose_matrix=[[1.0, 0.0, 0.0, 1.5], [0.0, 1.0, 0.0, -0.4], [0.0, 0.0, 1.0, 6.8], [0.0, 0.0, 0.0, 1.0]],
            camera_intrinsics={"width": 640, "height": 480, "fx": 320.0, "fy": 320.0, "cx": 320.0, "cy": 240.0},
            human_base_position=[1.5, -1.60, 4.0],
            human_base_yaw=0.0,
            human_pose_gt_world={"pelvis": [1.5, -0.80, 4.0]},
            action_class="fall_related",
            action_label="fall to the ground",
            babel_sid="3522",
            rgb_relative_path="runs/test/rgb/frame_000000.png",
            depth_relative_path="runs/test/depth/frame_000000.npy",
        )
        self.assertEqual(frame.frame_index, 0)
        self.assertEqual(frame.action_class, "fall_related")
        self.assertIn("pelvis", frame.human_pose_gt_world)

    def test_episode_to_dict_conversion(self):
        ep = Episode(
            episode_id="ep_001",
            scene_id="apartment_1",
            motion_id="fall_related_3522",
            action_class="fall_related",
            action_label="fall to the ground",
            num_frames=1,
            camera_view_id="standard_view",
            camera_initial_position=[1.5, -1.60, 6.8],
            camera_initial_yaw_deg=0.0,
            human_initial_position=[1.5, -1.60, 4.0],
            human_initial_yaw_deg=0.0,
            frames=[],
            metadata={"test": True},
        )
        d = ep.to_dict()
        self.assertEqual(d["episode_id"], "ep_001")
        self.assertEqual(d["scene_id"], "apartment_1")
        self.assertEqual(d["num_frames"], 1)


if __name__ == "__main__":
    unittest.main()
