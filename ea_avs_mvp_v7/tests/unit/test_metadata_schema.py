"""
Metadata Schema 单元测试 —— test_metadata_schema.py
===================================================
"""

import unittest


class TestMetadataSchema(unittest.TestCase):
    """统一 Metadata 字典与位姿解耦规范校验测试。"""

    def test_metadata_fields_and_pose_separation(self):
        sample_meta = {
            "scene_id": "apartment_1",
            "episode_id": "v7_demo_fall_3522",
            "humanoid_id": "neutral_0",
            "motion_id": "fall_related_3522",
            "source_dataset": "AMASS",
            "action_class": "fall_related",
            "action_label": "fall to the ground",
            "frame_count": 15,
            "fps": 120.0,
            "robot_pose": {
                "position": [0.0, 0.1, 2.0],
                "rotation": [0.0, 1.0, 0.0, 0.0],
            },
            "camera_pose": {
                "extrinsic": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                "intrinsic": {"fx": 320.0, "fy": 320.0, "cx": 320.0, "cy": 240.0},
            },
            "frames": [
                {
                    "frame_idx": 0,
                    "timestamp": 0.0,
                    "rgb_path": "visualizations/v7_demo/rgb/frame_000000.png",
                    "depth_path": "visualizations/v7_demo/depth/frame_000000.npy",
                    "human_pose_gt": {"pelvis": [0.0, 0.9, 0.0]},
                }
            ],
            "motion_metrics": {
                "height_change": 0.523,
                "pelvis_velocity": 0.31,
                "joint_motion_energy": 0.87,
                "torso_angle_change": 49.7,
                "orientation_change": 145.8,
                "dynamic_motion": True,
            },
        }

        # 校验顶层字段
        required_top_keys = [
            "scene_id", "episode_id", "humanoid_id", "motion_id", "source_dataset",
            "action_class", "action_label", "frame_count", "fps", "robot_pose",
            "camera_pose", "frames", "motion_metrics",
        ]
        for k in required_top_keys:
            self.assertIn(k, sample_meta)

        # 校验位姿解耦
        self.assertIn("position", sample_meta["robot_pose"])
        self.assertIn("rotation", sample_meta["robot_pose"])
        self.assertIn("extrinsic", sample_meta["camera_pose"])
        self.assertIn("intrinsic", sample_meta["camera_pose"])

        # 校验每帧 GT 完整性
        self.assertIn("human_pose_gt", sample_meta["frames"][0])


if __name__ == "__main__":
    unittest.main()
