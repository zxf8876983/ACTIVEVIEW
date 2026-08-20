"""
Keypoint Mapping 单元测试 —— test_keypoint_mapping.py
=====================================================
"""

import unittest
import numpy as np

from ea_avs_mvp_v7.human.keypoint_mapping import (
    HUMAN_16_KEYPOINTS,
    KEYPOINT_LINK_MAP,
    extract_human_keypoints_3d,
    validate_keypoints,
)


class MockSceneNode:
    def __init__(self, translation):
        self.transformation = type("Trans", (), {"translation": translation})()


class MockSimObj:
    def __init__(self):
        self.transformation = type("Trans", (), {"translation": [0.0, 0.0, 0.0]})()
        self.link_map = {
            "spine3": 1,
            "neck": 2,
            "head": 3,
            "left_shoulder": 4,
            "right_shoulder": 5,
            "left_elbow": 6,
            "right_elbow": 7,
            "left_wrist": 8,
            "right_wrist": 9,
            "left_hip": 10,
            "right_hip": 11,
            "left_knee": 12,
            "right_knee": 13,
            "left_ankle": 14,
            "right_ankle": 15,
        }

    def get_link_id_from_name(self, name):
        return self.link_map.get(name, -1)

    def get_link_scene_node(self, link_id):
        if link_id == 10:  # left_hip
            return MockSceneNode([-0.1, 0.9, 0.0])
        elif link_id == 11:  # right_hip
            return MockSceneNode([0.1, 0.9, 0.0])
        return MockSceneNode([0.0, float(link_id) * 0.1, 0.0])


class TestKeypointMapping(unittest.TestCase):
    """关键点映射与提取逻辑测试。"""

    def test_keypoint_list_integrity(self):
        self.assertEqual(len(HUMAN_16_KEYPOINTS), 16)
        self.assertIn("pelvis", HUMAN_16_KEYPOINTS)
        self.assertIn("head", HUMAN_16_KEYPOINTS)
        self.assertIn("spine3", HUMAN_16_KEYPOINTS)

    def test_keypoint_extraction_and_pelvis_derivation(self):
        mock_obj = MockSimObj()
        positions = extract_human_keypoints_3d(mock_obj)
        self.assertEqual(len(positions), 16)
        self.assertAlmostEqual(positions["pelvis"][0], 0.0, places=5)
        self.assertAlmostEqual(positions["pelvis"][1], 0.9, places=5)
        self.assertAlmostEqual(positions["pelvis"][2], 0.0, places=5)

        val_res = validate_keypoints(positions, min_joints=15)
        self.assertTrue(val_res["is_complete"])
        self.assertEqual(val_res["num_extracted"], 16)

    def test_insufficient_keypoints_raises_error(self):
        incomplete = {"head": [0.0, 1.7, 0.0]}
        with self.assertRaises(ValueError):
            validate_keypoints(incomplete, min_joints=15)


if __name__ == "__main__":
    unittest.main()
