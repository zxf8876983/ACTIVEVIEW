"""
Unit tests for v10 paths and core types.
"""

import unittest
from pathlib import Path
import tempfile
import shutil

from ea_avs_mvp_v10.core.paths import (
    get_data_root,
    get_repo_root,
    get_v10_dataset_root,
    init_v10_dataset_dirs,
)
from ea_avs_mvp_v10.core.types import (
    ActionClassV10,
    CameraIntrinsics,
    CameraPose,
    CandidateViewpointV10,
    V10Sample,
)


class TestV10PathsAndTypes(unittest.TestCase):
    def test_paths_resolution(self):
        repo_root = get_repo_root()
        self.assertTrue(repo_root.exists())
        self.assertTrue((repo_root / "ea_avs_mvp_v10").exists())

        data_root = get_data_root()
        self.assertTrue(data_root.exists())

        v10_root = get_v10_dataset_root()
        self.assertTrue(v10_root.exists())
        self.assertEqual(v10_root.name, "v10")

    def test_init_v10_dataset_dirs(self):
        with tempfile.TemporaryDirectory() as tmp_d:
            tmp_p = Path(tmp_d)
            dirs = init_v10_dataset_dirs(tmp_p)
            self.assertTrue(dirs["raw_rgb"].exists())
            self.assertTrue(dirs["raw_depth"].exists())
            self.assertTrue(dirs["raw_camera_pose"].exists())
            self.assertTrue(dirs["raw_scene_meta"].exists())
            self.assertTrue(dirs["gt_skeleton"].exists())
            self.assertTrue(dirs["gt_action"].exists())
            self.assertTrue(dirs["metadata"].exists())

    def test_action_class_enum(self):
        self.assertEqual(ActionClassV10.from_str("standing"), ActionClassV10.STANDING)
        self.assertEqual(ActionClassV10.from_str("walk"), ActionClassV10.WALKING)
        self.assertEqual(ActionClassV10.from_str("sitting_4336"), ActionClassV10.SITTING)
        self.assertEqual(ActionClassV10.from_str("bending_12956"), ActionClassV10.BENDING)
        self.assertEqual(ActionClassV10.from_str("reaching"), ActionClassV10.REACHING)
        self.assertEqual(ActionClassV10.from_str("fall_related_3522"), ActionClassV10.FALLING)

    def test_camera_types_serialization(self):
        intr = CameraIntrinsics(width=640, height=480, fx=320.0, fy=320.0, cx=320.0, cy=240.0)
        pose = CameraPose(position=[1.0, 1.2, 3.0], rotation_quat=[0.0, 0.0, 0.0, 1.0], yaw_deg=45.0, intrinsics=intr)
        d = pose.to_dict()
        self.assertIn("position", d)
        self.assertIn("intrinsics", d)
        self.assertEqual(d["yaw_deg"], 45.0)

    def test_sample_type_serialization(self):
        intr = CameraIntrinsics()
        pose = CameraPose(position=[0.0, 1.2, 2.0], rotation_quat=[0.0, 0.0, 0.0, 1.0], intrinsics=intr)
        sample = V10Sample(
            sample_id="test_sample_01",
            scene_id="apartment_1",
            motion_id="sitting_4336",
            action_label="sitting",
            frame_idx=0,
            view_id="vp_000_r1.5_a000",
            camera_pose=pose,
            rgb_path="raw/rgb/test.png",
            depth_path="raw/depth/test.npy",
        )
        s_dict = sample.to_dict()
        self.assertEqual(s_dict["sample_id"], "test_sample_01")
        self.assertEqual(s_dict["action_label"], "sitting")


if __name__ == "__main__":
    unittest.main()
