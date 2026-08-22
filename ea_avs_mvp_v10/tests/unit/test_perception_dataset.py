"""
Unit tests for perception dataset processing pipeline.
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path
import numpy as np

from ea_avs_mvp_v10.core.types import CameraIntrinsics, CameraPose, V10Sample
from ea_avs_mvp_v10.dataset.perception_dataset import V10PerceptionPipeline
from ea_avs_mvp_v10.perception.pose_estimator import MockPoseEstimator


class TestPerceptionDataset(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.mock_estimator = MockPoseEstimator()
        self.pipeline = V10PerceptionPipeline(
            pose_estimator=self.mock_estimator,
            dataset_root=self.tmp_dir,
        )

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_process_single_sample(self):
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        depth = np.ones((480, 640), dtype=np.float32) * 2.5
        intr = CameraIntrinsics()
        cam_pose = CameraPose(position=[0, 0, 0], rotation_quat=[0, 0, 0, 1], intrinsics=intr)

        sample = V10Sample(
            sample_id="test_perc_001",
            scene_id="apartment_1",
            motion_id="sitting_4336",
            action_label="sitting",
            frame_idx=0,
            view_id="vp_000_r1.5_a000",
            camera_pose=cam_pose,
            rgb_path="raw/rgb/dummy.png",
            depth_path="raw/depth/dummy.npy",
        )

        skel, record = self.pipeline.process_sample(
            sample=sample,
            rgb_image=rgb,
            depth_map=depth,
            save_outputs=True,
        )

        self.assertEqual(skel.joint_format, "COCO17")
        self.assertEqual(skel.joints_3d_cam.shape, (17, 3))
        self.assertEqual(skel.joints_3d_normalized.shape, (17, 3))
        self.assertEqual(record["sample_id"], "test_perc_001")
        self.assertEqual(record["num_valid_joints"], 17)

        # 检查文件是否落地
        root = Path(self.tmp_dir) / "perception"
        self.assertTrue((root / "pose2d" / "test_perc_001.json").exists())
        self.assertTrue((root / "pose3d" / "test_perc_001.json").exists())
        self.assertTrue((root / "normalized_pose3d" / "test_perc_001.json").exists())
        self.assertTrue((root / "confidence" / "test_perc_001.json").exists())


if __name__ == "__main__":
    unittest.main()
