"""
Unit tests for v10 SampleBuilder.
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ea_avs_mvp_v11.core.types import CameraIntrinsics, CameraPose
from ea_avs_mvp_v11.dataset.sample_builder import V10SampleBuilder


class TestSampleBuilder(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.builder = V10SampleBuilder(dataset_root=self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_build_and_save_sample(self):
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        depth = np.ones((480, 640), dtype=np.float32) * 2.5
        intr = CameraIntrinsics()
        cam_pose = CameraPose(position=[1.0, 1.2, 3.0], rotation_quat=[0.0, 0.0, 0.0, 1.0], intrinsics=intr)
        gt_skel = {"pelvis": [1.0, 0.0, 1.0], "head": [1.0, 1.6, 1.0]}

        sample = self.builder.build_and_save_sample(
            sample_id="test_sample_001",
            scene_id="apartment_1",
            motion_id="sitting_4336",
            action_label="sitting",
            frame_idx=10,
            view_id="vp_000_r1.5_a000",
            rgb_array=rgb,
            depth_array=depth,
            camera_pose=cam_pose,
            gt_skeleton=gt_skel,
        )

        # 检查文件生成
        root = Path(self.tmp_dir)
        self.assertTrue((root / "raw" / "rgb" / "test_sample_001.png").exists())
        self.assertTrue((root / "raw" / "depth" / "test_sample_001.npy").exists())
        self.assertTrue((root / "raw" / "camera_pose" / "test_sample_001.json").exists())
        self.assertTrue((root / "ground_truth" / "skeleton" / "test_sample_001.json").exists())
        self.assertTrue((root / "ground_truth" / "action" / "test_sample_001.json").exists())

        # 测试保存 manifest
        manifest_p = self.builder.save_dataset_manifest([sample])
        self.assertTrue(manifest_p.exists())
        with open(manifest_p, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["total_samples"], 1)


if __name__ == "__main__":
    unittest.main()
