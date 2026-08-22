"""
Unit tests for perception dataset processing pipeline.
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from ea_avs_mvp_v10.core.types import CameraIntrinsics, CameraPose, V10Sample
from ea_avs_mvp_v10.dataset.perception_dataset import V10PerceptionPipeline
from ea_avs_mvp_v10.perception.coordinate_validator import CoordinateValidator
from ea_avs_mvp_v10.perception.rgbd_skeleton_extractor import MockRGBDSkeletonExtractor
from ea_avs_mvp_v10.perception.skeleton_normalizer import SkeletonNormalizer


class TestPerceptionDataset(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.dataset_root = Path(self.tmp_dir)

        # 创建假数据
        raw_rgb = self.dataset_root / "raw" / "rgb"
        raw_depth = self.dataset_root / "raw" / "depth"
        meta_dir = self.dataset_root / "metadata"
        raw_rgb.mkdir(parents=True, exist_ok=True)
        raw_depth.mkdir(parents=True, exist_ok=True)
        meta_dir.mkdir(parents=True, exist_ok=True)

        img = Image.new("RGB", (640, 480), color=(100, 100, 100))
        img.save(raw_rgb / "test_perc_001.png")
        depth = np.ones((480, 640), dtype=np.float32) * 2.0
        np.save(raw_depth / "test_perc_001.npy", depth)

        self.sample = V10Sample(
            sample_id="test_perc_001",
            scene_id="test_scene",
            motion_id="test_motion",
            action_label="standing",
            frame_idx=0,
            view_id="vp_001",
            camera_pose=CameraPose(
                position=[0.0, 0.0, 0.0],
                rotation_quat=[0.0, 0.0, 0.0, 1.0],
                yaw_deg=0.0,
                intrinsics=CameraIntrinsics(640, 480, 320.0, 320.0, 320.0, 240.0),
            ),
            rgb_path="raw/rgb/test_perc_001.png",
            depth_path="raw/depth/test_perc_001.npy",
        )

        with open(meta_dir / "samples.json", "w", encoding="utf-8") as f:
            json.dump({"samples": [self.sample.to_dict()]}, f)

        self.pipeline = V10PerceptionPipeline(
            extractor=MockRGBDSkeletonExtractor(default_confidence=0.95),
            skeleton_normalizer=SkeletonNormalizer(),
            coordinate_validator=CoordinateValidator(),
            dataset_root=self.dataset_root,
        )

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_process_single_sample(self):
        skel, record = self.pipeline.process_sample(self.sample, save_outputs=True)
        self.assertEqual(skel.joint_format, "mediapipe_33")
        self.assertEqual(record["sample_id"], "test_perc_001")
        self.assertEqual(record["validation_status"], "VALID")

        # 检查最新规范目录结构
        root = Path(self.tmp_dir) / "perception"
        self.assertTrue((root / "skeleton_raw" / "test_perc_001.json").exists())
        self.assertTrue((root / "skeleton_normalized" / "test_perc_001.json").exists())
        self.assertTrue((root / "confidence" / "test_perc_001.json").exists())

    def test_process_dataset_batch(self):
        records = self.pipeline.process_dataset()
        self.assertEqual(len(records), 1)
        manifest_p = self.dataset_root / "perception" / "metadata" / "perception_manifest.json"
        self.assertTrue(manifest_p.exists())

        with open(manifest_p, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["phase"], "phase_2_perception")
        self.assertEqual(data["extractor_name"], "MediaPipeBlazePose3D")
        self.assertEqual(data["joint_format"], "MediaPipe33")


if __name__ == "__main__":
    unittest.main()
