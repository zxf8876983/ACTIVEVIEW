"""
Observation 模块单元测试 —— test_observation.py
===============================================
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ea_avs_mvp_v7.observation.metadata import FrameMetadata, SequenceMetadata
from ea_avs_mvp_v7.observation.recorder import ObservationRecorder


class TestObservation(unittest.TestCase):
    """ObservationRecorder 与 Metadata 测试。"""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_recorder_save_frame(self):
        recorder = ObservationRecorder(self.tmp_dir)
        target_dir = Path(self.tmp_dir) / "test_seq"

        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        depth = np.ones((480, 640), dtype=np.float32) * 2.5

        meta = FrameMetadata(
            frame_index=0,
            timestamp=0.0,
            action_class="standing",
            action_label="stand",
            babel_sid=10780,
            camera_position=[0.0, 1.2, 2.0],
            camera_yaw_deg=180.0,
            camera_pose_matrix=[[1, 0, 0, 0], [0, 1, 0, 1.2], [0, 0, 1, 2], [0, 0, 0, 1]],
            camera_intrinsics={"fx": 320.0, "fy": 320.0, "cx": 320.0, "cy": 240.0},
            human_base_position=[0.0, 0.0, 0.0],
            human_base_yaw=0.0,
            human_pose_gt_world={"pelvis": [0.0, 0.9, 0.0]},
        )

        res = recorder.record_frame(target_dir, 0, rgb, depth, meta)
        self.assertIsNotNone(res.rgb_relative_path)
        self.assertIsNotNone(res.depth_relative_path)
        self.assertTrue((target_dir / "rgb" / "0000.png").exists())
        self.assertTrue((target_dir / "depth" / "0000.npy").exists())
        self.assertTrue((target_dir / "metadata" / "0000.json").exists())

    def test_recorder_save_sequence_summary(self):
        recorder = ObservationRecorder(self.tmp_dir)
        target_dir = Path(self.tmp_dir) / "test_seq"
        seq_meta = SequenceMetadata(
            sequence_id="test_seq",
            action_class="standing",
            action_label="stand",
            babel_sid=10780,
            num_frames=1,
            camera_position=[0.0, 1.2, 2.0],
            camera_yaw_deg=180.0,
        )

        summary_p = recorder.record_sequence_summary(target_dir, seq_meta)
        self.assertTrue(summary_p.exists())
        with open(summary_p, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["sequence_id"], "test_seq")
        self.assertEqual(data["num_frames"], 1)


if __name__ == "__main__":
    unittest.main()
