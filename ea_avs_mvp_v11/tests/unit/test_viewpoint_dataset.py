"""
ViewpointDatasetGenerator 单元测试 —— test_viewpoint_dataset.py
=============================================================
"""

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ea_avs_mvp_v11.active_view.viewpoint_dataset import ACTION_CLASSES, ViewpointDatasetGenerator


class TestViewpointDataset(unittest.TestCase):
    """测试视点质量数据集生成、样本结构、概率分布与无泄漏划分。"""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = Path(tempfile.mkdtemp(prefix="test_vp_dataset_"))
        cls.generator = ViewpointDatasetGenerator(estimator_type="mock")
        # 生成小规模微型测试数据集 (每类 2 个动作实例)
        cls.stats = cls.generator.generate_viewpoint_dataset(
            output_dir=cls.temp_dir,
            samples_per_action=2,
            train_ratio=0.50,
            val_ratio=0.25,
            test_ratio=0.25,
            seed=42,
        )

    @classmethod
    def tearDownClass(cls):
        if cls.temp_dir.exists():
            shutil.rmtree(cls.temp_dir)

    def test_sample_json_format_and_completeness(self):
        """测试生成的单样本 JSON 结构字段完整性。"""
        samples_dir = self.temp_dir / "samples"
        sample_files = list(samples_dir.glob("*.json"))
        self.assertGreater(len(sample_files), 0)

        required_fields = [
            "sample_id", "action_label", "action_id", "human_id", "motion_id",
            "scene_id", "split", "viewpoint", "pose_confidence", "action_probability",
            "predicted_action", "predicted_action_id", "is_correct", "entropy",
            "normalized_entropy", "confidence"
        ]

        with open(sample_files[0], "r", encoding="utf-8") as f:
            data = json.load(f)

        for field in required_fields:
            self.assertIn(field, data, f"Missing required field: {field}")

        self.assertIn(data["action_label"], ACTION_CLASSES)
        self.assertIsInstance(data["is_correct"], bool)

    def test_viewpoint_metadata_completeness(self):
        """测试 Viewpoint 结构字段完整性。"""
        samples_dir = self.temp_dir / "samples"
        sample_file = next(samples_dir.glob("*.json"))
        with open(sample_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        vp = data["viewpoint"]
        vp_required = ["id", "position", "rotation", "yaw", "pitch", "distance", "angle", "camera_height", "navigation_cost"]
        for f_name in vp_required:
            self.assertIn(f_name, vp, f"Missing viewpoint field: {f_name}")

        self.assertEqual(len(vp["position"]), 3)
        self.assertEqual(len(vp["rotation"]), 2)
        self.assertGreater(vp["distance"], 0.0)

    def test_action_probability_and_entropy_validity(self):
        """测试动作 Softmax 概率分布与 Shannon 熵数学合法性。"""
        metadata_path = self.temp_dir / "metadata.json"
        with open(metadata_path, "r", encoding="utf-8") as f:
            samples = json.load(f)

        for s in samples:
            probs = s["action_probability"]
            self.assertEqual(len(probs), len(ACTION_CLASSES))
            # 概率非负且和为 1
            for p in probs:
                self.assertGreaterEqual(p, -1e-6)
                self.assertLessEqual(p, 1.0 + 1e-6)
            self.assertAlmostEqual(sum(probs), 1.0, places=3)

            # 信息熵非负
            self.assertGreaterEqual(s["entropy"], 0.0)
            self.assertGreaterEqual(s["normalized_entropy"], 0.0)
            self.assertLessEqual(s["normalized_entropy"], 1.0 + 1e-5)

            # 置信度等于最大概率
            self.assertAlmostEqual(s["confidence"], max(probs), places=3)

    def test_dataset_split_disjointness_no_data_leakage(self):
        """测试 Train/Val/Test 划分严格互斥，杜绝同一 Motion Instance 跨集泄漏。"""
        splits_dir = self.temp_dir / "splits"
        with open(splits_dir / "train.json", "r", encoding="utf-8") as f:
            train_ids = set(json.load(f)["sample_ids"])
        with open(splits_dir / "val.json", "r", encoding="utf-8") as f:
            val_ids = set(json.load(f)["sample_ids"])
        with open(splits_dir / "test.json", "r", encoding="utf-8") as f:
            test_ids = set(json.load(f)["sample_ids"])

        # 样本 ID 互斥
        self.assertEqual(len(train_ids.intersection(val_ids)), 0)
        self.assertEqual(len(train_ids.intersection(test_ids)), 0)
        self.assertEqual(len(val_ids.intersection(test_ids)), 0)

        # 检查 Motion Instance ID 互斥
        metadata_path = self.temp_dir / "metadata.json"
        with open(metadata_path, "r", encoding="utf-8") as f:
            samples = json.load(f)

        train_motions = {s["motion_id"] for s in samples if s["sample_id"] in train_ids}
        val_motions = {s["motion_id"] for s in samples if s["sample_id"] in val_ids}
        test_motions = {s["motion_id"] for s in samples if s["sample_id"] in test_ids}

        self.assertEqual(len(train_motions.intersection(val_motions)), 0, "Data leakage detected between train and val!")
        self.assertEqual(len(train_motions.intersection(test_motions)), 0, "Data leakage detected between train and test!")
        self.assertEqual(len(val_motions.intersection(test_motions)), 0, "Data leakage detected between val and test!")


if __name__ == "__main__":
    unittest.main()
