"""
ACTIVEVIEW v11.2.1 Metadata Enhancement 单元测试 —— test_v1121_metadata.py
========================================================================
"""

import json
import unittest
from pathlib import Path

from ea_avs_mvp_v11.core.paths import get_data_root


class TestV1121MetadataEnhancement(unittest.TestCase):
    """测试 v11.2.1 增强元数据的字段完整性、别名兼容性与几何有效性。"""

    @classmethod
    def setUpClass(cls):
        cls.data_root = get_data_root()
        cls.dataset_dir = cls.data_root / "v11_viewpoint_dataset"
        cls.metadata_path = cls.dataset_dir / "metadata.json"
        cls.stats_path = cls.dataset_dir / "dataset_statistics.json"

        if not cls.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {cls.metadata_path}")

        with open(cls.metadata_path, "r", encoding="utf-8") as f:
            cls.samples = json.load(f)

    def test_sample_count_preservation(self):
        """确认 8,400 个样本总数保持不变。"""
        self.assertEqual(len(self.samples), 8400)

    def test_current_viewpoint_structure(self):
        """测试所有 sample 均包含合法的 current_viewpoint 字段。"""
        required_cv_fields = ["position", "rotation", "yaw", "pitch", "distance_to_human", "angle_to_human"]

        for s in self.samples[:500]: # 抽样前 500 个深入校验
            self.assertIn("current_viewpoint", s)
            cv = s["current_viewpoint"]
            for field in required_cv_fields:
                self.assertIn(field, cv, f"Missing current_viewpoint field: {field}")
            self.assertEqual(len(cv["position"]), 3)
            self.assertEqual(len(cv["rotation"]), 2)
            self.assertGreater(cv["distance_to_human"], 0.0)

    def test_aliases_consistency(self):
        """测试 motion_instance_id 与 correctness 别名一致性。"""
        for s in self.samples[:500]:
            self.assertIn("motion_instance_id", s)
            self.assertEqual(s["motion_instance_id"], s["motion_id"])

            self.assertIn("correctness", s)
            self.assertEqual(s["correctness"], s["is_correct"])

    def test_candidate_pool_stats(self):
        """测试 sample 级与全局 candidate_pool 统计。"""
        for s in self.samples[:500]:
            self.assertIn("candidate_pool", s)
            cp = s["candidate_pool"]
            self.assertEqual(cp["raw_candidates"], 32)
            self.assertEqual(cp["feasible_candidates"], 28)

        with open(self.stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)

        self.assertIn("candidate_pool_statistics", stats)
        cp_stats = stats["candidate_pool_statistics"]
        self.assertEqual(cp_stats["raw_candidates"], 32)
        self.assertEqual(cp_stats["average_feasible_candidates"], 28.0)
        self.assertEqual(cp_stats["filtering_rate"], 0.125)


if __name__ == "__main__":
    unittest.main()
