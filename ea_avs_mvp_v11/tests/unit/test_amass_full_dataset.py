"""
Unit Tests for AMASS Full Dataset Registry & Action Management (v11.5)
"""
import sys
import unittest
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ea_avs_mvp_v11.active_view.action_registry import ActionRegistry, DEFAULT_ACTION_CATEGORIES


class TestAmassFullDataset(unittest.TestCase):
    """测试 AMASS 全量动作注册表与骨架格式。"""

    def setUp(self):
        self.registry = ActionRegistry()

    def test_categories_coverage(self):
        """测试 6 大主要动作类别完整注册。"""
        for cat in DEFAULT_ACTION_CATEGORIES:
            self.assertIn(cat, self.registry.categories)
            self.assertIn(cat, self.registry.category_to_id)

    def test_skeleton_sequence_shape(self):
        """测试骨架提取接口返回统一 (30, 33, 3) 形状。"""
        skel = self.registry.get_skeleton_sequence(action_id=0, instance_idx=0, split="train")
        self.assertEqual(skel.shape, (30, 33, 3))
        self.assertEqual(skel.dtype.name, "float32")

    def test_export_statistics(self):
        """测试动作统计导出。"""
        stats = self.registry.export_statistics()
        self.assertIn("total_motion_sequences", stats)
        self.assertIn("num_categories", stats)
        self.assertEqual(stats["num_categories"], len(self.registry.categories))
        self.assertEqual(stats["sequence_length"], 30)
        self.assertEqual(stats["joint_count"], 33)


if __name__ == "__main__":
    unittest.main()
