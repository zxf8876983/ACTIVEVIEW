"""
Unit Tests for Occlusion Analyzer & Difficulty Level Stratification (v11.5)
"""
import sys
import unittest
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ea_avs_mvp_v11.active_view.occlusion_analyzer import OcclusionAnalyzer, ViewpointOcclusionResult


class TestOcclusionAnalysis(unittest.TestCase):
    """测试遮挡分析器各指标与难度分类。"""

    def setUp(self):
        self.analyzer = OcclusionAnalyzer()

    def test_frontal_low_occlusion(self):
        """测试正前方视角 (0 deg, 1.5m) 处于 Easy 等级。"""
        res = self.analyzer.analyze_viewpoint_occlusion(
            angle_deg=0.0,
            distance_m=1.5,
            placement_difficulty=0.2,
        )
        self.assertIsInstance(res, ViewpointOcclusionResult)
        self.assertLess(res.occlusion_ratio, 0.15)
        self.assertEqual(res.occlusion_level, "Easy")
        self.assertGreaterEqual(res.visible_joint_ratio, 0.85)

    def test_back_high_occlusion(self):
        """测试背向视角 (180 deg, 3.0m) 处于 Hard 等级。"""
        res = self.analyzer.analyze_viewpoint_occlusion(
            angle_deg=180.0,
            distance_m=3.0,
            placement_difficulty=0.8,
        )
        self.assertGreaterEqual(res.occlusion_ratio, 0.40)
        self.assertEqual(res.occlusion_level, "Hard")
        self.assertLessEqual(res.visible_joint_ratio, 0.60)

    def test_difficulty_thresholds(self):
        """测试三级难度阈值连续性。"""
        # Easy: < 0.10
        # Medium: 0.10 <= occ < 0.40
        # Hard: >= 0.40
        res_med = self.analyzer.analyze_viewpoint_occlusion(
            angle_deg=90.0,
            distance_m=2.0,
            placement_difficulty=0.4,
        )
        self.assertEqual(res_med.occlusion_level, "Medium")
        self.assertGreaterEqual(res_med.occlusion_ratio, 0.10)
        self.assertLess(res_med.occlusion_ratio, 0.40)

    def test_dataset_occlusion_statistics_computation(self):
        """测试批量遮挡指标统计。"""
        mock_candidates = [
            {"occlusion_ratio": 0.05, "scene_id": "sc1", "action_label": "walking"},
            {"occlusion_ratio": 0.25, "scene_id": "sc1", "action_label": "walking"},
            {"occlusion_ratio": 0.55, "scene_id": "sc2", "action_label": "sitting"},
        ]
        stats = self.analyzer.compute_dataset_occlusion_statistics(mock_candidates)
        self.assertEqual(stats["total_viewpoints_analyzed"], 3)
        self.assertEqual(stats["difficulty_level_breakdown"]["easy"]["count"], 1)
        self.assertEqual(stats["difficulty_level_breakdown"]["medium"]["count"], 1)
        self.assertEqual(stats["difficulty_level_breakdown"]["hard"]["count"], 1)


if __name__ == "__main__":
    unittest.main()
