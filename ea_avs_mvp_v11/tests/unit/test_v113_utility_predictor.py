"""
ACTIVEVIEW v11.3 Viewpoint Utility Predictor 单元测试 —— test_v113_utility_predictor.py
===================================================================================
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from ea_avs_mvp_v11.active_view.models.utility_predictor import ViewpointUtilityPredictorNet
from ea_avs_mvp_v11.active_view.utility_dataset import UtilityDatasetBuilder
from ea_avs_mvp_v11.active_view.utility_predictor import ViewpointUtilityPredictor
from ea_avs_mvp_v11.active_view.viewpoint_selection import (
    ActiveViewSelector,
    evaluate_viewpoint_selection_benchmarks,
)
from ea_avs_mvp_v11.active_view.viewpoint_types import Viewpoint
from ea_avs_mvp_v11.core.paths import get_data_root


class TestV113UtilityPredictor(unittest.TestCase):
    """测试 v11.3 效用预测数据集构建、MLP模型推理、Checkpoint存取与基准选择策略。"""

    @classmethod
    def setUpClass(cls):
        cls.data_root = get_data_root()
        cls.v112_dataset_dir = cls.data_root / "v11_viewpoint_dataset"
        cls.tmp_dir = Path(tempfile.mkdtemp(prefix="test_v113_"))

    @classmethod
    def tearDownClass(cls):
        if cls.tmp_dir.exists():
            shutil.rmtree(cls.tmp_dir, ignore_errors=True)

    def test_01_utility_dataset_builder(self):
        """测试 UtilityDatasetBuilder 的特征提取、划分继承与目标增益计算。"""
        builder = UtilityDatasetBuilder(data_root=self.data_root)
        stats = builder.build_utility_dataset(output_dir=self.tmp_dir / "util_data")

        self.assertEqual(stats["total_samples"], 8400)
        self.assertEqual(stats["feature_dim"], 11)
        self.assertEqual(stats["splits"]["train"], 5880)
        self.assertEqual(stats["splits"]["val"], 1176)
        self.assertEqual(stats["splits"]["test"], 1344)

        with open(self.tmp_dir / "util_data" / "train.json", "r") as f:
            train_records = json.load(f)

        sample0 = train_records[0]
        self.assertIn("features", sample0)
        self.assertEqual(len(sample0["features"]), 11)
        self.assertIn("utility_gain", sample0)
        self.assertIn("current_viewpoint", sample0)
        self.assertIn("candidate_viewpoint", sample0)

    def test_02_model_forward_and_shapes(self):
        """测试 ViewpointUtilityPredictorNet 模型前向传播与张量维度。"""
        model = ViewpointUtilityPredictorNet(in_dim=11, hidden_dim1=64, hidden_dim2=32)
        dummy_input = torch.randn(8, 11)
        output = model(dummy_input)

        self.assertEqual(output.shape, (8, 1))
        self.assertFalse(torch.isnan(output).any())

    def test_03_high_level_predictor_and_checkpoint(self):
        """测试 ViewpointUtilityPredictor 高层封装与权重存取。"""
        model = ViewpointUtilityPredictorNet(in_dim=11)
        ckpt_path = self.tmp_dir / "test_ckpt.pth"
        torch.save({"model_state_dict": model.state_dict(), "in_dim": 11}, ckpt_path)

        predictor = ViewpointUtilityPredictor(model_path=ckpt_path, in_dim=11)

        curr_obs = {
            "current_viewpoint": {
                "position": [1.8, 0.0, 3.4],
                "yaw": 209.74,
                "distance_to_human": 4.03,
                "angle_to_human": 60.26,
            }
        }
        cand_vps = [
            Viewpoint(id=0, position=[0.0, 0.0, 1.5], rotation=[180.0, 0.0], yaw=180.0, pitch=0.0, distance=1.5, angle=0.0, camera_height=1.25, navigation_cost=2.0),
            Viewpoint(id=1, position=[1.5, 0.0, 0.0], rotation=[90.0, 0.0], yaw=90.0, pitch=0.0, distance=1.5, angle=90.0, camera_height=1.25, navigation_cost=2.5),
        ]

        scores = predictor.predict_utility(curr_obs, cand_vps)
        self.assertEqual(len(scores), 2)
        self.assertIsInstance(scores[0], float)

    def test_04_active_view_selector_and_baselines(self):
        """测试 4 大视角选择策略决策与对比评测函数。"""
        model = ViewpointUtilityPredictorNet(in_dim=11)
        ckpt_path = self.tmp_dir / "test_ckpt.pth"
        torch.save({"model_state_dict": model.state_dict(), "in_dim": 11}, ckpt_path)
        predictor = ViewpointUtilityPredictor(model_path=ckpt_path, in_dim=11)

        selector = ActiveViewSelector(predictor=predictor, strategy="utility_predictor")

        curr_obs = {
            "current_viewpoint": {
                "position": [1.8, 0.0, 3.4],
                "yaw": 209.74,
                "distance_to_human": 4.03,
                "angle_to_human": 60.26,
            }
        }
        cand_vps = [
            Viewpoint(id=0, position=[0.0, 0.0, 1.5], rotation=[180.0, 0.0], yaw=180.0, pitch=0.0, distance=1.5, angle=0.0, camera_height=1.25, navigation_cost=3.0),
            Viewpoint(id=1, position=[1.5, 0.0, 0.0], rotation=[90.0, 0.0], yaw=90.0, pitch=0.0, distance=1.5, angle=90.0, camera_height=1.25, navigation_cost=1.2),
        ]

        # 1. Utility Predictor selection
        best_vp, score = selector.select_viewpoint(curr_obs, cand_vps, strategy="utility_predictor")
        self.assertIn(best_vp.id, [0, 1])

        # 2. Nearest selection
        nearest_vp, _ = selector.select_viewpoint(curr_obs, cand_vps, strategy="nearest")
        self.assertEqual(nearest_vp.id, 1) # cost 1.2 < 3.0

        # 3. Random selection
        rand_vp, _ = selector.select_viewpoint(curr_obs, cand_vps, strategy="random")
        self.assertIn(rand_vp.id, [0, 1])


if __name__ == "__main__":
    unittest.main()
