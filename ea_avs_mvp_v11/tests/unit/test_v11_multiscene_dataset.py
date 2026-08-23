"""
ACTIVEVIEW v11.3 Multi-Scene Active View Dataset & Predictor Unit Tests
======================================================================
"""

import json
import math
import unittest
from pathlib import Path

import numpy as np
import torch

from ea_avs_mvp_v11.active_view.human_placement_generator import HumanPlacementGenerator
from ea_avs_mvp_v11.active_view.robot_start_sampler import RobotStartSampler
from ea_avs_mvp_v11.active_view.scene_manager import SceneManager
from ea_avs_mvp_v11.active_view.utility_dataset import UtilityDatasetBuilder
from ea_avs_mvp_v11.active_view.utility_predictor import ViewpointUtilityPredictor
from ea_avs_mvp_v11.core.paths import get_data_root, get_repo_root


class TestMultiSceneDataset(unittest.TestCase):
    """多场景主动感知数据集与效用预测单元测试。"""

    @classmethod
    def setUpClass(cls):
        cls.data_root = get_data_root()
        cls.repo_root = get_repo_root()
        cls.scene_mgr = SceneManager()
        cls.human_gen = HumanPlacementGenerator(scene_manager=cls.scene_mgr, seed=42)
        cls.robot_sampler = RobotStartSampler(scene_manager=cls.scene_mgr, seed=42)

    def test_01_scene_manager_discovery(self):
        """测试 1: 场景管理器多目录扫描与资产索引。"""
        scenes = self.scene_mgr.discover_scenes()
        self.assertGreaterEqual(len(scenes), 3, "Should discover at least 3 scenes")
        primary_scenes = self.scene_mgr.get_primary_scenes(count=3)
        self.assertEqual(len(primary_scenes), 3, "Primary scenes count should be 3")
        for s in primary_scenes:
            self.assertTrue(s.glb_path.exists(), f"GLB file should exist: {s.glb_path}")
            self.assertIsInstance(s.floor_height, float)

    def test_02_human_placement_randomness(self):
        """测试 2: 人体放置生成器多样性与空间约束。"""
        placements = self.human_gen.batch_sample_placements(scene_id="apartment_1", num_placements=20)
        self.assertEqual(len(placements), 20)
        positions = [p["human_position"] for p in placements]
        yaws = [p["yaw"] for p in placements]

        # 验证位置与朝向具备随机分布
        x_coords = [p[0] for p in positions]
        self.assertGreater(np.std(x_coords), 0.1, "Human X positions should be varied")
        self.assertGreater(np.std(yaws), 10.0, "Human Yaws should be randomized")

    def test_03_robot_start_sampler_constraints(self):
        """测试 3: 机器人初始观察位姿采样约束。"""
        hp = self.human_gen.sample_human_placement(scene_id="apartment_1")
        cv = self.robot_sampler.sample_robot_start(human_placement=hp, min_dist=2.0, max_dist=8.0)

        dist = cv["distance_to_human"]
        self.assertGreaterEqual(dist, 1.5, "Distance should respect lower bound margin")
        self.assertLessEqual(dist, 9.0, "Distance should respect upper bound margin")
        self.assertIn("yaw", cv)
        self.assertIn("angle_to_human", cv)
        self.assertEqual(len(cv["position"]), 3)

    def test_04_multiscene_dataset_episodes_integrity(self):
        """测试 4: 多场景 Episode 数据集结构与划分完整性。"""
        d_dir = self.data_root / "v11_multiscene_viewpoint_dataset"
        self.assertTrue(d_dir.exists(), f"Dataset directory should exist: {d_dir}")

        stats_file = d_dir / "dataset_statistics.json"
        self.assertTrue(stats_file.exists(), "dataset_statistics.json should exist")
        with open(stats_file, "r", encoding="utf-8") as f:
            stats = json.load(f)

        self.assertEqual(stats["total_episodes"], 300)
        self.assertEqual(stats["total_samples"], 9600)
        self.assertGreaterEqual(stats["num_scenes"], 3)
        self.assertGreaterEqual(stats["non_zero_entropy_ratio"], 0.80, "Most viewpoints should have non-zero entropy")


        # 检查划分文件
        for split in ["train", "val", "test"]:
            s_file = d_dir / "splits" / f"{split}.json"
            self.assertTrue(s_file.exists())
            with open(s_file, "r", encoding="utf-8") as sf:
                s_data = json.load(sf)
                self.assertGreater(s_data["count"], 0)

    def test_05_multiscene_utility_dataset_properties(self):
        """测试 5: 多场景 Utility Dataset 11D 特征与目标属性。"""
        u_dir = self.data_root / "v11_multiscene_utility_dataset"
        self.assertTrue(u_dir.exists(), f"Utility dataset dir should exist: {u_dir}")

        with open(u_dir / "test.json", "r", encoding="utf-8") as f:
            test_data = json.load(f)

        self.assertEqual(len(test_data), 1536)
        first_rec = test_data[0]
        self.assertEqual(len(first_rec["features"]), 11, "Feature vector must be 11-dimensional")
        self.assertIn("utility_gain", first_rec)
        self.assertIn("scene_id", first_rec)

    def test_06_trained_multiscene_utility_predictor(self):
        """测试 6: 多场景 Utility Predictor 权重加载与前向推理。"""
        ckpt_p = self.data_root / "checkpoints" / "v11_utility_multiscene" / "utility_predictor_best.pth"
        self.assertTrue(ckpt_p.exists(), f"Checkpoint should exist at {ckpt_p}")

        predictor = ViewpointUtilityPredictor(model_path=ckpt_p, in_dim=11)
        dummy_feat = [0.6, 0.0, 1.0, 0.0, 1.0, 0.3, 0.0, 1.0, 0.0, 1.0, 0.2]
        pred_u = predictor.predict_single(dummy_feat)
        self.assertIsInstance(pred_u, float)
        self.assertGreater(pred_u, -2.0)
        self.assertLess(pred_u, 5.0)


if __name__ == "__main__":
    unittest.main()
