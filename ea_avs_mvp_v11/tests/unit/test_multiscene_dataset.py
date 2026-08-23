"""
Unit Tests for Multi-Scene Household Dataset & Episode Generation (v11.5)
"""
import sys
import unittest
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ea_avs_mvp_v11.active_view.human_placement_generator import HumanPlacementGenerator
from ea_avs_mvp_v11.active_view.multiscene_dataset_generator import MultiSceneViewpointDatasetGenerator
from ea_avs_mvp_v11.active_view.robot_start_sampler import RobotStartSampler
from ea_avs_mvp_v11.active_view.scene_manager import SceneManager


class TestMultiSceneDataset(unittest.TestCase):
    """测试多场景复杂家庭环境数据集各模块。"""

    def setUp(self):
        self.scene_mgr = SceneManager()
        self.human_gen = HumanPlacementGenerator(scene_manager=self.scene_mgr, seed=42)
        self.robot_sampler = RobotStartSampler(scene_manager=self.scene_mgr, seed=42)

    def test_household_scene_discovery(self):
        """测试场景管理器发现并检索场景。"""
        scenes = self.scene_mgr.get_primary_scenes(count=10)
        self.assertGreaterEqual(len(scenes), 3)

    def test_human_placement_difficulty(self):
        """测试人体空间放置包含难度得分与附近家具信息。"""
        scene_id = "apartment_1"
        placement = self.human_gen.sample_human_placement(scene_id=scene_id)
        self.assertTrue(placement["is_valid"])
        self.assertIn("placement_difficulty", placement)
        self.assertIn("nearby_objects", placement)
        self.assertGreaterEqual(placement["placement_difficulty"], 0.0)
        self.assertLessEqual(placement["placement_difficulty"], 1.0)

    def test_robot_start_visibility_metadata(self):
        """测试机器人初始位姿采样包含初始可见性与遮挡类型。"""
        scene_id = "apartment_1"
        placement = self.human_gen.sample_human_placement(scene_id=scene_id)
        start_vp = self.robot_sampler.sample_robot_start(human_placement=placement)
        self.assertIn("visibility_before_action", start_vp)
        self.assertIn("initial_occlusion_ratio", start_vp)
        self.assertIn("initial_observation_type", start_vp)
        self.assertIn(start_vp["initial_observation_type"], ["unoccluded", "partially_occluded", "far_observation"])


if __name__ == "__main__":
    unittest.main()
