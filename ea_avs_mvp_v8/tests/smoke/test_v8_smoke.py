"""
v8 冒烟集成测试 —— test_v8_smoke.py
===================================
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from ea_avs_mvp_v8.human.human_placement import HumanPlacement
from ea_avs_mvp_v8.viewpoint.viewpoint_generator import ViewpointGenerator
from ea_avs_mvp_v8.constraints.constraint_checker import ConstraintChecker
from ea_avs_mvp_v8.visibility.visibility_evaluator import VisibilityEvaluator
from ea_avs_mvp_v8.dataset.v8_dataset_generator import save_v8_viewpoint_dataset


class TestV8Smoke(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_end_to_end_data_flow(self):
        placer = HumanPlacement()
        pose = placer.sample_position("apartment_1")

        vp_gen = ViewpointGenerator({"radii": [2.0], "num_angles": 4})
        candidates = vp_gen.generate_candidates(pose.position, pose.yaw_deg)

        checker = ConstraintChecker(config={"check_navmesh": False})
        checked = checker.filter_feasible_viewpoints(candidates)

        evaluator = VisibilityEvaluator()
        qualities = evaluator.evaluate_batch(checked, {"pelvis": [1.5, -0.8, 4.0]})

        out_path = save_v8_viewpoint_dataset(
            output_dir=self.tmp_dir,
            scene_id="apartment_1",
            episode_id="smoke_v8",
            human_pose=pose,
            action_info={"action_class": "standing", "avatar": "neutral_0"},
            robot_start_pos=[1.5, -1.6, 6.8],
            candidate_views=checked,
            view_qualities=qualities,
        )

        self.assertTrue((Path(out_path) / "candidate_views.json").exists())
        self.assertTrue((Path(out_path) / "visibility.json").exists())
        self.assertTrue((Path(out_path) / "metadata.json").exists())

        with open(Path(out_path) / "metadata.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            self.assertEqual(data["scene_id"], "apartment_1")
            self.assertEqual(data["candidate_views"]["total_candidates"], 4)


if __name__ == "__main__":
    unittest.main()
