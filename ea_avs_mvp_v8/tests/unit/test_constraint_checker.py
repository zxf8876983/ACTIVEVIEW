"""
Constraint Checker 单元测试 —— test_constraint_checker.py
========================================================
"""

import unittest
from ea_avs_mvp_v8.core.types import CandidateViewpoint
from ea_avs_mvp_v8.constraints.constraint_checker import ConstraintChecker


class TestConstraintChecker(unittest.TestCase):
    def setUp(self):
        # Offline/pure Python mode
        self.checker = ConstraintChecker(
            env_adapter=None,
            config={
                "check_navmesh": False,
                "check_line_of_sight": False,
                "check_human_visibility": True,
            },
        )

    def test_filter_viewpoints_pure_python(self):
        # vp1 at Z=6.0 looking towards human at Z=4.0 (yaw=0.0)
        vp1 = CandidateViewpoint(
            viewpoint_id="v1",
            position=[1.5, -1.60, 6.0],
            yaw_deg=0.0,
            radius=2.0,
            angle_deg=0.0,
        )
        # vp2 at X=3.5 looking towards human at X=1.5 (yaw=270.0)
        vp2 = CandidateViewpoint(
            viewpoint_id="v2",
            position=[3.5, -1.60, 4.0],
            yaw_deg=270.0,
            radius=2.0,
            angle_deg=90.0,
        )
        results = self.checker.filter_feasible_viewpoints(
            [vp1, vp2],
            human_position=[1.5, -1.60, 4.0],
        )
        self.assertEqual(len(results), 2)
        self.assertTrue(all(v.feasible for v in results))

    def test_filter_rejects_out_of_fov_viewpoint(self):
        # vp_backwards looking away from human
        vp_backwards = CandidateViewpoint(
            viewpoint_id="v_back",
            position=[1.5, -1.60, 6.0],
            yaw_deg=180.0,  # Looking away (+Z)
            radius=2.0,
            angle_deg=0.0,
        )
        checked = self.checker.check_viewpoint(vp_backwards, human_position=[1.5, -1.60, 4.0])
        self.assertFalse(checked.feasible)
        self.assertEqual(checked.reason, "human_out_of_fov")


if __name__ == "__main__":
    unittest.main()
