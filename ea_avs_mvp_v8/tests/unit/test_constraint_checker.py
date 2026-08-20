"""
Constraint Checker 单元测试 —— test_constraint_checker.py
========================================================
"""

import unittest
from ea_avs_mvp_v8.core.types import CandidateViewpoint
from ea_avs_mvp_v8.constraints.constraint_checker import ConstraintChecker


class TestConstraintChecker(unittest.TestCase):
    def setUp(self):
        # Offline/pure Python mode: env_adapter is None
        self.checker = ConstraintChecker(env_adapter=None, config={"check_navmesh": False})

    def test_filter_viewpoints_pure_python(self):
        vp1 = CandidateViewpoint(
            viewpoint_id="v1",
            position=[1.5, -1.60, 6.8],
            yaw_deg=0.0,
            radius=2.0,
            angle_deg=0.0,
        )
        vp2 = CandidateViewpoint(
            viewpoint_id="v2",
            position=[2.5, -1.60, 4.0],
            yaw_deg=90.0,
            radius=1.5,
            angle_deg=90.0,
        )
        results = self.checker.filter_feasible_viewpoints([vp1, vp2])
        self.assertEqual(len(results), 2)
        self.assertTrue(all(v.feasible for v in results))


if __name__ == "__main__":
    unittest.main()
