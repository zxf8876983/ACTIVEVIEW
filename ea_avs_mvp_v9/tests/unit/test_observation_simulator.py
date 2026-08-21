import unittest
from ea_avs_mvp_v9.core.types import ActionClass
from ea_avs_mvp_v9.features.observation_simulator import ObservationSimulator
from ea_avs_mvp_v9.training.dataset import create_mock_joints_for_action


class TestObservationSimulator(unittest.TestCase):
    def setUp(self):
        self.sim = ObservationSimulator()
        self.joints = create_mock_joints_for_action(ActionClass.SITTING, [1.5, -1.6, 4.0], yaw_deg=0.0)

    def test_simulate_no_occlusion(self):
        obs = self.sim.simulate_observation(
            gt_joints=self.joints,
            camera_pos=[1.5, -0.4, 6.0],  # 正前方 2.0m
            human_pos=[1.5, -1.6, 4.0],
            human_yaw_deg=0.0,
            degradation_mode="none",
        )
        self.assertGreater(obs.mean_confidence, 0.80)
        self.assertEqual(obs.missing_joint_count, 0)
        self.assertIn("head", obs.body_part_confidences)

    def test_simulate_self_occlusion(self):
        obs = self.sim.simulate_observation(
            gt_joints=self.joints,
            camera_pos=[1.5, -0.4, 2.0],  # 正后方
            human_pos=[1.5, -1.6, 4.0],
            human_yaw_deg=0.0,
            degradation_mode="self_occlusion",
        )
        self.assertLess(obs.body_part_confidences["left_hand"], 0.70)


if __name__ == "__main__":
    unittest.main()
