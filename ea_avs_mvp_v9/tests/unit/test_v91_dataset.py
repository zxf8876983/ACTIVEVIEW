import unittest
from ea_avs_mvp_v9.core.types import ActionClass
from ea_avs_mvp_v9.training.dataset import create_mock_joints_for_action, generate_scoring_dataset


class TestV91Dataset(unittest.TestCase):
    def test_create_mock_joints_for_all_actions(self):
        for act in ALL_ACTION_CLASSES_VALS:
            joints = create_mock_joints_for_action(act, root_pos=[1.5, -1.6, 4.0], yaw_deg=30.0)
            self.assertIn("pelvis", joints)
            self.assertIn("head", joints)
            self.assertIn("left_knee", joints)
            self.assertEqual(len(joints), 17)

    def test_generate_scoring_dataset_split(self):
        train_ds, val_ds = generate_scoring_dataset(num_episodes=20, seed=42)
        self.assertEqual(len(train_ds), 16)
        self.assertEqual(len(val_ds), 4)

        sample = train_ds[0]
        self.assertEqual(sample["pose_vec"].shape, (49,))
        self.assertEqual(sample["view_vecs"].shape[1], 13)
        self.assertEqual(sample["target_scores"].shape[0], sample["view_vecs"].shape[0])


ALL_ACTION_CLASSES_VALS = [
    ActionClass.FALL,
    ActionClass.SITTING,
    ActionClass.STANDING,
    ActionClass.BENDING,
    ActionClass.REACHING,
]

if __name__ == "__main__":
    unittest.main()
