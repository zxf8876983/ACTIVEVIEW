import unittest
import numpy as np
import torch
from ea_avs_mvp_v9.core.types import ObservationState
from ea_avs_mvp_v9.models.observation_encoder import ObservationEncoder, extract_observation_vector


class TestObservationEncoder(unittest.TestCase):
    def setUp(self):
        self.encoder = ObservationEncoder(input_dim=71, hidden_dim=64, embed_dim=32)
        joints = {
            "pelvis": [1.5, -1.6, 4.0],
            "head": [1.5, -0.3, 4.0],
            "left_hand": [1.2, -0.8, 4.2],
        }
        confs = {k: 0.9 for k in ["pelvis", "head", "left_hand"]}
        parts = {k: 0.85 for k in ["head", "torso", "pelvis", "left_hand", "right_hand", "left_leg", "right_leg"]}
        self.obs = ObservationState(
            estimated_joints_3d=joints,
            joint_confidences=confs,
            body_part_confidences=parts,
            mean_confidence=0.88,
            missing_joint_count=1,
        )

    def test_extract_observation_vector_shape(self):
        vec = extract_observation_vector(self.obs)
        self.assertEqual(vec.shape, (71,))
        self.assertEqual(vec.dtype, np.float32)

    def test_observation_encoder_forward(self):
        vec = extract_observation_vector(self.obs)
        t = torch.tensor(vec).unsqueeze(0)  # (1, 71)
        out = self.encoder(t)
        self.assertEqual(out.shape, (1, 32))


if __name__ == "__main__":
    unittest.main()
