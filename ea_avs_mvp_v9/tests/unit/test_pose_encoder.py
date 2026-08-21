import unittest
import numpy as np
import torch
from ea_avs_mvp_v9.models.pose_encoder import HumanPoseEncoder, extract_pose_vector


class TestPoseEncoder(unittest.TestCase):
    def setUp(self):
        self.mock_joints = {
            "pelvis": [1.5, -0.80, 4.0],
            "head": [1.5, -0.05, 4.0],
            "neck": [1.5, -0.20, 4.0],
            "spine": [1.5, -0.50, 4.0],
            "left_shoulder": [1.3, -0.30, 4.0],
            "right_shoulder": [1.7, -0.30, 4.0],
        }
        self.encoder = HumanPoseEncoder(input_dim=49, hidden_dim=64, embed_dim=32)

    def test_extract_pose_vector_dimension(self):
        vec = extract_pose_vector(self.mock_joints, human_yaw_deg=45.0)
        self.assertEqual(vec.shape, (49,))
        self.assertEqual(vec.dtype, np.float32)

    def test_pose_encoder_forward(self):
        vec = extract_pose_vector(self.mock_joints, human_yaw_deg=0.0)
        t = torch.tensor(vec).unsqueeze(0)  # (1, 49)
        out = self.encoder(t)
        self.assertEqual(out.shape, (1, 32))


if __name__ == "__main__":
    unittest.main()
