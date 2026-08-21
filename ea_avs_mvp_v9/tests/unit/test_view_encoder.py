import unittest
import numpy as np
import torch
from ea_avs_mvp_v9.core.types import ViewFeature
from ea_avs_mvp_v9.models.view_encoder import ViewFeatureEncoder, extract_view_vector


class TestViewEncoder(unittest.TestCase):
    def setUp(self):
        self.feat = ViewFeature(
            viewpoint_id="vp_001",
            distance=2.0,
            viewing_angle_deg=30.0,
            pose_coverage=0.9,
            visibility_loss_ratio=0.1,
            projected_area_ratio=0.04,
            body_part_visibilities={
                "head": 1.0, "torso": 0.9, "pelvis": 0.8,
                "left_hand": 0.85, "right_hand": 0.85,
                "left_leg": 0.75, "right_leg": 0.75,
            },
            feasible=True,
        )
        self.encoder = ViewFeatureEncoder(input_dim=13, hidden_dim=64, embed_dim=32)

    def test_extract_view_vector_dimension(self):
        vec = extract_view_vector(self.feat)
        self.assertEqual(vec.shape, (13,))
        self.assertEqual(vec.dtype, np.float32)

    def test_view_encoder_forward(self):
        vec = extract_view_vector(self.feat)
        t = torch.tensor(vec).unsqueeze(0)  # (1, 13)
        out = self.encoder(t)
        self.assertEqual(out.shape, (1, 32))

    def test_view_encoder_batch_multi_view(self):
        vec = extract_view_vector(self.feat)
        arr = np.array([vec, vec], dtype=np.float32)
        t = torch.tensor(arr).unsqueeze(0)  # (1, 2, 13)
        flat_t = t.view(-1, 13)
        out = self.encoder(flat_t)
        self.assertEqual(out.shape, (2, 32))


if __name__ == "__main__":
    unittest.main()
