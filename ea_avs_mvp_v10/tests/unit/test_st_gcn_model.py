"""
Unit test for ST-GCN network forward pass and dimension invariants.
"""

import unittest
import torch
import numpy as np

from ea_avs_mvp_v10.action_recognition.st_gcn_model import STGCN
from ea_avs_mvp_v10.perception.skeleton_definition import get_skeleton_definition


class TestSTGCNModel(unittest.TestCase):
    def setUp(self):
        self.skel_def = get_skeleton_definition()
        self.model = STGCN(
            in_channels=3,
            num_classes=6,
            graph_strategy="spatial",
            skel_def=self.skel_def,
            dropout=0.1,
        )

    def test_forward_pass_5d_tensor(self):
        # (N=2, C=3, T=30, V=33, M=1)
        x = torch.randn(2, 3, 30, 33, 1)
        out = self.model(x)

        self.assertEqual(out.shape, (2, 6))
        self.assertTrue(torch.all(torch.isfinite(out)))

    def test_forward_pass_4d_tensor(self):
        # (N=4, C=3, T=20, V=33)
        x = torch.randn(4, 3, 20, 33)
        out = self.model(x)

        self.assertEqual(out.shape, (4, 6))
        self.assertTrue(torch.all(torch.isfinite(out)))


if __name__ == "__main__":
    unittest.main()
