"""
Unit test for Action Classification and Shannon Entropy uncertainty estimation.
"""

import unittest
import numpy as np
import torch

from ea_avs_mvp_v11.action_recognition.action_classifier import ActionClassifier, ActionPredictionResult
from ea_avs_mvp_v11.action_recognition.st_gcn_model import STGCN
from ea_avs_mvp_v11.perception.skeleton_definition import get_skeleton_definition


class TestActionUncertainty(unittest.TestCase):
    def setUp(self):
        self.skel_def = get_skeleton_definition()
        self.classifier = ActionClassifier(skel_def=self.skel_def)

    def test_entropy_computation_deterministic(self):
        # 完全确定性概率分布: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        probs_det = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        ent, norm_ent, margin = ActionClassifier.compute_uncertainty_metrics(probs_det, num_classes=6)

        self.assertAlmostEqual(ent, 0.0, places=4)
        self.assertAlmostEqual(norm_ent, 0.0, places=4)
        self.assertAlmostEqual(margin, 1.0, places=4)

    def test_entropy_computation_uniform(self):
        # 完全不确定均匀分布: [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]
        probs_unif = np.ones(6) / 6.0
        ent, norm_ent, margin = ActionClassifier.compute_uncertainty_metrics(probs_unif, num_classes=6)

        self.assertAlmostEqual(norm_ent, 1.0, places=4)
        self.assertAlmostEqual(margin, 0.0, places=4)

    def test_predict_sequence_output_shape(self):
        # 随机生成骨架时序 (T=30, V=33, C=3)
        dummy_seq = np.random.randn(30, 33, 3).astype(np.float32)
        res = self.classifier.predict_sequence(dummy_seq)

        self.assertIsInstance(res, ActionPredictionResult)
        self.assertIn(res.predicted_label, self.classifier.action_classes)
        self.assertEqual(len(res.probabilities), 6)
        self.assertGreaterEqual(res.normalized_entropy, 0.0)
        self.assertLessEqual(res.normalized_entropy, 1.0)


if __name__ == "__main__":
    unittest.main()
