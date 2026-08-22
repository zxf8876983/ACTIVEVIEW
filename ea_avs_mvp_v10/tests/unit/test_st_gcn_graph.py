"""
Unit test for ST-GCN Graph topology construction & spatial partitioning.
"""

import unittest
import numpy as np

from ea_avs_mvp_v10.action_recognition.graph import Graph
from ea_avs_mvp_v10.perception.skeleton_definition import get_skeleton_definition


class TestSTGCNGraph(unittest.TestCase):
    def setUp(self):
        self.skel_def = get_skeleton_definition()

    def test_spatial_partitioning_graph(self):
        graph = Graph(strategy="spatial", max_hop=1, skel_def=self.skel_def)

        # 空间划分生成 3 个子图矩阵 (K=3, V=33, V=33)
        self.assertEqual(graph.A.shape, (3, 33, 33))
        # 矩阵元素非负且有限
        self.assertTrue(np.all(graph.A >= 0.0))
        self.assertTrue(np.all(np.isfinite(graph.A)))
        # 自环子图对角线应全为 1
        self.assertTrue(np.all(np.diag(graph.A[0]) > 0.0))

    def test_uniform_and_distance_graph(self):
        g_uni = Graph(strategy="uniform", max_hop=1, skel_def=self.skel_def)
        self.assertEqual(g_uni.A.shape, (1, 33, 33))

        g_dist = Graph(strategy="distance", max_hop=1, skel_def=self.skel_def)
        self.assertEqual(g_dist.A.shape, (2, 33, 33))


if __name__ == "__main__":
    unittest.main()
