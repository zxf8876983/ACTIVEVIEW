"""
Action Recognition Module for ACTIVEVIEW v10.0.
Provides ST-GCN graph definition, network architecture, training, and uncertainty-aware inference.
"""

from .action_classifier import ActionClassifier, ActionPredictionResult
from .action_dataset import ActionSkeletonDataset, create_action_dataloader
from .graph import Graph
from .st_gcn_model import STGCN, STGCNBlock
from .trainer import STGCNTrainer

__all__ = [
    "Graph",
    "STGCN",
    "STGCNBlock",
    "ActionClassifier",
    "ActionPredictionResult",
    "ActionSkeletonDataset",
    "create_action_dataloader",
    "STGCNTrainer",
]
