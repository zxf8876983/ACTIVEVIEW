"""
ST-GCN 动作骨架数据集加载器 —— action_dataset.py
=============================================

职责：
    1. 提供标准 PyTorch Dataset 封装 (`ActionSkeletonDataset`)；
    2. 支持加载 (N, C, T, V, M) 或 (N, T, V, C) 格式的 3D 骨架序列与 labels.npy；
    3. 支持时间序列固定长度重采样与对齐 (Temporal Resampling / Cropping / Padding)；
    4. 提供标准 DataLoader 工厂方法。
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


def align_temporal_sequence(
    sequence: np.ndarray,
    target_len: int = 30,
) -> np.ndarray:
    """
    将时序数据 (T, ...) 线性重采样或填充至固定时间步长 target_len。
    """
    curr_len = sequence.shape[0]
    if curr_len == target_len:
        return sequence
    elif curr_len > target_len:
        # 均匀子采样
        indices = np.linspace(0, curr_len - 1, target_len).astype(int)
        return sequence[indices]
    else:
        # 重复最后一帧进行填充
        pad_len = target_len - curr_len
        last_frame = sequence[-1:]
        padding = np.repeat(last_frame, pad_len, axis=0)
        return np.concatenate([sequence, padding], axis=0)


class ActionSkeletonDataset(Dataset):
    """用于 ST-GCN 训练与测试的 3D 动作骨骼数据集。"""

    def __init__(
        self,
        data: Union[np.ndarray, str, Path],
        labels: Union[np.ndarray, str, Path],
        target_time_steps: int = 30,
        transform: Optional[Any] = None,
    ):
        # 加载数据
        if isinstance(data, (str, Path)):
            self.data = np.load(data)
        else:
            self.data = np.array(data)

        # 加载标签
        if isinstance(labels, (str, Path)):
            self.labels = np.load(labels)
        else:
            self.labels = np.array(labels)

        assert len(self.data) == len(self.labels), f"Data length ({len(self.data)}) != Labels length ({len(self.labels)})"

        self.target_time_steps = target_time_steps
        self.transform = transform

        # 统一格式化为 (N, C, T, V, M)
        self.formatted_data = self._format_data(self.data)

    def _format_data(self, raw_data: np.ndarray) -> np.ndarray:
        # 输入可能是:
        # 1. (N, C, T, V, M)
        # 2. (N, C, T, V)
        # 3. (N, T, V, C)
        N = raw_data.shape[0]
        if raw_data.ndim == 5:
            # (N, C, T, V, M)
            N, C, T, V, M = raw_data.shape
            if T != self.target_time_steps:
                formatted = np.zeros((N, C, self.target_time_steps, V, M), dtype=np.float32)
                for n in range(N):
                    for m in range(M):
                        seq = raw_data[n, :, :, :, m].transpose(1, 2, 0) # (T, V, C)
                        seq_resampled = align_temporal_sequence(seq, self.target_time_steps)
                        formatted[n, :, :, :, m] = seq_resampled.transpose(2, 0, 1)
                return formatted
            return raw_data.astype(np.float32)
        elif raw_data.ndim == 4:
            if raw_data.shape[1] == 3:
                # (N, C, T, V) -> (N, C, T, V, 1)
                return self._format_data(raw_data[..., np.newaxis])
            else:
                # (N, T, V, C) -> (N, C, T, V, 1)
                transposed = np.transpose(raw_data, (0, 3, 1, 2))[..., np.newaxis] # (N, C, T, V, 1)
                return self._format_data(transposed)
        else:
            raise ValueError(f"Unsupported data shape: {raw_data.shape}")

    def __len__(self) -> int:
        return len(self.formatted_data)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample_tensor = torch.tensor(self.formatted_data[index], dtype=torch.float32)
        label_tensor = torch.tensor(int(self.labels[index]), dtype=torch.long)
        return sample_tensor, label_tensor


def create_action_dataloader(
    data: Union[np.ndarray, str, Path],
    labels: Union[np.ndarray, str, Path],
    batch_size: int = 16,
    shuffle: bool = True,
    target_time_steps: int = 30,
    num_workers: int = 0,
) -> DataLoader:
    """创建标准 PyTorch DataLoader。"""
    dataset = ActionSkeletonDataset(data, labels, target_time_steps=target_time_steps)
    return DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
    )
