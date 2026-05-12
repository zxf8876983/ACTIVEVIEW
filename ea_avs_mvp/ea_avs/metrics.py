"""
实验指标记录模块 —— metrics.py
================================

功能：
    负责将实验结果写入磁盘文件，包括：
    - metrics.csv：结构化的表格数据，每行代表一个（episode × policy）的结果
    - episodes.jsonl：每个 episode 的摘要信息，JSONL 格式，每行一个 JSON 对象

输出文件格式说明：
    metrics.csv
        - 每行对应一个 (episode_id × policy) 组合
        - 成功 episode 有 4 行（Fixed/Random/Nearest/Ours）
        - 失败 episode 没有对应的 metrics 行
        - 包含所有原始数值，供后续数据分析使用
    
    episodes.jsonl
        - 每行对应一个 episode 的摘要
        - 成功 episode 包含 fixed_score、ours_score 等摘要信息
        - 失败 episode 包含失败原因
        - 适合快速查看实验概览
"""

import csv
import json
import os
from typing import Dict, List, Optional

import numpy as np


def _convert_numpy(obj):
    """
    递归地将对象中的 numpy 类型转换为 Python 原生类型。
    
    用途：
        json.dumps 无法序列化 numpy 的类型（如 np.float32、np.bool_ 等），
        此函数在写入 JSONL 前递归转换所有 numpy 类型。
    
    支持的转换：
        - np.integer → int
        - np.floating → float
        - np.bool_ → bool
        - np.ndarray → list
        - dict/list/tuple → 递归处理其元素
    """
    if isinstance(obj, dict):
        return {k: _convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(_convert_numpy(v) for v in obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


class MetricsWriter:
    """
    指标写入器 —— 同时管理 metrics.csv 和 episodes.jsonl 两个输出文件。
    
    使用方式：
        1. 在实验开始时创建 MetricsWriter 实例
        2. 在每个 episode 中对每个 policy 调用 write_metric_row()
        3. 在每个 episode 结束时调用 write_episode_summary()
        4. 实验结束时调用 close() 关闭文件
    """

    # CSV 字段定义（也是列标题）
    # 每个字段的含义：
    #   episode_id:       episode 编号
    #   scene_id:         场景名称（从 .glb 文件名提取）
    #   policy:           策略名称（Fixed/Random/Nearest/Ours）
    #   status:           状态（success/failed）
    #   num_candidates:   有效候选点数量
    #   human_x/y/z:      人体目标位置
    #   robot_start_x/y/z: 机器人起始位置
    #   selected_x/y/z:   策略选中的视角位置
    #   selected_yaw:     策略选中的视角朝向（弧度）
    #   geodesic_distance: 测地距离（米）
    #   S_kp:             关键点可见率评分
    #   S_center:          居中度评分
    #   S_dist:            距离评分
    #   C_move:            运动代价
    #   Q:                 综合评分
    #   torso/下肢/头部可见率
    CSV_FIELDS = [
        "episode_id", "scene_id", "policy", "status",
        "num_candidates",
        "human_x", "human_y", "human_z",
        "robot_start_x", "robot_start_y", "robot_start_z",
        "selected_x", "selected_y", "selected_z",
        "selected_yaw",
        "geodesic_distance",
        "S_kp", "S_center", "S_dist", "C_move", "Q",
        "torso_visibility", "lower_body_visibility", "head_visibility",
    ]

    def __init__(self, output_dir: str):
        """
        初始化写入器，创建 CSV 和 JSONL 文件。

        参数：
            output_dir: 输出目录路径。如果目录不存在，自动创建。
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # ---------- 创建 metrics.csv ----------
        self._csv_file = open(
            os.path.join(output_dir, "metrics.csv"), "w", newline=""
        )
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self.CSV_FIELDS)
        self._csv_writer.writeheader()  # 写入列标题

        # ---------- 创建 episodes.jsonl ----------
        self._jsonl_file = open(
            os.path.join(output_dir, "episodes.jsonl"), "w"
        )

    def write_metric_row(self, row: dict):
        """
        写入一行数据到 metrics.csv。

        参数：
            row: 字典，包含 CSV_FIELDS 中定义的字段。
                 不在 CSV_FIELDS 中的字段会被忽略。

        注意：
            每写入一行立即 flush，防止程序崩溃时数据丢失。
        """
        # 只保留 CSV 定义中存在的字段，按顺序写入
        filtered = {k: row.get(k, "") for k in self.CSV_FIELDS}
        self._csv_writer.writerow(filtered)
        self._csv_file.flush()

    def write_episode_summary(self, summary: dict):
        """
        写入一行 episode 摘要到 episodes.jsonl。

        参数：
            summary: 字典，包含 episode 的摘要信息。
                     包括 episode_id, status, 评分等。
                     失败 episode 还包含 reason 字段。

        注意：
            写入前会自动转换 numpy 类型为 Python 原生类型。
        """
        clean = _convert_numpy(summary)  # 确保 JSON 可序列化
        self._jsonl_file.write(json.dumps(clean, ensure_ascii=False) + "\n")
        self._jsonl_file.flush()

    def close(self):
        """关闭所有打开的输出文件。应在实验结束后调用。"""
        if hasattr(self, "_csv_file") and not self._csv_file.closed:
            self._csv_file.close()
        if hasattr(self, "_jsonl_file") and not self._jsonl_file.closed:
            self._jsonl_file.close()
