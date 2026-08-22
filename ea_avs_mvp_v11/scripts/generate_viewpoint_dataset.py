#!/usr/bin/env python3
"""
ACTIVEVIEW v11.2 视点质量数据集生成脚本 —— generate_viewpoint_dataset.py
======================================================================

职责：
    1. 批量生成针对 6 大动作类别、多候选视点与室内环境约束的 Viewpoint Quality Dataset；
    2. 计算每个观察视点下的动作识别概率分布、Shannon 信息熵与正确性标签；
    3. 输出标准统计指标并保存 train/val/test 划分索引。
"""

import argparse
import logging
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ea_avs_mvp_v11.active_view.viewpoint_dataset import ViewpointDatasetGenerator
from ea_avs_mvp_v11.core.paths import get_data_root

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("generate_viewpoint_dataset")


def main():
    parser = argparse.ArgumentParser(description="Generate Viewpoint Quality Dataset for ACTIVEVIEW v11.2")
    parser.add_argument("--samples_per_action", type=int, default=50, help="Number of human motion instances per action class")
    parser.add_argument("--estimator_type", type=str, default="mock", choices=["mock", "mediapipe"], help="Pose estimator backend")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    args = parser.parse_args()

    data_root = get_data_root()
    output_dir = Path(args.output_dir) if args.output_dir else (data_root / "v11_viewpoint_dataset")

    generator = ViewpointDatasetGenerator(data_root=data_root, estimator_type=args.estimator_type)
    stats = generator.generate_viewpoint_dataset(
        output_dir=output_dir,
        samples_per_action=args.samples_per_action,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    logger.info("Dataset Summary:")
    logger.info("  Total Samples: %d", stats["total_samples"])
    logger.info("  Overall Accuracy: %.2f%%", stats["overall_accuracy"] * 100)
    logger.info("  Mean Entropy: %.4f", stats["mean_entropy"])
    logger.info("  Mean Confidence: %.4f", stats["mean_confidence"])


if __name__ == "__main__":
    main()
