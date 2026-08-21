"""
v10 Phase 1 数据集批量生成 CLI —— generate_v10_dataset.py
=========================================================

职责：
    1. 接收 CLI 命令行参数 (scene_id, max_motions, frame_step, output_dir 等)；
    2. 执行大规模 Habitat 多视角 RGB-D 数据集生成与元数据持久化。
"""

import argparse
import logging
import sys
from pathlib import Path

from ea_avs_mvp_v10.core.config import load_v10_config
from ea_avs_mvp_v10.dataset.v10_dataset_generator import V10DatasetGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("generate_v10_dataset")


def parse_args():
    parser = argparse.ArgumentParser(description="ACTIVEVIEW v10.0 Phase 1 Dataset Generator")
    parser.add_argument("--config", type=str, default=None, help="Path to config yaml")
    parser.add_argument("--output_dir", type=str, default=None, help="Output dataset directory")
    parser.add_argument("--frame_step", type=int, default=10, help="Frame downsampling step")
    parser.add_argument("--max_frames", type=int, default=5, help="Max frames per motion")
    parser.add_argument("--max_viewpoints", type=int, default=4, help="Max candidate viewpoints to render")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_v10_config(args.config)
    generator = V10DatasetGenerator(config=cfg, dataset_root=args.output_dir)

    samples = generator.generate_motion_dataset(
        frame_step=args.frame_step,
        max_frames_per_motion=args.max_frames,
        max_viewpoints=args.max_viewpoints,
    )
    print(f"Successfully generated {len(samples)} samples to: {generator.dataset_root}")
    sys.exit(0)


if __name__ == "__main__":
    main()
