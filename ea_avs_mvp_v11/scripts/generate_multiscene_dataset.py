#!/usr/bin/env python3
"""
ACTIVEVIEW v11.3 Multi-Scene Dataset Generation CLI Script
"""
import argparse
import logging
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ea_avs_mvp_v11.active_view.multiscene_dataset_generator import MultiSceneViewpointDatasetGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Generate Multi-Scene Active View Dataset")
    parser.add_argument("--output_dir", type=str, default=None, help="Output dataset directory")
    parser.add_argument("--total_episodes", type=int, default=300, help="Total episodes to generate")
    parser.add_argument("--estimator_type", type=str, default="oracle", help="Pose estimator type")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    generator = MultiSceneViewpointDatasetGenerator(estimator_type=args.estimator_type, seed=args.seed)
    generator.generate_multiscene_dataset(
        output_dir=args.output_dir,
        total_episodes=args.total_episodes,
    )


if __name__ == "__main__":
    main()
