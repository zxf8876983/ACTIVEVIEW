"""
2D 姿态检测后端冒烟测试脚本 —— smoke_test_pose_backend.py
======================================================

功能：
    验证 2D 姿态估计后端（TorchVision KeypointRCNN 或 Mock）在当前运行环境下的
    初始化、前向推理与 Schema 转换。
"""

import argparse
import time
import numpy as np

from ea_avs_v6.config import load_config
from ea_avs_v6.pose_backend import create_pose_backend


def main():
    parser = argparse.ArgumentParser(description="Smoke test for 2D Pose Backend")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/mvp60_estimated_state.yaml",
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"[SmokeTest] Loaded config: {args.config}")

    backend = create_pose_backend(config)
    print(f"[SmokeTest] Initialized PoseBackend: {backend.backend_name}")

    # 构造虚拟 RGB 测试图像 (480, 640, 3)
    dummy_rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    t0 = time.time()
    detections = backend.infer(dummy_rgb)
    latency_ms = (time.time() - t0) * 1000.0

    print(f"[SmokeTest] Inference completed in {latency_ms:.2f} ms")
    print(f"[SmokeTest] Detections count: {len(detections)}")

    if detections:
        det = detections[0]
        print(f"[SmokeTest] Top detection score: {det.score:.3f}")
        print(f"[SmokeTest] Detected keypoints count: {sum(1 for k in det.keypoints.values() if k.detected)}/15")
    else:
        print("[SmokeTest] (No person detected in random noise dummy image, which is expected)")

    print("[SmokeTest] PASS: Pose backend inference smoke test completed successfully.")


if __name__ == "__main__":
    main()
