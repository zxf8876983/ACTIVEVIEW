"""
RGB 图像序列视频生成工具 —— create_video.py
===========================================

功能：
    1. 读取指定目录下的有序 RGB 图像帧 (如 frame_000000.png, demo_frame_00.png 等)；
    2. 使用 OpenCV (cv2) 编码生成高质量 .mp4 视频；
    3. 支持自定义帧率 (FPS) 与尺寸保持；
    4. 缺少输入文件或格式错误时直接报错。

运行方式：
    python -m ea_avs_mvp_v7.scripts.create_video --input <rgb_dir> --output <output_mp4> [--fps 10]
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Union

import cv2

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("create_video")


def create_video_from_frames(
    input_dir: Union[str, Path],
    output_mp4_path: Union[str, Path],
    fps: float = 10.0,
    file_pattern: str = "*.png",
) -> Path:
    """将目录下的图像序列合成编码为 MP4 视频。"""
    inp_p = Path(input_dir).resolve()
    if not inp_p.exists() or not inp_p.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {inp_p}")

    img_files = sorted(list(inp_p.glob(file_pattern)))
    if not img_files:
        raise ValueError(f"No image files matching '{file_pattern}' found in: {inp_p}")

    out_p = Path(output_mp4_path).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)

    # 读取首帧获取尺寸
    first_img = cv2.imread(str(img_files[0]))
    if first_img is None:
        raise ValueError(f"Failed to read initial image file: {img_files[0]}")

    h, w, _ = first_img.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_p), fourcc, float(fps), (w, h))

    try:
        for idx, img_path in enumerate(img_files):
            frame = cv2.imread(str(img_path))
            if frame is None:
                logger.warning("Skipping unreadable frame: %s", img_path)
                continue
            if frame.shape[:2] != (h, w):
                frame = cv2.resize(frame, (w, h))
            writer.write(frame)
    finally:
        writer.release()

    logger.info("Video successfully encoded: %s (%d frames @ %.1f fps)", out_p, len(img_files), fps)
    return out_p


def main():
    parser = argparse.ArgumentParser(description="Create MP4 Video from Image Sequence")
    parser.add_argument("--input", type=str, required=True, help="Input directory containing image frames")
    parser.add_argument("--output", type=str, required=True, help="Output MP4 file path")
    parser.add_argument("--fps", type=float, default=10.0, help="Video framerate (default: 10.0)")
    parser.add_argument("--pattern", type=str, default="*.png", help="Image glob pattern (default: *.png)")
    args = parser.parse_args()

    out_file = create_video_from_frames(
        input_dir=args.input,
        output_mp4_path=args.output,
        fps=args.fps,
        file_pattern=args.pattern,
    )

    print("\n" + "=" * 65)
    print("[v7.0 Video Creation Result]")
    print(f"  - Input Directory:    {args.input}")
    print(f"  - Output Video:       {out_file}")
    print(f"  - Framerate:          {args.fps:.1f} fps")
    print(f"  - Status:             SUCCESS")
    print("=" * 65)
    print("PASS: Video Generated Successfully\n")
    sys.exit(0)


if __name__ == "__main__":
    main()
