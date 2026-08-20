"""
Video Generator 单元测试 —— test_video_generator.py
===================================================
"""

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from ea_avs_mvp_v7.scripts.create_video import create_video_from_frames


class TestVideoGenerator(unittest.TestCase):
    """图像序列转 MP4 视频编码测试。"""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.rgb_dir = Path(self.tmp_dir) / "rgb"
        self.rgb_dir.mkdir()

        # 生成 5 帧测试图片
        for i in range(5):
            img = Image.new("RGB", (320, 240), color=(i * 40, 100, 200))
            img.save(self.rgb_dir / f"frame_{i:06d}.png")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_video_creation_success(self):
        out_mp4 = Path(self.tmp_dir) / "test_out.mp4"
        created_path = create_video_from_frames(
            input_dir=self.rgb_dir,
            output_mp4_path=out_mp4,
            fps=5.0,
        )
        self.assertTrue(created_path.exists())
        self.assertGreater(created_path.stat().st_size, 0)

    def test_empty_dir_raises_error(self):
        empty_dir = Path(self.tmp_dir) / "empty"
        empty_dir.mkdir()
        with self.assertRaises(ValueError):
            create_video_from_frames(empty_dir, Path(self.tmp_dir) / "out.mp4")


if __name__ == "__main__":
    unittest.main()
