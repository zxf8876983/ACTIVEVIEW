"""
Motion Assets & Data Infrastructure 单元测试套件 —— test_motion_assets.py
==========================================================================

功能：
    在纯 Python 环境下测试动作筛选、路径解析、凭据防护、安全解压与数据映射逻辑。
"""

import io
import json
import os
import shutil
import tempfile
import unittest
import zipfile
import tarfile
from pathlib import Path

import numpy as np

from tools.motion_assets.data_paths import (
    get_repo_root,
    get_data_root,
    to_relative_data_path,
    from_relative_data_path,
)
from tools.motion_assets.select_babel_elderly_actions import (
    tokenize,
    contains_phrase_or_token,
    extract_amass_dataset,
    match_action,
    parse_babel_split,
    select_feasibility_set,
    load_query_config,
)
from tools.motion_assets.download_amass_required import (
    is_safe_path,
    safe_extract_archive,
    get_credentials,
    match_subdataset_to_resource,
    generate_manual_download_manifest,
)
from tools.motion_assets.index_amass_files import (
    inspect_npz_schema,
    compute_frame_range,
    normalize_path_key,
    find_matching_file,
)


class TestMotionAssets(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.query_cfg = load_query_config()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    # -------------------------------------------------------------
    # TEST 1: BABEL dense train/val parsing
    # -------------------------------------------------------------
    def test_01_babel_dense_train_val_parsing(self):
        fake_dense_data = {
            "101": {
                "babel_sid": 101,
                "url": "https://fake.url/101.mp4",
                "feat_p": "CMU/CMU/01/01_01_poses.npz",
                "dur": 5.0,
                "seq_ann": {
                    "labels": [{"raw_label": "stand", "proc_label": "stand", "act_cat": ["stand"]}]
                },
                "frame_ann": {
                    "labels": [
                        {"raw_label": "bend down", "proc_label": "bend down", "act_cat": ["bend"], "start_t": 1.0, "end_t": 3.0}
                    ]
                }
            }
        }
        json_file = Path(self.tmp_dir) / "train.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(fake_dense_data, f)

        cands = parse_babel_split("train", json_file, self.query_cfg)
        self.assertEqual(len(cands), 2)
        classes = [c["target_class"] for c in cands]
        self.assertIn("standing", classes)
        self.assertIn("bending", classes)

    # -------------------------------------------------------------
    # TEST 2: extra_train / extra_val parsing (seq_anns / frame_anns)
    # -------------------------------------------------------------
    def test_02_babel_extra_split_parsing(self):
        fake_extra_data = {
            "202": {
                "babel_sid": 202,
                "url": "https://fake.url/202.mp4",
                "feat_p": "KIT/KIT/10/10_01_poses.npz",
                "dur": 8.0,
                "seq_anns": [
                    {"labels": [{"raw_label": "sit", "proc_label": "sit", "act_cat": ["sit"]}]}
                ],
                "frame_anns": [
                    {"labels": [{"raw_label": "reach forward", "proc_label": "reach forward", "act_cat": ["reach"], "start_t": 2.0, "end_t": 5.0}]}
                ]
            }
        }
        json_file = Path(self.tmp_dir) / "extra_train.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(fake_extra_data, f)

        cands = parse_babel_split("extra_train", json_file, self.query_cfg)
        self.assertEqual(len(cands), 2)
        classes = [c["target_class"] for c in cands]
        self.assertIn("sitting", classes)
        self.assertIn("reaching", classes)

    # -------------------------------------------------------------
    # TEST 3: token-aware action matching (avoids substring bugs)
    # -------------------------------------------------------------
    def test_03_token_aware_action_matching(self):
        # "install software" 包含 "stall" 但不应匹配 "stand" 或 "sit"
        self.assertFalse(contains_phrase_or_token("install software", "stand"))
        self.assertFalse(contains_phrase_or_token("installing unit", "sit"))

        # 精确单词/短语匹配
        self.assertTrue(contains_phrase_or_token("person stands up slowly", "stand"))
        self.assertTrue(contains_phrase_or_token("pick up the dropped keys", "pick up"))
        self.assertTrue(contains_phrase_or_token("reach for something", "reach for"))

    # -------------------------------------------------------------
    # TEST 4: fall high-confidence vs lie manual-review
    # -------------------------------------------------------------
    def test_04_fall_high_confidence_vs_lie_review(self):
        # 1. 明确的 fall -> needs_manual_review = False
        res_fall = match_action(["fall"], "fall down", "fall on floor", self.query_cfg)
        self.assertIsNotNone(res_fall)
        self.assertEqual(res_fall[0], "fall_related")
        self.assertFalse(res_fall[1])

        # 2. 普通的 lie -> needs_manual_review = True
        res_lie = match_action(["lie"], "lie down", "lie on floor", self.query_cfg)
        self.assertIsNotNone(res_lie)
        self.assertEqual(res_lie[0], "fall_related")
        self.assertTrue(res_lie[1])

        # 3. 日常睡眠 / 休息 -> 显式排除
        res_sleep = match_action(["sleep"], "sleep in bed", "sleeping", self.query_cfg)
        self.assertIsNone(res_sleep)

    # -------------------------------------------------------------
    # TEST 5: feat_p -> amass_dataset extraction
    # -------------------------------------------------------------
    def test_05_extract_amass_dataset(self):
        self.assertEqual(extract_amass_dataset("CMU/CMU/141/141_05_poses.npz"), "CMU")
        self.assertEqual(extract_amass_dataset("EyesJapanDataset/Eyes_Japan_Dataset/shiono/01_poses.npz"), "EyesJapanDataset")
        self.assertEqual(extract_amass_dataset("BMLrub/BioMotionLab_NTroje/rub055/0020_poses.npz"), "BMLrub")
        self.assertEqual(extract_amass_dataset(""), "UNKNOWN")

    # -------------------------------------------------------------
    # TEST 6: repo-relative data root resolution
    # -------------------------------------------------------------
    def test_06_repo_relative_data_root_resolution(self):
        # 清除环境变量测试默认相对解析
        old_env = os.environ.pop("ACTIVEVIEW_DATA_ROOT", None)
        try:
            repo_root = get_repo_root()
            data_root = get_data_root()
            expected = (repo_root / ".." / ".." / "data" / "ActiveView").resolve()
            self.assertEqual(data_root, expected)
        finally:
            if old_env is not None:
                os.environ["ACTIVEVIEW_DATA_ROOT"] = old_env

    # -------------------------------------------------------------
    # TEST 7: ACTIVEVIEW_DATA_ROOT override
    # -------------------------------------------------------------
    def test_07_data_root_env_override(self):
        custom_dir = Path(self.tmp_dir) / "custom_data"
        custom_dir.mkdir(parents=True, exist_ok=True)
        old_env = os.environ.get("ACTIVEVIEW_DATA_ROOT")

        try:
            os.environ["ACTIVEVIEW_DATA_ROOT"] = str(custom_dir)
            self.assertEqual(get_data_root(), custom_dir.resolve())
        finally:
            if old_env is not None:
                os.environ["ACTIVEVIEW_DATA_ROOT"] = old_env
            else:
                os.environ.pop("ACTIVEVIEW_DATA_ROOT", None)

    # -------------------------------------------------------------
    # TEST 8: credentials absent behavior (fail gracefully, no crash)
    # -------------------------------------------------------------
    def test_08_credentials_absent_safe_behavior(self):
        old_email = os.environ.pop("AMASS_EMAIL", None)
        old_pass = os.environ.pop("AMASS_PASSWORD", None)

        try:
            email, password = get_credentials()
            self.assertIsNone(email)
            self.assertIsNone(password)

            # 测试生成 manual download manifest
            out_cache = Path(self.tmp_dir) / "cache"
            generate_manual_download_manifest(
                required_subdatasets=["CMU", "KIT"],
                feasibility_manifest=[],
                reason="credentials_missing",
                output_cache_dir=out_cache,
            )
            self.assertTrue((out_cache / "manual_download_required.json").exists())
            self.assertTrue((out_cache / "manual_download_required.md").exists())
        finally:
            if old_email is not None:
                os.environ["AMASS_EMAIL"] = old_email
            if old_pass is not None:
                os.environ["AMASS_PASSWORD"] = old_pass

    # -------------------------------------------------------------
    # TEST 9: HTML login response rejection (not valid archive)
    # -------------------------------------------------------------
    def test_09_html_response_rejection(self):
        html_file = Path(self.tmp_dir) / "fake_archive.tar.bz2"
        with open(html_file, "w", encoding="utf-8") as f:
            f.write("<!DOCTYPE html><html><body>Login Page</body></html>")

        self.assertFalse(tarfile.is_tarfile(str(html_file)))
        self.assertFalse(zipfile.is_zipfile(str(html_file)))
        self.assertFalse(safe_extract_archive(html_file, Path(self.tmp_dir) / "extract"))

    # -------------------------------------------------------------
    # TEST 10: archive safe extraction path traversal guard
    # -------------------------------------------------------------
    def test_10_safe_extraction_path_traversal_guard(self):
        base_dir = Path(self.tmp_dir) / "safe_extract"
        base_dir.mkdir(parents=True, exist_ok=True)

        safe_path = base_dir / "subdir" / "file.npz"
        unsafe_path = base_dir / ".." / "escaped.txt"

        self.assertTrue(is_safe_path(base_dir, safe_path))
        self.assertFalse(is_safe_path(base_dir, unsafe_path))

    # -------------------------------------------------------------
    # TEST 11: BABEL segment time -> frame index computation
    # -------------------------------------------------------------
    def test_11_frame_index_computation(self):
        # fps = 30.0, start_t = 1.0, end_t = 2.5, num_frames = 100
        s_f, e_f = compute_frame_range(1.0, 2.5, 30.0, 100)
        self.assertEqual(s_f, 30)
        self.assertEqual(e_f, 75)

        # 超出范围 clamp
        s_f_clamped, e_f_clamped = compute_frame_range(0.0, 10.0, 30.0, 50)
        self.assertEqual(s_f_clamped, 0)
        self.assertEqual(e_f_clamped, 49)

    # -------------------------------------------------------------
    # TEST 12: AMASS NPZ schema compatibility checker
    # -------------------------------------------------------------
    def test_12_npz_schema_compatibility(self):
        # 1. 构造兼容的 Mock NPZ
        valid_npz_path = Path(self.tmp_dir) / "valid.npz"
        np.savez(
            valid_npz_path,
            trans=np.zeros((10, 3)),
            root_orient=np.zeros((10, 3)),
            poses=np.zeros((10, 156)),
            mocap_frame_rate=np.array(30.0),
        )

        res_valid = inspect_npz_schema(valid_npz_path)
        self.assertTrue(res_valid["schema_compatible"])
        self.assertEqual(res_valid["fps"], 30.0)
        self.assertEqual(res_valid["num_frames"], 10)
        self.assertEqual(len(res_valid["missing_fields"]), 0)

        # 2. 构造缺失 trans 的 Incompatible NPZ
        invalid_npz_path = Path(self.tmp_dir) / "invalid.npz"
        np.savez(
            invalid_npz_path,
            poses=np.zeros((10, 156)),
        )

        res_invalid = inspect_npz_schema(invalid_npz_path)
        self.assertFalse(res_invalid["schema_compatible"])
        self.assertIn("trans", res_invalid["missing_fields"])


if __name__ == "__main__":
    unittest.main()
