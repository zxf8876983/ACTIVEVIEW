"""
Motion Assets & Data Infrastructure 单元测试套件 —— test_motion_assets.py
==========================================================================

功能：
    在纯 Python 环境下测试动作筛选、路径解析、凭据防护、安全解压、
    标准 AMASS 双 Schema 校验、文件名模糊匹配及 Indexer 严格判定逻辑。
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
    build_amass_disk_index,
    find_matching_file,
    index_and_validate_feasibility_manifest,
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
        self.assertFalse(contains_phrase_or_token("install software", "stand"))
        self.assertFalse(contains_phrase_or_token("installing unit", "sit"))

        self.assertTrue(contains_phrase_or_token("person stands up slowly", "stand"))
        self.assertTrue(contains_phrase_or_token("pick up the dropped keys", "pick up"))
        self.assertTrue(contains_phrase_or_token("reach for something", "reach for"))

    # -------------------------------------------------------------
    # TEST 4: fall high-confidence vs lie manual-review
    # -------------------------------------------------------------
    def test_04_fall_high_confidence_vs_lie_review(self):
        res_fall = match_action(["fall"], "fall down", "fall on floor", self.query_cfg)
        self.assertIsNotNone(res_fall)
        self.assertEqual(res_fall[0], "fall_related")
        self.assertFalse(res_fall[1])

        res_lie = match_action(["lie"], "lie down", "lie on floor", self.query_cfg)
        self.assertIsNotNone(res_lie)
        self.assertEqual(res_lie[0], "fall_related")
        self.assertTrue(res_lie[1])

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
        s_f, e_f = compute_frame_range(1.0, 2.5, 30.0, 100)
        self.assertEqual(s_f, 30)
        self.assertEqual(e_f, 75)

        s_f_clamped, e_f_clamped = compute_frame_range(0.0, 10.0, 30.0, 50)
        self.assertEqual(s_f_clamped, 0)
        self.assertEqual(e_f_clamped, 49)

    # -------------------------------------------------------------
    # TEST 12: Dual Schema Compatibility (explicit_root_orient & standard_amass)
    # -------------------------------------------------------------
    def test_12_npz_schema_compatibility(self):
        # 1. Explicit root_orient Schema (PASS)
        path_explicit = Path(self.tmp_dir) / "explicit.npz"
        np.savez(
            path_explicit,
            trans=np.zeros((10, 3)),
            root_orient=np.zeros((10, 3)),
            poses=np.zeros((10, 156)),
            mocap_frame_rate=np.array(30.0),
        )
        res_exp = inspect_npz_schema(path_explicit)
        self.assertTrue(res_exp["schema_compatible"])
        self.assertEqual(res_exp["schema_type"], "explicit_root_orient")
        self.assertEqual(res_exp["root_orient_source"], "root_orient")

        # 2. Standard AMASS Schema (poses[:, :3] with mocap_framerate) (PASS)
        path_std = Path(self.tmp_dir) / "std_amass.npz"
        np.savez(
            path_std,
            trans=np.zeros((15, 3)),
            poses=np.zeros((15, 156)),
            mocap_framerate=np.array(100.0),
        )
        res_std = inspect_npz_schema(path_std)
        self.assertTrue(res_std["schema_compatible"])
        self.assertEqual(res_std["schema_type"], "standard_amass")
        self.assertEqual(res_std["root_orient_source"], "poses[:, :3]")
        self.assertEqual(res_std["fps"], 100.0)

        # 3. Missing trans (FAIL)
        path_no_trans = Path(self.tmp_dir) / "no_trans.npz"
        np.savez(path_no_trans, poses=np.zeros((10, 156)), fps=30.0)
        res_no_trans = inspect_npz_schema(path_no_trans)
        self.assertFalse(res_no_trans["schema_compatible"])
        self.assertIn("trans", res_no_trans["missing_fields"])

        # 4. Missing poses (FAIL)
        path_no_poses = Path(self.tmp_dir) / "no_poses.npz"
        np.savez(path_no_poses, trans=np.zeros((10, 3)), fps=30.0)
        res_no_poses = inspect_npz_schema(path_no_poses)
        self.assertFalse(res_no_poses["schema_compatible"])
        self.assertIn("poses", res_no_poses["missing_fields"])

        # 5. Missing fps (FAIL)
        path_no_fps = Path(self.tmp_dir) / "no_fps.npz"
        np.savez(path_no_fps, trans=np.zeros((10, 3)), poses=np.zeros((10, 156)))
        res_no_fps = inspect_npz_schema(path_no_fps)
        self.assertFalse(res_no_fps["schema_compatible"])
        self.assertIn("frame_rate", res_no_fps["missing_fields"])

    # -------------------------------------------------------------
    # TEST 13: Path matching (_poses.npz vs _stageii.npz & nested folders)
    # -------------------------------------------------------------
    def test_13_amass_path_matching_and_stageii(self):
        amass_root = Path(self.tmp_dir) / "amass_mock"
        amass_root.mkdir(parents=True, exist_ok=True)

        # 构造不同命名的 mock 文件
        target_stageii = amass_root / "BioMotionLab_NTroje" / "rub054" / "0019_lifting_heavy1_stageii.npz"
        target_stageii.parent.mkdir(parents=True, exist_ok=True)
        np.savez(target_stageii, trans=np.zeros((5, 3)), poses=np.zeros((5, 156)), fps=30.0)

        target_cmu = amass_root / "15" / "15_04_poses.npz"
        target_cmu.parent.mkdir(parents=True, exist_ok=True)
        np.savez(target_cmu, trans=np.zeros((5, 3)), poses=np.zeros((5, 156)), fps=30.0)

        disk_idx = build_amass_disk_index(amass_root)

        # 1. BABEL feat_p 为 _poses.npz，磁盘上为 _stageii.npz
        p1 = find_matching_file("BMLrub/BioMotionLab_NTroje/rub054/0019_lifting_heavy1_poses.npz", disk_idx, amass_root)
        self.assertIsNotNone(p1)
        self.assertEqual(p1.resolve(), target_stageii.resolve())

        # 2. BABEL feat_p 包含 CMU/CMU/15/15_04_poses.npz，磁盘为 15/15_04_poses.npz
        p2 = find_matching_file("CMU/CMU/15/15_04_poses.npz", disk_idx, amass_root)
        self.assertIsNotNone(p2)
        self.assertEqual(p2.resolve(), target_cmu.resolve())

    # -------------------------------------------------------------
    # TEST 14: Indexer strict exit validation & relative manifest paths
    # -------------------------------------------------------------
    def test_14_indexer_strict_validation(self):
        amass_root = Path(self.tmp_dir) / "amass_data"
        amass_root.mkdir(parents=True, exist_ok=True)

        # 创建一条 mock 动作数据
        mock_file = amass_root / "CMU" / "15" / "15_04_poses.npz"
        mock_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez(mock_file, trans=np.zeros((10, 3)), poses=np.zeros((10, 156)), fps=30.0)

        items_partial = [
            {
                "target_class": "standing",
                "babel_sid": 10780,
                "proc_label": "stand",
                "raw_label": "standing",
                "act_cat": ["stand"],
                "start_t": 0.0,
                "end_t": 1.0,
                "feat_p": "CMU/15/15_04_poses.npz",
                "amass_dataset": "CMU",
                "needs_manual_review": False,
            },
            {
                "target_class": "sitting",
                "babel_sid": 4336,
                "proc_label": "sit",
                "raw_label": "sit",
                "act_cat": ["sit"],
                "start_t": 0.0,
                "end_t": 1.0,
                "feat_p": "BMLrub/rub101/0014_sitting1_poses.npz",
                "amass_dataset": "BMLrub",
                "needs_manual_review": False,
            }
        ]

        manifest_out = Path(self.tmp_dir) / "manifest.json"
        compat_csv = Path(self.tmp_dir) / "compat.csv"
        index_json = Path(self.tmp_dir) / "idx.json"

        # 1. 存在缺失文件时 -> is_all_passed 为 False
        all_passed, stats = index_and_validate_feasibility_manifest(
            feasibility_items=items_partial,
            amass_dir=amass_root,
            output_manifest_path=manifest_out,
            output_compat_csv=compat_csv,
            output_index_json=index_json,
        )
        self.assertFalse(all_passed)
        self.assertEqual(stats["file_found_count"], 1)
        self.assertEqual(stats["total_count"], 2)

        # 2. 检查输出 manifest 中路径为相对路径，不包含绝对硬编码开发机根
        with open(manifest_out, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
        for m in manifest_data:
            if m["local_motion_path"]:
                self.assertFalse(m["local_motion_path"].startswith("/home/"))
                self.assertFalse(m["local_motion_path"].startswith("C:\\"))


if __name__ == "__main__":
    unittest.main()
