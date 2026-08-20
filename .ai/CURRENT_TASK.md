# ACTIVEVIEW Current Task

Status: COMPLETED

## Task
ACTIVEVIEW v7.0 Pre-Development — AMASS Hugging Face Download & Motion Asset Infrastructure Validation

## Objective
在 ACTIVEVIEW 数据准备阶段完成 AMASS 真实数据从 Hugging Face 的下载、校验与前置基础设施修正：
1. 数据来源切换为 Hugging Face 镜像源 (`tuguobin/AMASS`)，实现自动化选择性下载与安全解压；
2. 真实下载 5 个子库 (`BMLrub`, `CMU`, `EKUT`, `EyesJapanDataset`, `KIT`) 归档并解压到 `../../data/ActiveView/datasets/amass/`；
3. 统计 5 个子库各包含的 `.npz` 文件数与磁盘占用（共计 7,862 个 `.npz` 文件，13.75 GB）；
4. 增强 Indexer 兼容标准 AMASS Schema A (`explicit_root_orient`) 与 Schema B (`standard_amass` 即 `poses[:, :3]`)，支持多帧率键；
5. 执行 `python3 -m tools.motion_assets.index_amass_files`，实现 `file_found = 17/17`、`schema_compatible = 17/17` (PASS, exit code 0)；
6. 严格复核 5 条 `fall_related` 动作均为真实 dynamic fall（无普通 lie/sleep）；
7. 保证 `motion_asset_manifest.json` 中 `local_motion_path` 为相对 POSIX 路径，无机器硬编码；
8. 运行 14 项 pure Python 单元测试 (`tools/motion_assets/tests/test_motion_assets.py`) 全部通过；
9. 保证 v6.0 代码严格保持只读，Git 仓库零大文件污染。

## Current Version Status
- **Active Code Version**: v6.0 (6.0.0 — CLOSED / FINALIZED, Read-Only)
- **Pre-Development Stage**: v7.0 Elderly Motion Asset Infrastructure (AMASS Downloaded & Validated)
- **Active Specification**: `EA_AVS_MVP60_Code_Generation_Document.md`

## Work Completed
1. **Hugging Face Downloader (`tools/motion_assets/download_amass_hf.py`)**:
   - 切换下载源为 Hugging Face (`tuguobin/AMASS`)。
   - 实现了多线程流式下载、断点续传、归档校验与防 Path Traversal 解压。
2. **Real AMASS Download & Extraction**:
   - `BioMotionLab_NTroje` (BMLrub): 443 `.npz` 文件 (663.0 MB)
   - `CMU`: 2088 `.npz` 文件 (4474.7 MB)
   - `EKUT`: 349 `.npz` 文件 (235.5 MB)
   - `Eyes_Japan_Dataset`: 750 `.npz` 文件 (3643.4 MB)
   - `KIT`: 4232 `.npz` 文件 (5065.7 MB)
   - **Total**: 7,862 `.npz` files (13.75 GB on disk).
3. **AMASS Indexer Dual-Schema Support (`tools/motion_assets/index_amass_files.py`)**:
   - 同时支持 `standard_amass` (`poses[:, :3]`) 与 `explicit_root_orient`。
   - 支持 `mocap_frame_rate`, `mocap_framerate`, `frame_rate`, `framerate`, `fps` 多种键名。
   - 导出 POSIX 相对路径 manifest (`motion_asset_manifest.json`)。
   - 增加严谨 exit code：17/17 PASS 返回 0，有缺失返回 1。
4. **Unit Tests (`tools/motion_assets/tests/test_motion_assets.py`)**:
   - 14 项单元测试全部通过 (Ran 14 tests, OK)。
5. **v6.0 Regression Validation**:
   - 21 项 v6.0 纯 Python 与 No-GT-Leakage 守卫测试全部通过。

## Validation Plan Execution Results
- [x] Motion Assets 14 项单元测试: `python3 -m unittest tools.motion_assets.tests.test_motion_assets` (PASS, 14/14)
- [x] AMASS Hugging Face 下载与解压: 5 个子库 7,862 个 `.npz` 文件完成落地 (PASS)
- [x] AMASS Indexer 校验: `python3 -m tools.motion_assets.index_amass_files` (PASS: file_found=17/17, schema_compatible=17/17, exit=0)
- [x] 相对路径检查: `motion_asset_manifest.json` 中无绝对路径或机器路径硬编码 (PASS)
- [x] v6.0 回归测试: `python3 ea_avs_mvp_v6/scripts/test_v60_pure_python.py` & `test_no_gt_leakage.py` (PASS, 21/21)
