# ACTIVEVIEW Current Task

Status: COMPLETED

## Task
ACTIVEVIEW v7.0 Pre-Development — Elderly Motion Asset Preparation & Infrastructure

## Objective
建立面向 v7.0 助老场景动作感知的 BABEL / AMASS 动作资产准备与前置基础设施：
1. 建立统一外部数据根目录规范（基于相对路径 `../../data/ActiveView`，支持 `ACTIVEVIEW_DATA_ROOT` 覆盖）；
2. 同步并解析 BABEL annotations（train, val, extra_train, extra_val）；
3. 筛选面向室内老人监护的 5 类典型动作（standing, sitting, bending, reaching, fall_related）；
4. 选定第一轮 17 条高质量 Motion Feasibility Set（涉及 5 个核心 AMASS 子库：BMLrub, CMU, EKUT, EyesJapanDataset, KIT）；
5. 实现 AMASS 官方 Portal 自动化登录、资源映射与 Dry-Run 预检工具，并生成完整手动下载指引清单；
6. 实现本地 AMASS 文件索引与 Habitat MotionConverterSMPLX 数据 Schema 兼容性检查工具；
7. 建立 12 项纯 Python 单元测试套件 (`tools/motion_assets/tests/test_motion_assets.py`)；
8. 严格保证 v6.0 代码与科学规范 100% 不受影响，第三方大数据不入库，无凭据泄露。

## Current Version Status
- **Active Code Version**: v6.0 (6.0.0 — CLOSED / FINALIZED, Read-Only)
- **Pre-Development Stage**: v7.0 Elderly Motion Asset Infrastructure
- **Active Specification**: `EA_AVS_MVP60_Code_Generation_Document.md`

## Work Completed
1. **Data Paths & Unified Root (`tools/motion_assets/data_paths.py`)**:
   - 解析默认相对路径 `../../data/ActiveView`，支持 `ACTIVEVIEW_DATA_ROOT` 环境变量覆盖。
   - 建立 `datasets/`, `assets/`, `cache/`, `runs/`, `logs/`, `tmp/` 等标准子目录。
2. **BABEL Sync & Elderly Action Selector (`tools/motion_assets/select_babel_elderly_actions.py`)**:
   - 解析 10,370 条候选动作片段，涵盖 5 类老人典型动作。
   - 选定 17 条高质量 feasibility 动作片段（standing: 3, sitting: 3, bending: 3, reaching: 3, fall_related: 5）。
   - 产出候选池 JSON/CSV、`feasibility_manifest.json/csv`、`summary.json`、`amass_subdatasets.txt`。
3. **AMASS Portal Downloader & Verifier (`tools/motion_assets/download_amass_required.py`)**:
   - 仅从环境变量 `AMASS_EMAIL` / `AMASS_PASSWORD` 读取凭据，杜绝文件硬编码与密码打印。
   - 模拟官方登录认证，成功实现 `LOGIN_OK`，并从 Downloads 页面自动解析匹配 5 个子库的下载接口。
   - 实现安全解压防 Path Traversal 逃逸与归档校验。
   - 生成 `manual_download_required.md/json` 手动下载指引。
4. **AMASS Indexer & Habitat Schema Compatibility (`tools/motion_assets/index_amass_files.py`)**:
   - 建立 BABEL `feat_p` 到本地 NPZ 文件的鲁棒匹配。
   - 检查 `trans`, `root_orient`, `poses`, `mocap_frame_rate` 等 Habitat 所需关键字段。
   - 计算时间戳到 frame index 映射，导出 `motion_asset_manifest.json` 与 `habitat_motion_compatibility.csv`。
5. **Unit Tests Suite (`tools/motion_assets/tests/test_motion_assets.py`)**:
   - 包含 12 项严谨性单元测试，全部通过（12/12 PASS）。
6. **Documentation (`tools/motion_assets/README.md`)**:
   - 编写完整的工具使用指南与安全规范。

## Validation Plan Execution Results
- [x] Motion Assets 12 项单元测试: `python3 -m unittest tools.motion_assets.tests.test_motion_assets` (PASS, 12/12)
- [x] BABEL 动作筛选器运行: `python3 -m tools.motion_assets.select_babel_elderly_actions` (PASS, 10370 segments, 17 feasibility items)
- [x] AMASS Portal Downloader Dry-Run: `python3 -m tools.motion_assets.download_amass_required --dry-run` (PASS, LOGIN_OK, 5 resources mapped)
- [x] AMASS Indexer & Schema Checker: `python3 -m tools.motion_assets.index_amass_files` (PASS, manifest exported)
- [x] v6.0 回归测试: `python3 ea_avs_mvp_v6/scripts/test_v60_pure_python.py` & `test_no_gt_leakage.py` (PASS, 21/21)
