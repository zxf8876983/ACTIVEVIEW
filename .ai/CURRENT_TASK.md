# ACTIVEVIEW Current Task

Status: COMPLETED (v10.0 Phase 1: Habitat RGB-D Dataset Generation Verified & Closed)

## Current Phase
v10.0 Phase 1 — Habitat RGB-D Dataset Generation.

## Phase 1 Accomplished Items
1. **数据目录与边界初始化**：
   - 严格继承运行时数据根目录 `/home/zxf/WorkSpace/code/data/ActiveView/`；
   - 建立 `datasets/v10/` 标准子目录：`raw/` (rgb, depth, camera_pose, scene_meta), `ground_truth/` (skeleton, action), `metadata/` (samples.json)。
2. **RGB-D 采集模块 (`ea_avs_mvp_v10/sensors/rgbd_capture.py`)**：
   - 采集 640x480 RGB 图像 (uint8 PNG) 与 Depth 深度图 (float32 NPY / Vis PNG)；
   - 提取相机外参 (CameraPose) 与内参 (CameraIntrinsics, fx=320, fy=320, cx=320, cy=240, hfov=90°)。
3. **Motion 数据接口与 6 大动作类别管理 (`ea_avs_mvp_v10/motion/motion_manager.py`)**：
   - 支持 6 大动作类别：`standing`, `walking`, `sitting`, `bending`, `reaching`, `falling`；
   - 索引 `data/ActiveView/assets/motions/converted/` 动作文件。
4. **多视角采样 (`ea_avs_mvp_v10/viewpoint/candidate_generator.py`)**：
   - 生成多半径、多方位的候选观察视点并计算注视角度。
5. **数据集生成引擎与演示 (`ea_avs_mvp_v10/dataset/`, `scripts/`, `examples/`)**：
   - 自动化生成 48 个多视角 RGB-D 样本并构建 `metadata/samples.json`；
   - 生成 `examples/v10_phase1_demo/sample_visualization.png` 与 `README.md`。

## Validation Plan Execution Results
- [x] Python Compile Check: `python -m compileall ea_avs_mvp_v10` (PASS)
- [x] v10.0 Unit Tests: `python3 -m unittest discover -s ea_avs_mvp_v10/tests -p "test_*.py"` (PASS, 9/9)
- [x] v9.1 Regression Tests: `python3 -m unittest discover -s ea_avs_mvp_v9/tests -p "test_*.py"` (PASS, 34/34)
- [x] v8.2 Regression Tests: `python3 -m unittest discover -s ea_avs_mvp_v8/tests -p "test_*.py"` (PASS, 17/17)
- [x] v7.0 Regression Tests: `python3 -m unittest discover -s ea_avs_mvp_v7/tests -p "test_*.py"` (PASS, 30/30)
- [x] Phase 1 Demo Script: `conda run -n habitat python -m ea_avs_mvp_v10.scripts.run_v10_phase1_demo` (PASS, 48 samples generated)
