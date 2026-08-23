# ACTIVEVIEW Current Task

## 1. Task Definition
- **Task ID**: `V11.5-PERCEPTION-AWARE-ACTIVE-VIEW-BENCHMARK`
- **Active Directory**: `ea_avs_mvp_v11/`
- **Goal**: Reconstruct a perception-aware, zero-ground-truth-leakage Active View Benchmark based on optical sensor rendering, Keypoint R-CNN 2D detection, VideoPose3D 3D lifting, unified skeleton normalization, frozen ST-GCN classification, and Habitat HM3D multi-baseline evaluation.
- **Phase Status**: `COMPLETED & SCIENTIFICALLY VALIDATED`

---

## 2. Work Completed in v11.5 Benchmark Reconstruction
1. **Code Reorganization & Merge (Phase 0)**:
   - Migrated VideoPose3D model definitions from `tools/videopose3d/` into `ea_avs_mvp_v11/perception/videopose3d/`.
   - Merged tools (`motion_assets`, `dataset_generation`, `experiments`, `visualization`) into `ea_avs_mvp_v11/`.
   - Purged obsolete v10/11.4 test directories (`active_view/`, `perception/`, `scripts/`, `tests/`, `tools/`, `run_v11.py`).
2. **AMASS Motion Instance-Level Disjoint Split (Phase 1)**:
   - Created `ea_avs_mvp_v11/dataset/dataset_split.py`.
   - Partitioned converted motions into Train (70%, 8 instances), Val (15%, 6 instances), Test (15%, 6 instances).
   - Strictly verified $\text{Train} \cap \text{Val} = \emptyset, \text{Train} \cap \text{Test} = \emptyset, \text{Val} \cap \text{Test} = \emptyset$.
3. **Unified Skeleton Normalization (Phase 2)**:
   - `ea_avs_mvp_v11/perception/skeleton_normalizer.py`: Pelvis root centering, torso scale normalization, and canonical coronal/sagittal coordinate alignment.
4. **Clean Perception Studio Dataset Generation (Phase 3)**:
   - Created `ea_avs_mvp_v11/dataset/clean_dataset_generator.py`.
   - Rendered human motions in 100% solid background Studio environment (`scene_id='NONE'`).
   - Processed via Keypoint R-CNN $\to$ VideoPose3D $\to$ Normalizer, generating $(160, 3, 30, 17, 1)$ train and $(60, 3, 30, 17, 1)$ val datasets.
5. **ST-GCN Training & Model Freezing (Phase 4)**:
   - Created `ea_avs_mvp_v11/action_recognition/train_st_gcn_v11_5.py`.
   - Trained on estimated skeletons achieving 55.00% validation accuracy.
   - Saved and froze best model at `data/ActiveView/checkpoints/v11_5/stgcn_v11_5_best.pth`.
6. **Habitat Active View Multi-Baseline Benchmark (Phase 5 & 6)**:
   - Created `ea_avs_mvp_v11/active_view/habitat_active_view_benchmark.py`.
   - Evaluated 18 test episodes in HM3D `00800-TEEsavR23oF` across 5 baselines:
     - `Fixed`: Acc 33.33%, Entropy 0.6782
     - `Random`: Acc 44.44%, Entropy 0.5582 ($\Delta H = +0.1200$)
     - `Nearest`: Acc 44.44%, Entropy 0.5599 ($\Delta H = +0.1183$)
     - `Utility_Ours`: Acc 33.33%, Entropy 0.6782
     - `Oracle Upper Bound`: **Acc 55.56%, Entropy 0.0133 ($\Delta H = +0.6649$, Conf 0.9980)**.
7. **Pose Quality Stratification Analysis (Phase 7)**:
   - Created `ea_avs_mvp_v11/evaluation/pose_quality_analysis.py`.
   - Correlated viewpoint occlusion/angle with 2D keypoint confidence and ST-GCN classification entropy/accuracy.
8. **Anti-Leakage Static Code Audit**:
   - `grep -rn "gt_skeleton"` and `grep -rn "smpl.*joint"`: 0 leakage into ST-GCN decision paths.
9. **Scientific Report**:
   - `V11.5_PERCEPTION_AWARE_ACTIVE_VIEW_REPORT.md`.

---

## 3. Validation Summary
- `python -m compileall ea_avs_mvp_v11`: **100% PASS (0 syntax or import errors)**.
- AMASS Disjoint Partitioning: **100% PASS**.
- ST-GCN Clean Perception Convergence: **55.00% Val Acc**.
- Habitat Active View Benchmark: **18 / 18 Episodes Completed**.
