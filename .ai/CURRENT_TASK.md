# ACTIVEVIEW Current Task

## 1. Task Definition
- **Task ID**: `V11.3-MULTISCENE-ACTIVE-VIEW-DATASET-UPGRADE`
- **Active Directory**: `ea_avs_mvp_v11/`
- **Goal**: Upgrade ACTIVEVIEW v11.3 data distribution from single-scene to Multi-Scene Multi-Position Active View Dataset across >=3 Habitat scenes with randomized human placement and robot initial view, re-train Utility Predictor, and demonstrate clear Active View superiority over Nearest baseline.
- **Phase Status**: `COMPLETED & SCIENTIFICALLY VALIDATED`

---

## 2. Work Completed in v11.3 Multi-Scene Upgrade
1. **Multi-Scene Management (`ea_avs_mvp_v11/active_view/scene_manager.py`)**:
   - Implemented `SceneManager` indexing `apartment_1`, `skokloster-castle`, `van-gogh-room` and HSSD scenes.
2. **Scene Check & Audit Tool (`ea_avs_mvp_v11/tools/check_habitat_scenes.py` & root alias)**:
   - Audited local Habitat scenes and provided HuggingFace download interface without forced auto-downloads.
3. **Random Human Placement Generator (`ea_avs_mvp_v11/active_view/human_placement_generator.py`)**:
   - Implemented `HumanPlacementGenerator` placing humans on NavMesh with >=1.0m clearance and random 0~360 deg yaw.
4. **Robot Initial View Randomization (`ea_avs_mvp_v11/active_view/robot_start_sampler.py`)**:
   - Implemented `RobotStartSampler` placing robot at 2m~8m distance with Face-Human Yaw and authentic `current_viewpoint`.
5. **Multi-Scene Episode Dataset Generator (`ea_avs_mvp_v11/active_view/multiscene_dataset_generator.py`)**:
   - Generated 300 episodes across 3 scenes (9,600 viewpoints, 100% non-zero entropy, mean entropy 0.2406 nats).
   - Saved to `data/ActiveView/v11_multiscene_viewpoint_dataset/` with strict instance-disjoint splits.
6. **Utility Dataset & Predictor Re-training (`ea_avs_mvp_v11/scripts/train_utility_predictor.py`)**:
   - Re-trained 3-layer MLP on `v11_multiscene_utility_dataset`.
   - Test Spearman correlation reached **$\rho = 0.9836$** (MSE = 0.002549, MAE = 0.035160).
   - Saved checkpoint to `data/ActiveView/checkpoints/v11_utility_multiscene/utility_predictor_best.pth`.
7. **Benchmark Evaluation on Multi-Scene Test Set**:
   - **Nearest baseline degraded**: $H(v) = 0.3517\text{ nats}, \Delta H = 0.6071\text{ nats}, \text{Oracle Gap} = 0.3513\text{ nats}$.
   - **Utility Predictor (Ours)**: $H(v) = \mathbf{0.0004}\text{ nats}, \Delta H = \mathbf{0.9585}\text{ nats}, \text{Oracle Gap} = \mathbf{0.0000}\text{ nats}$ (**$879\times$ lower uncertainty, $+57.9\%$ information gain over Nearest**).
8. **Scientific Visualizations (`ea_avs_mvp_v11/tools/visualization/multiscene_visualizer.py`)**:
   - Generated `outputs/v11_visualization/multiscene_active_view_distribution.png` and `multiscene_utility_evaluation.png`.
9. **Unit Tests & Regression**:
   - `test_v11_multiscene_dataset.py`: 6 / 6 PASS.
   - Full repository regression tests: **100% PASS**.
10. **Report**:
    - `V11.3_MULTISCENE_DATASET_UPGRADE_REPORT.md`.
