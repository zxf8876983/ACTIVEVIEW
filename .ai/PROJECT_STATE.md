# ACTIVEVIEW Project State

Last Updated: 2026-08-19  
Current Git HEAD: 02bc45e (main)  
Target Audience: Coding Agents (Codex, DeepSeek, Claude Code, Gemini) & Researchers

---

## 1. Current Stage
- **项目阶段**：科研原型 MVP 快速迭代阶段。
- **架构组织**：处于版本化递进过渡期（`v1 -> v2 -> v3 -> v4 -> v5 -> 未来v6`），各版本保留独立目录与设计文档。核心功能全链路跑通前暂不重构为单一 `src/`。

---

## 2. Active Version
- **Version**: v5.0 (5.0.0)
- **Active Code Directory**: `ea_avs_mvp_v5/`
- **Active Version Specification**: `EA_AVS_MVP50_Code_Generation_Document.md`
- **Active Config**: `ea_avs_mvp_v5/configs/mvp50_humanoid.yaml`
- **Active Main Script**: `ea_avs_mvp_v5/scripts/run_mvp50_humanoid.py`

---

## 3. Research Goal
面向室内场景下移动机器人对老人动作的主动感知（Elderly Action Active View Selection, EA-AVS），在已知或估计的人体先验与环境地图约束下，通过单步局部候选视角采样与预测评分，选择能够最大化人体动作关键部位可见性、最小化环境/人体自遮挡的下一最佳观察位姿（Next-Best-View），从而服务于下游动作识别任务。

---

## 4. Version Evolution
- **v1.0 (`ea_avs_mvp/`)**: 几何版候选视角采样与可达性过滤，基于抽象骨架计算 FOV 可见性与移动代价，对比 Fixed/Random/Nearest/Ours。
- **v2.0 (`ea_avs_mvp_v2/`)**: 严格确立 `Pred`（决策前预测评分）与 `True`（到达后真实评估）信息边界，严禁选择阶段偷看未来图像，引入原地保留（stay）机制。
- **v3.0 (`ea_avs_mvp_v3/`)**: 引入多姿态骨架（standing, sitting, lying_fallen, bending）、人体朝向建模（`human_yaw`）与动作关键部位加权评分（`S_action_part`）。
- **v4.0 (`ea_avs_mvp_v4/`)**: 引入物理引擎 Ray Casting 实现环境障碍物遮挡检测，提出遮挡感知评分（`S_action_occ`）、消融策略体系与同口径 Oracle 离线上界。
- **v5.0 (`ea_avs_mvp_v5/`)**: 接入真实 Habitat KinematicHumanoid（`neutral_0`）与 RGB-D/语义渲染，由 URDF link 派生 15 关节 GT 骨架，实现 5 类遮挡判定与决策闭环。

---

## 5. Current v5 Capabilities (Confirmed)
1. **真实 Humanoid 仿真**：在 Habitat 场景中成功嵌入 `neutral_0` 人体网格，支持官方 standing 与 walking 动作。
2. **严格 GT 骨架提取**：从 Humanoid 实际 URDF link transforms 实时提取 15 个关键点（`habitat_humanoid_gt`），杜绝手工坐标漂移。
3. **真实 RGB-D 与语义渲染**：机器人相机朝向人体渲染，支持语义分割 ID（`semantic_id: 100`）与 GT 锚定 depth 像素有效性验证。
4. **5 分类射线遮挡判定**：区分 `none`, `target_surface`, `environment`, `humanoid_self`, `unknown`，防止自身表面误判为遮挡。
5. **严谨的决策与评估闭环**：
   - 决策阶段：`PredictiveEvaluator` 基于 GT 骨架与地图几何计算 `Q_pred`，无未来图像泄漏；
   - 评估阶段：`TrueEvaluator` 在选中位姿渲染后计算 `Q_true`；
   - 策略竞争：`OursPolicy` 将当前视角与候选视角公平竞优，支持分数占优留原地（`stay_by_score`）及全失效安全兜底（`stay_by_fallback`）。
6. **基线、消融与 Oracle 体系**：包含 Fixed, Random, Nearest, 6 种消融策略，以及受 depth_coverage 门控的同口径 Oracle 离线上界。
7. **纯 Python 策略测试**：`ea_avs_mvp_v5/scripts/test_v50_closure_policy.py` 覆盖决策有效性与兜底逻辑。

---

## 6. Current Incomplete / Open Items
- **Research 姿态素材未接入**：`sitting`, `bending`, `lying_fallen` 在 yaml 中为 `null`（待后续 AMASS/SMPL-X 动作资源转换，当前使用 official standing/walking）。
- **无上游视觉姿态估计前端**：v5 仍使用 GT-State 支路，未接入 raw RGB-D -> 2D Pose (YOLO-Pose/RTMPose) -> 3D 姿态估计网络（规划留待后续版本）。
- **无下游动作识别增益评测**：动作分类器下游增益尚未接入端到端评测闭环。
- **自遮挡统计判定标记**：`raycast_self_occlusion_status` 在 yaml 中标记为 `inconclusive`（机制已实现并实测 8 方向，但复杂自遮挡样本受限于当前站立姿态多样性）。
- **外部环境依赖**：依赖本地 Habitat-Sim 场景网格及 Habitat-Lab Humanoid 资产路径。

---

## 7. Important Scientific Constraints
1. **One-shot Active Re-observation**：当前定位为单步主动观察重定位，非多步强化学习或长程导航。
2. **严禁未来观测泄漏**：策略决策阶段（`OursPolicy` 等）只能使用 `Q_pred`，绝对禁止调用渲染器获取候选点的未来 RGB、Depth 或真实遮挡状态。
3. **公平原地选择**：`OursPolicy` 必须包含当前视角竞优，允许且合理处理 `stay` 决策。
4. **Oracle 同口径门控**：Oracle 上界仅在候选点与选中点均满足深度有效覆盖（`depth_coverage >= min_depth_coverage`）时计算 gap，禁止跨口径比较。
5. **GT 骨架严谨性**：正式实验要求 15/15 关键点均来自 Humanoid 真实 link，禁止静默退化到手工坐标。

---

## 8. Current Stable Point
- **Active Branch**: `main`
- **Current HEAD**: `02bc45e` - `refactor(v5.0): final closure - current-vs-candidate competition, invalid-occlusion validity, geometry cause counts`
- **Recent Important Commits**:
  - `02bc45e`: v5.0 final closure (current-vs-candidate competition, invalid-occlusion validity, closure unit test)
  - `aab699a`: v5.0 closure fix (unknown validity, target-surface occlusion, semantic invisible, oracle same-aperture gap)
  - `46a615e`: v5.0 rigor fix round 3 (object IDs, 5-way self-occlusion, semantic mask, depth coverage)
  - `869653b`: v5.0 rigor fixes for humanoid visibility/yaw/GT
  - `76e0e7e`: Initial v5.0 implementation

---

## 9. Expected Next Phase
- **后续版本演进 (v6/v7)**：计划引入真实视觉前端（RGB-D -> 姿态估计 -> 不确定性建模）与动作分类网络增益验证。
- **工程重构计划**：在全链路科学闭环验证完成后，再统一进行单主线 `src/` 代码重构。

---

## 10. Historical Versions (Read-Only)
- `ea_avs_mvp/` (v1.0), `ea_avs_mvp_v2/` (v2.0), `ea_avs_mvp_v3/` (v3.0), `ea_avs_mvp_v4/` (v4.0) 为历史快照，默认保持只读，不参与当前任务修改。
