# ACTIVEVIEW AI Handoff

Status: CLEAN

v8.1 (Local Active View Planning Baseline) has been fully completed, verified, and frozen. The repository is in a clean, stable state and ready for future v9 (Action-aware Active View Selection) research.

---

## Completed Phase: v8.1 Local Active View Planning Baseline
- **Active Code Directory**: `ea_avs_mvp_v8/`
- **Specification Document**: `EA_AVS_MVP80_Code_Generation_Document.md`
- **Key Capabilities Implemented**:
  1. Local polar grid candidate viewpoint generation around known human location ($1.5\text{m} \sim 3.0\text{m}$).
  2. Three-stage spatial and observation constraint pipeline: `NavigationConstraint` (NavMesh), `LineOfSightConstraint` (RayCast), `HumanVisibilityConstraint` (FOV, $A_{proj} \ge 1.5\%$, `pose_visibility_score`).
  3. Transparent View Quality scoring: $Q(v) = w_1 \cdot \text{vis} + w_2 \cdot \text{cov} - w_3 \cdot \text{dist} - w_4 \cdot \text{occlusion}$.
  4. Explicit `evaluation_mode: "oracle"` and `pose_source: "oracle"` boundary marking.
  5. Standardized baseline strategy interface: `select_view(strategy)` supporting `random`, `nearest`, `geometry_best`.
  6. Structured experiment reports: `view_selection_report.json`, `candidate_statistics.json`, `candidate_views.json`, `best_view.json`, and `best_view_rgb.png`.

---

## Next Version Scope (v9.0 Roadmap)
- **Title**: Action-aware Active View Selection
- **Core Research Goal**: Moving beyond static geometric quality to action-aware utility prediction ($Q_{action}(v)$) under dynamic human actions (e.g. falling, sitting, standing) and estimated states.
- **Rules to Preserve**:
  - Keep `ea_avs_mvp_v7/` and `ea_avs_mvp_v8/` as read-only historical baselines.
  - Implement v9 in dedicated directory `ea_avs_mvp_v9/`.
  - Maintain the decision-time vs evaluation-time anti-leakage boundary.
