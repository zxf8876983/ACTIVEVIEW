# ParaHome feasibility audit

## Dataset and temporal capacity
- ParaHome contains **207 sequences**, **38 subjects**, and **8.11 hours** (486.33 minutes) at the inferred 30 fps.
- The audit covers **5476 annotated intervals**. Duration median is **3.00s**; fractions >=2s and >=5s are **0.847** and **0.313**.
- Conservative normalized taxonomy has **16 classes**; **16** meet the 30/15/8 candidate threshold and **16** meet the 50/20/10 stricter threshold.
- Unresolved classes excluded from viability gating: none.
- Capacity counts use `floor(max(annotation_duration - 2s, 0) / decision_interval)`; this explicitly reserves a 2s HAR window before repeated decisions.

## Representation compatibility
- ParaHome provides 23 body joints as 6D rotations, 73 global joint positions (23 body + 50 hand), a 4x4 body-to-world transform, and optional SMPL-X axis-angle body/root/hand pose.
- The existing ACTIVEVIEW `MotionConverter` accepts the SMPL-X representation after a thin adapter that pads the 21-body-joint pose and inserts 15+15 hand rotations into the converter's expected 54-joint layout. Root coordinates require one rigid ParaHome-to-Habitat alignment; this is not a direct coordinate identity.
- The three representative conversions report finite tensors, valid 6D rotations, and an explicit rigid alignment error; details are in `representation_audit.json`.

## Habitat clean replay
- Replay status: **PASS**; human replay **PASS**, rigid-object replay **PASS**.
- The replay uses `scene_id=NONE`, a diagnostic floor, the existing male_0 Habitat humanoid, and 4 kinematic scanned OBJ objects. It rendered 15 snapshots (five timestamps × three cameras).
- This validates a clean-scene replay path only; it does not claim HM3D integration or complete articulated-object conversion.

## Six required questions
1. **Scale:** The 207-sequence/38-subject scale is sufficient for a dataset audit and matches the published aggregate, but taxonomy viability depends on the normalized class thresholds above.
2. **Viable classes:** 16 classes satisfy >=30 instances, >=15 sequences and >=8 subjects; stricter count is 16.
3. **Repeated decisions:** With a 2s HAR window, **9 / 16** classes have at least half of their intervals supporting two 1s decision cycles; the corresponding 2s-cycle count is **5 / 16**. Raw per-class counts are in `duration_analysis.json`.
4. **Motion compatibility:** **Thin adapter**, not direct identity: SMPL-X pose is compatible with the existing converter, while ParaHome world coordinates need a single rigid alignment and ParaHome hand/body data need explicit mapping.
5. **Replay:** **PASS** for the selected clean-scene sequence; human/object snapshots and alignment checks are saved under `visualizations/`.
6. **Verdict:** **CONDITIONAL PROMOTE** — the motion representation and minimal clean replay are feasible, but taxonomy duration/coverage and the coordinate/gender/scene adapter must be addressed before making ParaHome the continuous Active HAR mainline.

## Hard-gate notes
- No policy Test, BABEL/reduced12 formal artifact, large-scale RGB generation, HM3D integration, or HAR model training was used.
- ParaHome's own README warns that under-sink items can be manually filled and may contain physical alignment/penetration errors; object replay should therefore be treated as research data, not guaranteed physical ground truth.
- The verdict does not count sliding windows as independent annotated instances.
