# ACTIVEVIEW Current Task

Status: IDLE

There is currently no persisted coding task.

For small iteration tasks, the user may provide a temporary prompt directly without updating this file.

For:
- complex tasks,
- multi-session tasks,
- model switching,
- interrupted tasks,

store the current task here before execution.

---

## Task Template

```markdown
# Task: [Task Name / Short Identifier]

Task ID: [e.g. TASK-20260819-01]
Date: [YYYY-MM-DD]
Author / Initiator: [User / Model Name]
Status: [PENDING / IN_PROGRESS / BLOCKED / COMPLETED]

### Objective
[1-3 sentences describing the high-level goal of this task]

### Background & Scientific Context
[Relevant background, motivation, or design rationale]

### Relevant Files
- Target Code: `ea_avs_mvp_v5/...`
- Target Configs: `ea_avs_mvp_v5/configs/...`
- Specification: `EA_AVS_MVP50_Code_Generation_Document.md`

### Required Changes
1. [Change item 1]
2. [Change item 2]
3. [Change item 3]

### Do Not Change (Strict Invariants)
- [e.g., Do not touch v1-v4 code]
- [e.g., Do not break Pred/True separation]
- [e.g., Do not modify evaluation protocol]

### Validation Plan
- [ ] Syntax check: `python -m compileall ea_avs_mvp_v5`
- [ ] Policy unit test: `python ea_avs_mvp_v5/scripts/test_v50_closure_policy.py`
- [ ] Smoke test: `cd ea_avs_mvp_v5 && python scripts/run_mvp50_humanoid.py --episodes 1` (If Habitat available)

### Deliverables
- Code changes in target files
- Execution verification / Test logs
- Update to `.ai/HANDOFF.md` if incomplete or handoff required
```
