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
- Active Code Directory: [Read from .ai/PROJECT_STATE.md]
- Target Files:
- Target Configs:
- Active Version Specification: [Read from .ai/PROJECT_STATE.md]

### Required Changes
1. [Change item 1]
2. [Change item 2]
3. [Change item 3]

### Do Not Change (Strict Invariants)
- [e.g., Do not touch historical read-only version directories]
- [e.g., Do not break Pred/True separation]
- [e.g., Do not modify evaluation protocol]

### Validation Plan
- [ ] Syntax / import check:
- [ ] Unit / logic test:
- [ ] Smoke / integration test:
- [ ] Environment-dependent validation:

> Note: Validation commands must be resolved from the current active version; do not reuse old-version commands blindly.

### Deliverables
- Code changes in target files
- Execution verification / Test logs
- Update to `.ai/HANDOFF.md` if incomplete or handoff required
```
