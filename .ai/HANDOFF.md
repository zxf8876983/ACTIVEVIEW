# ACTIVEVIEW AI Handoff

Status: CLEAN

No unfinished cross-agent task is currently recorded.

---

## Standard Handoff Protocol & Template

When pausing a task, switching coding models (e.g. Claude Code → DeepSeek / Codex / Gemini), or reaching session limits, the active agent MUST update this document using the template below.

```markdown
## Active Handoff

Last Updated: [YYYY-MM-DD HH:MM]
Status: [IN_PROGRESS / BLOCKED / READY_FOR_REVIEW]

### Current Task
[Brief description of what was being implemented or debugged]

### Previous Executor
[e.g., Claude Code / DeepSeek / Codex / Human Developer]

### Git State
- Branch: `main`
- HEAD: `[git commit hash]`
- Uncommitted Changes: [None / List of modified files]

### Completed
- [x] [Item 1]
- [x] [Item 2]

### Files Changed
- `path/to/modified_file.py`: [Brief summary of modification]

### Validation Performed
- [x] [Test command executed and result, e.g., `python ea_avs_mvp_v5/scripts/test_v50_closure_policy.py` -> PASS]

### Validation Not Performed (with reason)
- [ ] [e.g., `run_mvp50_humanoid.py`: NOT RUN (Habitat GPU environment unavailable in current terminal)]

### Important Findings & Gotchas
- [Critical technical details discovered during implementation that the next agent must know]

### Remaining Work
1. [Next step 1]
2. [Next step 2]

### Risks / Invariants (Do Not Revert)
- [Explicitly list invariants that must NOT be broken or reverted]

### Recommended Next Action
[Specific, actionable instruction for the incoming agent on what command or file to tackle first]
```

---

## Handoff Rules for Incoming and Outgoing Agents
1. **No Chat Dumps**: Keep handoff summaries structured, technical, and concise. Do not dump raw conversation logs.
2. **Honest Validation**: Never claim tests passed if they were not actually executed. Explicitly state `NOT RUN: <reason>`.
3. **Clean Up**: When a handed-off task is fully completed, verified, and committed by the subsequent agent, reset the top-level status back to `Status: CLEAN`.
