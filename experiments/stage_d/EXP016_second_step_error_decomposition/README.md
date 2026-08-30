# EXP016 — Second-Step Decision Error Decomposition

**Status: PREPARED (analysis-only; real Val run not authorized).**

## Scientific question

After the frozen Stage C-v0 first Move/Stay decision and Top-1 proposal, is
the corrected EXP014 gap primarily caused by the second-step Stay/Move gate,
by p2/p3 candidate ranking, or by an interaction of both?

## Design

EXP016 is a counterfactual Val-only decomposition.  It does not train a model
or regenerate perception.  Every variant preserves the frozen v0 first action:
v0 Stay remains Stay at `s0`; v0 Move keeps the same `p1`.  Only the decision
after the cached `p1` observation is replaced.

| Variant | Gate | Candidate |
|---|---|---|
| EXP014 | learned EXP014 predictions | learned EXP014 predictions |
| OracleGate + LearnedCandidate | true U2 | learned EXP014 predictions |
| LearnedGate + OracleCandidate | learned EXP014 predictions | true U2 |
| Fixed-first Second-Step Oracle | true U2 | true U2 |

True U2 is used only for offline oracle branches and never as a learned-policy
input.  The fixed-first oracle compares `Stay=0`, `true U2(p2)` and
`true U2(p3)` with the existing deterministic geodesic/viewpoint-ID tie-break.

## Frozen inputs

- Stage C-v0 Val trajectories/predictions;
- corrected EXP014 Val second-step predictions;
- corrected EXP014 Stage D Val cache;
- Stage B Val utility labels for offline U2/oracle evaluation.

Stage A/B/C-v0, EXP014 and EXP015 artifacts, geometry, utility targets and
first-step protocol are unchanged.  Test is not accepted by the CLI.

## Execution contract

`run.sh` is provided for a separately authorized Val analysis.  This task only
implements and tests the code with synthetic fixtures; the real Val analysis
has **not** been executed.
