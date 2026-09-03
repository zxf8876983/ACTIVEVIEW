# EXP057 — Final Method Freeze Before Test

## Frozen canonical method

`WM-E + Multi-Positive Joint Revision + Closed-Loop H2`

The protocol starts from frozen Stage-C `s0/s1`, predicts imagined legal
candidates with the frozen recognition-aware WM-E, selects with the frozen
Multi-positive Joint Revision (seed 42), reveals real archived observations,
recomputes the current-view candidate graph and geometry for step 2, and uses
the real terminal archived skeleton through frozen ST-GCN for final HAR.  The
candidate budget is `ALL_LEGAL`; visited viewpoints are removed.

## Frozen Val reference (recorded, not re-evaluated)

EXP055 seed 42:

- Moving subset: Accuracy `0.683022`, Macro-F1 `0.647338`
- Full population: Accuracy `0.700722`, Macro-F1 `0.650955`

EXP056 provides objective-stability evidence: Multi-positive wins Accuracy
3/3 seeds and Macro-F1 3/3 seeds.  Its paired Original JR is an
ALL_LEGAL-controlled retraining baseline, not an exact reproduction of the
historical EXP050-R1 multi-budget checkpoint; those deltas are stability
evidence, not a replacement for the EXP051-R2 absolute baseline.

## Predefined Final Test comparison

Only these methods are registered: Initial/frozen Stage-C baseline, H1 real
observation policy, Original JR H2, and the frozen Multi-positive JR H2.  The
primary population is FULL; MOVING is secondary mechanism analysis.  Primary
metrics are Accuracy and Macro-F1, with only the three predeclared final-method
deltas.

No Test split has been read or evaluated in EXP057.  No model was trained and
no perception, Habitat, skeleton, or ST-GCN artifact was regenerated.
