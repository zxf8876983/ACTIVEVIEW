# EXP040 — Sequential Belief-Space Active HAR

Val-only sequential diagnostic over the fixed 4-radius × 8-azimuth graph.
Visited viewpoint ST-GCN beliefs are acquired only after an actual transition;
unvisited viewpoint outputs and true CE are never planner inputs. Terminal-view
HAR and cumulative fused-belief HAR are reported separately for H=1/2/3.
For this rapid pilot, candidate risk/correctness heads are scored from the
initial legal Stage-D state; only the ST-GCN belief is updated after each
visited transition. This approximation is recorded explicitly in result.json.
