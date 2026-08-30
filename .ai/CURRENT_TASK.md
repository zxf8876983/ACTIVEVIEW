# Current Task

## EXP017 — second-step gate calibration completed

Stage C-v0, corrected EXP014, corrected EXP015 and EXP016 remain frozen.
EXP017 applied exactly one intervention: selecting a strict scalar
`gate_score > tau` threshold on Train from frozen EXP014 predictions, then
applying that frozen threshold once to Val. The learned p2/p3 candidate ranking
and the Stage C-v0 first action/p1 proposal were unchanged.

## Result

Train selected `tau=-0.08218251913785934` from 29,133 episodes. On 13,987 Val
episodes, EXP014 tau=0 achieved Accuracy 0.658254, Macro-F1 0.610153 and mean
regret 1.422463; EXP017 achieved 0.650962, 0.598102 and 1.477153. Headroom
decreased from 0.783313 to 0.777146. The threshold changed 2,838 second-step
Stay decisions to Move, with no Move-to-Stay changes. Candidate identity was
unchanged in all 2,356 episodes where both policies moved.

Train calibration improved gate balanced accuracy on Val (0.571825 →
0.609873) and Move recall (0.304729 → 0.629359), but the trajectory metrics
worsened. A single global threshold is therefore rejected as a deployable
intervention; see the EXP017 `analysis.md` and runtime result under
`ACTIVEVIEW_DATA_ROOT/experiments/stage_d/EXP017_second_step_gate_calibration/`.

## Protocol boundaries

- EXP017 used Train for threshold fitting and Val for one evaluation only.
- Test was not read or used.
- No training, Habitat rendering, perception regeneration, ST-GCN retraining,
  Stage A/B/C-v0 modification or EXP014/EXP015/EXP016 rerun was performed.

## Status

EXP017 is **REJECTED** as a scalar-threshold policy and remains a completed
diagnostic. Do not start another experiment automatically.
