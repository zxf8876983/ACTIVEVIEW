# Current Task

## Oracle Candidate Occlusion Context — completed 2026-09-07

Implemented and ran the privileged reduced14 + eight-placement Oracle Candidate
Occlusion Context diagnostic using Train/Val only. Static Habitat physics
raycasts produced a 10-D descriptor for 5,376 scene/placement/viewpoint
entries (21 scenes × 8 placements × 32 viewpoints): eight fixed body-landmark
LOS indicators, visible-keypoint ratio, and mean normalized obstruction
distance. No action label, recognition output, real candidate prediction or
Test row was used to build the descriptor.

The Action-Discriminative WM-E candidate conditioning was extended from 9-D to
19-D (the old 9-D geometry plus the optional 10-D context). The old checkpoint
was used only for initialization and remains untouched. The new WM-E was run
for 12 Train epochs (44,248 contexts, seed 42, CUDA RTX 4090), with Val
checkpoint selection by the existing belief-entropy Pearson / margin
Spearman rule. The checkpoint is stored separately at
`/home/zxf/WorkSpace/code/data/ActiveView/checkpoints/activeview_reduced14_eight_placement_v1/wm_e_oracle_occlusion_context_best.pth`.

On 14,809 Val moving contexts / 426,474 legal candidates, the oracle-context
model reached recognition agreement 0.585435, true-class Pearson/Spearman
0.564185/0.687090, feature cosine 0.907745, belief KL 0.315069, and
Top-1/Top-3 positive hit 0.588561/0.726585. The imagined H1 min-entropy and
max-margin selectors reached Accuracy/F1 0.396651/0.400672 and
0.352623/0.367313, respectively; both remain below Frozen H1
0.482815/0.496501. The result is recorded in
`experiments/reduced14_eight_placement_v1/oracle_occlusion_context/`.

Test was not read. Formal WM-E, JR, ST-GCN and taxonomy checkpoints were not
overwritten. The task is complete; do not start another experiment
automatically.
