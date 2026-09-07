# Current Task

## Action-Discriminative WM-E — completed 2026-09-07

Implemented and ran the reduced14 Train/Val Action-Discriminative WM-E. The
existing candidate-conditioned WM-E skeleton head and pose/velocity objective
were retained; optional 256-D ST-GCN feature and 14-way recognition heads were
added. Training used 44,248 Train contexts for 12 epochs (seed 42, CUDA RTX
4090). The fixed objective was
`L_pose + 0.1*L_rec + 0.1*L_feat + 0.2*L_belief`, with frozen ST-GCN and
pretrained History Identity teachers. The selected checkpoint is epoch 10 by
Val belief-entropy Pearson and is stored separately at
`/home/zxf/WorkSpace/code/data/ActiveView/checkpoints/activeview_reduced14_eight_placement_v1/wm_e_action_discriminative_best.pth`.

On 14,809 Val moving contexts and 426,474 legal candidates, WM-E diagnostics
were recognition agreement 0.573531, candidate true-class Pearson/Spearman
0.538782/0.667349, feature cosine 0.903356, belief KL 0.326315, belief
entropy Pearson/Spearman 0.775088/0.771148, and belief-margin
Pearson/Spearman 0.615788/0.503336. Candidate Top-1/Top-3 positive hits were
0.578567/0.720035 (0.663569/0.825821 conditioned on 12,912
oracle-positive contexts). Final Train loss was 0.170400.

Using the new WM-imagined H1 beliefs, min-entropy/max-margin selectors reached
identity Accuracy/Macro-F1 0.392734/0.395487 and 0.360119/0.368381. Frozen H1
was 0.482815/0.496501; real-observation privileged min-entropy/max-margin
were 0.512594/0.504617 and 0.515970/0.510083; IdentityOracle was
0.899588/0.900413. The new imagined selectors therefore improved over old
WM-E but did not exceed Frozen H1. Results are in
`experiments/reduced14_eight_placement_v1/action_discriminative_wm_e/`.

Only Train/Val artifacts were read for the experiment; Test was not read.
The old WM-E, JR, ST-GCN and History Identity checkpoints were not overwritten.

Status: CLEAN. No follow-up experiment is authorized automatically.
