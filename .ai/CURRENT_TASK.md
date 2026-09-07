# Current Task

## GT-conditioned World Model — completed 2026-09-08

Implemented and ran the privileged reduced14 GT-conditioned Action-
Discriminative WM-E diagnostic using Train/Val only. A 14-D one-hot ground-
truth action is fused into the candidate-conditioned latent through a small
embedding and projection; the existing pose/feature/recognition/belief losses
and frozen ST-GCN/History Identity teachers are retained. The old WM-E remains
untouched. Training used 44,248 Train contexts for 12 epochs (seed 42, CUDA
RTX 4090), with Val selection by the existing belief-entropy Pearson / margin
Spearman rule. The separate checkpoint is stored at
`/home/zxf/WorkSpace/code/data/ActiveView/checkpoints/activeview_reduced14_eight_placement_v1/wm_e_gt_conditioned_best.pth`.

On 14,809 Val moving contexts / 426,474 legal candidate samples, the
GT-conditioned model reached recognition agreement 0.592210, true-class
Pearson/Spearman 0.542778/0.679059, feature cosine 0.905771, belief KL
0.317047, and Top-1/Top-3 positive hit 0.587481/0.729219. Imagined H1
min-entropy/max-margin reached identity Accuracy/Macro-F1
0.411304/0.419345 and 0.382403/0.394278, respectively; both remain below
Frozen H1 0.482815/0.496501. Results are recorded in
`experiments/reduced14_eight_placement_v1/gt_conditioned_world_model/`.

Test was not read. Formal WM-E, JR, ST-GCN, taxonomy and split artifacts were
not modified. The task is complete; do not start another experiment
automatically.
