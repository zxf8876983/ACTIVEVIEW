# Notes: reduced12 RGB-conditioned WM-E restoration

## Integrity audit

- Frozen reduced12 ST-GCN SHA256: `078cff9490fb51fe52ebaa7d3fa4ac2802ff6c3eebba44de71c8276cd5e86a71`.
- Conda environment: `/home/zxf/anaconda3/envs/habitat/bin/python`.
- CUDA: available on `NVIDIA GeForce RTX 4090`.
- Val-only archive audit: 24 Stage-B rows, 149 viewpoints; all archived ST-GCN predicted labels matched Stage-B labels. Maximum absolute error for stored true-class log probability was `0.00094986`.
- Val Stage-B/H0 contexts: 15,540; H0 legal candidate count min/mean/median/max = 1/6.5878/6/21.
- Stage-D Train/Val contexts: 30,580/10,080; H2 remaining candidate count min/mean/median/max = 1/1.9198/2/2.
- Current reduced12 scene set: 20 scenes. Previous reduced14 diagnostic scene set: 21 scenes; intersection 3, reduced12-only 17, reduced14-only 18. This is a documented protocol population difference, not a label/viewpoint/cache mismatch.
- Test files were not read.

## Cache plan

- RGB output: `datasets/rgb_reduced12_eight_placement_v1/visited_s0_s1/`.
- DINO output: `features/dinov2_vitb14_spatial4x4_reduced12_eight_placement/initial_history/`.
- Only unique Train/Val s0/s1 keys are allowed; future candidate RGB is forbidden.
