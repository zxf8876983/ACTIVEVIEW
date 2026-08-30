# EXP024 — DINOv2 RGB Context Utility Regression Pilot

EXP024 tests whether legal, already visited RGB observations improve raw
executed-candidate utility regression over the frozen EXP022 representation.
Only Stage-D `s0` and `s1` RGB views are embedded; p2/p3 RGB is never requested.

The RGB encoder is the official Hugging Face `facebook/dinov2-base` (DINOv2
ViT-B/14), frozen in evaluation mode. Its 768-D global CLS embedding is stored
as float16 in the external cache, projected through a shared trainable
`Linear(768,128) → GELU`, and concatenated with the frozen EXP014 contextual
token and predicted utility to form a 513-D input. The only trainable head is
the fixed 513→128→64→1 GELU MLP with SmoothL1Loss on raw `true_U2(c_hat)`.

Train uses 30 fixed epochs (Adam, lr 1e-3, batch 256, seed 42), with no Val
selection or threshold tuning. Val is evaluated once using
`predicted_U_exec > 0`; Stage C-v0 first action and EXP014 candidate ranking
remain frozen. Test is locked.

Run:

```bash
bash experiments/stage_d/EXP024_dinov2_rgb_context/run.sh
```

Runtime embeddings, checkpoint, predictions and full result are external:
`/home/zxf/MG08/robot/ActiveView/`.
