# EXP050-R1

The original EXP050 Joint Revision protocol was retrained once with the frozen
configuration (seed 42, 20 epochs, batch 512, AdamW, 1e-3, weight decay
1e-4).  The runtime checkpoint is recorded by SHA256 in
`checkpoint_manifest.json` and is not committed to Git.

The ALL_LEGAL Val reproduction is Accuracy 0.685065 and Macro-F1 0.642002,
within the original reference performance level (0.686566/0.646064).  This
checkpoint is the frozen Joint Revision used by EXP051-R1.
