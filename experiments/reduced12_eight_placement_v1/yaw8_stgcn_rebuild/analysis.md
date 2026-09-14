Experiment Yaw8 ST-GCN Recognizer Rebuild
Purpose mismatch: test whether clean recognizer training is mismatched to random eight-direction scene yaw.
Identical original record split; yaw augmentation 8 values: 0,45,90,135,180,225,270,315.
Augmentation at Habitat humanoid rendering before RGB; perception unchanged.
Preprocessing unchanged align_canonical=True; old assets preserved.
Policy Train/Val/Test not used for model selection; Policy Test not read.

## Results
- Original clean Val: old Acc/F1=0.767347/0.781069; new=0.734694/0.740031.
- Yaw8 Val overall: old Acc/F1=0.583163/0.545710; new=0.725510/0.727956.
- Mean yaw accuracy: old=0.583163, new=0.725510, MeanYawGain=+0.142347.
- Worst yaw accuracy: old=0.510204, new=0.714286, WorstYawGain=+0.204082.
- Yaw robustness gap (max-min): old=0.257143, new=0.020408.
- Original-clean change (new-old)=-0.032653.

## Decision
Final decision: **PARTIALLY CONFIRMED**.
This is a recognizer-only Train/Val result. Do not rebuild Policy automatically; freeze the new recognizer and rebuild Policy logits/utility only after an explicit decision.
