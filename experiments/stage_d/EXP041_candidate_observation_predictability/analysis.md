# EXP041

The frozen EXP014 evaluator gate passed at Accuracy 0.6582540931 and
Macro-F1 0.6101526052.  Source archives contain 32 unique viewpoint IDs and
the perceived target shape is `[3,30,17]`; target identity audit is PASS for
29,133 Train and 9,742 Val contexts.  No Test data was read.

Full baseline computation is exposed by `run_stage_d_exp041_044.py` and was
not silently substituted with ground-truth pose.
