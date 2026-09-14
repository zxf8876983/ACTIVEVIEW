# Yaw8 ST-GCN Recognizer Rebuild — completed

The reduced12 clean ST-GCN recognizer was rebuilt with eight fixed humanoid
scene yaws while preserving the original BABEL record split and perception
preprocessing. Train/Val generation used eight CUDA worker processes and
completed with 17,696/1,960 samples and zero failures. The new recognizer
reached 0.725510/0.727956 Accuracy/Macro-F1 on Yaw8 Val versus
0.583163/0.545710 for the old checkpoint; original clean Val changed by
-3.265pp Accuracy. Policy artifacts and Policy Test remained out of scope.
Runtime data/checkpoints are external under `ACTIVEVIEW_DATA_ROOT`; only the
protocol, result and analysis artifacts are tracked here.
