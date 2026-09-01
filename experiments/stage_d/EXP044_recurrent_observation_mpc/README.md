# EXP044 — Recurrent Observation MPC

Receding-horizon offline diagnostic.  After each selected move the real
visited skeleton is appended and the history is re-encoded; imagined future
skeletons are never treated as observed transitions.  Test is locked.
