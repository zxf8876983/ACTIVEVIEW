# EXP055 — Multi-Positive Joint Revision

Train-only multi-positive action supervision for the frozen H2 closed-loop
protocol.  A context contributes all legal actions whose archived recognition
is correct; when none is positive, the fallback action is the highest true
label log-probability.  The model is evaluated once on Val using real archived
ST-GCN terminal recognition.  No Test, perception regeneration, Habitat
rendering, or additional model variant is used.

Run with `run.sh` after the Conda `habitat`/CUDA preflight. Runtime checkpoint
and rollout artifacts remain under the external data root; compact result and
analysis files are tracked here.
