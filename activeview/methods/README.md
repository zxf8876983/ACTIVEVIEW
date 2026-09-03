# Methods layer

`methods/` contains the paper method and comparison policies:

- `world_model/`: Candidate Observation World Model (WM-E);
- `joint_revision/`: candidate-set scoring and multi-positive revision;
- `active_view/`: frozen initial policy, geometry, training helpers and closed-loop rollout;
- `baselines/`: NoMove, Random, FrozenStageCv0 and oracle behavior.

Methods consume prepared data and frozen recognition outputs; data generation
and metric aggregation remain in their dedicated layers.
