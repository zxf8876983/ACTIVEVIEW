# EXP042

The lazy candidate-conditioned world-model implementation is prepared.  It
uses only observed s0/s1 skeletons, optional frozen beliefs and visited RGB;
future perceived skeletons are targets only.  Fixed Train protocol is recorded
in `config.yaml`.  Real training/evaluation is deferred because the active
runtime has no CUDA device; no Test data was read.
