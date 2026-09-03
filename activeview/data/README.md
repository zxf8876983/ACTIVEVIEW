# Data layer

`data/` owns the offline data chain and keeps responsibilities separate:

- `motion/`: AMASS/BABEL loading, filtering, joint mapping and Habitat motion conversion;
- `generation/`: offline policy episodes and Stage-B utility labels;
- `preprocessing/`: feature, geometry, RGB and skeleton caches;
- `splits/`: deterministic Train/Val/Test policy split metadata.

This layer reads and writes runtime artifacts through `activeview.core.paths`.
Model architectures and evaluation metrics do not belong here.
