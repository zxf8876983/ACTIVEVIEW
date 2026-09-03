# Script entry points

Scripts are intentionally thin command-line wrappers around the package:

- `data/`: dataset generation, split preparation and cache construction;
- `train/`: recognition and active-view training entry points;
- `eval/`: validators, baselines and final evaluation;
- `visualize/`: read-only scene and observation visualizations.

Reusable implementation belongs under `activeview/data`, `perception`,
`recognition`, `methods` or `evaluation`, not inside these wrappers.
