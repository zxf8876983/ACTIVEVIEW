# EXP042-R1 results

Train contexts: 29133; Val contexts: 9742; dense targets: 932256 / 311744.

Corrected Val baselines and A–F model metrics are in `result.json`; velocity/acceleration use temporal axis 1 and H36M-17 edges. E/F use frozen ST-GCN KL weight 0.10 (EXP045).

| Variant | Train loss | p2/p3 coord MAE | p2/p3 coord RMSE | p2/p3 bone rel. error | STGCN agreement |
|---|---:|---:|---:|---:|---:|
| A | 0.040575 | 0.140780 | 0.214764 | 0.173974 | 0.3137 |
| B | 0.041623 | 0.140348 | 0.214585 | 0.166248 | 0.2992 |
| C | 0.030048 | 0.127001 | 0.196296 | 0.144160 | 0.3845 |
| D | 0.036818 | 0.126943 | 0.199075 | 0.151808 | 0.4321 |
| E | 0.146497 | 0.182172 | 0.253557 | 0.353550 | 0.6059 |
| F | 0.166195 | 0.182776 | 0.259424 | 0.398901 | 0.5744 |

No Test data, perception regeneration, or GT pose input was used.