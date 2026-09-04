# Analysis

本轮建立了与旧 reduced12 独立的 15 类协议。Official Train 先按每类最多
300 条进行 diversity-first 选择，再按类别执行 90/10 ST-GCN development
划分；因此最终 Train=2,926、Val=325。所有样本使用 canonical 30-frame
resampling，未使用 temporal segment jitter。

| split | records | unique sources | unique subjects | AMASS datasets | duration total (s) |
|---|---:|---:|---:|---:|---:|
| Train | 2,926 | 1,757 | 389 | 17 | 11,147.192 |
| Val | 325 | 282 | 180 | 17 | 1,278.837 |

ST-GCN 使用 CUDA:0、seed=42、batch=64、Adam 1e-3、tempered inverse-
frequency oversampling（power=0.5），最多 200 epochs，依据 train loss
进行 early stopping；最终 epoch=164。最终 Train accuracy/Macro-F1 为
0.997949/0.998383。Val 指标为 posthoc 记录，未参与 epoch 或超参数选择：
accuracy=0.630769，Macro-F1=0.665723，balanced accuracy=0.675079。

该 checkpoint 与旧 reduced12、selected16 路径分离；本轮未访问 Test，也未启用
temporal candidate pool。
