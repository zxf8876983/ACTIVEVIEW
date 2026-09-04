# Analysis

将原 reduced15 中的 `stretch` 与 `take/pick something up` 删除，并加入
`bend`、`eat`、`telephone call`，形成独立 16 类协议。Official Train/Val
仍按每类 300/100 上限和 diversity-first 规则选择；Official Train 以 90/10
划分 ST-GCN development Train/Val。由于 `eat` 在 BABEL 中极少，实际仅有
11 个 Train 和 1 个 Val 样本；该稀疏性保留在结果中，没有人为补样。

| split | records | unique sources | unique subjects | AMASS datasets | duration total (s) |
|---|---:|---:|---:|---:|---:|
| Train | 2,716 | 1,569 | 390 | 17 | 10,791.932 |
| Val | 301 | 272 | 166 | 16 | 1,271.340 |

骨架生成使用 16 个独立 Habitat worker、CUDA:0、固定 30-frame resampling，
未启用 temporal segment jitter。ST-GCN 使用 seed=42、batch=64、Adam
1e-3、oversample power=0.5，最多 200 epochs，并仅依据 Train loss early
stopping；最终 epoch=184。最终 Train Accuracy/Macro-F1 为
0.997054/0.997775，posthoc Val Accuracy/Macro-F1 为 0.664452/0.620068。
Val 指标不参与模型选择。本轮没有生成或读取 Test skeleton，也没有进行
Test 评估。
