# Analysis

旧 reduced15 runtime 数据与 checkpoint 已删除并由本独立 revised protocol
替代。类别为 walk、sit、stand up、bend、crawl、stumble、wave、clap、throw、
clean something、jump、kick、knock、punch、touching face。每类 Official
Train/Val 上限 300/100，按 unique source、subject、AMASS dataset 与
duration-bin 优先选择，Official Train 再按 90/10 划分 ST-GCN development。

| split | records | unique sources | unique subjects | AMASS datasets | duration total (s) |
|---|---:|---:|---:|---:|---:|
| Train | 2,796 | 1,597 | 389 | 17 | 10,976.310 |
| Val | 310 | 278 | 163 | 16 | 1,284.878 |

骨架使用 16 个独立 Habitat worker 生成，固定 30-frame resampling，未启用
temporal segment jitter。ST-GCN 使用 CUDA:0、seed=42、batch=64、Adam
1e-3、oversample power=0.5，依据 Train loss early stopping；最终 epoch=172。
最终 Train Accuracy/Macro-F1 为 0.995708/0.996167；posthoc development Val
Accuracy/Macro-F1 为 0.651613/0.672486，未用于模型选择。

`touching face` 在当前筛选后为 Train=140、Val=15，显著比原先被删除的
`eat`/`telephone call` 更适合纳入训练。Test skeleton 未生成，Test 未读取或
评估；原始 BABEL/AMASS 数据保持不变。
