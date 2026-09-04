# Analysis

旧 revised reduced15 runtime 数据与 checkpoint 已删除（原始 BABEL/AMASS 保留）。
新协议只把 `wave` 替换为 `kneel`，使用 cap=300/100、seed=42 和 diversity-first
选择，Official Train 再按 90/10 生成 ST-GCN development。

| split | records | unique sources | unique subjects | AMASS datasets | duration total (s) |
|---|---:|---:|---:|---:|---:|
| Train | 2,700 | 1,535 | 389 | 17 | 10,628.138 |
| Val | 300 | 268 | 161 | 16 | 1,230.223 |

完整 cap-100 Official Val 主动感知 manifest 共 1036 条，已使用相同 16-worker
CUDA 生成固定 30 帧骨架至 `activeview_official_val/`。协议的
`activeview/val.json` 仅是 60/20/20 切分中的 209 条子集，未将其作为完整
Official Val 报告；没有生成任何 Test skeleton。

各类别 development 数量（Train/Val）为：walk 270/30，sit 270/30，stand up
270/30，bend 270/30，crawl 63/7，stumble 104/12，kneel 87/10，clap 88/10，
throw 270/30，clean something 131/15，jump 270/30，kick 270/30，knock 68/7，
punch 129/14，touching face 140/15。

骨架使用 16 个独立 CUDA Habitat worker 生成，固定 30-frame resampling、无
temporal segment jitter。ST-GCN 使用 CUDA:0、seed=42、Adam 1e-3、batch=64、
weight decay=1e-4、oversample power=0.5，训练 200 epochs；最终 Train
Accuracy/Macro-F1 为 0.992963/0.991418，posthoc development Val
Accuracy/Macro-F1 为 0.700000/0.710511。Val 仅作训练后报告，未用于选择。

Test skeleton 未生成，Test 未读取或评估；既有 reduced12、reduced16、原始
BABEL/AMASS 及其他 frozen protocol 未修改。
