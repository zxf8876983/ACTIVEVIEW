# Revised Reduced-15 BABEL ST-GCN

本实验删除旧 15 类中的 `stretch` 与 `take/pick something up`，加入
`touching face`，得到新的 15 类：`walk`, `sit`, `stand up`, `bend`, `crawl`,
`stumble`, `wave`, `clap`, `throw`, `clean something`, `jump`, `kick`,
`knock`, `punch`, `touching face`。

Official Train/Val 每类上限为 300/100，使用 source、subject、AMASS dataset
和 duration-bin diversity-first 选择；Official Train 按 90/10 生成 ST-GCN
development Train/Val。数据固定 30-frame resampling，无 temporal segment
jitter。生成使用 16 个独立 Habitat worker。

仅生成和训练 ST-GCN development Train/Val；没有生成、读取或评估 Test。
