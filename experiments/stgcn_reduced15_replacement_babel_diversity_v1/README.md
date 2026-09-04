# Reduced-15 BABEL ST-GCN replacement

本实验在独立运行时目录中将 `wave` 替换为 `kneel`，保留其余 14 类：
`walk`, `sit`, `stand up`, `bend`, `crawl`, `stumble`, `clap`, `throw`,
`clean something`, `jump`, `kick`, `knock`, `punch`, `touching face`。

Official Train/Val 每类上限为 300/100，按 unique source、subject、AMASS
dataset 和 duration-bin diversity-first 选择；Official Train 按 90/10 生成
ST-GCN development Train/Val。骨架固定 30-frame resampling，无 temporal
segment jitter，使用 16 个独立 Habitat worker 和 CUDA:0。

本任务生成并训练 ST-GCN development Train/Val，并生成完整 cap-100
Official Val 主动感知骨架（1036 条）至 `activeview_official_val/`。
协议默认的 60/20/20 子集 manifest 仍保留作记录，但不把其中 20% 子集
误称为完整 Official Val。没有生成 Test skeleton、读取或评估 Test。

运行时数据位于 `/home/zxf/WorkSpace/code/data/ActiveView/`，不纳入 Git。
