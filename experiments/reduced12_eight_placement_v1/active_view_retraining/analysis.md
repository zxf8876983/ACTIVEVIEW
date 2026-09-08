# Reduced12 + Eight-placement ActiveView retraining

本轮只使用 reduced12 的 Train/Val，未读取 policy Test，也没有重新生成 perception。数据为 20 个 HM3D 场景、每场景 8 个 furniture-anchored placements、每条 archive 32 viewpoints。Stage-D moving contexts 为 Train 30,580、Val 10,080；完整 Val 为 15,540。

## Val benchmark

| Method | Full Accuracy | Full Macro-F1 | Moving Accuracy | Moving Macro-F1 |
|---|---:|---:|---:|---:|
| NoMove | 0.329601 | 0.329175 | 0.254266 | 0.235500 |
| Random legal-view | 0.341570 | 0.341245 | 0.330060 | 0.327350 |
| FrozenStageCv0 | 0.459331 | 0.449009 | 0.454266 | 0.444782 |
| H0 CandidateOracle | 0.699936 | 0.698727 | 0.709325 | 0.704330 |
| H0 SafeOracle | 0.730824 | 0.728527 | 0.727976 | 0.721495 |
| Multi-positive H2 | 0.487259 | 0.477739 | 0.497321 | 0.492134 |
| FixedH1-H2 SafeOracle | 0.755727 | 0.751001 | 0.911210 | 0.908047 |

相对 FrozenStageCv0，Multi-positive H2 在 Full 上为 +2.793 accuracy points、+2.873 Macro-F1 points；Moving 上为 +4.306 / +4.735 points。FixedH1-H2 SafeOracle 是单独的 H1 固定后 oracle 上界，不能与 H0 SafeOracle 混为一谈。

## H0 与 WM-E 诊断

- H0 SafeOracle：Full 0.730824 / 0.728527；AnyCorrect rate 0.731338。
- 合法 viewpoint 的平均 correct-view ratio 为 0.335912，平均正确合法 view 数为 2.2517，平均合法 view 数为 6.5878。
- WM-E legal candidate recognition agreement：0.371491。
- WM-E true-class Pearson / Spearman：0.360785 / 0.403005。
- WM-E Top-1 / Top-3 positive hit：0.498301 / 0.739231；Oracle-positive contexts 为 9,123。

## 训练与科学判断

Recognition-aware WM-E 使用固定的 SmoothL1 pose + 0.25 velocity + 0.10 frozen ST-GCN recognition KL；12 epochs 中最佳 Val loss 为 0.1572399（epoch 11）。JR 使用原 Multi-positive objective、BCE 权重 0.25、posterior CE 权重 0.05，20 epochs，最终 Train loss 1.554696。

12 类的 H0 SafeOracle/AnyCorrect 表明可观测视角仍有明显 perception ceiling，ActiveView 并非没有可恢复空间。Multi-positive H2 相比 FrozenStageCv0 有稳定但有限的 Val 提升；结合 WM-E agreement 与 ranking correlation，当前更接近 WM candidate fidelity + selector 联合瓶颈，而不是单纯 taxonomy ceiling。暂不重新引入后续 14 类实验优化，先保留该 12 类正式基线作为独立比较。

当前 eight-placement 数据没有可复用的 DINO spatial cache，因此本轮 checkpoint 明确记录 `use_rgb=false`、`rgb_dino_available=false`；没有在本轮偷偷重渲染 RGB 或引入未来观测。

`test_used=false`；未读取 policy Test，未覆盖 reduced14/旧16结果。
