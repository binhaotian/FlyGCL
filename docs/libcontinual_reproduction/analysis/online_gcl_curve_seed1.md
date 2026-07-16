# Seed 1 在线 GCL 曲线

CSV 文件：

```text
reproduce/flyprompt/analysis/libcontinual_v6_amp_seed1_online_curve.csv
```

曲线图：

```text
reproduce/flyprompt/analysis/online_gcl_curve_seed1.png
```

| 项目 | 数值 |
| --- | ---: |
| 评估点数量 | 50 |
| 第一个样本计数 | 1000 |
| 最后一个样本计数 | 50000 |
| 第一个 GCL accuracy | 0.530217 |
| 最终 GCL accuracy | 0.846200 |
| 最佳 GCL accuracy | 0.856571，出现在 20000 samples |

曲线横轴是已经消耗的训练样本数。PNG 图中的竖线标记 10000 样本间隔的 session 边界。

本地 FlyGCL v6 note 只记录了最终 summary 指标，没有包含每 1000 个样本一次的 online curve。因此目前只能做 summary-level 对齐；如果后续找回原始 FlyGCL `training_progress_seed_1.jsonl`，可以再补充逐点评估曲线对齐图。
