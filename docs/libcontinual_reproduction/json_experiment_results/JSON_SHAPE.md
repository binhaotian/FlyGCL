# FlyPrompt JSON 字段说明

本文件说明归档的 `flyprompt_gcl_seed_*.json` 实验结果文件结构。可优先参考下面这个主结果文件：

```text
local_reference_json/all_local_json/results_balanced_v6_amp__flyprompt_gcl_seed_1.json
```

## 顶层字段

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `A_auc` | float | 所有 online GCL evaluation points 的平均 accuracy。报告表格中通常乘以 100 后以百分数展示。 |
| `A_avg` | float | 各 session/task 结束时 GCL accuracy 的平均值。 |
| `A_last` | float | 最后一个 session 结束后的最终 GCL accuracy。 |
| `F_last` | float | 根据 per-class accuracy history 计算的最终遗忘值。 |
| `BWT_last` | float | 最终 backward transfer 指标。 |
| `task_acc` | list of float | session/task 结束时的 GCL accuracy。默认 5-session 设置下长度为 5。 |
| `online_test_acc` | list of float | 周期性 online evaluation 的 GCL accuracy。默认 CIFAR-100 主实验中长度为 50。 |
| `online_data_cnt` | list of int | 与 `online_test_acc` 对齐的已消耗训练样本数，默认是 `1000, 2000, ..., 50000`。 |
| `cls_acc` | nested list of float | per-class accuracy history，用于计算 forgetting/BWT 等指标。 |
| `seen_classes` | nested list of int | 各评估或 session 记录中已见类别 id。 |

## 使用建议

如果只想核对报告主表，优先读取 `A_auc`、`A_last`、`A_avg`、`F_last` 和 `BWT_last`。如果要复查 online curve，则读取 `online_data_cnt` 与 `online_test_acc`。
