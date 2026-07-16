# Seed 1 任务级对齐记录

本文件总结 LibContinual seed 1 在 FlyPrompt CIFAR-100 / Sup-21K 作者代码对齐配置下的任务级证据。

## LibContinual Task-End GCL Accuracy

| session | task-end GCL accuracy |
| ---: | ---: |
| 0 | 0.826333 |
| 1 | 0.856571 |
| 2 | 0.854250 |
| 3 | 0.849889 |
| 4 | 0.846200 |

## LibContinual 最终 Average Acc 表

| session | task 0 | task 1 | task 2 | task 3 | task 4 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 91.80 | 0.00 | 0.00 | 0.00 | 0.00 |
| 1 | 86.70 | 88.60 | 0.00 | 0.00 | 0.00 |
| 2 | 85.05 | 84.40 | 87.00 | 0.00 | 0.00 |
| 3 | 85.65 | 85.30 | 84.90 | 85.75 | 0.00 |
| 4 | 85.45 | 84.65 | 83.55 | 84.35 | 85.10 |

## 作者代码证据可用性

本地 FlyGCL v6 note 包含最终 summary 指标，但没有完整 per-task accuracy table，也没有 online curve samples。因此，在找回原始 FlyGCL 运行目录之前，本轮无法做逐行作者代码表格对齐。

| metric | FlyGCL v6 note seed 1 | LibContinual v6 AMP seed 1 | gap |
| --- | ---: | ---: | ---: |
| A_auc | 0.821871 | 0.819422 | -0.002448 |
| A_avg | 0.841053 | 0.846649 | +0.005595 |
| A_last | 0.849900 | 0.846200 | -0.003700 |
| F_last | 0.045556 | 0.038000 | -0.007556 |
| BWT_last | 0.011333 | 0.015222 | +0.003889 |

## 解释

- LibContinual seed 1 最终 `A_last = 0.846200`。
- 最强 task-end 结果出现在 session 1，accuracy 为 0.856571。
- 最终 session 略低于最佳中间 task-end 值，因此残余差距不只是最终评估实现造成的。
