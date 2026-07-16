# AMP 消融实验

| 运行设置 | 配置 | seed | A_auc | A_avg | A_last |
| --- | --- | ---: | ---: | ---: | ---: |
| 不启用 AMP | `config/flyprompt_cifar100_sup21k_balanced_v6.yaml` | 1 | 0.808177 | 0.831766 | 0.838900 |
| 启用 AMP | `config/flyprompt_cifar100_sup21k_balanced_v6_amp.yaml` | 1 | 0.819422 | 0.846649 | 0.846200 |

| 差值 | A_auc | A_avg | A_last |
| --- | ---: | ---: | ---: |
| variant - base | +0.011245 | +0.014883 | +0.007300 |

## 解释

在 seed 1 上，启用 AMP 后 `A_auc`、`A_avg` 和 `A_last` 都有提升。因此最终与作者代码口径对齐的配置保留 `use_amp: True`。
