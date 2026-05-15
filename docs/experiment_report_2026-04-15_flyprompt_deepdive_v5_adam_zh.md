# FlyPrompt v5_adam 实验整理（中文）

## 1. 实验背景与目的
本次实验是在 `paper_align_sup21k_seed1_head_deepdive_v4` 的基础上进行口径修正。v4 使用了 `sgd`，导致整体结果低于历史高结果。为与历史实验对齐，本次将优化器改为 `adam`，其余核心参数保持一致，并继续保留多 EMA 头深挖分析。

实验目录：
- results/logs/cifar100/paper_align_sup21k_seed1_head_deepdive_v5_adam

## 2. 关键配置
- method: flyprompt
- dataset: cifar100
- backbone: vit_base_patch16_224
- n_tasks: 5
- Si-Blurry: n=50, m=10
- batchsize: 64
- lr: 0.005
- num_epochs: 1
- online_iter: 3
- eval_period: 1000
- optimizer: adam
- ema_ratio: [0.9, 0.99]
- ensemble_method: softmax_max_prob
- analysis_expert_similarity: True
- seed: 1

## 3. 总体结果（Summary）
- A_auc: 0.8196096666015553
- A_avg: 0.8204214077958089
- A_last: 0.8431
- F_last: 0.06804597275010471
- BWT_last: 0.0378160889249766

结论：
- 指标与历史高结果实验（paper_align_sup21k_seed1_headviz）对齐。
- 说明此前 v4 与高结果差距的主要来源确实是优化器口径差异。

## 4. EMA 详细分析结果
数据来源：
- head_output_analysis_seed_1.json
- similarity_seed_1.npy / cka_seed_1.npy
- residual_similarity_seed_1.npy / residual_cka_seed_1.npy

### 4.1 三个头的基础表现
| Head | Standalone Acc | Selected Count | Acc When Selected |
|---|---:|---:|---:|
| online | 0.8082 | 3126 | 0.7892 |
| ema_1 | 0.8208 | 3863 | 0.8760 |
| ema_2 | 0.8234 | 3011 | 0.8569 |

- Overall selected acc: 0.8431

解释：
- 两个 EMA 头的 standalone 准确率都高于 online。
- 被选中时，ema_1 与 ema_2 都明显优于 online，其中 ema_1 最强。

### 4.2 Pairwise 指标
Agreement Rate
|  | online | ema_1 | ema_2 |
|---|---:|---:|---:|
| online | 1.0000 | 0.8489 | 0.8407 |
| ema_1 | 0.8489 | 1.0000 | 0.9629 |
| ema_2 | 0.8407 | 0.9629 | 1.0000 |

Mean JSD
|  | online | ema_1 | ema_2 |
|---|---:|---:|---:|
| online | 0.0000 | 0.0811 | 0.0856 |
| ema_1 | 0.0811 | 0.0000 | 0.0068 |
| ema_2 | 0.0856 | 0.0068 | 0.0000 |

解释：
- ema_1 与 ema_2 的一致性非常高（agreement 高、JSD 极低），表现出强同质性。
- online 与 EMA 头之间分歧更大，体现 online 头更“即时”、波动更大。

### 4.3 条件概率（你关注的“选中且对错后其他头表现”）
Other Correct Given Selected Wrong
| selected\other | online | ema_1 | ema_2 |
|---|---:|---:|---:|
| online | NaN | 0.2777 | 0.3126 |
| ema_1 | 0.1566 | NaN | 0.0397 |
| ema_2 | 0.1369 | 0.0557 | NaN |

解释：
- online 头选错时，EMA 头仍有较强概率给出正确答案（约 27.8% / 31.3%）。
- 某个 EMA 头选错时，另一个 EMA 头纠错空间较小，进一步支持“两个 EMA 头相似度高”的结论。

### 4.4 特征层相似度 / CKA
- similarity（off-diagonal）整体较高，说明专家特征有较强共享。
- CKA 也保持高相关，说明主表示空间差异不大。
- residual 相似度与 residual CKA 显示去掉公共部分后仍有结构性差异，不是完全塌缩到同一方向。

## 5. 与论文叙事的对应关系
在仓库可见材料中：
- README 明确提到 FlyPrompt 的核心是 temporal-ensemble experts（时间集成专家）。
- 论文公开口径主要是总体性能指标（A_last/A_avg 等）。
- 本次 deepdive 的细粒度指标（pairwise/conditional/CKA）更偏机制解释，属于对论文叙事的补充证据，而非论文主表格的逐项复刻。

## 6. 总结（自然语言）
这次 v5_adam 实验说明：当优化器口径与历史高结果保持一致时，FlyPrompt 能稳定达到 A_last=84.31%。从多 EMA 头视角看，EMA 头整体优于 online 头，且 ema_1 在“被选中时准确率”上最稳健。ema_1 与 ema_2 在预测层与特征层都高度一致，说明 EMA 机制主要贡献是稳定化，而非形成强互补的异质专家。online 头保留了更多即时更新特征，因此与 EMA 头分歧更大，也因此在其出错时常可被 EMA 头纠正。这与论文中 temporal-ensemble experts 的整体叙事一致，并由本次深挖统计提供了更细粒度的支持。
