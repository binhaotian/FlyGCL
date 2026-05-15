# FlyPrompt EMA 详细分析报告（v5_adam，2026-04-15）

## 实验背景
本次实验是对 `paper_align_sup21k_seed1_head_deepdive_v4` 的口径修正版本。v4 使用了 `sgd`，导致整体指标显著低于此前高结果实验。为保证与历史高结果（`paper_align_sup21k_seed1_headviz`）口径一致，本次将优化器改回 `adam`，其余核心参数保持一致。

实验目录：
- `results/logs/cifar100/paper_align_sup21k_seed1_head_deepdive_v5_adam`

## 运行配置（核心）
- method: `flyprompt`
- dataset: `cifar100`
- backbone: `vit_base_patch16_224`
- n_tasks: `5`
- n/m: `50/10`
- batchsize: `64`
- lr: `0.005`
- online_iter: `3`
- eval_period: `1000`
- opt_name: `adam`
- ema_ratio: `[0.9, 0.99]`
- ensemble_method: `softmax_max_prob`
- analysis_expert_similarity: `True`
- seed: `1`

## 总体结果
日志 Summary：
- `A_auc = 0.8196096666015553`
- `A_avg = 0.8204214077958089`
- `A_last = 0.8431`
- `F_last = 0.06804597275010471`
- `BWT_last = 0.0378160889249766`

结论：
- 本次结果与历史高结果 `paper_align_sup21k_seed1_headviz` 数值对齐（同一 seed 下几乎逐点一致），说明优化器口径是此前性能差异的主因。

## EMA 头详细分析
数据来源：
- `head_output_analysis_seed_1.json`
- `similarity_seed_1.npy`, `cka_seed_1.npy`
- `residual_similarity_seed_1.npy`, `residual_cka_seed_1.npy`

### 1) 三个头的独立性能（standalone）
- online: `0.8082`
- ema_1: `0.8208`
- ema_2: `0.8234`

解释：
- 两个 EMA 头独立精度略高于 online 头，说明 EMA 平滑在该配置下有稳定增益。

### 2) 被选中情况（selected）
- 被选中次数：
  - online: `3126`
  - ema_1: `3863`
  - ema_2: `3011`
- 被选中时准确率：
  - online: `0.7892`
  - ema_1: `0.8760`
  - ema_2: `0.8569`
- overall selected acc: `0.8431`

解释：
- 选择器更偏好 `ema_1`，且 `ema_1` 被选中时精度最高。
- 与 v4（sgd）相比，`ema_2` 的表现从“被选中质量较差”恢复到“与 ema_1 接近”。

### 3) 头间一致性与分歧
- agreement rate：
  - online vs ema_1: `0.8489`
  - online vs ema_2: `0.8407`
  - ema_1 vs ema_2: `0.9629`
- mean JSD：
  - online vs ema_1: `0.0811`
  - online vs ema_2: `0.0856`
  - ema_1 vs ema_2: `0.0068`

解释：
- 两个 EMA 头之间高度一致（agreement 高、JSD 极低）。
- online 与 EMA 头存在更明显预测分歧，说明 online 头更多反映即时更新噪声。

### 4) 条件概率（selected correct/wrong 条件下其他头表现）
- `other_correct_given_selected_wrong`：
  - selected=online 错时，ema_1/ema_2 正确概率约 `0.278 / 0.313`
  - selected=ema_1 错时，online/ema_2 正确概率约 `0.157 / 0.040`
  - selected=ema_2 错时，online/ema_1 正确概率约 `0.137 / 0.056`

解释：
- 当 online 选错时，EMA 头仍有较强纠错能力。
- 当某个 EMA 头选错时，另一个 EMA 头“补救”的空间较小，侧面印证两个 EMA 头同质性较强。

### 5) 特征层相似度与 CKA
- `similarity_seed_1.npy`（5 个 expert）显示整体较高正相关；
- `cka_seed_1.npy` 也整体较高（多数组合 > 0.83）；
- residual 相似度出现正负混合，residual CKA 显示中低相关，提示去掉公共部分后差异性更明显。

解释：
- 专家间共享了较多公共表征；
- 但 residual 空间仍保留了部分可区分结构，不是完全塌缩到同一个方向。

## 和论文（仓库可见信息）的对应关系
在仓库可见材料中（README 与 docs）：
- 明确提到 FlyPrompt 的思想是 `temporal-ensemble experts`；
- 论文级公开口径主要体现为总体性能指标（如 A_last/A_avg 等）；
- 没有看到与本次 deepdive 等价的细粒度指标（如 pairwise agreement/JSD、conditional 概率、CKA/residual CKA）作为论文主结果表格内容。

因此，本次 EMA 细粒度分析属于对论文结论的“机制补充解释”，不是对论文主表格的逐项复刻。

## 最终结论（自然语言）
这次 v5_adam 实验表明：在与历史高结果一致的优化器口径下，FlyPrompt 能稳定恢复到 `A_last=84.31%`。从多 EMA 头细节看，EMA 头整体强于在线头，且 `ema_1` 在被选中时表现最稳健；`ema_1` 与 `ema_2` 高度一致，更多体现为稳定化而非强互补。online 头保留了更“激进”的即时更新特征，因此与 EMA 头分歧更大，也带来一定可被 EMA 修正的错误。这与“temporal-ensemble experts 提升稳定性”的论文叙事一致，同时本次 deepdive 为该叙事提供了更细粒度的数据证据。
