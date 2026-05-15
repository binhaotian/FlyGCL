# FlyPrompt 实验记录（v7 Adam Balanced + OnlineHeadBank）

- 日期: 2026-04-15
- 运行 note: paper_align_sup21k_seed1_head_deepdive_v7_adam_balanced_onlineheadbank
- 状态: Completed
- 改动点: 每个 task 结束时保存该 task 的 online 头；评估时按路由 expert 使用对应 task 的 online 头。

## 配置

- method: flyprompt
- dataset: cifar100
- backbone: vit_base_patch16_224
- n_tasks: 5
- n/m: 50/10
- optimizer: adam
- lr: 0.005
- batchsize: 64
- num_epochs: 1
- online_iter: 3
- eval_period: 1000
- rp_dim: 10000
- seed: 1
- extra: --use_amp --analysis_expert_similarity --no_rnd_NM

## v7 最终 Summary

- A_auc: 0.8061551716111885
- A_avg: 0.8340503174603174
- A_last: 0.8287
- F_last: 0.02511111232969496
- BWT_last: 0.012222220169173346

## 与 v6（balanced baseline）对比

v6:
- A_auc: 0.8218706687822536
- A_avg: 0.8410534126984126
- A_last: 0.8499
- BWT_last: 0.011333332790268793

Delta (v7 - v6):
- A_auc: -0.015715
- A_avg: -0.007003
- A_last: -0.0212
- BWT_last: +0.000889

结论:
- 在该种子与该配置下，OnlineHeadBank 改动使最终精度下降（A_last 下降约 2.12 个百分点）。

## 多头行为变化（来自 head_output_analysis_seed_1.json）

v6:
- overall_selected_acc: 0.8499
- standalone_acc: [0.8278, 0.8246, 0.8297]
- selected_acc: [0.8128, 0.8744, 0.8631]
- selected_count: [3456, 3702, 2842]
- pairwise_agreement_offdiag_mean: 0.895233
- pairwise_jsd_offdiag_mean: 0.050656

v7:
- overall_selected_acc: 0.8287
- standalone_acc: [0.8256, 0.8265, 0.8290]
- selected_acc: [0.8509, 0.8144, 0.8096]
- selected_count: [4339, 2452, 3209]
- pairwise_agreement_offdiag_mean: 0.970067
- pairwise_jsd_offdiag_mean: 0.005011

解释:
- 各头 standalone 准确率变化很小，但头间输出差异显著减小（agreement 上升、JSD 下降），表现为多头更同质化。
- 选择器在 v7 更偏向 online 头（selected_count 从 3456 升至 4339），而 ema_1/ema_2 被选中减少。
- 同质化与选择偏移共同导致 ensemble 增益下降，最终 A_last 低于 v6。

## 产物路径

- 日志目录: results/logs/cifar100/paper_align_sup21k_seed1_head_deepdive_v7_adam_balanced_onlineheadbank
- 主日志: results/logs/cifar100/paper_align_sup21k_seed1_head_deepdive_v7_adam_balanced_onlineheadbank/stdout.log
- 最终权重: results/logs/cifar100/paper_align_sup21k_seed1_head_deepdive_v7_adam_balanced_onlineheadbank/final_model_seed_1.pth
- 训练进度: results/logs/cifar100/paper_align_sup21k_seed1_head_deepdive_v7_adam_balanced_onlineheadbank/training_progress_seed_1.jsonl
- 多头分析: results/logs/cifar100/paper_align_sup21k_seed1_head_deepdive_v7_adam_balanced_onlineheadbank/head_output_analysis_seed_1.json
