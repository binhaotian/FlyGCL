# FlyPrompt 实验记录（v6 Adam Balanced）

- 日期: 2026-04-15
- 运行 note: `paper_align_sup21k_seed1_head_deepdive_v6_adam_balanced`
- 状态: Completed
- 说明: 本次为 `--no_rnd_NM` 的均分 task 实验（每个 task 约 10k 样本）；该实验启动于“per-task online head snapshot”代码改动之前。
主要的改动就是同意了每一个task的样本量

## 配置

- method: `flyprompt`
- dataset: `cifar100`
- backbone: `vit_base_patch16_224`
- n_tasks: `5`
- n/m: `50/10`
- optimizer: `adam`
- lr: `0.005`
- batchsize: `64`
- num_epochs: `1`
- online_iter: `3`
- eval_period: `1000`
- rp_dim: `10000`
- seed: `1`
- extra: `--use_amp --analysis_expert_similarity --no_rnd_NM`

## 最终 Summary

- A_auc: `0.8218706687822536`
- A_avg: `0.8410534126984126`
- A_last: `0.8499`
- F_last: `0.045555553171369764`
- BWT_last: `0.011333332790268793`

## 关键输出目录

- 日志目录: `results/logs/cifar100/paper_align_sup21k_seed1_head_deepdive_v6_adam_balanced`
- 主日志: `results/logs/cifar100/paper_align_sup21k_seed1_head_deepdive_v6_adam_balanced/stdout.log`
- 最终权重: `results/logs/cifar100/paper_align_sup21k_seed1_head_deepdive_v6_adam_balanced/final_model_seed_1.pth`
- 训练进度: `results/logs/cifar100/paper_align_sup21k_seed1_head_deepdive_v6_adam_balanced/training_progress_seed_1.jsonl`
- 多头分析: `head_output_analysis_seed_1.json` 及相关 PNG/NPY 产物
