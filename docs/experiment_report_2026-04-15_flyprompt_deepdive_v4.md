# FlyPrompt Deepdive v4 实验整理（2026-04-15）

## 1. 实验目的
- 在 CIFAR-100 的 Si-Blurry 设定下完成 FlyPrompt 全流程复跑。
- 落盘并核验多 EMA 头分析结果（头选择、pairwise、conditional、相似度、CKA、residual）。
- 与历史高结果实验（paper_align_sup21k_seed1_headviz）做同仓口径对比，定位性能差异来源。

## 2. 运行配置（v4）
- note: paper_align_sup21k_seed1_head_deepdive_v4
- method: flyprompt
- dataset: cifar100
- data_dir: /root/autodl-tmp/data
- backbone: vit_base_patch16_224
- seeds: [1]
- n_tasks: 5
- n: 50
- m: 10
- batchsize: 64
- lr: 0.005
- num_epochs: 1
- online_iter: 3
- eval_period: 1000
- n_worker: 8
- use_amp: true
- optimizer: sgd
- rp_dim: 10000
- rp_ridge: 10000.0
- ema_ratio: [0.9, 0.99]
- ensemble_method: softmax_max_prob
- analysis_expert_similarity: true

证据来源：
- results/logs/cifar100/paper_align_sup21k_seed1_head_deepdive_v4/stdout.log 开头 Configuration 行。

## 3. 训练完成情况
- 训练进度到 100%，并出现 run_end。
- 最终模型已保存：
  - results/logs/cifar100/paper_align_sup21k_seed1_head_deepdive_v4/final_model_seed_1.pth
- 关键摘要：
  - A_auc = 0.5421205708323248
  - A_avg = 0.5052710928898436
  - A_last = 0.7528
  - F_last = 0.030229886372884113
  - BWT_last = -0.02000000226129403

## 4. 多 EMA 头分析（已完成）
主文件：
- results/logs/cifar100/paper_align_sup21k_seed1_head_deepdive_v4/head_output_analysis_seed_1.json

图与矩阵文件：
- head_selection_stats_seed_1.png
- head_standalone_acc_seed_1.png
- head_pairwise_metrics_seed_1.png
- head_conditional_metrics_seed_1.png
- similarity_seed_1.npy / similarity_seed_1.png
- cka_seed_1.npy / cka_seed_1.png
- residual_similarity_seed_1.npy / residual_similarity_seed_1.png
- residual_cka_seed_1.npy / residual_cka_seed_1.png

分析摘要（v4）：
- 头 standalone acc：
  - online: 0.7481
  - ema_1: 0.7421
  - ema_2: 0.7293
- 被选中次数：
  - online: 4267
  - ema_1: 4846
  - ema_2: 887
- 被选中时准确率：
  - online: 0.8158
  - ema_1: 0.7765
  - ema_2: 0.3202
- 总体 selected acc：0.7528
- pairwise agreement：
  - online vs ema_1: 0.9030
  - online vs ema_2: 0.8775
  - ema_1 vs ema_2: 0.9608
- pairwise mean JSD：
  - online vs ema_1: 0.01680
  - online vs ema_2: 0.02226
  - ema_1 vs ema_2: 0.00571
- 专家特征相似度矩阵 off-diagonal：
  - cosine mean: 0.9998856
  - cosine min/max: 0.9996928 / 0.9999998
  - CKA mean: 0.9992788
  - CKA min/max: 0.9980304 / 0.9999993

解读：
- 多头之间高度一致（尤其 ema_1 与 ema_2）。
- ema_2 在该 run 中被选中较少且被选中时准确率偏低，提示路由到 ema_2 的样本更困难或该头在该配置下偏弱。

## 5. 与 headviz 历史高结果对比
对照实验：
- note: paper_align_sup21k_seed1_headviz
- summary：
  - A_auc = 0.8196096666015553
  - A_avg = 0.8204214077958089
  - A_last = 0.8431
  - BWT_last = 0.0378160889249766

参数对比要点：
- 共同点：backbone、数据设定、任务设定、批大小、学习率、online_iter 基本一致。
- 主要差异：
  - v4 使用 optimizer=sgd
  - headviz 使用 optimizer=adam
  - v4 开启 analysis_expert_similarity（只影响后处理时间，不直接改变训练梯度）

初步结论：
- v4 与 headviz 的性能差异主要由优化器差异引起的可能性很高。

## 6. 后续动作
- 在保持其余超参数一致前提下，将 optimizer 切换为 adam 复跑一次（建议 note: paper_align_sup21k_seed1_head_deepdive_v5_adam）。
- 若 A_last 回升接近 0.84，则可基本确认性能差异主因来自优化器口径不一致。
