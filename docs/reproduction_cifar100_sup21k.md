# FlyPrompt CIFAR-100 / Sup-21K 复现说明

本文件是 FlyGCL 辅助参考目录中的复现说明。最终中文主报告不在这里，而在：

```text
LibContinual/reproduce/flyprompt/FULL_REPRODUCTION_REPORT.md
```

## 论文与代码

- 论文：FlyPrompt: Insect-brain Inspired Prompt Learning for General Continual Learning
- arXiv：https://arxiv.org/abs/2602.01976
- 论文中的官方代码链接：https://github.com/AnAppleCore/FlyGCL

## 主实验协议

- 数据集：CIFAR-100
- 设置：Si-Blurry GCL
- disjoint class ratio：`r_D = 50%`，对应命令参数 `--n 50`
- blurry sample ratio：`r_B = 10%`，对应命令参数 `--m 10`
- session/task 数：`5`，对应 `--n_tasks 5`
- backbone：ViT-B/16，ImageNet-21K / Sup-21K 预训练，对应 `--backbone vit_base_patch16_224`
- 方法：FlyPrompt，对应 `--method flyprompt`
- prompt 长度：`5`
- prompt 插入层：ViT 前 5 层，即 `--pos_prompt 0 1 2 3 4`
- 随机投影维度：`10000`
- ridge 参数：`10000`
- EMA heads：`0.9, 0.99`
- 优化器：Adam
- 学习率：`0.005`
- batch size：`64`
- epoch：`1`
- online iteration：`3`
- 完整报告 seed：`1 2 3 4 5`

## 注意事项

- FlyGCL 代码中的 `eval_period` 按累计训练样本数触发，不是按 batch 数触发；使用 `--eval_period 1000` 是按代码口径复现。
- 直接运行 `main.py` 默认参数并不完全等于论文协议；建议显式传入协议参数，或使用 baseline 脚本。
- 如果只想快速看 `A_last`，可以使用 `--eval_period -1` 跳过周期评估；但这样不会得到有效的 `A_auc`。
- 本提交包不包含数据集和 checkpoint，运行前需要自行准备。

## 单 seed 参考命令

```bash
python main.py \
  --method flyprompt \
  --backbone vit_base_patch16_224 \
  --dataset cifar100 \
  --data_dir ./data/CIFAR \
  --n_tasks 5 --n 50 --m 10 --rnd_NM \
  --seeds 1 \
  --batchsize 64 \
  --num_epochs 1 \
  --online_iter 3 \
  --opt_name adam \
  --lr 0.005 \
  --eval_period 1000 \
  --note reproduce_cifar100_sup21k_seed1
```

## 快速 A_last pilot 命令

```bash
python main.py \
  --method flyprompt \
  --backbone vit_base_patch16_224 \
  --dataset cifar100 \
  --data_dir ./data/CIFAR \
  --n_tasks 5 --n 50 --m 10 --rnd_NM \
  --seeds 1 \
  --batchsize 64 \
  --num_epochs 1 \
  --online_iter 3 \
  --opt_name adam \
  --lr 0.005 \
  --eval_period -1 \
  --note pilot_cifar100_sup21k_seed1_alast
```
