# FlyGCL 复现进度记录（2026-04-14）

## 本次目标
- 修复 FlyPrompt 的首个阻塞错误。
- 完成 1-batch profile 验证（能启动并前向/反向）。
- 完成 CIFAR-100 的 1-seed smoke。
- 尝试扩展到 5-seeds。

## 已执行工作

### 1) 修复语法阻塞（已完成）
- 文件：`methods/flyprompt.py`
- 问题：函数 docstring 缩进层级错误，触发 `IndentationError`。
- 处理：统一修复 4 处函数体内 docstring 缩进。
- 结果：该文件语法检查通过。

### 2) 处理权重加载阻塞（临时兜底，已完成）
- 文件：`models/flyprompt.py`
- 问题：backbone 预训练加载时报 `Expected hasRecord("version")`（缓存/权重格式不兼容）。
- 处理：为 smoke 目的临时改为 `pretrained=False`，先保证训练链路可跑。
- 说明：这是临时措施，不是最终论文复现实验配置。

### 3) profile 验证（已完成）
- 命令（简化）：
  - `python main.py --method flyprompt --dataset cifar100 --backbone vit_tiny_patch16_224 --profile ...`
- 结果：训练日志正常出现 `Train | Sample # ...`，说明入口链路打通。

### 4) 1-seed smoke（已完成）
- 命令（实际）：
  - `python main.py --method flyprompt --dataset cifar100 --data_dir /root/autodl-tmp/data --backbone vit_tiny_patch16_224 --n_tasks 2 --n 100 --m 0 --batchsize 8 --lr 0.005 --num_epochs 1 --online_iter 1 --eval_period 100000 --n_worker 0 --rp_dim 512 --seeds 1 --note smoke_flyprompt_cifar100_tiny_rp512`
- 输出摘要（日志末尾）：
  - `A_auc nan | A_avg 0.0662546 | A_last 0.0609 | F_last 0.0176471`
- 结果目录：
  - `results/logs/cifar100/smoke_flyprompt_cifar100_tiny_rp512`

### 5) 5-seeds 扩展（第一次尝试中断）
- 命令：
  - `... --seeds 1 2 3 4 5 --note smoke_flyprompt_cifar100_tiny_rp512_s1to5`
- 结果：进程被 `KeyboardInterrupt` 中断。
- 证据：日志末尾含 `KeyboardInterrupt`，且目录为空。
- 目录：
  - `results/logs/cifar100/smoke_flyprompt_cifar100_tiny_rp512_s1to5`

### 6) 5-seeds 扩展（第二次成功）
- 命令（提升吞吐）：
  - `python main.py --method flyprompt --dataset cifar100 --data_dir /root/autodl-tmp/data --backbone vit_tiny_patch16_224 --n_tasks 2 --n 100 --m 0 --batchsize 64 --lr 0.005 --num_epochs 1 --online_iter 1 --eval_period 100000 --n_worker 8 --rp_dim 512 --seeds 1 2 3 4 5 --note smoke_flyprompt_cifar100_tiny_rp512_s1to5_bs64`
- 结果目录：
  - `results/logs/cifar100/smoke_flyprompt_cifar100_tiny_rp512_s1to5_bs64`
- 该目录已生成 `seed_1.npy` 到 `seed_5.npy` 及对应 eval 文件。

## 5-seed 汇总（来自 `seed_*.npy`）
- seed=1: A_avg=0.071429, A_last=0.067800, F_last=0.007257
- seed=2: A_avg=0.082909, A_last=0.064000, F_last=0.037818
- seed=3: A_avg=0.121259, A_last=0.067000, F_last=0.108517
- seed=4: A_avg=0.095834, A_last=0.070100, F_last=0.051469
- seed=5: A_avg=0.069629, A_last=0.066400, F_last=0.006457
- Mean: A_avg=0.088212, A_last=0.067060, F_last=0.042304

## 关键解释

### 为什么 A_auc 是 nan
- 当前 smoke 配置用了 `eval_period=100000`。
- 在 2-task、50k 样本这个短流程里，在线评估几乎不会触发，`eval_results['test_acc']` 为空。
- 代码里对空数组求均值会得到 `nan`（属于配置现象，不是训练崩溃）。

### 为什么“可跑通”但“不是论文最终配置”
- 可跑通：指代码链路完整可执行，能训练、切任务、落盘结果。
- 不是最终配置：当前为了 smoke 做了两类简化：
  - backbone 用了 `vit_tiny_patch16_224`（而论文主结果多是 ViT-B/16）。
  - 临时关闭 pretrained 加载（`pretrained=False`），不符合论文主实验设定。

## 正式复现前必须补齐
1. 恢复 `pretrained=True` 流程并修好权重加载格式兼容。
2. 准备主实验权重文件与目录布局（含多种 backbone）。
3. 按论文配置回到主参数：backbone、rp_dim、n_tasks、seeds、数据集组合。
4. 合理设置 `eval_period`（例如 1000）以得到有效 A_auc。

## 备注
- 本文档记录的是“链路验证阶段”状态，便于多窗口协作。
