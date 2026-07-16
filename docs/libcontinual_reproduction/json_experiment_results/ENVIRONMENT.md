# 复现实验环境

本文件记录 FlyPrompt 迁移复现实验使用的 LibContinual 工作环境。关键点是：实验使用 `/root/autodl-tmp/envs/flygcl` Python 环境，而不是系统默认 Python。

| 项目 | 数值 |
| --- | --- |
| 记录日期 | 2026-07-16 |
| 工作目录 | `/root/autodl-tmp/LibContinual` |
| Git commit | `a13f7bf6728e76ac1f3f97cf7e42f72a856189ad` |
| Python 可执行文件 | `/root/autodl-tmp/envs/flygcl/bin/python` |
| Python 版本 | `3.10.20 (main, Jun 11 2026, 15:17:37) [GCC 14.3.0]` |
| torch | `2.5.1+cu124` |
| timm | `0.6.12` |
| CUDA available | `True` |
| CUDA version | `12.4` |
| cuDNN | `90100` |
| GPU 数量 | `1` |
| GPU 0 | `NVIDIA GeForce RTX 4080 SUPER` |
| GPU 0 显存 | `31.48 GiB` |
| 主配置 | `config/flyprompt_cifar100_sup21k_balanced_v6_amp.yaml` |
| 原始主结果目录 | `results_balanced_v6_amp/log/FlyPrompt/` |
| 提交包归档 JSON 目录 | `reproduce/flyprompt/json_experiment_results/local_reference_json/all_local_json/` |
| Sup-21K checkpoint | `/root/autodl-tmp/FlyGCL/checkpoints/ViT-B_16.npz` |

## 说明

提交包不包含原始 `results*` 目录、checkpoint 或数据集。报告数值的本地证据以 `local_reference_json/all_local_json/` 中归档的 JSON 为准。
