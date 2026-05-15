# FlyGCL 实验记录 - 2026-04-11

## 环境信息

- 运行平台：AutoDL
- GPU：NVIDIA GeForce RTX 4080 SUPER，32GB
- CPU：16 核
- 内存：80GB
- 系统盘：30GB
- 数据盘：/root/autodl-tmp，50GB

## 仓库与环境

- 仓库已拉取到：/root/FlyGCL
- Python 环境：/root/miniconda3/envs/flygcl
- Python 版本：3.10.20
- 依赖已安装：requirements.txt

## 数据与缓存

- CIFAR-100 缓存已存在于 /root/autodl-tmp/data/cifar-100-python
- CIFAR-100 训练数据可直接复用，不需要再从公网下载
- ImageNet-1K 体量较大，通常远超 50GB，不适合放在系统盘或 50GB 数据盘中
- ViT 预训练缓存路径：/root/.cache/torch/hub/checkpoints

## 本次 smoke run

- 任务：先跑 CIFAR
- 数据集：cifar100
- 方法：l2p
- backbone：vit_tiny_patch16_224
- 数据目录：/root/autodl-tmp/data
- 运行命令：

```bash
cd /root/FlyGCL
source /etc/network_turbo
/root/miniconda3/envs/flygcl/bin/python main.py \
  --method l2p \
  --dataset cifar100 \
  --data_dir /root/autodl-tmp/data \
  --backbone vit_tiny_patch16_224 \
  --n_tasks 2 --n 100 --m 0 \
  --batchsize 8 --lr 0.01 \
  --num_epochs 1 --online_iter 1 \
  --eval_period 100000 --n_worker 0 \
  --note smoke_run_cifar100
```

## 结果

- 训练已完整跑通 2 个 task
- 最终摘要：A_avg 0.08175，A_last 0.0573，F_last 0.0732
- 说明数据加载、建模、训练、任务切换和结果汇总链路均正常

## 过程中做过的临时处理

- 为解决预训练权重加载兼容问题，临时将 [models/l2p.py](models/l2p.py) 中的 ViT 初始化改为 `pretrained=False`
- 修改位置：[models/l2p.py](models/l2p.py#L133) 到 [models/l2p.py](models/l2p.py#L138)
- 这是为了让 smoke run 先跑通；如果后续需要恢复预训练加载，可以再单独调整

## 后续建议

- 如果继续做 CIFAR 实验，建议优先把数据集放在 /root/autodl-tmp 下
- 如果要跑 ImageNet-1K，建议不要放在 50GB 数据盘，容量大概率不够
- 如果要恢复官方预训练流程，需要再处理权重格式兼容问题