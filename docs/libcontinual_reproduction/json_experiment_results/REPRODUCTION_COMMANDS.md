# 复现命令说明

本文件记录用于复现和验证 FlyPrompt JSON 证据的主要命令。命令应从 LibContinual 仓库根目录执行：

```bash
cd /root/autodl-tmp/LibContinual
source /root/autodl-tmp/envs/flygcl/bin/activate
```

## 主实验五种子运行

```bash
CONFIG=flyprompt_cifar100_sup21k_balanced_v6_amp.yaml \
RESULTS_ROOT=./results_balanced_v6_amp/log/FlyPrompt \
bash reproduce/flyprompt/run_cifar100_sup21k.sh
```

运行完成后，原始主结果 JSON 通常位于：

```text
results_balanced_v6_amp/log/FlyPrompt/flyprompt_gcl_seed_1.json
results_balanced_v6_amp/log/FlyPrompt/flyprompt_gcl_seed_2.json
results_balanced_v6_amp/log/FlyPrompt/flyprompt_gcl_seed_3.json
results_balanced_v6_amp/log/FlyPrompt/flyprompt_gcl_seed_4.json
results_balanced_v6_amp/log/FlyPrompt/flyprompt_gcl_seed_5.json
```

提交包不保留上述原始 `results*` 目录，而是保留已经复制并改名后的归档 JSON：

```text
reproduce/flyprompt/json_experiment_results/local_reference_json/all_local_json/
```

## 聚合某个结果目录

```bash
python reproduce/flyprompt/collect_gcl_results.py \
  --results-root ./results_balanced_v6_amp/log/FlyPrompt
```

## 生成报告辅助材料

```bash
python reproduce/flyprompt/export_gcl_analysis.py
```

## 成本统计

```bash
python reproduce/flyprompt/measure_flyprompt_cost.py
```

## 注意事项

重新运行实验前需要自行准备数据集和 Sup-21K checkpoint。提交包只保留代码、配置、报告和归档 JSON 证据。
