# FlyPrompt JSON 实验结果归档索引

本目录归档 FlyPrompt 迁移到 LibContinual 后产生的实验 JSON、配置、脚本、聚合表、图和报告快照。归档创建于 2026-07-15，用于支撑中文复现报告中的主要 LibContinual 数值。

## 归档范围

| 分组 | 含义 | 目录 |
| --- | --- | --- |
| 本地参考 JSON | 从本工作区各 result root 中复制出的、有效且不重复的 FlyPrompt 结果 JSON，包括主实验和 planned/report 实验。 | `local_reference_json/all_local_json/` |

主实验 CIFAR-100 / Sup-21K / balanced v6 AMP 的五种子结果已经归档为：

```text
local_reference_json/all_local_json/results_balanced_v6_amp__flyprompt_gcl_seed_1.json
local_reference_json/all_local_json/results_balanced_v6_amp__flyprompt_gcl_seed_2.json
local_reference_json/all_local_json/results_balanced_v6_amp__flyprompt_gcl_seed_3.json
local_reference_json/all_local_json/results_balanced_v6_amp__flyprompt_gcl_seed_4.json
local_reference_json/all_local_json/results_balanced_v6_amp__flyprompt_gcl_seed_5.json
```

## 目录结构

| 目录或文件 | 内容 |
| --- | --- |
| `local_reference_json/all_local_json/` | 归档 JSON，共 128 个。 |
| `configs/` | 与归档结果对应的配置文件。 |
| `aggregated_tables/` | 由 JSON 聚合出的 Markdown/CSV 表格。 |
| `figures/` | 报告使用的基础图。 |
| `scripts/` | 聚合、配置生成和成本统计脚本副本。 |
| `TABLE_TO_JSON_MAP.md` | 报告表格行到 JSON 证据的映射。 |
| `JSON_SHAPE.md` | JSON 字段说明。 |
| `REPRODUCTION_COMMANDS.md` | 复现命令说明。 |
| `SHA256SUMS` | 文件校验信息。 |

## 说明

提交包不包含原始 `results*` 目录；所有可追溯实验数值以本目录中的归档 JSON 为准。
