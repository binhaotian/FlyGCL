# FlyPrompt 复现目录说明

本目录保存 FlyPrompt 在 LibContinual 中的迁移实现说明、复现报告、运行脚本和证据归档。最终提交时，本目录是 `LibContinual/` 中最重要的复现材料目录。

## 优先阅读

完整中文主报告：

```text
FULL_REPRODUCTION_REPORT.md
```

该报告已经覆盖：

- 迁移背景与目标
- FlyPrompt 方法理解
- LibContinual 迁移设计
- 实验协议
- 复现过程中的困难与异常处理
- 主结果与消融分析
- Threats to Validity
- 完整结果表
- JSON 证据文件与交付清单

## 证据归档

主要证据目录：

```text
json_experiment_results/
```

关键文件：

- `json_experiment_results/local_reference_json/all_local_json/`：归档 JSON，共 128 个。
- `json_experiment_results/TABLE_TO_JSON_MAP.md`：报告表格行到 JSON 的映射。
- `json_experiment_results/aggregated_tables/`：从 JSON 聚合出的表格。
- `json_experiment_results/figures/`：报告配套图。
- `json_experiment_results/MANIFEST.md`：归档文件索引。
- `json_experiment_results/JSON_SHAPE.md`：JSON 字段说明。
- `json_experiment_results/REPRODUCTION_COMMANDS.md`：复现命令说明。
- `json_experiment_results/SHA256SUMS`：归档文件校验信息。

## 代码与脚本

- `run_cifar100_sup21k.sh`：CIFAR-100 / Sup-21K 主实验多 seed 启动脚本。
- `run_planned_experiments.sh`：补充实验启动脚本。
- `collect_gcl_results.py`：聚合 `flyprompt_gcl_seed_*.json` 的工具。
- `export_gcl_analysis.py`：导出曲线、表格和分析材料的工具。
- `generate_planned_configs.py`：生成 planned 实验配置的工具。
- `measure_flyprompt_cost.py`：成本统计辅助脚本。
- `smoke_test.py`：轻量链路检查脚本。

## 其他说明文件

- `REPORT.md`：较早版本的短报告。
- `PAPER_TABLE_REPRO_FRAMEWORK.md`：论文表格复现框架。
- `RIGOROUS_REPRODUCTION_TASKS.md`：严谨复现任务清单。
- `DETAILED_REPRODUCTION_REPORT_DRAFT.md`：早期详细报告草稿。
- `PLANNED_EXPERIMENTS_TABLES_AND_SCRIPTS.md`：补充实验计划说明。
- `analysis/`：seed-1 曲线、AMP ablation、transform ablation、随机性记录等辅助分析。

## 提交包口径

本提交包保留报告、配置、代码和归档 JSON 证据；不保留原始 `results*` 运行目录、训练日志、checkpoint 或数据集。报告中的数值主张应优先以 `TABLE_TO_JSON_MAP.md` 中标记为 `ARCHIVED_JSON` 的行作为本地证据来源。
