# 表格到 JSON 证据的映射

本文件把复现报告中的表格行映射到 `local_reference_json/all_local_json/` 下的归档 JSON 证据。

标记为 `NO_ARCHIVED_JSON_MAPPING` 的行表示：该行出现在报告表格中，但当前 LibContinual 归档里没有对应的本地配置或结果 JSON。标记为 `ARCHIVED_JSON` 的行表示：该行有本地归档 JSON 证据。

## A.1 Overall Performance

| Row | Result Group | JSON Count | JSON Files | Evidence Status |
| --- | --- | ---: | --- | --- |
| Sup-21K / CIFAR-100 | results_balanced_v6_amp | 5 | results_balanced_v6_amp__flyprompt_gcl_seed_1.json<br>results_balanced_v6_amp__flyprompt_gcl_seed_2.json<br>results_balanced_v6_amp__flyprompt_gcl_seed_3.json<br>results_balanced_v6_amp__flyprompt_gcl_seed_4.json<br>results_balanced_v6_amp__flyprompt_gcl_seed_5.json | ARCHIVED_JSON; Paper FlyPrompt |
| Sup-21K / ImageNet-R | results_plan_imagenetr_sup21k_default_amp | 5 | results_plan_imagenetr_sup21k_default_amp__flyprompt_gcl_seed_1.json<br>results_plan_imagenetr_sup21k_default_amp__flyprompt_gcl_seed_2.json<br>results_plan_imagenetr_sup21k_default_amp__flyprompt_gcl_seed_3.json<br>results_plan_imagenetr_sup21k_default_amp__flyprompt_gcl_seed_4.json<br>results_plan_imagenetr_sup21k_default_amp__flyprompt_gcl_seed_5.json | ARCHIVED_JSON; Paper FlyPrompt |
| Sup-21K / CUB-200 | results_plan_cub200_sup21k_default_amp | 5 | results_plan_cub200_sup21k_default_amp__flyprompt_gcl_seed_1.json<br>results_plan_cub200_sup21k_default_amp__flyprompt_gcl_seed_2.json<br>results_plan_cub200_sup21k_default_amp__flyprompt_gcl_seed_3.json<br>results_plan_cub200_sup21k_default_amp__flyprompt_gcl_seed_4.json<br>results_plan_cub200_sup21k_default_amp__flyprompt_gcl_seed_5.json | ARCHIVED_JSON; Paper FlyPrompt |
| Sup-21K/1K / all datasets |  | 0 |  | NO_ARCHIVED_JSON_MAPPING; Paper table row exists; no local archived JSON/config mapping |
| iBOT-21K / all datasets |  | 0 |  | NO_ARCHIVED_JSON_MAPPING; Paper table row exists; no local archived JSON/config mapping |
| iBOT-1K / all datasets |  | 0 |  | NO_ARCHIVED_JSON_MAPPING; Paper table row exists; no local archived JSON/config mapping |
| DINO-1K / all datasets |  | 0 |  | NO_ARCHIVED_JSON_MAPPING; Paper table row exists; no local archived JSON/config mapping |
| MoCo v3-1K / all datasets |  | 0 |  | NO_ARCHIVED_JSON_MAPPING; Paper table row exists; no local archived JSON/config mapping |

## A.2 Average Accuracy and Forgetting

| Row | Result Group | JSON Count | JSON Files | Evidence Status |
| --- | --- | ---: | --- | --- |
| Sup-21K / CIFAR-100 | results_balanced_v6_amp | 5 | results_balanced_v6_amp__flyprompt_gcl_seed_1.json<br>results_balanced_v6_amp__flyprompt_gcl_seed_2.json<br>results_balanced_v6_amp__flyprompt_gcl_seed_3.json<br>results_balanced_v6_amp__flyprompt_gcl_seed_4.json<br>results_balanced_v6_amp__flyprompt_gcl_seed_5.json | ARCHIVED_JSON; Paper FlyPrompt |
| Sup-21K / ImageNet-R | results_plan_imagenetr_sup21k_default_amp | 5 | results_plan_imagenetr_sup21k_default_amp__flyprompt_gcl_seed_1.json<br>results_plan_imagenetr_sup21k_default_amp__flyprompt_gcl_seed_2.json<br>results_plan_imagenetr_sup21k_default_amp__flyprompt_gcl_seed_3.json<br>results_plan_imagenetr_sup21k_default_amp__flyprompt_gcl_seed_4.json<br>results_plan_imagenetr_sup21k_default_amp__flyprompt_gcl_seed_5.json | ARCHIVED_JSON; Paper FlyPrompt |
| Sup-21K / CUB-200 | results_plan_cub200_sup21k_default_amp | 5 | results_plan_cub200_sup21k_default_amp__flyprompt_gcl_seed_1.json<br>results_plan_cub200_sup21k_default_amp__flyprompt_gcl_seed_2.json<br>results_plan_cub200_sup21k_default_amp__flyprompt_gcl_seed_3.json<br>results_plan_cub200_sup21k_default_amp__flyprompt_gcl_seed_4.json<br>results_plan_cub200_sup21k_default_amp__flyprompt_gcl_seed_5.json | ARCHIVED_JSON; Paper FlyPrompt |
| Sup-21K/1K rows |  | 0 |  | NO_ARCHIVED_JSON_MAPPING; Paper table rows exist; no local archived JSON/config mapping |

## A.3 Component Ablation

| Row | Result Group | JSON Count | JSON Files | Evidence Status |
| --- | --- | ---: | --- | --- |
| RP analytic classifier | results_plan_cifar100_ranpac_dagger_amp | 5 | results_plan_cifar100_ranpac_dagger_amp__flyprompt_gcl_seed_1.json<br>results_plan_cifar100_ranpac_dagger_amp__flyprompt_gcl_seed_2.json<br>results_plan_cifar100_ranpac_dagger_amp__flyprompt_gcl_seed_3.json<br>results_plan_cifar100_ranpac_dagger_amp__flyprompt_gcl_seed_4.json<br>results_plan_cifar100_ranpac_dagger_amp__flyprompt_gcl_seed_5.json | ARCHIVED_JSON; Paper RP analytic classifier |
| No prompt, no EMA | results_plan_cifar100_no_prompt_no_ema_amp | 5 | results_plan_cifar100_no_prompt_no_ema_amp__flyprompt_gcl_seed_1.json<br>results_plan_cifar100_no_prompt_no_ema_amp__flyprompt_gcl_seed_2.json<br>results_plan_cifar100_no_prompt_no_ema_amp__flyprompt_gcl_seed_3.json<br>results_plan_cifar100_no_prompt_no_ema_amp__flyprompt_gcl_seed_4.json<br>results_plan_cifar100_no_prompt_no_ema_amp__flyprompt_gcl_seed_5.json | ARCHIVED_JSON; Paper no/no |
| No prompt, EMA | results_plan_cifar100_no_prompt_ema_amp | 5 | results_plan_cifar100_no_prompt_ema_amp__flyprompt_gcl_seed_1.json<br>results_plan_cifar100_no_prompt_ema_amp__flyprompt_gcl_seed_2.json<br>results_plan_cifar100_no_prompt_ema_amp__flyprompt_gcl_seed_3.json<br>results_plan_cifar100_no_prompt_ema_amp__flyprompt_gcl_seed_4.json<br>results_plan_cifar100_no_prompt_ema_amp__flyprompt_gcl_seed_5.json | ARCHIVED_JSON; Paper no/yes |
| No REAR, no EMA | results_plan_cifar100_no_rear_no_ema_amp | 5 | results_plan_cifar100_no_rear_no_ema_amp__flyprompt_gcl_seed_1.json<br>results_plan_cifar100_no_rear_no_ema_amp__flyprompt_gcl_seed_2.json<br>results_plan_cifar100_no_rear_no_ema_amp__flyprompt_gcl_seed_3.json<br>results_plan_cifar100_no_rear_no_ema_amp__flyprompt_gcl_seed_4.json<br>results_plan_cifar100_no_rear_no_ema_amp__flyprompt_gcl_seed_5.json | ARCHIVED_JSON; Paper no REAR/no EMA |
| No REAR, EMA | results_plan_cifar100_no_rear_ema_amp | 5 | results_plan_cifar100_no_rear_ema_amp__flyprompt_gcl_seed_1.json<br>results_plan_cifar100_no_rear_ema_amp__flyprompt_gcl_seed_2.json<br>results_plan_cifar100_no_rear_ema_amp__flyprompt_gcl_seed_3.json<br>results_plan_cifar100_no_rear_ema_amp__flyprompt_gcl_seed_4.json<br>results_plan_cifar100_no_rear_ema_amp__flyprompt_gcl_seed_5.json | ARCHIVED_JSON; Paper no REAR/EMA |
| REAR + prompt, no EMA | results_plan_cifar100_no_ema_amp | 5 | results_plan_cifar100_no_ema_amp__flyprompt_gcl_seed_1.json<br>results_plan_cifar100_no_ema_amp__flyprompt_gcl_seed_2.json<br>results_plan_cifar100_no_ema_amp__flyprompt_gcl_seed_3.json<br>results_plan_cifar100_no_ema_amp__flyprompt_gcl_seed_4.json<br>results_plan_cifar100_no_ema_amp__flyprompt_gcl_seed_5.json | ARCHIVED_JSON; Paper no EMA |
| Full FlyPrompt | results_balanced_v6_amp | 5 | results_balanced_v6_amp__flyprompt_gcl_seed_1.json<br>results_balanced_v6_amp__flyprompt_gcl_seed_2.json<br>results_balanced_v6_amp__flyprompt_gcl_seed_3.json<br>results_balanced_v6_amp__flyprompt_gcl_seed_4.json<br>results_balanced_v6_amp__flyprompt_gcl_seed_5.json | ARCHIVED_JSON; Paper full |

## A.4 Ensemble Aggregation

| Row | Result Group | JSON Count | JSON Files | Evidence Status |
| --- | --- | ---: | --- | --- |
| Mean | results_plan_cifar100_ensemble_mean_amp | 5 | results_plan_cifar100_ensemble_mean_amp__flyprompt_gcl_seed_1.json<br>results_plan_cifar100_ensemble_mean_amp__flyprompt_gcl_seed_2.json<br>results_plan_cifar100_ensemble_mean_amp__flyprompt_gcl_seed_3.json<br>results_plan_cifar100_ensemble_mean_amp__flyprompt_gcl_seed_4.json<br>results_plan_cifar100_ensemble_mean_amp__flyprompt_gcl_seed_5.json | ARCHIVED_JSON; Paper Mean |
| Max Prob. | results_plan_cifar100_ensemble_max_prob_amp | 5 | results_plan_cifar100_ensemble_max_prob_amp__flyprompt_gcl_seed_1.json<br>results_plan_cifar100_ensemble_max_prob_amp__flyprompt_gcl_seed_2.json<br>results_plan_cifar100_ensemble_max_prob_amp__flyprompt_gcl_seed_3.json<br>results_plan_cifar100_ensemble_max_prob_amp__flyprompt_gcl_seed_4.json<br>results_plan_cifar100_ensemble_max_prob_amp__flyprompt_gcl_seed_5.json | ARCHIVED_JSON; Paper Max Prob. |
| Min Entropy |  | 0 |  | NO_ARCHIVED_JSON_MAPPING; Report value exists; no implemented local ensemble method / JSON mapping |
| SoftMax+Mean | results_plan_cifar100_ensemble_softmax_mean_amp | 5 | results_plan_cifar100_ensemble_softmax_mean_amp__flyprompt_gcl_seed_1.json<br>results_plan_cifar100_ensemble_softmax_mean_amp__flyprompt_gcl_seed_2.json<br>results_plan_cifar100_ensemble_softmax_mean_amp__flyprompt_gcl_seed_3.json<br>results_plan_cifar100_ensemble_softmax_mean_amp__flyprompt_gcl_seed_4.json<br>results_plan_cifar100_ensemble_softmax_mean_amp__flyprompt_gcl_seed_5.json | ARCHIVED_JSON; Paper SoftMax+Mean |
| SoftMax+Max Prob. | results_balanced_v6_amp | 5 | results_balanced_v6_amp__flyprompt_gcl_seed_1.json<br>results_balanced_v6_amp__flyprompt_gcl_seed_2.json<br>results_balanced_v6_amp__flyprompt_gcl_seed_3.json<br>results_balanced_v6_amp__flyprompt_gcl_seed_4.json<br>results_balanced_v6_amp__flyprompt_gcl_seed_5.json | ARCHIVED_JSON; Paper SoftMax+Max Prob. |
| SoftMax+Min Entropy |  | 0 |  | NO_ARCHIVED_JSON_MAPPING; Report value exists; no implemented local ensemble method / JSON mapping |

## A.5 EMA Decay

| Row | Result Group | JSON Count | JSON Files | Evidence Status |
| --- | --- | ---: | --- | --- |
| Online head only | results_plan_cifar100_ema_online_only_amp | 5 | results_plan_cifar100_ema_online_only_amp__flyprompt_gcl_seed_1.json<br>results_plan_cifar100_ema_online_only_amp__flyprompt_gcl_seed_2.json<br>results_plan_cifar100_ema_online_only_amp__flyprompt_gcl_seed_3.json<br>results_plan_cifar100_ema_online_only_amp__flyprompt_gcl_seed_4.json<br>results_plan_cifar100_ema_online_only_amp__flyprompt_gcl_seed_5.json | ARCHIVED_JSON; Paper online only |
| +0.9 | results_plan_cifar100_ema_09_amp | 5 | results_plan_cifar100_ema_09_amp__flyprompt_gcl_seed_1.json<br>results_plan_cifar100_ema_09_amp__flyprompt_gcl_seed_2.json<br>results_plan_cifar100_ema_09_amp__flyprompt_gcl_seed_3.json<br>results_plan_cifar100_ema_09_amp__flyprompt_gcl_seed_4.json<br>results_plan_cifar100_ema_09_amp__flyprompt_gcl_seed_5.json | ARCHIVED_JSON; Paper +0.9 |
| +0.99 | results_plan_cifar100_ema_099_amp | 5 | results_plan_cifar100_ema_099_amp__flyprompt_gcl_seed_1.json<br>results_plan_cifar100_ema_099_amp__flyprompt_gcl_seed_2.json<br>results_plan_cifar100_ema_099_amp__flyprompt_gcl_seed_3.json<br>results_plan_cifar100_ema_099_amp__flyprompt_gcl_seed_4.json<br>results_plan_cifar100_ema_099_amp__flyprompt_gcl_seed_5.json | ARCHIVED_JSON; Paper +0.99 |
| +0.999 | results_plan_cifar100_ema_0999_amp | 5 | results_plan_cifar100_ema_0999_amp__flyprompt_gcl_seed_1.json<br>results_plan_cifar100_ema_0999_amp__flyprompt_gcl_seed_2.json<br>results_plan_cifar100_ema_0999_amp__flyprompt_gcl_seed_3.json<br>results_plan_cifar100_ema_0999_amp__flyprompt_gcl_seed_4.json<br>results_plan_cifar100_ema_0999_amp__flyprompt_gcl_seed_5.json | ARCHIVED_JSON; Paper +0.999 |
| +0.9,0.99 | results_balanced_v6_amp | 5 | results_balanced_v6_amp__flyprompt_gcl_seed_1.json<br>results_balanced_v6_amp__flyprompt_gcl_seed_2.json<br>results_balanced_v6_amp__flyprompt_gcl_seed_3.json<br>results_balanced_v6_amp__flyprompt_gcl_seed_4.json<br>results_balanced_v6_amp__flyprompt_gcl_seed_5.json | ARCHIVED_JSON; Paper +0.9,0.99 |
| +0.9,0.99,0.999 | results_plan_cifar100_ema_09_099_0999_amp | 5 | results_plan_cifar100_ema_09_099_0999_amp__flyprompt_gcl_seed_1.json<br>results_plan_cifar100_ema_09_099_0999_amp__flyprompt_gcl_seed_2.json<br>results_plan_cifar100_ema_09_099_0999_amp__flyprompt_gcl_seed_3.json<br>results_plan_cifar100_ema_09_099_0999_amp__flyprompt_gcl_seed_4.json<br>results_plan_cifar100_ema_09_099_0999_amp__flyprompt_gcl_seed_5.json | ARCHIVED_JSON; Paper +0.9,0.99,0.999 |

## A.6 Si-Blurry Variants

| Row | Result Group | JSON Count | JSON Files | Evidence Status |
| --- | --- | ---: | --- | --- |
| r_D=0, r_B=10 | results_plan_cifar100_rd0_rb10_amp | 5 | results_plan_cifar100_rd0_rb10_amp__flyprompt_gcl_seed_1.json<br>results_plan_cifar100_rd0_rb10_amp__flyprompt_gcl_seed_2.json<br>results_plan_cifar100_rd0_rb10_amp__flyprompt_gcl_seed_3.json<br>results_plan_cifar100_rd0_rb10_amp__flyprompt_gcl_seed_4.json<br>results_plan_cifar100_rd0_rb10_amp__flyprompt_gcl_seed_5.json | ARCHIVED_JSON; Paper r_D=0 |
| r_D=50, r_B=10 | results_balanced_v6_amp | 5 | results_balanced_v6_amp__flyprompt_gcl_seed_1.json<br>results_balanced_v6_amp__flyprompt_gcl_seed_2.json<br>results_balanced_v6_amp__flyprompt_gcl_seed_3.json<br>results_balanced_v6_amp__flyprompt_gcl_seed_4.json<br>results_balanced_v6_amp__flyprompt_gcl_seed_5.json | ARCHIVED_JSON; Paper r_D=50 |
| r_D=100, r_B=10 | results_plan_cifar100_rd100_rb10_amp | 5 | results_plan_cifar100_rd100_rb10_amp__flyprompt_gcl_seed_1.json<br>results_plan_cifar100_rd100_rb10_amp__flyprompt_gcl_seed_2.json<br>results_plan_cifar100_rd100_rb10_amp__flyprompt_gcl_seed_3.json<br>results_plan_cifar100_rd100_rb10_amp__flyprompt_gcl_seed_4.json<br>results_plan_cifar100_rd100_rb10_amp__flyprompt_gcl_seed_5.json | ARCHIVED_JSON; Paper r_D=100 |
| r_D=50, r_B=30 | results_plan_cifar100_rd50_rb30_amp | 5 | results_plan_cifar100_rd50_rb30_amp__flyprompt_gcl_seed_1.json<br>results_plan_cifar100_rd50_rb30_amp__flyprompt_gcl_seed_2.json<br>results_plan_cifar100_rd50_rb30_amp__flyprompt_gcl_seed_3.json<br>results_plan_cifar100_rd50_rb30_amp__flyprompt_gcl_seed_4.json<br>results_plan_cifar100_rd50_rb30_amp__flyprompt_gcl_seed_5.json | ARCHIVED_JSON; Paper r_B=30 |
| r_D=50, r_B=50 | results_plan_cifar100_rd50_rb50_amp | 5 | results_plan_cifar100_rd50_rb50_amp__flyprompt_gcl_seed_1.json<br>results_plan_cifar100_rd50_rb50_amp__flyprompt_gcl_seed_2.json<br>results_plan_cifar100_rd50_rb50_amp__flyprompt_gcl_seed_3.json<br>results_plan_cifar100_rd50_rb50_amp__flyprompt_gcl_seed_4.json<br>results_plan_cifar100_rd50_rb50_amp__flyprompt_gcl_seed_5.json | ARCHIVED_JSON; Paper r_B=50 |

## A.7 Mask Types

| Row | Result Group | JSON Count | JSON Files | Evidence Status |
| --- | --- | ---: | --- | --- |
| No Mask |  | 0 |  | NO_ARCHIVED_JSON_MAPPING; Report value exists; no local mask_type=no_mask config / JSON mapping |
| Random Mask |  | 0 |  | NO_ARCHIVED_JSON_MAPPING; Report value exists; no local mask_type=random config / JSON mapping |
| Seen-Class Mask | results_plan_cifar100_mask_seen_class_amp | 5 | results_plan_cifar100_mask_seen_class_amp__flyprompt_gcl_seed_1.json<br>results_plan_cifar100_mask_seen_class_amp__flyprompt_gcl_seed_2.json<br>results_plan_cifar100_mask_seen_class_amp__flyprompt_gcl_seed_3.json<br>results_plan_cifar100_mask_seen_class_amp__flyprompt_gcl_seed_4.json<br>results_plan_cifar100_mask_seen_class_amp__flyprompt_gcl_seed_5.json | ARCHIVED_JSON; Paper Seen-Class Mask |
| Batch Seen-Class Mask | results_balanced_v6_amp | 5 | results_balanced_v6_amp__flyprompt_gcl_seed_1.json<br>results_balanced_v6_amp__flyprompt_gcl_seed_2.json<br>results_balanced_v6_amp__flyprompt_gcl_seed_3.json<br>results_balanced_v6_amp__flyprompt_gcl_seed_4.json<br>results_balanced_v6_amp__flyprompt_gcl_seed_5.json | ARCHIVED_JSON; Paper Batch Seen-Class Mask |

## A.8 Routing Algorithms

| Row | Result Group | JSON Count | JSON Files | Evidence Status |
| --- | --- | ---: | --- | --- |
| Prototype Similarity |  | 0 |  | NO_ARCHIVED_JSON_MAPPING; Report value exists; no local router implementation / JSON mapping |
| Naive Bayes |  | 0 |  | NO_ARCHIVED_JSON_MAPPING; Report value exists; no local router implementation / JSON mapping |
| MLP |  | 0 |  | NO_ARCHIVED_JSON_MAPPING; Report value exists; no local router implementation / JSON mapping |
| K-Means |  | 0 |  | NO_ARCHIVED_JSON_MAPPING; Report value exists; no local router implementation / JSON mapping |
| Ridge Regression | results_balanced_v6_amp | 5 | results_balanced_v6_amp__flyprompt_gcl_seed_1.json<br>results_balanced_v6_amp__flyprompt_gcl_seed_2.json<br>results_balanced_v6_amp__flyprompt_gcl_seed_3.json<br>results_balanced_v6_amp__flyprompt_gcl_seed_4.json<br>results_balanced_v6_amp__flyprompt_gcl_seed_5.json | ARCHIVED_JSON; Paper Ridge Regression |

## A.9 Cost Tables

| Row | Result Group | JSON Count | JSON Files | Evidence Status |
| --- | --- | ---: | --- | --- |
| Method cost / default FlyPrompt | results_balanced_v6_amp | 5 | results_balanced_v6_amp__flyprompt_gcl_seed_1.json<br>results_balanced_v6_amp__flyprompt_gcl_seed_2.json<br>results_balanced_v6_amp__flyprompt_gcl_seed_3.json<br>results_balanced_v6_amp__flyprompt_gcl_seed_4.json<br>results_balanced_v6_amp__flyprompt_gcl_seed_5.json | ARCHIVED_JSON; Metric JSON exists; timing/storage raw artifact is separate |
| Component cost |  | 0 |  | NO_ARCHIVED_JSON_MAPPING; No seed JSON expected; requires parameter/storage measurement artifact |
