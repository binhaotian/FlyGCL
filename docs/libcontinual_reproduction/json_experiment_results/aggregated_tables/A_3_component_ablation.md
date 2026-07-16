# A.3 Component Ablation

| row | result_group | n_json | evidence_status | paper_A_auc | json_A_auc_mean | json_A_auc_std | paper_A_last | json_A_last_mean | json_A_last_std | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RP analytic classifier | results_plan_cifar100_ranpac_dagger_amp | 5 | ARCHIVED_JSON | 69.91 | 68.44 | 0.72 | 79.92 | 77.83 | 0.51 | Paper RP analytic classifier |
| No prompt, no EMA | results_plan_cifar100_no_prompt_no_ema_amp | 5 | ARCHIVED_JSON | 71.33 | 69.82 | 0.66 | 73.22 | 71.15 | 0.54 | Paper no/no |
| No prompt, EMA | results_plan_cifar100_no_prompt_ema_amp | 5 | ARCHIVED_JSON | 71.69 | 70.21 | 0.63 | 73.3 | 71.44 | 0.50 | Paper no/yes |
| No REAR, no EMA | results_plan_cifar100_no_rear_no_ema_amp | 5 | ARCHIVED_JSON | 80.75 | 78.96 | 0.68 | 83.65 | 81.67 | 0.59 | Paper no REAR/no EMA |
| No REAR, EMA | results_plan_cifar100_no_rear_ema_amp | 5 | ARCHIVED_JSON | 82.17 | 80.28 | 0.64 | 83.75 | 82.20 | 0.52 | Paper no REAR/EMA |
| REAR + prompt, no EMA | results_plan_cifar100_no_ema_amp | 5 | ARCHIVED_JSON | 81.9 | 80.15 | 0.65 | 84.23 | 82.71 | 0.48 | Paper no EMA |
| Full FlyPrompt | results_balanced_v6_amp | 5 | ARCHIVED_JSON | 83.24 | 81.47 | 0.61 | 86.76 | 84.59 | 0.37 | Paper full |
