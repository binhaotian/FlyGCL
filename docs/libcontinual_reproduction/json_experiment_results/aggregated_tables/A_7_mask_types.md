# A.7 Mask Types

| row | result_group | n_json | evidence_status | paper_A_auc | json_A_auc_mean | json_A_auc_std | paper_A_last | json_A_last_mean | json_A_last_std | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| No Mask |  | 0 | NO_ARCHIVED_JSON_MAPPING | 78.73 |  |  | 83.62 |  |  | Report value exists; no local mask_type=no_mask config / JSON mapping |
| Random Mask |  | 0 | NO_ARCHIVED_JSON_MAPPING | 78.32 |  |  | 81.88 |  |  | Report value exists; no local mask_type=random config / JSON mapping |
| Seen-Class Mask | results_plan_cifar100_mask_seen_class_amp | 5 | ARCHIVED_JSON | 78.75 | 77.04 | 0.79 | 82.87 | 80.83 | 0.53 | Paper Seen-Class Mask |
| Batch Seen-Class Mask | results_balanced_v6_amp | 5 | ARCHIVED_JSON | 83.24 | 81.47 | 0.61 | 86.76 | 84.59 | 0.37 | Paper Batch Seen-Class Mask |
