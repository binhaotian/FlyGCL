# Tensor Transform 消融实验

| 运行设置 | 配置 | seed | A_auc | A_avg | A_last |
| --- | --- | ---: | ---: | ---: | ---: |
| 默认 PIL/torchvision transform | `config/flyprompt_cifar100_sup21k_balanced_v6_amp.yaml` | 1 | 0.819422 | 0.846649 | 0.846200 |
| FlyGCL 风格 tensor transform | `config/flyprompt_cifar100_sup21k_balanced_v6_tensor.yaml` | 1 | 0.807849 | 0.831604 | 0.836600 |

| 差值 | A_auc | A_avg | A_last |
| --- | ---: | ---: | ---: |
| variant - base | -0.011573 | -0.015045 | -0.009600 |

## 解释

Tensor-transform 路径是为了检查与 FlyGCL 作者代码的数据预处理路径是否一致而实现的。但在 seed 1 上，它低于默认 transform 路径。因此最终复现配置保留 `flygcl_tensor_transform: False`。
