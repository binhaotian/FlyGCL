# FlyPrompt 随机性记录

本文件记录当前 LibContinual FlyPrompt 复现路径中的随机性来源和控制方式。

| 项目 | 当前状态 | 剩余风险或处理方式 |
| --- | --- | --- |
| CLI seed | `run_trainer.py --seed` 会在 trainer 构造前覆盖 `config['seed']`。 | 每次运行由命令行 seed 控制。 |
| Python random | `core.utils.init_seed` 调用 `random.seed(seed)`。 | 在 device 初始化和 epoch 训练开始时重新设置。 |
| NumPy | `core.utils.init_seed` 调用 `np.random.seed(seed)`。 | 对使用 NumPy 全局 RNG 的代码有效。 |
| torch CPU | `core.utils.init_seed` 调用 `torch.manual_seed(seed)`。 | 已控制。 |
| torch CUDA | `core.utils.init_seed` 调用 `torch.cuda.manual_seed` 和 `manual_seed_all`。 | 在 run/epoch 边界控制。 |
| PYTHONHASHSEED | `core.utils.init_seed` 设置 `os.environ['PYTHONHASHSEED']`。 | 运行时设置，不是在解释器启动前设置。 |
| Si-Blurry 数据流划分 | `core/data/si_blurry.py` 使用 `torch.Generator().manual_seed(seed)` 构造类别和样本流。 | 对当前实现的数据流采样器有效。 |
| DataLoader worker seed | 当前没有 FlyPrompt 专用 `worker_init_fn`。 | `num_workers` 会影响 worker RNG 行为；仅在不破坏现有 loader 的前提下再补充 worker seeding。 |
| torchvision 训练 transform | RandomCrop/HorizontalFlip 使用 PyTorch/torchvision RNG。 | 受全局 RNG 和 worker RNG 控制。 |
| deterministic flag | `config['deterministic']` 传入 `init_seed`；为 false 时 cuDNN benchmark 开启，deterministic mode 关闭。 | 当前复现优先保持作者代码性能口径，而不是严格逐 bit 确定性。 |
| AMP | `use_amp: True` 为正式 v6 对齐配置启用 CUDA AMP 和 GradScaler。 | 允许小幅数值非确定性。 |

建议：报告使用当前 performance-aligned 配置；如果需要 CI 中的严格稳定性，可另建 deterministic smoke 配置。不要在不重跑结果的情况下静默修改正式五种子配置。
