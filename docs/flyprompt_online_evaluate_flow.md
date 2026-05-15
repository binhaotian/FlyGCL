# FlyPrompt online_evaluate 调用图

这个图对应 [assets/flyprompt_online_evaluate_flow.mmd](../assets/flyprompt_online_evaluate_flow.mmd)。

## 你要看的主线

1. `main.py` 解析参数并构建 `FlyPrompt` trainer。
2. `_Trainer.run()` 进入训练主循环。
3. `_Trainer.main_worker()` 在评估周期或任务结束时调用 `online_evaluate(...)`。
4. `methods/flyprompt.py` 里的 `online_evaluate()` 先刷新路由器，再做 expert 路由和多头推理。
5. `models/flyprompt.py` 里的模型层负责真正的 forward、collect、update。

## 图里每一层的意思

- 入口层：程序怎么启动。
- 训练器层：什么时候触发评估。
- 方法层：FlyPrompt 这一方法自己定义了什么评估逻辑。
- 模型层：具体 forward / collect / update 如何委托到更底层模块。
- 叶子实现：最底层的数值计算，例如 RPFC 闭式求解、Prompt 插入、EMA 初始化。

## 重点理解

- `online_evaluate()` 本身不直接做模型前向，它通过 `model_without_ddp.forward_with_rp()` 和 `forward_with_ema()` 间接进入模型层。
- `model_without_ddp.update()` 的作用是先把路由器参数结算好，再做评估。
- `collect()` 是训练期的统计累积入口，`update()` 是评估期把统计变成路由器权重。

## 怎么预览

在 VS Code 里直接打开 `.mmd` 文件即可预览 Mermaid 图。
