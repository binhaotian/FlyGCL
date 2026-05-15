# 输入数据流可视化说明

这个文档对应脚本 [scripts/visualize_input_stream.py](../scripts/visualize_input_stream.py)。

## 作用

它用来观察 FlyGCL 在默认设置下，训练输入是怎么被 `OnlineSampler` 切成多个 task 的，
每个 task 里混入了多少来自其他 task 的样本，以及每个 class 在不同 task 中的出现情况。

默认参数是：

- dataset: `cifar100`
- n_tasks: `5`
- n: `50`
- m: `10`
- seed: `1`
- rnd_NM: `True`

## 生成内容

- `task_origin_heatmap.png`
  - 看每个当前 task 的样本，分别来自哪些 owner task。
  - 对角线越深，说明这个 task 的样本越集中在自己的 class 段里。

- `task_origin_stacked_bar.png`
  - 看每个 task 的样本来源比例。
  - 便于快速比较哪个 task 被混得更厉害。

- `task_0_stream_timeline.png` 到 `task_4_stream_timeline.png`
  - 看某个 task 内部，样本在顺序上是怎么分布的。
  - 横轴是流中的样本顺序，纵轴是 class id，颜色表示 class 属于哪个 owner task。

- `class_mixing_heatmap.png`
  - 看每个 class 的样本最终分散到了哪些 task 流里。
  - 适合回答“某个 class 是不是主要出现在自己的 task 里，还是被打散到别的 task 里了”。

## 怎么看默认结果

如果你看到热力图主对角线最深，说明大多数样本还是留在自己的 task 中。
如果某个 task 的非对角线颜色更明显，说明这个 task 受到了更多跨 task 混杂。
当前默认设置下，混杂通常是“少量跨 task 样本插入”，不是完全均匀打散。

## 运行方式

```bash
cd /root/FlyGCL
python scripts/visualize_input_stream.py --data_root /root/autodl-tmp/data --output_dir results/input_stream/default_seed1
```

输出会保存在 `results/input_stream/default_seed1/`。