import argparse
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torchvision.datasets import CIFAR10, CIFAR100


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.OnlineIterDataset import OnlineIterDataset
from utils.onlinesampler import OnlineSampler


def build_dataset(dataset_name: str, data_root: str):
    dataset_name = dataset_name.lower()
    if dataset_name == "cifar100":
        return CIFAR100(root=data_root, train=True, download=False, transform=torchvision_to_tensor)
    if dataset_name == "cifar10":
        return CIFAR10(root=data_root, train=True, download=False, transform=torchvision_to_tensor)
    raise ValueError(f"Unsupported dataset for this visualization script: {dataset_name}")


def torchvision_to_tensor(image):
    return torch.as_tensor(np.array(image)).permute(2, 0, 1).float() / 255.0


def get_default_args():
    parser = argparse.ArgumentParser(description="Visualize FlyGCL input stream mixing")
    parser.add_argument("--dataset", type=str, default="cifar100")
    parser.add_argument("--data_root", type=str, default="/root/autodl-tmp/data")
    parser.add_argument("--n_tasks", type=int, default=5)
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--m", type=int, default=10)
    parser.add_argument("--rnd_seed", type=int, default=1)
    parser.add_argument("--rnd_NM", action="store_true", default=True)
    parser.add_argument("--output_dir", type=str, default="results/input_stream/default_seed1")
    parser.add_argument("--timeline_task", type=int, default=0)
    parser.add_argument("--timeline_limit", type=int, default=300)
    return parser.parse_args()


def summarize_stream(dataset, sampler, n_tasks: int):
    task_classes = [sorted(map(int, sampler.disjoint_classes[t] + sampler.blurry_classes[t])) for t in range(n_tasks)]
    class_owner = {}
    for task_id, cls_list in enumerate(task_classes):
        for cls in cls_list:
            class_owner[int(cls)] = task_id

    rows = []
    matrix = np.zeros((n_tasks, n_tasks), dtype=np.int64)
    class_matrix = np.zeros((len(dataset.classes), n_tasks), dtype=np.int64)
    total_samples_per_task = []

    for cur_task in range(n_tasks):
        indices = sampler.indices[cur_task]
        labels = [int(dataset.targets[i]) for i in indices]
        total_samples_per_task.append(len(labels))

        for lbl in labels:
            owner = class_owner[int(lbl)]
            matrix[cur_task, owner] += 1
            class_matrix[int(lbl), cur_task] += 1

        same_task = matrix[cur_task, cur_task]
        foreign = len(labels) - same_task
        rows.append(
            {
                "task": cur_task,
                "total_samples": len(labels),
                "same_task_samples": same_task,
                "foreign_samples": foreign,
                "same_task_ratio": same_task / max(1, len(labels)),
                "foreign_ratio": foreign / max(1, len(labels)),
            }
        )

    summary = pd.DataFrame(rows)
    matrix_ratio = matrix / matrix.sum(axis=1, keepdims=True)
    class_row_sum = class_matrix.sum(axis=1, keepdims=True)
    class_ratio = np.divide(
        class_matrix,
        np.maximum(class_row_sum, 1),
        out=np.zeros_like(class_matrix, dtype=float),
        where=class_row_sum > 0,
    )
    return task_classes, class_owner, matrix, matrix_ratio, class_matrix, class_ratio, summary


def plot_origin_heatmap(matrix, matrix_ratio, output_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix_ratio, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(matrix_ratio.shape[1]))
    ax.set_yticks(np.arange(matrix_ratio.shape[0]))
    ax.set_xticklabels([f"owner {i}" for i in range(matrix_ratio.shape[1])])
    ax.set_yticklabels([f"task {i}" for i in range(matrix_ratio.shape[0])])
    for i in range(matrix_ratio.shape[0]):
        for j in range(matrix_ratio.shape[1]):
            ax.text(j, i, f"{matrix_ratio[i, j]:.1%}", ha="center", va="center", color="black", fontsize=9)
    fig.colorbar(im, ax=ax, label="Share within current task")
    ax.set_xlabel("Owner task of the class")
    ax.set_ylabel("Current task / stream position")
    ax.set_title("FlyGCL default input stream mixing")
    fig.tight_layout()
    fig.savefig(output_dir / "task_origin_heatmap.png", dpi=200)
    plt.close(fig)


def plot_origin_stacked_bar(matrix_ratio, output_dir: Path):
    n_tasks = matrix_ratio.shape[0]
    x = np.arange(n_tasks)
    bottoms = np.zeros(n_tasks)
    colors = [plt.get_cmap("tab10")(i % 10) for i in range(n_tasks)]

    fig, ax = plt.subplots(figsize=(9, 5))
    for owner_task in range(n_tasks):
        vals = matrix_ratio[:, owner_task]
        ax.bar(x, vals, bottom=bottoms, color=colors[owner_task], label=f"owner {owner_task}")
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels([f"task {i}" for i in range(n_tasks)])
    ax.set_ylabel("Ratio")
    ax.set_ylim(0, 1)
    ax.set_title("Per-task sample origin ratio")
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "task_origin_stacked_bar.png", dpi=200)
    plt.close(fig)


def plot_timeline(dataset, sampler, class_owner, task_id: int, timeline_limit: int, output_dir: Path):
    indices = sampler.indices[task_id][:timeline_limit]
    labels = [int(dataset.targets[i]) for i in indices]
    owners = [class_owner[lbl] for lbl in labels]

    fig, ax = plt.subplots(figsize=(12, 4))
    colors = [plt.get_cmap("tab10")(i % 10) for i in range(max(owners) + 1)]
    ax.scatter(np.arange(len(owners)), labels, c=[colors[o] for o in owners], s=22, alpha=0.85)
    ax.set_xlabel("Sample order inside the task stream")
    ax.set_ylabel("Class id")
    ax.set_title(f"Task {task_id} stream preview (first {len(owners)} samples)")
    fig.tight_layout()
    fig.savefig(output_dir / f"task_{task_id}_stream_timeline.png", dpi=200)
    plt.close(fig)


def plot_class_mixing_heatmap(class_matrix, class_ratio, output_dir: Path):
    fig, ax = plt.subplots(figsize=(10, 12))
    im = ax.imshow(class_ratio, aspect="auto", cmap="OrRd", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(class_ratio.shape[1]))
    ax.set_xticklabels([f"task {i}" for i in range(class_ratio.shape[1])])
    ax.set_yticks(np.arange(class_ratio.shape[0]))
    ax.set_yticklabels([str(i) for i in range(class_ratio.shape[0])], fontsize=7)
    ax.set_xlabel("Stream task")
    ax.set_ylabel("Class id")
    ax.set_title("Per-class appearance ratio across task streams")
    fig.colorbar(im, ax=ax, label="Appearance ratio")
    fig.tight_layout()
    fig.savefig(output_dir / "class_mixing_heatmap.png", dpi=200)
    plt.close(fig)


def main():
    args = get_default_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset(args.dataset, args.data_root)
    online_dataset = OnlineIterDataset(dataset, 1)
    sampler = OnlineSampler(
        online_dataset,
        args.n_tasks,
        args.m,
        args.n,
        args.rnd_seed,
        0,
        args.rnd_NM,
        None,
        None,
    )

    task_classes, class_owner, matrix, matrix_ratio, class_matrix, class_ratio, summary = summarize_stream(dataset, sampler, args.n_tasks)
    summary_path = output_dir / "task_origin_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("Default input stream summary")
    print(summary.to_string(index=False))
    print()
    print("Class ownership by task:")
    for i, cls_list in enumerate(task_classes):
        print(f"task {i}: {len(cls_list)} classes -> {cls_list}")
    print()
    print(f"Saved summary to: {summary_path}")

    plot_origin_heatmap(matrix, matrix_ratio, output_dir)
    plot_origin_stacked_bar(matrix_ratio, output_dir)
    plot_class_mixing_heatmap(class_matrix, class_ratio, output_dir)

    for task_id in range(args.n_tasks):
        plot_timeline(dataset, sampler, class_owner, task_id, args.timeline_limit, output_dir)

    np.save(output_dir / "task_origin_matrix.npy", matrix)
    np.save(output_dir / "task_origin_ratio.npy", matrix_ratio)
    np.save(output_dir / "class_origin_matrix.npy", class_matrix)
    np.save(output_dir / "class_origin_ratio.npy", class_ratio)
    print(f"Saved plots to: {output_dir}")


if __name__ == "__main__":
    main()