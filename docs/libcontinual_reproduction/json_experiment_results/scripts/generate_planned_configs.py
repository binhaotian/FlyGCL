import copy
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG = REPO_ROOT / "config" / "flyprompt_cifar100_sup21k_balanced_v6_amp.yaml"
OUT_DIR = REPO_ROOT / "config" / "flyprompt_planned"


def deep_update(base, patch):
    result = copy.deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_with_includes(path):
    with open(path, "r", encoding="utf-8") as fin:
        data = yaml.safe_load(fin) or {}

    merged = {}
    for include in data.get("includes", []):
        include_path = REPO_ROOT / "config" / include
        merged = deep_update(merged, load_with_includes(include_path))

    data = {k: v for k, v in data.items() if k != "includes"}
    return deep_update(merged, data)


def write_config(name, patch):
    base = load_with_includes(BASE_CONFIG)
    config = deep_update(base, patch)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    with open(path, "w", encoding="utf-8") as fout:
        fout.write("# 由 reproduce/flyprompt/generate_planned_configs.py 自动生成\n")
        yaml.safe_dump(config, fout, sort_keys=False)
    return path


def norm_transform(mean, std):
    return [
        {"Resize": {"size": [224, 224]}},
        {"RandomCrop": {"size": 224, "padding": 4}},
        {"RandomHorizontalFlip": {"p": 0.5}},
        {"ToTensor": {}},
        {"Normalize": {"mean": mean, "std": std}},
    ]


def test_transform(mean, std):
    return [
        {"Resize": {"size": [224, 224]}},
        {"ToTensor": {}},
        {"Normalize": {"mean": mean, "std": std}},
    ]


def main():
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    cub_mean = [0.5071, 0.4867, 0.4408]
    cub_std = [0.2675, 0.2565, 0.2761]

    specs = {
        "flyprompt_plan_cifar100_rd0_rb10_amp.yaml": {
            "save_path": "./results_plan_cifar100_rd0_rb10_amp",
            "si_blurry": {"disjoint_ratio": 0, "blurry_ratio": 10, "randomized": False, "shuffle_train": False},
        },
        "flyprompt_plan_cifar100_rd100_rb10_amp.yaml": {
            "save_path": "./results_plan_cifar100_rd100_rb10_amp",
            "si_blurry": {"disjoint_ratio": 100, "blurry_ratio": 10, "randomized": False, "shuffle_train": False},
        },
        "flyprompt_plan_cifar100_rd50_rb30_amp.yaml": {
            "save_path": "./results_plan_cifar100_rd50_rb30_amp",
            "si_blurry": {"disjoint_ratio": 50, "blurry_ratio": 30, "randomized": False, "shuffle_train": False},
        },
        "flyprompt_plan_cifar100_rd50_rb50_amp.yaml": {
            "save_path": "./results_plan_cifar100_rd50_rb50_amp",
            "si_blurry": {"disjoint_ratio": 50, "blurry_ratio": 50, "randomized": False, "shuffle_train": False},
        },
        "flyprompt_plan_cifar100_no_ema_amp.yaml": {
            "save_path": "./results_plan_cifar100_no_ema_amp",
            "classifier": {"kwargs": {"ema_ratio": []}},
        },
        "flyprompt_plan_cifar100_no_prompt_no_ema_amp.yaml": {
            "save_path": "./results_plan_cifar100_no_prompt_no_ema_amp",
            "classifier": {"kwargs": {"use_prompt_experts": False, "ema_ratio": []}},
        },
        "flyprompt_plan_cifar100_no_prompt_ema_amp.yaml": {
            "save_path": "./results_plan_cifar100_no_prompt_ema_amp",
            "classifier": {"kwargs": {"use_prompt_experts": False, "ema_ratio": [0.9, 0.99]}},
        },
        "flyprompt_plan_cifar100_no_rear_no_ema_amp.yaml": {
            "save_path": "./results_plan_cifar100_no_rear_no_ema_amp",
            "classifier": {"kwargs": {"rp_dim": 0, "router_solver": "direct", "ema_ratio": []}},
        },
        "flyprompt_plan_cifar100_no_rear_ema_amp.yaml": {
            "save_path": "./results_plan_cifar100_no_rear_ema_amp",
            "classifier": {"kwargs": {"rp_dim": 0, "router_solver": "direct", "ema_ratio": [0.9, 0.99]}},
        },
        "flyprompt_plan_cifar100_ranpac_dagger_amp.yaml": {
            "save_path": "./results_plan_cifar100_ranpac_dagger_amp",
            "classifier": {
                "name": "RanPACGCL",
                "kwargs": {
                    "num_class": 100,
                    "task_num": 5,
                    "feat_dim": 768,
                    "rp_dim": 10000,
                    "rp_ridge": 100000.0,
                    "train_first_task_head": True,
                    "use_batch_mask": True,
                },
            },
        },
        "flyprompt_plan_cifar100_no_prompt_proxy_amp.yaml": {
            "save_path": "./results_plan_cifar100_no_prompt_proxy_amp",
            "classifier": {"kwargs": {"prompt_length": 0}},
        },
        "flyprompt_plan_cifar100_no_rear_proxy_amp.yaml": {
            "save_path": "./results_plan_cifar100_no_rear_proxy_amp",
            "classifier": {"kwargs": {"rp_dim": 0, "router_solver": "direct"}},
        },
        "flyprompt_plan_cifar100_ensemble_mean_amp.yaml": {
            "save_path": "./results_plan_cifar100_ensemble_mean_amp",
            "classifier": {"kwargs": {"ensemble_method": "mean"}},
        },
        "flyprompt_plan_cifar100_ensemble_max_prob_amp.yaml": {
            "save_path": "./results_plan_cifar100_ensemble_max_prob_amp",
            "classifier": {"kwargs": {"ensemble_method": "max_prob"}},
        },
        "flyprompt_plan_cifar100_ensemble_softmax_mean_amp.yaml": {
            "save_path": "./results_plan_cifar100_ensemble_softmax_mean_amp",
            "classifier": {"kwargs": {"ensemble_method": "softmax_mean"}},
        },
        "flyprompt_plan_cifar100_ema_online_only_amp.yaml": {
            "save_path": "./results_plan_cifar100_ema_online_only_amp",
            "classifier": {"kwargs": {"ema_ratio": []}},
        },
        "flyprompt_plan_cifar100_ema_09_amp.yaml": {
            "save_path": "./results_plan_cifar100_ema_09_amp",
            "classifier": {"kwargs": {"ema_ratio": [0.9]}},
        },
        "flyprompt_plan_cifar100_ema_099_amp.yaml": {
            "save_path": "./results_plan_cifar100_ema_099_amp",
            "classifier": {"kwargs": {"ema_ratio": [0.99]}},
        },
        "flyprompt_plan_cifar100_ema_0999_amp.yaml": {
            "save_path": "./results_plan_cifar100_ema_0999_amp",
            "classifier": {"kwargs": {"ema_ratio": [0.999]}},
        },
        "flyprompt_plan_cifar100_ema_09_099_0999_amp.yaml": {
            "save_path": "./results_plan_cifar100_ema_09_099_0999_amp",
            "classifier": {"kwargs": {"ema_ratio": [0.9, 0.99, 0.999]}},
        },
        "flyprompt_plan_cifar100_mask_seen_class_amp.yaml": {
            "save_path": "./results_plan_cifar100_mask_seen_class_amp",
            "classifier": {"kwargs": {"use_batch_mask": False}},
        },
        "flyprompt_plan_imagenetr_sup21k_default_amp.yaml": {
            "save_path": "./results_plan_imagenetr_sup21k_default_amp",
            "dataset": "si-blurry-imagenet-r",
            "data_root": "/root/autodl-tmp/FlyGCL/data/imagenet-r",
            "total_cls_num": 200,
            "init_cls_num": 40,
            "inc_cls_num": 40,
            "train_trfms": norm_transform(imagenet_mean, imagenet_std),
            "test_trfms": test_transform(imagenet_mean, imagenet_std),
            "backbone": {"kwargs": {"num_classes": 200}},
            "classifier": {"kwargs": {"num_class": 200}},
        },
        "flyprompt_plan_cub200_sup21k_default_amp.yaml": {
            "save_path": "./results_plan_cub200_sup21k_default_amp",
            "dataset": "si-blurry-cub200",
            "data_root": "/root/autodl-tmp/FlyGCL/data/CUB_200_2011",
            "total_cls_num": 200,
            "init_cls_num": 40,
            "inc_cls_num": 40,
            "train_trfms": norm_transform(cub_mean, cub_std),
            "test_trfms": test_transform(cub_mean, cub_std),
            "backbone": {"kwargs": {"num_classes": 200}},
            "classifier": {"kwargs": {"num_class": 200}},
        },
    }

    for name, patch in specs.items():
        path = write_config(name, patch)
        print(path.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
