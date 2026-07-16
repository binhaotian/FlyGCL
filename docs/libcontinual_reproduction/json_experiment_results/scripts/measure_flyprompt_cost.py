import argparse
import glob
import json
import time
from pathlib import Path

import torch

import core.model as arch
from core.config import Config
from core.utils import get_instance


def find_config(name):
    name = name + ".yaml" if not name.endswith(".yaml") else name
    matches = glob.glob(f"./config/**/{name}", recursive=True)
    if len(matches) != 1:
        raise FileNotFoundError(f"配置 {name} 应该只匹配到一个文件，实际匹配结果为：{matches}")
    return matches[0]


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_block(fn, device, warmup, iters):
    for _ in range(warmup):
        fn()
    synchronize(device)
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    synchronize(device)
    return (time.perf_counter() - start) / max(1, iters)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="flyprompt_cifar100_sup21k_balanced_v6_amp.yaml")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    config = Config(find_config(args.config)).get_config_dict()
    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{args.device}" if use_cuda else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    try:
        backbone = get_instance(arch, "backbone", config, **{"device": device})
    except TypeError:
        backbone = get_instance(arch, "backbone", config)
    model = get_instance(arch, "classifier", config, **{"device": device, "backbone": backbone}).to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    batch_size = args.batch_size or int(config.get("batch_size", 64))
    image_size = int(config.get("image_size", 224))
    num_classes = int(config["classifier"]["kwargs"]["num_class"])

    x = torch.randn(batch_size, 3, image_size, image_size, device=device)
    y = torch.randint(0, num_classes, (batch_size,), device=device)
    data = {"image": x, "label": y}

    optimizer = torch.optim.Adam(model.get_parameters(config), lr=config["optimizer"]["kwargs"]["lr"])
    use_amp = bool(config.get("use_amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    def train_step():
        model.train()
        optimizer.zero_grad(set_to_none=True)
        model.observe_with_optimizer(data, optimizer, scaler=scaler, use_amp=use_amp)

    with torch.no_grad():
        model.cur_task_id = 0
        model.max_seen_task = 0
        model.seen_classes.update(range(min(num_classes, 20)))
        model._router_ready = True
        if hasattr(model, "router") and hasattr(model.router, "fc"):
            model.router.fc.weight.zero_()

    def infer_step():
        model.eval()
        with torch.no_grad():
            model.inference(data)

    train_time = time_block(train_step, device, args.warmup, args.iters)
    inference_time = time_block(infer_step, device, args.warmup, args.iters)

    result = {
        "config": args.config,
        "device": str(device),
        "batch_size": batch_size,
        "warmup": args.warmup,
        "iters": args.iters,
        "total_params": total_params,
        "total_params_m": total_params / 1_000_000,
        "trainable_params": trainable_params,
        "trainable_params_m": trainable_params / 1_000_000,
        "train_time_sec_per_batch": train_time,
        "inference_time_sec_per_batch": inference_time,
    }

    print(json.dumps(result, indent=2, sort_keys=True))
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
