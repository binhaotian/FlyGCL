import math
import numpy as np
import torch
from collections import Counter

try:
    from sklearn.metrics import normalized_mutual_info_score
except Exception:
    normalized_mutual_info_score = None


def _entropy(probs):
    probs = np.array(probs, dtype=np.float64)
    probs = probs[probs > 0]
    if probs.size == 0:
        return 0.0
    return -np.sum(probs * np.log(probs))


def gini_coefficient(xs):
    xs = np.array(xs, dtype=np.float64)
    if xs.size == 0:
        return 0.0
    xs = xs.flatten()
    if np.all(xs == 0):
        return 0.0
    xs = xs + 1e-12
    xs = np.sort(xs)
    n = xs.size
    index = np.arange(1, n + 1)
    return (2.0 * np.sum(index * xs) / np.sum(xs) - (n + 1.0)) / n


def compute_router_metrics(labels, routes, margins=None, num_routes=None):
    """
    labels: array-like shape [N]
    routes: array-like shape [N] route ids (int)
    margins: optional array-like shape [N]
    num_routes: if provided, use as denominator for unused routes

    returns dict with nmi, cond_entropy, util_entropy, dead_route_ratio, avg_margin, gini
    """
    labels = np.array(labels)
    routes = np.array(routes)
    N = labels.size

    if num_routes is None:
        num_routes = int(max(routes.max() + 1, 1)) if routes.size > 0 else 0

    if N == 0 or num_routes <= 0:
        return {
            "nmi": float("nan"),
            "cond_entropy": float("nan"),
            "util_entropy": float("nan"),
            "dead_route_ratio": float("nan"),
            "avg_margin": float("nan"),
            "gini": float("nan"),
        }

    # NMI
    if normalized_mutual_info_score is not None:
        try:
            nmi = float(normalized_mutual_info_score(labels, routes))
        except Exception:
            nmi = float('nan')
    else:
        nmi = float('nan')

    # H(Route | Class): conditional entropy
    class_ids = np.unique(labels)
    ent_sum = 0.0
    total = 0
    for c in class_ids:
        idx = labels == c
        if idx.sum() == 0:
            continue
        route_counts = Counter(routes[idx])
        probs = np.array([route_counts.get(r, 0) / idx.sum() for r in range(num_routes)])
        ent = _entropy(probs)
        ent_sum += ent * idx.sum()
        total += idx.sum()
    cond_entropy = ent_sum / max(1, total)

    # Utilization entropy H(Route)/log(K)
    route_counts_all = Counter(routes)
    probs_all = np.array([route_counts_all.get(r, 0) / N for r in range(num_routes)])
    H_route = _entropy(probs_all)
    util_entropy = H_route / (math.log(num_routes + 1e-12) + 1e-12) if num_routes > 1 else 0.0

    # dead route ratio
    dead = sum(1 for r in range(num_routes) if route_counts_all.get(r, 0) == 0)
    dead_route_ratio = dead / max(1.0, num_routes)

    # avg margin
    if margins is None:
        avg_margin = float('nan')
    else:
        margins = np.array(margins, dtype=np.float64)
        avg_margin = float(np.nanmean(margins))

    # gini of route utilization
    gini = float(gini_coefficient(np.array([route_counts_all.get(r, 0) for r in range(num_routes)])))

    return {
        "nmi": nmi,
        "cond_entropy": float(cond_entropy),
        "util_entropy": float(util_entropy),
        "dead_route_ratio": float(dead_route_ratio),
        "avg_margin": float(avg_margin),
        "gini": float(gini),
    }


def _as_float(x):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().item()
    return float(x)


def compute_oracle_accuracies(per_route_logits, labels, n_classes=None):
    """Compute post-hoc route oracle diagnostics from route-forced logits.

    Args:
        per_route_logits: sequence of tensors, one [N, C] tensor per route.
        labels: [N] tensor/list of integer labels in the model's class index space.
        n_classes: optional class count. Defaults to logits.shape[-1].

    Returns:
        A dict with sample/class oracle and worst-router accuracies plus the
        selected routes. The oracle routes are diagnostic upper/lower bounds;
        they are not meant to be interpreted as ground-truth prompt labels.
    """
    if len(per_route_logits) == 0:
        raise ValueError("per_route_logits must contain at least one route")

    labels = torch.as_tensor(labels, dtype=torch.long).cpu()
    stacked = torch.stack([x.detach().cpu() for x in per_route_logits], dim=0)
    if stacked.ndim != 3:
        raise ValueError(f"Expected [R, N, C] logits, got shape {tuple(stacked.shape)}")

    num_routes, num_samples, inferred_classes = stacked.shape
    if labels.numel() != num_samples:
        raise ValueError(
            f"labels length {labels.numel()} does not match logits samples {num_samples}"
        )
    if n_classes is None:
        n_classes = inferred_classes

    sample_idx = torch.arange(num_samples)
    true_label_logits = stacked[:, sample_idx, labels]

    best_sample_routes = torch.argmax(true_label_logits, dim=0)
    sample_oracle_logits = stacked[best_sample_routes, sample_idx]
    sample_oracle_pred = torch.argmax(sample_oracle_logits, dim=1)
    A_sample_oracle = (sample_oracle_pred == labels).float().mean().item()

    worst_sample_routes = torch.argmin(true_label_logits, dim=0)
    worst_logits = stacked[worst_sample_routes, sample_idx]
    worst_pred = torch.argmax(worst_logits, dim=1)
    A_worst = (worst_pred == labels).float().mean().item()

    class_best_routes = torch.zeros(n_classes, dtype=torch.long)
    for c in range(n_classes):
        cls_mask = labels == c
        if not torch.any(cls_mask):
            continue
        cls_logits = stacked[:, cls_mask, :]
        cls_labels = labels[cls_mask]
        cls_pred = torch.argmax(cls_logits, dim=2)
        route_acc = (cls_pred == cls_labels.unsqueeze(0)).float().mean(dim=1)

        # Break accuracy ties by the mean true-label logit, matching the
        # sample-oracle objective as closely as possible.
        mean_true_logit = true_label_logits[:, cls_mask].mean(dim=1)
        tie_break_score = route_acc + mean_true_logit * 1e-12
        class_best_routes[c] = int(torch.argmax(tie_break_score).item())

    class_routes_per_sample = class_best_routes[labels]
    class_oracle_logits = stacked[class_routes_per_sample, sample_idx]
    class_oracle_pred = torch.argmax(class_oracle_logits, dim=1)
    A_class_oracle = (class_oracle_pred == labels).float().mean().item()

    return {
        "A_sample_oracle": float(A_sample_oracle),
        "A_class_oracle": float(A_class_oracle),
        "A_worst": float(A_worst),
        "sample_oracle_routes": best_sample_routes.tolist(),
        "class_oracle_routes": class_best_routes.tolist(),
        "class_oracle_routes_per_sample": class_routes_per_sample.tolist(),
        "worst_routes": worst_sample_routes.tolist(),
        "sample_oracle_pred": sample_oracle_pred.tolist(),
        "class_oracle_pred": class_oracle_pred.tolist(),
        "worst_pred": worst_pred.tolist(),
    }


def build_router_quality_summary(
    method,
    dataset,
    seed,
    A_learned,
    A_random,
    A_single=None,
    A_class_oracle=None,
    A_sample_oracle=None,
    A_worst=None,
    metrics=None,
):
    eps = 1e-8
    metrics = metrics or {}

    def maybe_gap(a, b):
        if a is None or b is None:
            return None
        return float(a - b)

    summary = {
        "method": method,
        "dataset": dataset,
        "seed": int(seed),
        "A_learned": _as_float(A_learned),
        "A_random": _as_float(A_random),
        "A_single": _as_float(A_single),
        "A_class_oracle": _as_float(A_class_oracle),
        "A_sample_oracle": _as_float(A_sample_oracle),
        "A_worst": _as_float(A_worst),
        "RQI_class": None,
        "RQI_sample": None,
        "router_utility": maybe_gap(A_learned, A_random),
        "oracle_gap_sample": maybe_gap(A_sample_oracle, A_learned),
        "capacity_gap_sample": maybe_gap(A_sample_oracle, A_single),
    }
    if A_class_oracle is not None:
        summary["RQI_class"] = float((A_learned - A_random) / (A_class_oracle - A_random + eps))
    if A_sample_oracle is not None:
        summary["RQI_sample"] = float((A_learned - A_random) / (A_sample_oracle - A_random + eps))

    summary.update(metrics)
    return summary


if __name__ == '__main__':
    # quick smoke test
    labels = [0,0,1,1,2,2]
    routes = [0,0,1,1,1,2]
    print(compute_router_metrics(labels, routes, margins=[0.1,0.2,0.3,0.1,0.5,0.4], num_routes=4))
