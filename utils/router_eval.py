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


if __name__ == '__main__':
    # quick smoke test
    labels = [0,0,1,1,2,2]
    routes = [0,0,1,1,1,2]
    print(compute_router_metrics(labels, routes, margins=[0.1,0.2,0.3,0.1,0.5,0.4], num_routes=4))
