import argparse
import json
import os

import numpy as np


DISPLAY_COLUMNS = [
    ("A_random", "A_random"),
    ("A_learned", "A_learned"),
    ("A_single", "A_single"),
    ("A_class_oracle", "A_class_oracle"),
    ("A_sample_oracle", "A_sample_oracle"),
    ("RQI_class", "RQI_class"),
    ("RQI_sample", "RQI_sample"),
    ("nmi", "NMI"),
    ("cond_entropy", "H(R|Y)"),
    ("util_entropy", "Util"),
    ("avg_margin", "Margin"),
]


def _format_mean_std(values):
    arr = np.array(
        [np.nan if value is None else value for value in values],
        dtype=np.float64,
    )
    if np.all(np.isnan(arr)):
        return "nan +/- nan", float("nan"), float("nan")
    mean = float(np.nanmean(arr))
    std = float(np.nanstd(arr))
    return f"{mean:.4f} +/- {std:.4f}", mean, std


def _load_rows(root, method=None):
    if not os.path.exists(root):
        raise FileNotFoundError(f"No logs at {root}")

    rows = []
    for file_name in sorted(os.listdir(root)):
        if not file_name.startswith("router_quality_summary_seed_"):
            continue
        if not file_name.endswith(".json"):
            continue
        path = os.path.join(root, file_name)
        with open(path, "r", encoding="utf-8") as f:
            row = json.load(f)
        if method is not None and row.get("method") != method:
            continue
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Summarize RouterBench-GCL seed JSON files.")
    parser.add_argument("--log_dir", type=str, default="results/logs", help="Root logs dir.")
    parser.add_argument("--dataset", type=str, default="cifar100")
    parser.add_argument("--note", type=str, default="")
    parser.add_argument("--method", type=str, default=None)
    args = parser.parse_args()

    root = os.path.join(args.log_dir, args.dataset, args.note)
    rows = _load_rows(root, method=args.method)
    if len(rows) == 0:
        print(f"No matching router quality summaries found in {root}")
        raise SystemExit(1)

    headers = ["Method"] + [display for _, display in DISPLAY_COLUMNS]
    print(" | ".join(headers))
    print(" | ".join(["---"] * len(headers)))

    aggregate = {"root": root, "method_summaries": {}}
    for method in sorted(set(row["method"] for row in rows)):
        subset = [row for row in rows if row["method"] == method]
        output_cells = [method]
        aggregate["method_summaries"][method] = {"num_seeds": len(subset)}
        for key, _display in DISPLAY_COLUMNS:
            text, mean, std = _format_mean_std([row.get(key) for row in subset])
            output_cells.append(text)
            aggregate["method_summaries"][method][key] = {
                "mean": mean,
                "std": std,
            }
        print(" | ".join(output_cells))

    out_path = os.path.join(root, "router_quality_aggregate.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2)
    print(f"Saved aggregate to {out_path}")


if __name__ == "__main__":
    main()
