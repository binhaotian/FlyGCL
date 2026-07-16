import argparse
import json
from pathlib import Path

import numpy as np


METRICS = ("A_auc", "A_avg", "A_last", "F_last", "BWT_last")


def load_results(results_root):
    paths = sorted(Path(results_root).rglob("flyprompt_gcl_seed_*.json"))
    rows = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as fin:
            item = json.load(fin)
        seed = path.stem.replace("flyprompt_gcl_seed_", "")
        row = {"seed": seed, "path": str(path)}
        for metric in METRICS:
            row[metric] = float(item.get(metric, np.nan))
        rows.append(row)
    return rows


def print_table(rows):
    if not rows:
        print("No flyprompt_gcl_seed_*.json files found.")
        return

    header = ["seed", *METRICS, "path"]
    print("\t".join(header))
    for row in rows:
        print("\t".join([
            row["seed"],
            *[f"{row[m]:.6f}" for m in METRICS],
            row["path"],
        ]))

    print("\nsummary")
    for metric in METRICS:
        values = np.array([row[metric] for row in rows], dtype=np.float64)
        values = values[~np.isnan(values)]
        if values.size == 0:
            print(f"{metric}\tmean=nan\tstd=nan\tn=0")
        else:
            print(f"{metric}\tmean={values.mean():.6f}\tstd={values.std(ddof=0):.6f}\tn={values.size}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="./results/log/FlyPrompt")
    args = parser.parse_args()
    print_table(load_results(args.results_root))


if __name__ == "__main__":
    main()
