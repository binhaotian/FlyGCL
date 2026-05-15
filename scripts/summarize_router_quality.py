import os
import json
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--log_dir', type=str, default='results/logs', help='Root logs dir')
parser.add_argument('--dataset', type=str, default='cifar100')
parser.add_argument('--note', type=str, default='')
parser.add_argument('--method', type=str, default=None)
args = parser.parse_args()

root = os.path.join(args.log_dir, args.dataset, args.note)
if not os.path.exists(root):
    print('No logs at', root)
    exit(1)

files = [f for f in os.listdir(root) if f.startswith('router_quality_summary_seed_') and f.endswith('.json')]
if len(files) == 0:
    print('No summary files found in', root)
    exit(1)

rows = []
for f in files:
    p = os.path.join(root, f)
    try:
        with open(p, 'r', encoding='utf-8') as fh:
            data = json.load(fh)
            if args.method is not None and data.get('method') != args.method:
                continue
            rows.append(data)
    except Exception as e:
        print('Failed to read', p, e)

if len(rows) == 0:
    print('No matching rows')
    exit(1)

keys = ['A_random', 'A_learned', 'A_single', 'A_class_oracle', 'A_sample_oracle', 'RQI_class', 'RQI_sample', 'nmi', 'cond_entropy', 'util_entropy', 'avg_margin']
print('Method | ' + ' | '.join(keys))
for method in sorted(set(r['method'] for r in rows)):
    subset = [r for r in rows if r['method'] == method]
    vals = {}
    for k in keys:
        arr = np.array([r.get(k, np.nan) if r.get(k, None) is not None else np.nan for r in subset], dtype=np.float64)
        mean = np.nanmean(arr)
        std = np.nanstd(arr)
        vals[k] = f"{mean:.4f} ± {std:.4f}"
    print(method + ' | ' + ' | '.join(vals[k] for k in keys))

# Save aggregate
out = {}
out['method_summaries'] = {}
for method in sorted(set(r['method'] for r in rows)):
    subset = [r for r in rows if r['method'] == method]
    out['method_summaries'][method] = {}
    for k in keys:
        arr = np.array([r.get(k, np.nan) if r.get(k, None) is not None else np.nan for r in subset], dtype=np.float64)
        out['method_summaries'][method][k] = {'mean': float(np.nanmean(arr)), 'std': float(np.nanstd(arr))}

with open(os.path.join(root, 'router_quality_aggregate.json'), 'w', encoding='utf-8') as f:
    json.dump(out, f, indent=2)
print('Saved aggregate to', os.path.join(root, 'router_quality_aggregate.json'))
