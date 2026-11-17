#!/usr/bin/env python3
"""Mean and standard deviation over seeds of every group of runs.

Every ``summary.json`` under ``--root`` is collected. Runs are grouped by
their folder path without the last component (the seed folder, e.g.
``hortaea_werneckii/cief_net/5shot`` for ``.../5shot/s42``), and the mean and
sample standard deviation of the main metrics are written to ``--out``.

Example::

    python scripts/aggregate_runs.py --root runs/fewshot --out runs/fewshot/fewshot_summary.csv
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

METRICS = ["pq_mean", "aji_mean", "dq_mean", "sq_mean", "boundary_iou_mean", "dice_mean",
           "count_mae", "map_50", "map", "ref_side"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary-name", default="summary.json")
    args = ap.parse_args()
    root = Path(args.root)

    groups = defaultdict(list)
    for path in sorted(root.rglob(args.summary_name)):
        rel = path.parent.relative_to(root)
        if len(rel.parts) < 2:
            continue
        with open(path) as f:
            groups[str(rel.parent)].append((rel.parts[-1], json.load(f)))

    rows = []
    for group, runs in sorted(groups.items()):
        row = {"group": group, "n_runs": len(runs), "runs": " ".join(r for r, _ in runs)}
        for m in METRICS:
            vals = [s[m] for _, s in runs if isinstance(s.get(m), (int, float))]
            if vals:
                row[f"{m}_mean"] = float(np.mean(vals))
                row[f"{m}_sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        rows.append(row)
    if not rows:
        print(f"no {args.summary_name} found under {root}")
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        if "pq_mean_mean" in r:
            print(f"  {r['group']:55s} n={r['n_runs']}  PQ={r['pq_mean_mean']:.4f} +- {r['pq_mean_sd']:.4f}")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
