#!/usr/bin/env python3
"""Paired image-level comparisons of a reference model against other models.

Every model is given as ``NAME=PATTERN``, where ``PATTERN`` is a run folder or
a glob matching several run folders (e.g. the four support seeds of a few-shot
setting). Each folder must contain ``per_image_metrics.csv``. When a model has
several runs, the per-image values are first averaged over its runs; only
images present in every run of every model are analysed.

For every metric and comparator the script reports the mean advantage of the
reference (for counting error, comparator minus reference, i.e. conidia of
error saved per image), a 95% bootstrap confidence interval (10 000 resamples
of images, seed 0), the number of images in which the reference is better, a
two-sided Wilcoxon signed-rank test on the non-zero differences, and the Holm
adjustment across the comparators of that metric. This is the protocol of the
paper.

Example (H. werneckii, five support images, four seeds)::

    python scripts/paired_tests.py \\
        --reference "cief_net=runs/fewshot/hortaea_werneckii/cief_net/5shot/s*" \\
        --compare "base=runs/fewshot/hortaea_werneckii/base/5shot/s*" \\
        --compare "frozen_refiner=runs/fewshot/hortaea_werneckii/cief_net_frozen_refiner/5shot/s*" \\
        --out results/paired_hortaea_5shot.csv

Baseline methods are compared the same way, provided that their runs write a
``per_image_metrics.csv`` with the same columns.
"""

import argparse
import csv
import glob
import math
import sys
from pathlib import Path

import numpy as np

METRICS = [("aji", True), ("pq", True), ("dq", True), ("sq", True), ("boundary_iou", True),
           ("dice", True), ("count_error", False)]
_LEGACY_COLUMNS = {"nombre_archivo": "image", "error_conteo": "count_error"}
_FILES = ("per_image_metrics.csv", "metricas_por_imagen.csv")


def read_per_image(folder, metrics) -> dict:
    """``{image file name: {metric: value}}`` from a run folder, or ``None``."""
    folder = Path(folder)
    path = next((folder / f for f in _FILES if (folder / f).exists()), None)
    if path is None:
        return None
    out = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            row = {_LEGACY_COLUMNS.get(k, k): v for k, v in row.items()}
            out[Path(row["image"]).name] = {m: float(row[m]) for m in metrics
                                            if row.get(m, "") not in ("", "nan")}
    return out


def wilcoxon_p(d) -> float:
    """Two-sided Wilcoxon signed-rank p-value on the non-zero differences
    (normal approximation if SciPy is unavailable); NaN with fewer than 6."""
    d = np.asarray(d, float)
    d = d[d != 0]
    if len(d) < 6:
        return float("nan")
    try:
        from scipy.stats import wilcoxon
        return float(wilcoxon(d).pvalue)
    except Exception:
        r = np.argsort(np.argsort(np.abs(d))) + 1.0
        n = len(d)
        z = (r[d > 0].sum() - n * (n + 1) / 4) / math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
        return float(math.erfc(abs(z) / math.sqrt(2)))


def holm(p) -> np.ndarray:
    """Holm step-down adjustment."""
    p = np.asarray(p, float)
    adj, running = np.empty_like(p), 0.0
    for i, k in enumerate(np.argsort(p)):
        running = max(running, min(1.0, (len(p) - i) * p[k]))
        adj[k] = running
    return adj


def load_model_runs(spec: str, metrics) -> tuple:
    if "=" not in spec:
        sys.exit(f"expected NAME=PATTERN, got '{spec}'")
    name, pattern = spec.split("=", 1)
    folders = sorted(glob.glob(pattern)) or [pattern]
    runs = [r for r in (read_per_image(f, metrics) for f in folders) if r]
    if not runs:
        sys.exit(f"{name}: no per_image_metrics.csv found for '{pattern}'")
    common = set.intersection(*[set(r) for r in runs])
    averaged = {img: {m: float(np.mean([r[img][m] for r in runs])) for m in metrics
                      if all(m in r[img] for r in runs)} for img in common}
    print(f"  {name}: {len(runs)} run(s), {len(common)} images")
    return name, averaged


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", required=True, help="NAME=PATTERN of the reference model")
    ap.add_argument("--compare", action="append", required=True, help="NAME=PATTERN (repeatable)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    keys = [m for m, _ in METRICS]
    ref_name, ref = load_model_runs(args.reference, keys)
    others = [load_model_runs(s, keys) for s in args.compare]
    images = sorted(set(ref).intersection(*[set(o) for _, o in others]))
    print(f"  {len(images)} images shared by all models")

    rng = np.random.default_rng(args.seed)
    rows = []
    for metric, higher_better in METRICS:
        block = []
        for name, data in others:
            imgs = [i for i in images if metric in ref[i] and metric in data[i]]
            if not imgs:
                continue
            a = np.array([ref[i][metric] for i in imgs])
            b = np.array([data[i][metric] for i in imgs])
            d = (a - b) if higher_better else (b - a)
            boot = d[rng.integers(0, len(d), size=(args.n_boot, len(d)))].mean(1)
            block.append({"metric": metric, "reference": ref_name, "versus": name, "n_images": len(d),
                          "mean_advantage": float(d.mean()),
                          "ci95_low": float(np.percentile(boot, 2.5)),
                          "ci95_high": float(np.percentile(boot, 97.5)),
                          "better_in": int((d > 0).sum()), "p_wilcoxon": wilcoxon_p(d)})
        for b, ph in zip(block, holm([x["p_wilcoxon"] if x["p_wilcoxon"] == x["p_wilcoxon"] else 1.0
                                      for x in block])):
            b["p_holm"] = float(ph)
        rows += block

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        if r["metric"] in ("pq", "sq", "count_error"):
            print(f"  {r['metric']:12s} vs {r['versus']:18s} {r['mean_advantage']:+.4f} "
                  f"[{r['ci95_low']:+.4f}, {r['ci95_high']:+.4f}]  {r['better_in']}/{r['n_images']}  "
                  f"p_Holm={r['p_holm']:.2g}")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
