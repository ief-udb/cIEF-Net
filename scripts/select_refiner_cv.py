#!/usr/bin/env python3
"""Select the refinement configuration by four-fold cross-validation.

Aggregates the runs of ``experiments/04_refiner_selection_cv.sh`` and applies
the selection rule fixed *before* the experiment:

    choose the refiner variant with the highest mean out-of-fold PQ over the
    116 training images (each evaluated once, when its fold was held out).
    The test set is never used.

It also reports, per variant, the out-of-fold PQ, SQ, AJI, DQ, Boundary IoU,
Dice and counting error, the PQ of every fold (mean +- SD over folds), the
number of folds won, and paired image-level tests of the selected variant
against every other one (mean difference, 95% bootstrap CI with 10 000
resamples, two-sided Wilcoxon signed-rank test, Holm correction across the
comparisons of each metric).

Usage::

    python scripts/select_refiner_cv.py --root runs/cv_refiner_selection
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paired_tests import holm, read_per_image, wilcoxon_p  # noqa: E402

METRICS = [("pq", "PQ"), ("sq", "SQ"), ("aji", "AJI"), ("dq", "DQ"),
           ("boundary_iou", "Boundary IoU"), ("dice", "Dice"), ("count_error", "Count error")]
PAIRED = ["pq", "sq", "aji", "boundary_iou"]
LABELS = {"no_refiner": "no refiner (base + containment)", "pose": "affine pose",
          "k2": "harmonic K=2", "k5": "harmonic K=5", "k12": "harmonic K=12", "k24": "harmonic K=24"}


def eval_dir(root, fold, variant):
    """Evaluation folder of one fold and variant (current or legacy names)."""
    if variant == "no_refiner":
        cands = [root / f"fold{fold}" / "base" / d for d in ("eval_containment_val", "eval_contencion_val")]
    else:
        cands = [root / f"fold{fold}" / f"ref_{variant}" / d
                 for d in ("eval_refiner_val", "eval_con_refinador_val")]
    return next((c for c in cands if c.exists()), cands[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--variants", nargs="+", default=["no_refiner", "pose", "k2", "k5", "k24"])
    ap.add_argument("--folds", nargs="+", default=["0", "1", "2", "3"])
    ap.add_argument("--chosen-on-test", default="k24",
                    help="variant originally chosen on the test set, for the consistency check")
    args = ap.parse_args()
    root = Path(args.root)
    rng = np.random.default_rng(0)
    metric_keys = [m for m, _ in METRICS]

    data = {}                                   # variant -> fold -> {image: metrics}
    for v in args.variants:
        for f in args.folds:
            d = read_per_image(eval_dir(root, f, v), metric_keys)
            if d:
                data.setdefault(v, {})[f] = d
    if not data:
        sys.exit(f"no evaluation found under {root}")

    complete = [v for v in args.variants if v in data and len(data[v]) == len(args.folds)]
    missing = {v: sorted(set(args.folds) - set(data.get(v, {}))) for v in args.variants}
    missing = {v: f for v, f in missing.items() if f}
    if missing:
        print("[warning] missing folds: " + "; ".join(f"{v}: {','.join(f)}" for v, f in missing.items()))
        print("          the final selection is valid only with all folds of all variants\n")

    images = None
    for v in complete:
        s = set().union(*(set(d) for d in data[v].values()))
        images = s if images is None else images & s
    images = sorted(images or [])

    rows, per_fold = [], []
    for v in args.variants:
        if v not in data:
            continue
        row = {"variant": v, "description": LABELS.get(v, v), "folds": len(data[v])}
        oof = {}
        for f, d in data[v].items():
            oof.update(d)
            per_fold.append({"variant": v, "fold": f, "n_images": len(d),
                             **{n: round(float(np.mean([x[m] for x in d.values() if m in x])), 4)
                                for m, n in METRICS}})
        row["n_images"] = len(oof)
        for m, n in METRICS:
            row[f"{n} (out of fold)"] = round(float(np.mean([x[m] for x in oof.values() if m in x])), 4)
        pq_folds = [np.mean([x["pq"] for x in data[v][f].values()]) for f in args.folds if f in data[v]]
        for f in args.folds:
            row[f"PQ fold {f}"] = round(float(np.mean([x["pq"] for x in data[v][f].values()])), 4) \
                if f in data[v] else ""
        row["PQ mean over folds"] = round(float(np.mean(pq_folds)), 4)
        row["PQ SD over folds"] = round(float(np.std(pq_folds, ddof=1)), 4) if len(pq_folds) > 1 else ""
        rows.append(row)

    candidates = [r for r in rows if r["variant"] in complete and r["variant"] != "no_refiner"]
    chosen = max(candidates, key=lambda r: r["PQ (out of fold)"])["variant"] if candidates else None

    wins = {v: 0 for v in complete if v != "no_refiner"}
    for f in args.folds:
        vals = {v: np.mean([x["pq"] for x in data[v][f].values()]) for v in wins}
        if vals:
            wins[max(vals, key=vals.get)] += 1
    for r in rows:
        r["folds won (PQ)"] = wins.get(r["variant"], "")
        r["selected"] = "yes" if r["variant"] == chosen else ""

    tests = []
    if chosen and images:
        others = [v for v in complete if v != chosen]
        oof = {v: {k: x for d in data[v].values() for k, x in d.items()} for v in complete}
        for m in PAIRED:
            block = []
            for o in others:
                d = np.array([oof[chosen][i][m] - oof[o][i][m] for i in images])
                boot = d[rng.integers(0, len(d), size=(10000, len(d)))].mean(1)
                block.append({"metric": m, "selected": chosen, "versus": o, "n_images": len(d),
                              "mean_advantage": round(float(d.mean()), 4),
                              "ci95_low": round(float(np.percentile(boot, 2.5)), 4),
                              "ci95_high": round(float(np.percentile(boot, 97.5)), 4),
                              "better_in": int((d > 0).sum()), "p_wilcoxon": wilcoxon_p(d)})
            for b, ph in zip(block, holm([x["p_wilcoxon"] if x["p_wilcoxon"] == x["p_wilcoxon"] else 1.0
                                          for x in block])):
                b["p_holm"] = round(float(ph), 4)
                b["p_wilcoxon"] = round(b["p_wilcoxon"], 4) if b["p_wilcoxon"] == b["p_wilcoxon"] else ""
                b["significant"] = "yes" if ph < 0.05 else "no"
            tests += block

    for name, table in (("refiner_selection_cv.csv", rows), ("refiner_selection_cv_per_fold.csv", per_fold),
                        ("refiner_selection_cv_paired.csv", tests)):
        if table:
            with open(root / name, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(table[0]))
                w.writeheader()
                w.writerows(table)

    print(f"Refiner selection by cross-validation ({len(images)} out-of-fold images)")
    for r in rows:
        print(f"  {r['variant']:>11s}: PQ_oof={r['PQ (out of fold)']:.4f}  SQ_oof={r['SQ (out of fold)']:.4f}  "
              f"AJI_oof={r['AJI (out of fold)']:.4f}  BIoU_oof={r['Boundary IoU (out of fold)']:.4f}  "
              f"PQ folds={r['PQ mean over folds']} +- {r['PQ SD over folds']}  "
              f"won={r['folds won (PQ)']} {'<- selected' if r['selected'] else ''}")
    if tests:
        print(f"\nPaired image-level tests: {chosen} versus each variant (> 0 = {chosen} better; * p_Holm < 0.05)")
        for t in tests:
            print(f"  {t['metric']:>13s} vs {t['versus']:>11s}: {t['mean_advantage']:+.4f} "
                  f"[{t['ci95_low']:+.4f}, {t['ci95_high']:+.4f}]  better in {t['better_in']}/{t['n_images']}  "
                  f"p_Holm={t['p_holm']}{' *' if t['significant'] == 'yes' else ''}")
    if chosen:
        print(f"\nRULE (highest out-of-fold PQ): {chosen} ({LABELS.get(chosen, chosen)}).")
        print("Same variant as originally chosen on the test set." if chosen == args.chosen_on_test else
              f"Differs from the variant chosen on the test set ({args.chosen_on_test}).")
    print(f"\nTables in {root}")


if __name__ == "__main__":
    main()
