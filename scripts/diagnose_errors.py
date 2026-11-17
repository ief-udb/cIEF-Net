#!/usr/bin/env python3
"""Decompose the errors of a checkpoint without retraining.

Runs exactly the inference of ``scripts/evaluate.py`` and compares every
prediction with the annotations (see :mod:`cief_net.diagnostics` for the
quantities). This is the analysis behind the spectral decomposition of the
contour error and the stratified results of the paper.

Outputs in ``--out``:

``errors_per_gt.csv``          one row per annotated conidium
``errors_per_prediction.csv``  one row per prediction
``gt_overlap_pairs.csv``       pairs of overlapping annotations
``error_summary.json``         aggregates and stratified tables
``containment_sweep.csv``      (``--sweep-containment``) sensitivity of the
                               suppression thresholds, evaluated on the same
                               unsuppressed predictions

Example::

    python scripts/diagnose_errors.py --checkpoint runs/cief_net_s42/checkpoint.pt \\
        --images-dir data/alternaria_alternata/images \\
        --split-csv data/splits/alternaria_alternata.csv --subset test \\
        --paper-postprocessing --out runs/cief_net_s42/diagnosis
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cief_net import diagnostics as dg  # noqa: E402
from cief_net.calibration import ref_side_from_run  # noqa: E402
from cief_net.cli import (add_architecture_args, add_inference_args, add_paper_containment_flag,  # noqa: E402
                          apply_paper_postprocessing, inference_config, load_model_for_inference,
                          resolve_device)
from cief_net.containment import apply_containment, decide_suppression, pairwise_overlaps  # noqa: E402
from cief_net.inference import predict_annotated_image  # noqa: E402
from cief_net.io import REF_SIDE_PX, find_image_annotation_pairs, names_in_subset, read_partition  # noqa: E402

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable


def print_report(res):
    s = res["shape_of_matched"]
    print("\n================ ERROR DECOMPOSITION ================")
    print(f"images={res['n_images']}  GT={res['n_gt']}  pred={res['n_pred']}  matched(IoU>0.5)={res['n_matched']}")
    print(f"SQ (mean per image) = {res['SQ_mean_per_image']:.4f}   (equals sq_mean of the evaluation)")
    print(f"SQ global = {res['SQ_global']:.4f}   DQ global = {res['DQ_global']:.4f}   "
          f"PQ global = {res['PQ_global']:.4f}")
    print("\n-- shape of matched conidia --")
    print(f"  mean log(area_pred/area_gt) : {s['log_area_ratio_mean']:+.4f}  "
          f"({100 * s['frac_smaller_predictions']:.1f}% of predictions smaller than the annotation)")
    print(f"  radial bias / radius        : {s['rel_radial_bias_mean']:+.4f}  (systematic)")
    print(f"  radial noise / radius       : {s['rel_radial_noise_mean']:.4f}   (contour jitter)")
    print(f"  displacement / radius       : {s['rel_displacement_mean']:.4f}")
    print(f"  IoU gained if centred       : {s['iou_gain_if_centred']:+.4f}")
    print("\n-- detection --")
    print(f"  unmatched GT        : {res['unmatched_gt']}")
    print(f"  unmatched predictions: {res['unmatched_pred']}")
    for key, title in (("by_size", "SIZE"), ("by_elongation", "ELONGATION"), ("by_neighbours", "NEIGHBOURS")):
        if res[key]:
            print(f"\n-- by {title} (quartiles) --")
            for k, v in res[key].items():
                sq = f"{v['sq']:.4f}" if v["sq"] is not None else "   -  "
                print(f"  {k:32s} n={v['n_gt']:6d}  recall={v['recall_iou50']:.3f}  SQ={sq}")
    spec = res.get("spectrum_and_oracle")
    if spec:
        bands = [n for n, _, _ in dg.SPECTRAL_BANDS]
        print("\n-- radial error spectrum (fraction of mean dr^2) and oracle IoU gains --")
        print(f"  {'stratum':26s} {'n':>5s} {'rms':>6s} " + " ".join(f"{b[:9]:>9s}" for b in bands)
              + "  " + " ".join(f"{'+k<=' + str(K):>7s}" for K in dg.ORACLE_K))
        for name, b in spec.items():
            fr = " ".join(f"{b['energy_fraction'][x]:9.3f}" for x in bands)
            ga = " ".join(f"{b['oracle_gain'][f'k<={K}']:+7.4f}" for K in dg.ORACLE_K)
            print(f"  {name:26s} {b['n']:5d} {b['rms_rel_radial_error']:6.3f} {fr}  {ga}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--split-csv", required=True)
    ap.add_argument("--subset", default="test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--duplicate-thresh", type=float, default=0.90,
                    help="IoU (or containment) between two annotations flagged as a possible duplicate")
    ap.add_argument("--sweep-containment", action="store_true")
    ap.add_argument("--sweep-taus", default="0.6,0.7,0.8,0.9")
    ap.add_argument("--sweep-rhos", default="0.5,0.7")
    ap.add_argument("--device", default=None)
    add_architecture_args(ap)
    add_inference_args(ap, ref_side_default=None)
    add_paper_containment_flag(ap)
    args = ap.parse_args()
    apply_paper_postprocessing(args)

    ckpt = Path(args.checkpoint)
    if args.ref_side is None:
        args.ref_side = ref_side_from_run(ckpt, REF_SIDE_PX)
    device = resolve_device(args.device)
    pairs = find_image_annotation_pairs(args.images_dir)
    names = names_in_subset(read_partition(args.split_csv), args.subset)[:args.max_images]
    model, _ = load_model_for_inference(args, ckpt, device)

    # predictions are produced WITHOUT suppression; it is applied afterwards so
    # that the sweep and the diagnosis share the same raw predictions
    cfg = inference_config(args)
    containment = dict(containment_thresh=cfg.containment, max_area_ratio=cfg.containment_area_ratio,
                       protect_merges=cfg.protect_merges)
    use_containment, cfg.containment = cfg.containment is not None, None

    configs = [None]
    if args.sweep_containment:
        configs += [(t, r, pm) for t in map(float, args.sweep_taus.split(","))
                    for r in map(float, args.sweep_rhos.split(",")) for pm in (False, True)]
    acc = {c: [] for c in configs}
    removed = {"n_fragments_removed": 0, "n_merges_removed": 0}

    rows_gt, rows_pr, gt_pairs = [], [], []
    with torch.no_grad():
        for name in tqdm(names, desc="diagnosing", unit="img"):
            _, gt_masks, _, pred = predict_annotated_image(model, name, pairs, device, cfg)
            if args.sweep_containment:
                dg.sweep_containment_image(name, gt_masks, list(pred["masks"]), configs, acc,
                                           pairwise_overlaps, decide_suppression)
            if use_containment:
                pred, info = apply_containment(pred, **containment)
                for k in removed:
                    removed[k] += info[k]
            scores = np.asarray(pred["scores"]).tolist()
            g, p, gp = dg.analyze_image(name, gt_masks, list(pred["masks"]), scores,
                                        dup_thresh=args.duplicate_thresh)
            rows_gt += g
            rows_pr += p
            gt_pairs += gp

    df_gt, df_pr = pd.DataFrame(rows_gt), pd.DataFrame(rows_pr)
    for c in ("iou", "log_area_ratio", "frac_missing", "frac_excess", "rel_displacement",
              "rel_radial_bias", "rel_radial_noise", "iou_centred"):
        if c not in df_gt:
            df_gt[c] = np.nan
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df_gt.to_csv(out / "errors_per_gt.csv", index=False)
    df_pr.to_csv(out / "errors_per_prediction.csv", index=False)
    df_pairs = pd.DataFrame(gt_pairs, columns=["image", "gt_a", "gt_b", "iou", "containment",
                                               "area_a", "area_b", "type"])
    df_pairs.sort_values(["type", "iou"], ascending=[True, False]).to_csv(out / "gt_overlap_pairs.csv",
                                                                          index=False)
    res = dg.summarize(df_gt, df_pr)
    res["gt_overlaps"] = dg.summarize_overlaps(df_pairs, df_gt, args.duplicate_thresh)
    res["checkpoint"], res["ref_side"] = str(ckpt), args.ref_side
    if use_containment:
        res["containment"] = {**containment, **removed}
    if args.sweep_containment:
        rows = dg.sweep_rows(acc)
        pd.DataFrame(rows).to_csv(out / "containment_sweep.csv", index=False)
        res["containment_sweep"] = rows
    with open(out / "error_summary.json", "w") as f:
        json.dump(res, f, indent=2)
    print_report(res)
    print(f"\nSaved to {out.resolve()}")


if __name__ == "__main__":
    main()
