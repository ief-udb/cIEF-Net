#!/usr/bin/env python3
"""Evaluate a trained checkpoint without retraining.

The images to evaluate are selected from a partition manifest: ``--subset``
is matched against the ``subset`` column of a few-shot manifest or the ``fold``
column of a source-domain manifest (e.g. ``test``).

Examples::

    # cIEF-Net (base + containment + refiner) on the A. alternata test set
    python scripts/evaluate.py --checkpoint runs/cief_net_s42/checkpoint.pt \\
        --images-dir data/alternaria_alternata/images \\
        --split-csv data/splits/alternaria_alternata.csv --subset test \\
        --paper-postprocessing --out runs/cief_net_s42/eval_test

    # ablation on the same checkpoint: refiner loaded but not applied
    python scripts/evaluate.py ... --paper-postprocessing --no-refine --out .../eval_no_refiner

For few-shot checkpoints, the calibrated working side stored in their
``summary.json`` is used unless ``--ref-side`` is given.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cief_net.calibration import ref_side_from_run  # noqa: E402
from cief_net.cli import (add_architecture_args, add_inference_args, add_paper_containment_flag,  # noqa: E402
                          apply_paper_postprocessing, inference_config, load_model_for_inference,
                          resolve_device)
from cief_net.inference import evaluate_images  # noqa: E402
from cief_net.io import REF_SIDE_PX, find_image_annotation_pairs, names_in_subset, read_partition  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--split-csv", required=True, help="partition manifest (fold or subset layout)")
    ap.add_argument("--subset", default="test", help="subset or fold to evaluate")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default=None)
    add_architecture_args(ap)
    add_inference_args(ap, ref_side_default=None)
    add_paper_containment_flag(ap)
    args = ap.parse_args()
    apply_paper_postprocessing(args)

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        sys.exit(f"--checkpoint does not exist: {ckpt}")
    if args.ref_side is None:
        args.ref_side = ref_side_from_run(ckpt, REF_SIDE_PX)
    print(f"ref_side = {args.ref_side}")

    device = resolve_device(args.device)
    pairs = find_image_annotation_pairs(args.images_dir)
    names = names_in_subset(read_partition(args.split_csv), args.subset)
    missing = [n for n in names if n not in pairs]
    if missing:
        sys.exit(f"{len(missing)} images of the manifest have no image+JSON pair (e.g. {missing[:3]})")
    print(f"evaluating {len(names)} images of '{args.subset}'")

    model, arch = load_model_for_inference(args, ckpt, device)
    cfg = inference_config(args)
    summary, per_image = evaluate_images(model, names, pairs, device, cfg, max_det=args.max_det)
    summary["refiner"] = ("off" if not arch["use_refiner"] else
                          "loaded, not applied" if args.no_refine else "on")
    if arch["use_refiner"]:
        summary["refiner_type"] = arch["refiner_type"]
        summary["refiner_kmax"] = arch["refiner_kmax"]
    summary["ref_side"] = args.ref_side
    if cfg.containment is not None:
        summary["containment"] = {"tau": cfg.containment, "rho": cfg.containment_area_ratio,
                                  "protect_merges": cfg.protect_merges}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    per_image.to_csv(out / "per_image_metrics.csv", index=False)
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nAJI={summary['aji_mean']:.4f}  PQ={summary['pq_mean']:.4f}  DQ={summary['dq_mean']:.4f}  "
          f"SQ={summary['sq_mean']:.4f}  Boundary IoU={summary['boundary_iou_mean']:.4f}  "
          f"mAP50={summary.get('map_50', float('nan')):.4f}  count MAE={summary['count_mae']:.2f}")
    print(f"Saved to {out.resolve()}")


if __name__ == "__main__":
    main()
