#!/usr/bin/env python3
"""Train cIEF-Net.

Modes
-----
``holdout``  train on folds 0-3 of ``--split-csv`` and evaluate on ``test``
             (the source-domain benchmark).
``cv``       four-fold cross-validation on folds 0-3 (``test`` is not used).
``fewshot``  fine-tune from a source-domain checkpoint (``--init-checkpoint``)
             on the support set of ``--fewshot-csv`` and evaluate on its test
             set, with support-derived scale calibration.

Every run writes to ``--out``: ``checkpoint.pt`` (last epoch, used for every
reported result), ``model_config.json``, ``summary.json``,
``per_image_metrics.csv``, ``history.json`` and ``diameter_calibration.json``.

Paper configuration (see ``experiments/``)::

    # stage 1: base network
    python scripts/train.py --mode holdout --images-dir data/alternaria_alternata/images \\
        --split-csv data/splits/alternaria_alternata.csv --out runs/base_s42 \\
        --epochs 30 --lr-t-max 60 --seed 42

    # stage 2: harmonic refiner (K = 24) on the frozen base
    python scripts/train.py --mode holdout ... --out runs/cief_net_s42 \\
        --init-checkpoint runs/base_s42/checkpoint.pt --use-refiner --refiner-only \\
        --epochs 15 --lr-new 3e-4 --paper-postprocessing
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

from cief_net.augmentations import AugConfig  # noqa: E402
from cief_net.calibration import (CALIBRATION_NAME, calibrate_ref_side, find_calibration,  # noqa: E402
                                  save_calibration)
from cief_net.checkpoints import save_model_config  # noqa: E402
from cief_net.cli import (add_architecture_args, add_inference_args, add_paper_containment_flag,  # noqa: E402
                          apply_paper_postprocessing, arch_from_args, parse_target_modules,
                          resolve_device)
from cief_net.data import TRAIN_FOLDS, build_loaders_cv, build_loaders_holdout  # noqa: E402
from cief_net.engine import train_run  # noqa: E402
from cief_net.io import (find_image_annotation_pairs, fewshot_to_fold_manifest, names_in_folds,  # noqa: E402
                         names_in_subset, read_partition)


def _loader_kwargs(args, aug):
    return dict(tile_size=args.tile_size, overlap=args.overlap, ref_side=args.ref_side,
                min_area_frac=args.min_area_frac, n_rays=args.n_rays, grid_stride=args.grid_stride,
                batch_size=args.batch_size, num_workers=args.num_workers,
                augment=not args.no_augmentation, aug_config=aug)


def _write_outputs(out_dir: Path, summary, per_image, history, args):
    summary.setdefault("ref_side", args.ref_side)
    per_image.to_csv(out_dir / "per_image_metrics.csv", index=False)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    save_model_config(out_dir, arch_from_args(args))


def run_holdout(args, device):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    aug = AugConfig.disabled() if args.no_augmentation else AugConfig()
    loader, names_test, pairs = build_loaders_holdout(args.images_dir, args.split_csv,
                                                      **_loader_kwargs(args, aug))
    if not args.no_scale_calibration:
        save_calibration(names_in_folds(read_partition(args.split_csv), TRAIN_FOLDS), pairs,
                         args.ref_side, out / CALIBRATION_NAME)
    _, summary, per_image, history = train_run(loader, names_test, pairs, args, device, "holdout_test",
                                               checkpoint_path=out / "checkpoint.pt")
    _write_outputs(out, summary, per_image, history, args)
    print(f"\nSaved to {out.resolve()}")


def run_cv(args, device):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    aug = AugConfig.disabled() if args.no_augmentation else AugConfig()
    summaries = []
    for val_fold in TRAIN_FOLDS:
        loader, names_val, pairs = build_loaders_cv(args.images_dir, args.split_csv, val_fold,
                                                    **_loader_kwargs(args, aug))
        label, fold_dir = f"val_fold={val_fold}", out / f"fold_{val_fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        if not args.no_scale_calibration:
            save_calibration(names_in_folds(read_partition(args.split_csv),
                                            [f for f in TRAIN_FOLDS if f != val_fold]),
                             pairs, args.ref_side, fold_dir / CALIBRATION_NAME)
        try:
            model, summary, per_image, history = train_run(loader, names_val, pairs, args, device, label,
                                                           checkpoint_path=fold_dir / "checkpoint.pt")
        except Exception as e:  # one failing fold must not stop the others
            print(f"\n  {label}: FAILED ({type(e).__name__}: {e}); continuing with the next fold\n")
            continue
        _write_outputs(fold_dir, summary, per_image, history, args)
        summaries.append(summary)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not summaries:
        print("\nno fold completed its evaluation")
        return
    df = pd.DataFrame(summaries)
    num = df.select_dtypes(include=[np.number]).columns
    agg = {"mean": df[num].mean().to_dict(), "std": df[num].std().to_dict(),
           "per_fold": df.to_dict(orient="records")}
    with open(out / "summary_cv.json", "w") as f:
        json.dump(agg, f, indent=2)
    print("\n=== 4-fold CV (mean +- SD) ===")
    for col in num:
        print(f"  {col}: {agg['mean'][col]:.4f} +- {agg['std'][col]:.4f}")


def run_fewshot(args, device):
    """Few-shot fine-tuning. Before building the loaders, ``args.ref_side`` is
    replaced by the support-calibrated side, so that training, the periodic
    evaluations and the final evaluation all use the same scale."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pairs = find_image_annotation_pairs(args.images_dir)
    support = names_in_subset(read_partition(args.fewshot_csv), "train")
    if not support:
        sys.exit(f"{args.fewshot_csv} has no support image (subset 'train')")

    ref_side_original = args.ref_side
    if args.no_scale_calibration:
        print(f"  scale calibration disabled; ref_side = {args.ref_side}")
    else:
        reference = args.calibration_reference or find_calibration(args.init_checkpoint)
        if reference is None or not Path(reference).exists():
            sys.exit(f"no diameter calibration found next to {args.init_checkpoint}. Every training "
                     f"run writes one; for older checkpoints create it with "
                     f"scripts/measure_reference_diameter.py and pass --calibration-reference, or use "
                     f"--no-scale-calibration")
        args.ref_side = calibrate_ref_side(args.ref_side, reference, support, pairs)
        # calibration of this checkpoint at its own (calibrated) scale, so that a
        # chained few-shot run can start from it
        save_calibration(support, pairs, args.ref_side, out / CALIBRATION_NAME)

    aug = AugConfig.disabled() if args.no_augmentation else AugConfig()
    split = fewshot_to_fold_manifest(args.fewshot_csv, out / "_fewshot_split_folds.csv")
    loader, names_test, pairs = build_loaders_holdout(args.images_dir, str(split),
                                                      **_loader_kwargs(args, aug))
    _, summary, per_image, history = train_run(loader, names_test, pairs, args, device, "fewshot",
                                               checkpoint_path=out / "checkpoint.pt")
    summary["ref_side_original"] = ref_side_original
    summary["ref_side_calibrated"] = args.ref_side
    _write_outputs(out, summary, per_image, history, args)
    print(f"\nSaved to {out.resolve()}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="holdout", choices=["holdout", "cv", "fewshot"])
    ap.add_argument("--images-dir", required=True, help="folder with images and LabelMe JSON files")
    ap.add_argument("--split-csv", help="source-domain manifest (image,fold); holdout and cv modes")
    ap.add_argument("--fewshot-csv", help="few-shot manifest (image,species,subset); fewshot mode")
    ap.add_argument("--out", required=True)
    ap.add_argument("--init-checkpoint", default=None, help="warm start (required in fewshot mode)")
    ap.add_argument("--calibration-reference", default=None,
                    help="diameter calibration of the initial checkpoint (default: found next to it)")
    ap.add_argument("--no-scale-calibration", action="store_true")
    ap.add_argument("--random-backbone", action="store_true",
                    help="random encoder weights (tests only, never for real training)")

    add_architecture_args(ap)

    g = ap.add_argument_group("optimization")
    g.add_argument("--epochs", type=int, default=30)
    g.add_argument("--batch-size", type=int, default=8)
    g.add_argument("--lr-backbone", type=float, default=1e-5, help="learning rate of the DoRA adapters")
    g.add_argument("--lr-new", type=float, default=1e-4, help="learning rate of decoder, head and refiner")
    g.add_argument("--weight-decay", type=float, default=0.05)
    g.add_argument("--scheduler", default="cosine", choices=["cosine", "step"])
    g.add_argument("--lr-t-max", type=int, default=None, help="cosine period (default: --epochs)")
    g.add_argument("--lr-step", type=int, default=10)
    g.add_argument("--lr-gamma", type=float, default=0.1)
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--num-workers", type=int, default=2)
    g.add_argument("--no-augmentation", action="store_true")
    g.add_argument("--min-area-frac", type=float, default=0.3,
                   help="discard tile instances keeping less than this fraction of their area")

    g = ap.add_argument_group("loss")
    g.add_argument("--radial-loss", default="log", choices=["log", "l1"])
    g.add_argument("--no-boundary-weighting", action="store_true",
                   help="plain radial loss without boundary-proximity weighting (StarDist)")
    g.add_argument("--w-prob", type=float, default=1.0)
    g.add_argument("--w-radial", type=float, default=1.0)
    g.add_argument("--w-density", type=float, default=0.0)
    g.add_argument("--w-offset", type=float, default=0.5)
    g.add_argument("--w-refiner", type=float, default=1.0)

    g = ap.add_argument_group("refiner training")
    g.add_argument("--refiner-only", action="store_true",
                   help="freeze everything except the refiner (stage 2; needs --init-checkpoint)")
    g.add_argument("--freeze-refiner", action="store_true",
                   help="keep the refiner of the initial checkpoint fixed (few-shot ablation)")
    g.add_argument("--reset-refiner", action="store_true",
                   help="discard the refiner weights of the initial checkpoint")
    g.add_argument("--refiner-p-gt", type=float, default=0.3)
    g.add_argument("--refiner-max-instances", type=int, default=512)

    g = ap.add_argument_group("monitoring")
    g.add_argument("--eval-every", type=int, default=5,
                   help="intermediate evaluation every N epochs (0 = final evaluation only)")
    g.add_argument("--best-metric", default="pq_mean")
    g.add_argument("--best-mode", default="max", choices=["max", "min"])
    g.add_argument("--save-best-checkpoint", action="store_true",
                   help="also keep checkpoint_best.pt (never used for reporting)")

    add_inference_args(ap)
    add_paper_containment_flag(ap)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    apply_paper_postprocessing(args)
    args.dora_target_modules = parse_target_modules(args.dora_target_modules)

    if not Path(args.images_dir).exists():
        sys.exit(f"--images-dir does not exist: {args.images_dir}")
    if args.mode == "fewshot":
        if not args.fewshot_csv or not Path(args.fewshot_csv).exists():
            sys.exit("--mode fewshot requires an existing --fewshot-csv")
        if not args.init_checkpoint:
            sys.exit("--mode fewshot requires --init-checkpoint")
    elif not args.split_csv or not Path(args.split_csv).exists():
        sys.exit(f"--mode {args.mode} requires an existing --split-csv")
    if args.refiner_only and not args.use_refiner:
        sys.exit("--refiner-only requires --use-refiner")
    if args.refiner_only and not args.init_checkpoint:
        sys.exit("--refiner-only without --init-checkpoint would train the refiner on an untrained base")
    if args.freeze_refiner and (args.refiner_only or not args.use_refiner):
        sys.exit("--freeze-refiner requires --use-refiner and is incompatible with --refiner-only")
    if args.tile_size % args.patch_size:
        sys.exit(f"--tile-size ({args.tile_size}) must be a multiple of --patch-size ({args.patch_size})")
    if args.adaptive_nms and not args.use_density_head:
        sys.exit("--adaptive-nms requires --use-density-head")

    device = resolve_device(args.device)
    refiner = (f"refiner={args.refiner_type}"
               f"{f' K={args.refiner_kmax}' if args.refiner_type.startswith('harmonic') else ''}"
               f"{' (refiner only)' if args.refiner_only else ''}" if args.use_refiner else "no refiner")
    print(f"device={device} | mode={args.mode} | backbone={args.backbone} | DoRA r={args.dora_r} | "
          f"tile={args.tile_size} stride={args.grid_stride} rays={args.n_rays} | {refiner} | "
          f"radial loss={args.radial_loss} | epochs={args.epochs} | seed={args.seed}"
          + (f" | warm start {args.init_checkpoint}" if args.init_checkpoint else ""))

    {"holdout": run_holdout, "cv": run_cv, "fewshot": run_fewshot}[args.mode](args, device)


if __name__ == "__main__":
    main()
