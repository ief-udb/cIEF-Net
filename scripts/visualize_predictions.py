#!/usr/bin/env python3
"""Side-by-side figures of annotations (left) and predictions (right).

Uses exactly the inference of ``scripts/evaluate.py``. Useful as a quick
visual check, e.g. with ``--max-images 10`` before a full evaluation.

Example::

    python scripts/visualize_predictions.py --checkpoint runs/cief_net_s42/checkpoint.pt \\
        --images-dir data/alternaria_alternata/images \\
        --split-csv data/splits/alternaria_alternata.csv --subset test \\
        --paper-postprocessing --max-images 10 --out runs/cief_net_s42/figures
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cief_net.calibration import ref_side_from_run  # noqa: E402
from cief_net.cli import (add_architecture_args, add_inference_args, add_paper_containment_flag,  # noqa: E402
                          apply_paper_postprocessing, inference_config, load_model_for_inference,
                          resolve_device)
from cief_net.inference import predict_annotated_image  # noqa: E402
from cief_net.io import REF_SIDE_PX, find_image_annotation_pairs, names_in_subset, read_partition  # noqa: E402


def instance_overlay(masks, height, width, alpha=0.45):
    """RGBA overlay with one colour per instance (cycling the tab20 palette)."""
    out = np.zeros((height, width, 4), dtype=np.float32)
    cmap = plt.get_cmap("tab20")
    for i, m in enumerate(masks):
        out[m, :3] = cmap(i % 20)[:3]
        out[m, 3] = alpha
    return out


def plot_comparison(name, image, gt_masks, pred_masks, out_path, dpi=110):
    h, w = image.shape[:2]
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    for ax, masks, title in ((axes[0], gt_masks, "Annotation"), (axes[1], pred_masks, "Prediction")):
        ax.imshow(image)
        ax.imshow(instance_overlay(masks, h, w))
        ax.set_title(f"{title} (n={len(masks)})")
        ax.axis("off")
    fig.suptitle(name)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--split-csv", required=True)
    ap.add_argument("--subset", default="test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--dpi", type=int, default=110)
    ap.add_argument("--device", default=None)
    add_architecture_args(ap)
    add_inference_args(ap, ref_side_default=None)
    add_paper_containment_flag(ap)
    args = ap.parse_args()
    apply_paper_postprocessing(args)
    if args.ref_side is None:
        args.ref_side = ref_side_from_run(args.checkpoint, REF_SIDE_PX)

    device = resolve_device(args.device)
    pairs = find_image_annotation_pairs(args.images_dir)
    names = names_in_subset(read_partition(args.split_csv), args.subset)[:args.max_images]
    model, _ = load_model_for_inference(args, args.checkpoint, device)
    cfg = inference_config(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in names:
        image, gt_masks, _, pred = predict_annotated_image(model, name, pairs, device, cfg)
        plot_comparison(name, image, gt_masks, list(pred["masks"]),
                        out / f"{Path(name).stem}_comparison.png", dpi=args.dpi)
        print(f"  {name}: {len(gt_masks)} annotated, {len(pred['masks'])} predicted")
    print(f"Saved to {out.resolve()}")


if __name__ == "__main__":
    main()
