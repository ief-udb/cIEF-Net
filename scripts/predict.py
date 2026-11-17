#!/usr/bin/env python3
"""Segment conidia in new (unannotated) images.

For every image in ``--input`` (a file or a folder, searched recursively) the
script writes to ``--out``:

* ``<stem>_labels.png``: 16-bit instance label image at the original
  resolution (0 = background, 1..N = conidia);
* ``<stem>_overlay.png`` (``--save-overlay``): instances drawn on the image;
* ``<stem>.json`` (``--labelme``): predicted polygons in LabelMe format at the
  original resolution, ready for manual correction;
* ``instances.csv``: one row per conidium with score, area, equivalent
  diameter, centroid and elongation (pixels of the original image);
* ``counts.csv``: number of conidia per image.

The paper post-processing (containment suppression with tau = 0.8, rho = 0.7
and merge protection) is applied by default. The working scale defaults to
the calibrated side stored with a few-shot checkpoint, or 1536 pixels.

Example::

    python scripts/predict.py --checkpoint checkpoints/cief_net_alternaria.pt \\
        --input my_micrographs/ --out predictions/ --save-overlay --labelme
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cief_net.calibration import ref_side_from_run  # noqa: E402
from cief_net.cli import (add_architecture_args, add_inference_args, inference_config,  # noqa: E402
                          load_model_for_inference, resolve_device)
from cief_net.inference import predict_image  # noqa: E402
from cief_net.io import IMAGE_EXTENSIONS, REF_SIDE_PX, find_images, load_image_rescaled  # noqa: E402

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable


def masks_to_label_image(masks: np.ndarray, scores: np.ndarray, shape) -> np.ndarray:
    """Paint instances in ascending score order so that the highest-scoring
    instance wins any residual overlap."""
    labels = np.zeros(shape, dtype=np.uint16)
    for rank, i in enumerate(np.argsort(scores, kind="stable"), start=1):
        labels[masks[i]] = rank
    return labels


def instance_table(labels: np.ndarray, scores_by_label: dict, image_name: str) -> list:
    rows = []
    for lab in np.unique(labels):
        if lab == 0:
            continue
        ys, xs = np.nonzero(labels == lab)
        area = len(xs)
        elong = float("nan")
        if area >= 3:
            ev = np.linalg.eigvalsh(np.cov(np.stack([xs, ys])))
            if ev[0] > 1e-9:
                elong = float(np.sqrt(ev[1] / ev[0]))
        rows.append({"image": image_name, "instance": int(lab), "score": scores_by_label.get(int(lab)),
                     "area_px": area, "equivalent_diameter_px": 2.0 * math.sqrt(area / math.pi),
                     "centroid_x": float(xs.mean()), "centroid_y": float(ys.mean()),
                     "elongation": elong})
    return rows


def labelme_polygons(masks: np.ndarray, factor: float, tolerance: float = 0.5) -> list:
    """Outer contour of every working-scale mask, mapped to original pixels."""
    from skimage.measure import approximate_polygon, find_contours
    shapes = []
    for m in masks:
        ys, xs = np.nonzero(m)
        if len(xs) == 0:
            continue
        y0, x0 = ys.min(), xs.min()
        crop = np.pad(m[y0:ys.max() + 1, x0:xs.max() + 1], 1)
        contours = find_contours(crop.astype(np.uint8), 0.5)
        if not contours:
            continue
        c = approximate_polygon(max(contours, key=len), tolerance)
        pts = [[float((x + x0 - 1) / factor), float((y + y0 - 1) / factor)] for y, x in c[:-1]]
        if len(pts) >= 3:
            shapes.append({"label": "conidium", "points": pts, "group_id": None,
                           "shape_type": "polygon", "flags": {}})
    return shapes


def overlay(image: np.ndarray, labels: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    rng = np.random.default_rng(0)
    colours = rng.integers(40, 256, size=(int(labels.max()) + 1, 3)).astype(np.float32)
    out = image.astype(np.float32).copy()
    fg = labels > 0
    out[fg] = (1 - alpha) * out[fg] + alpha * colours[labels[fg]]
    return out.clip(0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--input", required=True, help="an image file or a folder of images")
    ap.add_argument("--out", required=True)
    ap.add_argument("--save-overlay", action="store_true")
    ap.add_argument("--labelme", action="store_true", help="also export LabelMe JSON polygons")
    ap.add_argument("--no-paper-postprocessing", action="store_true",
                    help="disable containment suppression (enabled by default here)")
    ap.add_argument("--device", default=None)
    add_architecture_args(ap)
    add_inference_args(ap, ref_side_default=None)
    args = ap.parse_args()
    if not args.no_paper_postprocessing and args.containment is None:
        args.containment, args.containment_area_ratio, args.protect_merges = 0.8, 0.7, True

    src = Path(args.input)
    images = [src] if src.is_file() and src.suffix.lower() in IMAGE_EXTENSIONS else find_images(src)
    if not images:
        sys.exit(f"no image found in {src}")
    if args.ref_side is None:
        args.ref_side = ref_side_from_run(args.checkpoint, REF_SIDE_PX)
    device = resolve_device(args.device)
    print(f"{len(images)} images | ref_side={args.ref_side} | device={device}")

    model, _ = load_model_for_inference(args, args.checkpoint, device)
    cfg = inference_config(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    all_rows, counts = [], []
    for path in tqdm(images, desc="predicting", unit="img"):
        image, factor = load_image_rescaled(path, cfg.ref_side)
        pred = predict_image(model, image, device, cfg)
        masks, scores = pred["masks"], np.asarray(pred["scores"], dtype=float)

        with Image.open(path) as im:
            w0, h0 = im.size
            original = np.array(im.convert("RGB")) if args.save_overlay else None
        labels_work = masks_to_label_image(masks, scores, image.shape[:2])
        labels = np.array(Image.fromarray(labels_work).resize((w0, h0), Image.NEAREST))
        order = np.argsort(scores, kind="stable")
        scores_by_label = {rank: float(scores[i]) for rank, i in enumerate(order, start=1)}

        Image.fromarray(labels).save(out / f"{path.stem}_labels.png")
        if args.save_overlay:
            Image.fromarray(overlay(original, labels)).save(out / f"{path.stem}_overlay.png")
        if args.labelme:
            with open(out / f"{path.stem}.json", "w") as f:
                json.dump({"version": "5.0.0", "flags": {}, "shapes": labelme_polygons(masks, factor),
                           "imagePath": path.name, "imageData": None,
                           "imageHeight": h0, "imageWidth": w0}, f, indent=1)
        rows = instance_table(labels, scores_by_label, path.name)
        all_rows += rows
        counts.append({"image": path.name, "n_conidia": len(rows)})

    for name, rows in (("instances.csv", all_rows), ("counts.csv", counts)):
        if rows:
            with open(out / name, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
    print(f"{sum(c['n_conidia'] for c in counts)} conidia in {len(counts)} images; saved to {out.resolve()}")


if __name__ == "__main__":
    main()
