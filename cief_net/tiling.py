"""Sliding-window tiling for large, dense micrographs.

Images are first rescaled to a common working scale (longest side
``ref_side``) and then processed in overlapping square tiles, in the spirit of
slicing-aided inference (SAHI). Without tiling, the internal resizing of most
detectors would shrink high-resolution micrographs until the smallest conidia
span only a few pixels.

This module contains geometry only (NumPy/PIL) and can be tested without a
model or a GPU. It serves two purposes:

1. Training: :func:`crop_instances_to_window` clips the annotation polygons
   (already in working-scale image coordinates) to a tile and rasterizes the
   instance masks.
2. Inference: :func:`merge_tile_predictions` merges the predictions of all
   tiles of an image into a single full-image result, resolving duplicates in
   the overlap regions with box non-maximum suppression.
"""

import numpy as np
from PIL import Image, ImageDraw

try:
    import torch
    from torchvision.ops import nms as _tv_nms
    _HAS_TORCH = True
except ImportError:  # geometry helpers remain usable without torch
    _HAS_TORCH = False


# =============================================================================
# Windows
# =============================================================================

def generate_windows(width: int, height: int, tile_size: int, overlap: float):
    """Top-left corners ``(x0, y0)`` of windows covering the image with the
    requested fractional overlap. The last window of each axis is aligned with
    the image border. If the image is smaller than ``tile_size`` along an axis,
    a single window at 0 is returned for that axis (padding is handled by the
    consumer)."""
    stride = max(1, int(round(tile_size * (1 - overlap))))

    def _coords(dim):
        if dim <= tile_size:
            return [0]
        coords = list(range(0, dim - tile_size + 1, stride))
        last = dim - tile_size
        if coords[-1] != last:
            coords.append(last)
        return coords

    xs, ys = _coords(width), _coords(height)
    return [(x0, y0) for y0 in ys for x0 in xs]


def crop_pixels(image: np.ndarray, x0: int, y0: int, tile_size: int) -> np.ndarray:
    """Crop ``[y0:y0+tile, x0:x0+tile]``, zero-padding outside the image."""
    h, w = image.shape[:2]
    x1, y1 = min(x0 + tile_size, w), min(y0 + tile_size, h)
    crop = image[y0:y1, x0:x1]
    if crop.shape[0] == tile_size and crop.shape[1] == tile_size:
        return crop
    canvas = np.zeros((tile_size, tile_size) + image.shape[2:], dtype=image.dtype)
    canvas[:crop.shape[0], :crop.shape[1], ...] = crop
    return canvas


# =============================================================================
# Polygon area (shoelace formula) and clipping/rasterization to a window
# =============================================================================

def polygon_area(poly: list) -> float:
    """Area of a simple polygon given as ``[[x, y], ...]`` (shoelace formula)."""
    pts = np.asarray(poly, dtype=float)
    if len(pts) < 3:
        return 0.0
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def crop_instances_to_window(polygons: list, x0: int, y0: int, tile_size: int,
                             min_area_frac: float = 0.3):
    """Rasterize every polygon on a ``tile_size x tile_size`` canvas in tile
    coordinates; the canvas itself performs the clipping. An instance is
    discarded if nothing of it falls inside the tile, or if it keeps less than
    ``min_area_frac`` of its original area (truncated instances at the tile
    border would otherwise teach the model nearly invisible fragments).

    Returns a list of ``(mask (tile, tile) bool, bbox [x0, y0, x1, y1])``.
    """
    out = []
    for poly in polygons:
        area_full = polygon_area(poly)
        if area_full <= 0:
            continue
        local = [(x - x0, y - y0) for x, y in poly]
        canvas = Image.new("L", (tile_size, tile_size), 0)
        ImageDraw.Draw(canvas).polygon(local, outline=1, fill=1)
        mask = np.array(canvas, dtype=bool)

        area_clip = mask.sum()
        if area_clip == 0 or area_clip / area_full < min_area_frac:
            continue
        ys, xs = np.where(mask)
        bbox = [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]
        out.append((mask, bbox))
    return out


# =============================================================================
# Merging the predictions of several tiles
# =============================================================================

def shift_prediction(pred: dict, x0: int, y0: int, width: int, height: int) -> dict:
    """Move a tile prediction (``boxes`` (N, 4), ``scores`` (N,), ``labels``
    (N,), ``masks`` (N, tile, tile) bool) from tile to full-image coordinates.
    Masks are pasted on full-image canvases so that masks from different tiles
    can be compared during merging."""
    n = len(pred["boxes"])
    boxes = pred["boxes"].copy()
    if n > 0:
        boxes[:, [0, 2]] += x0
        boxes[:, [1, 3]] += y0
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height)

    masks_full = np.zeros((n, height, width), dtype=bool)
    tile_h, tile_w = pred["masks"].shape[1:] if n > 0 else (0, 0)
    for i in range(n):
        y1 = min(y0 + tile_h, height)
        x1 = min(x0 + tile_w, width)
        masks_full[i, y0:y1, x0:x1] = pred["masks"][i, :y1 - y0, :x1 - x0]

    return {"boxes": boxes, "scores": pred["scores"], "labels": pred["labels"],
            "masks": masks_full}


def merge_tile_predictions(predictions: list, height: int, width: int,
                           iou_thresh: float = 0.5):
    """Concatenate the shifted predictions of all tiles and remove duplicates
    from overlapping tiles with box NMS: a conidium detected in two
    overlapping tiles must be counted once."""
    boxes_l, scores_l, labels_l, masks_l = [], [], [], []
    for p in predictions:
        if len(p["boxes"]) == 0:
            continue
        boxes_l.append(p["boxes"])
        scores_l.append(p["scores"])
        labels_l.append(p["labels"])
        masks_l.append(p["masks"])

    if not boxes_l:
        return {"boxes": np.zeros((0, 4)), "scores": np.zeros((0,)),
                "labels": np.zeros((0,), dtype=np.int64),
                "masks": np.zeros((0, height, width), dtype=bool)}

    boxes = np.concatenate(boxes_l, axis=0)
    scores = np.concatenate(scores_l, axis=0)
    labels = np.concatenate(labels_l, axis=0)
    masks = np.concatenate(masks_l, axis=0)

    if not _HAS_TORCH:
        raise ImportError("merge_tile_predictions requires torch/torchvision for NMS")
    keep = _tv_nms(torch.as_tensor(boxes, dtype=torch.float32),
                   torch.as_tensor(scores, dtype=torch.float32), iou_thresh).numpy()
    return {"boxes": boxes[keep], "scores": scores[keep], "labels": labels[keep],
            "masks": masks[keep]}
