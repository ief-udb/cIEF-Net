"""Tiled inference and evaluation.

For every tile, the star-convex head is decoded into instances with the
functions of the ``stardist`` package (``non_maximum_suppression`` and
``polygons_to_label``); the refiner, when present, corrects every polygon kept
by NMS before rasterization. Tile predictions are shifted to full-image
coordinates and merged (:mod:`cief_net.tiling`), and containment suppression
(:mod:`cief_net.containment`) optionally removes tile-border fragments.

``polygons_to_label`` assigns label ``i`` to ``points[i-1]``/``probs[i-1]``,
so the score of every final instance is read from ``probs`` by index.

One instance per connected component (enabled by default)
---------------------------------------------------------
An elongated (obclavate) conidium can be split into two detections at its
narrowest part: polygon NMS only suppresses overlapping polygons, and the two
halves may not overlap enough. The region ``prob > prob_thresh`` usually
remains connected between both halves, so at most one detection is kept per
connected component of that region (4-connectivity), the one with the highest
score. In very dense scenes, truly distinct conidia that touch can share a
component; the rule can be disabled (``one_per_component=False``).

Exploratory options (not used by the final model)
-------------------------------------------------
* Density-adaptive NMS (requires the density head), an adaptation of
  Adaptive-NMS (Liu et al., 2019) in which the IoU threshold grows with the
  local predicted density. It never outperformed the fixed-threshold base.
* Sub-grid offset correction of the NMS points (requires the offset head).
"""

from dataclasses import dataclass

import numpy as np
import torch
from scipy import ndimage
from stardist import dist_to_coord, non_maximum_suppression, polygons_to_label

from . import tiling
from .containment import apply_containment
from .data import full_image_instances
from .io import REF_SIDE_PX, load_image_and_polygons
from .metrics import InstanceEvaluator

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable


@dataclass
class InferenceConfig:
    """Inference settings. Defaults reproduce the evaluation of the paper
    except for containment suppression, which must be requested explicitly
    (``containment=0.8, protect_merges=True`` for cIEF-Net)."""
    tile_size: int = 512
    overlap: float = 0.2
    ref_side: int = REF_SIDE_PX
    grid_stride: int = 4
    prob_thresh: float = 0.5
    nms_thresh: float = 0.5
    iou_merge: float = 0.5
    one_per_component: bool = True
    refine: bool = True                   # apply the refiner if the model has one
    containment: float = None             # containment threshold tau (None = off)
    containment_area_ratio: float = 0.7   # rho
    protect_merges: bool = False
    adaptive_nms: bool = False            # exploratory (density head)
    density_ref: object = "auto"
    iou_max_adaptive: float = 0.95


def _empty_prediction(tile_size):
    return {"boxes": np.zeros((0, 4), dtype=np.float32), "scores": np.zeros((0,), dtype=np.float32),
            "labels": np.zeros((0,), dtype=np.int64),
            "masks": np.zeros((0, tile_size, tile_size), dtype=bool)}


def _label_map_to_instances(label_map: np.ndarray, probs: np.ndarray):
    """Masks, boxes and scores from a ``polygons_to_label`` map; instances whose
    mask vanished during rasterization are dropped."""
    masks, boxes, scores = [], [], []
    for e in np.unique(label_map):
        if e == 0:
            continue
        m = label_map == e
        if m.sum() == 0:
            continue
        ys, xs = np.where(m)
        boxes.append([float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)])
        masks.append(m)
        idx = int(e) - 1
        scores.append(float(probs[idx]) if 0 <= idx < len(probs) else 0.0)
    return masks, boxes, scores


# =============================================================================
# Exploratory: density-adaptive NMS and sub-grid offsets
# =============================================================================

def _polygon_mask(coords: np.ndarray, tile_size: int) -> np.ndarray:
    """Rasterize a polygon given as (2, n_rays), row 0 = y, row 1 = x."""
    from PIL import Image, ImageDraw
    canvas = Image.new("L", (tile_size, tile_size), 0)
    ImageDraw.Draw(canvas).polygon(list(zip(coords[1].tolist(), coords[0].tolist())), outline=1, fill=1)
    return np.array(canvas, dtype=bool)


def _mask_iou_matrix(masks: np.ndarray) -> np.ndarray:
    n = masks.shape[0]
    flat = masks.reshape(n, -1).astype(np.float32)
    inter = flat @ flat.T
    areas = flat.sum(axis=1)
    union = areas[:, None] + areas[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(union), where=union > 0)


def density_adaptive_nms(masks: np.ndarray, scores: np.ndarray, local_density: list,
                         iou_base: float = 0.5, iou_max: float = 0.95,
                         density_ref: float = 0.1) -> list:
    """Greedy mask NMS whose IoU threshold rises linearly from ``iou_base`` to
    ``iou_max`` with the local density (normalized by ``density_ref``)."""
    iou = _mask_iou_matrix(masks)
    order = list(np.argsort(-np.asarray(scores)))
    suppressed = [False] * len(scores)
    keep = []
    for idx in order:
        if suppressed[idx]:
            continue
        keep.append(idx)
        c = min(max(local_density[idx] / density_ref, 0.0), 1.0) if density_ref > 0 else 0.0
        thresh = iou_base + (iou_max - iou_base) * c
        for j in order:
            if j != idx and not suppressed[j] and iou[idx, j] > thresh:
                suppressed[j] = True
    return keep


@torch.no_grad()
def calibrate_density_ref(model, names: list, pairs: dict, device, cfg: InferenceConfig,
                          percentile: float = 90.0, max_images: int = 10) -> float:
    """Percentile of the predicted density inside detected instances
    (``prob > prob_thresh``) over a few images; background cells are excluded
    because they would drag the percentile towards zero."""
    model.eval()
    values = []
    for name in names[:max_images]:
        img_path, json_path = pairs[name]
        image, _, _ = load_image_and_polygons(img_path, json_path, cfg.ref_side)
        h, w = image.shape[:2]
        for (x0, y0) in tiling.generate_windows(w, h, cfg.tile_size, cfg.overlap):
            crop = tiling.crop_pixels(image, x0, y0, cfg.tile_size)
            x = torch.as_tensor(crop, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
            out = model(x.to(device))
            if "density" not in out:
                raise RuntimeError("density calibration requires the density head")
            prob = out["prob"][0, 0].cpu().numpy()
            dens = out["density"][0, 0].cpu().numpy()[prob > cfg.prob_thresh]
            if dens.size:
                values.append(dens)
    if not values:
        print("  WARNING: no cell above the probability threshold; density_ref = 0.1")
        return 0.1
    allv = np.concatenate(values)
    value = float(np.percentile(allv, percentile))
    print(f"  density_ref calibrated: {value:.4f} (percentile {percentile}, {len(allv)} cells)")
    return value


def _apply_subgrid_offset(points, offset_map, grid_stride, tile_size):
    h_red, w_red = offset_map.shape[1:]
    half = grid_stride / 2.0
    out = points.astype(np.float32).copy()
    for i, (py, px) in enumerate(points):
        gy, gx = min(int(py) // grid_stride, h_red - 1), min(int(px) // grid_stride, w_red - 1)
        out[i, 0] += offset_map[0, gy, gx] * half
        out[i, 1] += offset_map[1, gy, gx] * half
    out[:, 0] = np.clip(out[:, 0], 0, tile_size - 1)
    out[:, 1] = np.clip(out[:, 1], 0, tile_size - 1)
    return out


# =============================================================================
# Decoding
# =============================================================================

def one_instance_per_component(points: np.ndarray, probs: np.ndarray, prob_map: np.ndarray,
                               grid_stride: int, prob_thresh: float) -> np.ndarray:
    """Indices of the detections to keep: the highest-scoring detection of
    every connected component of ``prob_map > prob_thresh`` (reduced grid,
    4-connectivity). A point whose cell falls on the background (possible
    near a border because of rounding) is kept on its own."""
    n = len(points)
    if n == 0:
        return np.arange(0)
    components, n_comp = ndimage.label(prob_map > prob_thresh)
    if n_comp == 0:
        return np.arange(n)
    h_red, w_red = prob_map.shape
    best, lone = {}, []
    for i in range(n):
        py, px = points[i]
        comp = int(components[min(int(py) // grid_stride, h_red - 1),
                              min(int(px) // grid_stride, w_red - 1)])
        if comp == 0:
            lone.append(i)
            continue
        if comp not in best or probs[i] > probs[best[comp]]:
            best[comp] = i
    return np.array(sorted(set(best.values()) | set(lone)), dtype=np.int64)


@torch.no_grad()
def predict_tile(model, crop: np.ndarray, device, cfg: InferenceConfig,
                 density_ref: float = 0.1) -> dict:
    """Run the model on one tile and decode instances in tile coordinates
    (``boxes``, ``scores``, ``labels``, ``masks``)."""
    tile_size = crop.shape[0]
    s = cfg.grid_stride
    x = torch.as_tensor(crop, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
    out = model(x.to(device))

    prob = out["prob"][0, 0].detach().cpu().numpy()
    dist = out["dist"][0].permute(1, 2, 0).detach().cpu().numpy()          # (Hr, Wr, K)
    offset_map = out["offset"][0].detach().cpu().numpy() if "offset" in out else None

    if cfg.adaptive_nms:
        if "density" not in out:
            raise RuntimeError("adaptive NMS requires a model with the density head")
        density = out["density"][0, 0].detach().cpu().numpy()
        # candidates with almost no suppression, then density-adaptive suppression
        points, probs, dists = non_maximum_suppression(dist, prob, grid=(s, s),
                                                       prob_thresh=cfg.prob_thresh, nms_thresh=0.98)
        if len(points) == 0:
            return _empty_prediction(tile_size)
        coords = dist_to_coord(dists, points)
        cand = np.stack([_polygon_mask(coords[i], tile_size) for i in range(len(points))])
        h_red, w_red = density.shape
        local = [float(density[min(int(py) // s, h_red - 1), min(int(px) // s, w_red - 1)])
                 for py, px in points]
        keep = density_adaptive_nms(cand, probs, local, iou_base=cfg.nms_thresh,
                                    iou_max=cfg.iou_max_adaptive, density_ref=density_ref)
        points, probs, dists = points[keep], probs[keep], dists[keep]
    else:
        points, probs, dists = non_maximum_suppression(dist, prob, grid=(s, s),
                                                       prob_thresh=cfg.prob_thresh,
                                                       nms_thresh=cfg.nms_thresh)
    if len(points) == 0:
        return _empty_prediction(tile_size)

    if cfg.one_per_component:
        keep = one_instance_per_component(points, probs, prob, s, cfg.prob_thresh)
        points, probs, dists = points[keep], probs[keep], dists[keep]
        if len(points) == 0:
            return _empty_prediction(tile_size)

    if offset_map is not None:
        points = _apply_subgrid_offset(points, offset_map, s, tile_size)

    # Polar contour refinement of every polygon kept by NMS. The refined rays
    # are returned in the stardist convention, so everything below is unchanged.
    if cfg.refine and getattr(model, "refiner", None) is not None:
        points, dists = model.refine_inference(out, points, dists)

    label_map = polygons_to_label(dists, points, prob=probs, shape=(tile_size, tile_size))
    masks, boxes, scores = _label_map_to_instances(label_map, probs)
    n = len(masks)
    if n == 0:
        return _empty_prediction(tile_size)
    return {"boxes": np.array(boxes, dtype=np.float32), "scores": np.array(scores, dtype=np.float32),
            "labels": np.ones((n,), dtype=np.int64), "masks": np.stack(masks)}


@torch.no_grad()
def predict_image(model, image: np.ndarray, device, cfg: InferenceConfig,
                  density_ref: float = 0.1) -> dict:
    """Predict instances on a full image already at the working scale.
    Returns the merged prediction dict (``boxes``, ``scores``, ``labels``,
    ``masks`` (N, H, W) bool)."""
    h, w = image.shape[:2]
    shifted = []
    for (x0, y0) in tiling.generate_windows(w, h, cfg.tile_size, cfg.overlap):
        crop = tiling.crop_pixels(image, x0, y0, cfg.tile_size)
        tile_pred = predict_tile(model, crop, device, cfg, density_ref)
        shifted.append(tiling.shift_prediction(tile_pred, x0, y0, w, h))
    merged = tiling.merge_tile_predictions(shifted, h, w, iou_thresh=cfg.iou_merge)
    if cfg.containment is not None:
        # Within one tile there is no containment (polygons_to_label returns
        # disjoint masks); fragments arise only across tiles.
        merged, _ = apply_containment(merged, containment_thresh=cfg.containment,
                                      max_area_ratio=cfg.containment_area_ratio,
                                      protect_merges=cfg.protect_merges)
    return merged


def predict_annotated_image(model, name: str, pairs: dict, device, cfg: InferenceConfig,
                            density_ref: float = 0.1):
    """Load an annotated image at the working scale and predict it. Returns
    ``(image, gt_masks, gt_boxes, prediction)``."""
    img_path, json_path = pairs[name]
    image, polygons, _ = load_image_and_polygons(img_path, json_path, cfg.ref_side)
    h, w = image.shape[:2]
    gt_masks, gt_boxes = full_image_instances(polygons, h, w)
    return image, gt_masks, gt_boxes, predict_image(model, image, device, cfg, density_ref)


def resolve_density_ref(model, names, pairs, device, cfg: InferenceConfig) -> float:
    if not cfg.adaptive_nms:
        return 0.1  # unused
    try:
        return float(cfg.density_ref)
    except (TypeError, ValueError):
        return calibrate_density_ref(model, names, pairs, device, cfg)


@torch.no_grad()
def evaluate_images(model, names: list, pairs: dict, device, cfg: InferenceConfig,
                    max_det: int = 1000, show_progress: bool = True):
    """Predict every image of ``names`` and compute the metrics.
    Returns ``(summary dict, per-image DataFrame)``."""
    model.eval()
    density_ref = resolve_density_ref(model, names, pairs, device, cfg)
    evaluator = InstanceEvaluator(score_thresh=cfg.prob_thresh, max_det=max_det)
    iterator = tqdm(names, desc="  evaluating", unit="img") if show_progress else names
    for name in iterator:
        image, gt_masks, gt_boxes, pred = predict_annotated_image(model, name, pairs, device,
                                                                  cfg, density_ref)
        h, w = image.shape[:2]
        n_gt = len(gt_masks)
        target = {
            "boxes": torch.as_tensor(np.array(gt_boxes, dtype=np.float32) if n_gt
                                     else np.zeros((0, 4), dtype=np.float32)),
            "labels": torch.ones((n_gt,), dtype=torch.int64),
            "masks": torch.as_tensor(np.stack(gt_masks).astype(np.uint8) if n_gt
                                     else np.zeros((0, h, w), dtype=np.uint8)),
            "image_name": name,
        }
        pred_t = [{"boxes": torch.as_tensor(pred["boxes"], dtype=torch.float32),
                   "scores": torch.as_tensor(pred["scores"], dtype=torch.float32),
                   "labels": torch.as_tensor(pred["labels"], dtype=torch.int64),
                   "masks": torch.as_tensor(pred["masks"], dtype=torch.float32).unsqueeze(1)}]
        evaluator.update(pred_t, [target])
    return evaluator.compute()
