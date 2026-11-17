"""Evaluation metrics for conidial instance segmentation.

* mAP@50 and mAP@50:95 (``torchmetrics``, ``iou_type="segm"``).
* Aggregated Jaccard index (AJI; Kumar et al., 2017), which penalizes merged
  and split instances in dense scenes.
* Panoptic quality PQ = DQ x SQ (Kirillov et al., 2019): DQ measures whether
  conidia are found and separated, SQ how precisely they are delineated.
* Boundary IoU (foreground-level variant, see :func:`compute_boundary_iou`).
* Dice coefficient of the image foreground.
* Absolute counting error per image (reported as MAE).

Predictions with a confidence score of at least 0.5 are evaluated as
individual binary instance masks at the working resolution; overlaps between
predicted instances are not resolved. Instance functions take lists of
boolean (H, W) arrays so that they can be tested in isolation.
"""

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import binary_erosion
from scipy.optimize import linear_sum_assignment

PER_IMAGE_COLUMNS = ["image", "aji", "pq", "dq", "sq", "boundary_iou", "dice",
                     "n_pred", "n_gt", "count_error"]


# =============================================================================
# IoU utilities
# =============================================================================

def _flatten(masks: list) -> np.ndarray:
    if len(masks) == 0:
        return np.zeros((0, 0), dtype=np.float32)
    return np.stack([m.reshape(-1) for m in masks]).astype(np.float32)


def iou_matrix(masks_a: list, masks_b: list):
    """IoU between every pair ``(a_i, b_j)``.
    Returns ``(iou, intersection, area_a, area_b)``."""
    a, b = _flatten(masks_a), _flatten(masks_b)
    if a.shape[0] == 0 or b.shape[0] == 0:
        inter = np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    else:
        inter = a @ b.T
    area_a = a.sum(axis=1) if a.shape[0] else np.zeros((0,), dtype=np.float32)
    area_b = b.sum(axis=1) if b.shape[0] else np.zeros((0,), dtype=np.float32)
    union = area_a[:, None] + area_b[None, :] - inter
    iou = np.divide(inter, union, out=np.zeros_like(union), where=union > 0)
    return iou, inter, area_a, area_b


# =============================================================================
# AJI
# =============================================================================

def compute_aji(masks_pred: list, masks_gt: list) -> float:
    """Aggregated Jaccard index with greedy one-to-one matching in annotation
    order: every annotation takes the not yet used prediction of highest IoU.
    Unused predictions add their area to the denominator (false positives);
    annotations without an overlapping prediction add their full area (false
    negatives)."""
    n_gt, n_pred = len(masks_gt), len(masks_pred)
    if n_gt == 0:
        return 1.0 if n_pred == 0 else 0.0
    if n_pred == 0:
        return 0.0

    iou, inter, area_gt, area_pred = iou_matrix(masks_gt, masks_pred)
    union = area_gt[:, None] + area_pred[None, :] - inter

    used = set()
    num, den = 0.0, 0.0
    for i in range(n_gt):
        row = iou[i].copy()
        for p in used:
            row[p] = -1.0
        j = int(np.argmax(row))
        if row[j] <= 0:
            den += area_gt[i]
            continue
        used.add(j)
        num += inter[i, j]
        den += union[i, j]
    for p in set(range(n_pred)) - used:
        den += area_pred[p]
    return float(num / den) if den > 0 else 1.0


# =============================================================================
# PQ / DQ / SQ
# =============================================================================

def compute_pq(masks_pred: list, masks_gt: list, iou_thresh: float = 0.5) -> dict:
    """Panoptic quality with optimal one-to-one (Hungarian) matching restricted
    to pairs with IoU >= ``iou_thresh`` (above 0.5 the matching is unique)."""
    n_gt, n_pred = len(masks_gt), len(masks_pred)
    if n_gt == 0 and n_pred == 0:
        return {"pq": 1.0, "dq": 1.0, "sq": 1.0, "tp": 0, "fp": 0, "fn": 0}
    if n_gt == 0:
        return {"pq": 0.0, "dq": 0.0, "sq": 0.0, "tp": 0, "fp": n_pred, "fn": 0}
    if n_pred == 0:
        return {"pq": 0.0, "dq": 0.0, "sq": 0.0, "tp": 0, "fp": 0, "fn": n_gt}

    iou, _, _, _ = iou_matrix(masks_gt, masks_pred)
    rows, cols = linear_sum_assignment(-iou)
    matched = [iou[r, c] for r, c in zip(rows, cols) if iou[r, c] >= iou_thresh]
    tp = len(matched)
    fp, fn = n_pred - tp, n_gt - tp
    sq = float(np.mean(matched)) if matched else 0.0
    denom = tp + 0.5 * fp + 0.5 * fn
    dq = tp / denom if denom > 0 else 0.0
    return {"pq": dq * sq, "dq": dq, "sq": sq, "tp": tp, "fp": fp, "fn": fn}


# =============================================================================
# Foreground-level Boundary IoU and Dice
# =============================================================================

def _inner_boundary(mask: np.ndarray, width: int = 2) -> np.ndarray:
    """Inner boundary band ``X \\ erode(X)``, ``width`` erosions with the
    four-connected 3x3 structuring element."""
    if mask.sum() == 0:
        return mask
    return mask & ~binary_erosion(mask, iterations=width, border_value=0)


def compute_boundary_iou(fg_pred: np.ndarray, fg_gt: np.ndarray, width: int = 2) -> float:
    """Boundary IoU of the image foregrounds (union of all predicted and of
    all annotated instance masks), with a band of ``width`` pixels."""
    b_pred, b_gt = _inner_boundary(fg_pred, width), _inner_boundary(fg_gt, width)
    union = np.logical_or(b_pred, b_gt).sum()
    return float(np.logical_and(b_pred, b_gt).sum() / union) if union > 0 else 1.0


def compute_dice(fg_pred: np.ndarray, fg_gt: np.ndarray) -> float:
    denom = fg_pred.sum() + fg_gt.sum()
    return float(2 * np.logical_and(fg_pred, fg_gt).sum() / denom) if denom > 0 else 1.0


# =============================================================================
# Accumulating evaluator
# =============================================================================

class InstanceEvaluator:
    """Accumulates mAP (torchmetrics) and per-image instance metrics.

    ``max_det`` caps the number of detections per image used by torchmetrics
    for *all* mAP/mAR figures. The COCO default (100) is insufficient for
    these micrographs (the densest image has 693 conidia). The thresholds are
    ``[1, 100, max_det]``: pycocotools depends internally on the value 100 and
    silently returns -1 for mAP@50:95 when it is absent.
    """

    def __init__(self, score_thresh: float = 0.5, mask_thresh: float = 0.5,
                 iou_thresh_pq: float = 0.5, boundary_width: int = 2, max_det: int = 1000):
        from torchmetrics.detection.mean_ap import MeanAveragePrecision
        self.score_thresh = score_thresh
        self.mask_thresh = mask_thresh
        self.iou_thresh_pq = iou_thresh_pq
        self.boundary_width = boundary_width
        if max_det <= 100:          # torchmetrics requires three distinct values
            max_det = 1000
        self.map_metric = MeanAveragePrecision(iou_type="segm",
                                               max_detection_thresholds=[1, 100, max_det])
        self.records = []

    @torch.no_grad()
    def update(self, preds: list, targets: list):
        """``preds``: dicts with ``boxes``, ``scores``, ``labels`` and ``masks``
        (N, 1, H, W) probabilities or binary values; ``targets``: dicts with
        ``boxes``, ``labels``, ``masks`` (N, H, W) and ``image_name``."""
        preds_tm, targets_tm = [], []
        for p in preds:
            if len(p["masks"]) > 0:
                masks_bin = (p["masks"][:, 0] > self.mask_thresh).to(torch.uint8)
            else:
                masks_bin = torch.zeros((0, *p["masks"].shape[-2:]), dtype=torch.uint8)
            preds_tm.append({"boxes": p["boxes"].detach().cpu(), "scores": p["scores"].detach().cpu(),
                             "labels": p["labels"].detach().cpu(),
                             "masks": masks_bin.detach().cpu().bool()})
        for t in targets:
            targets_tm.append({"boxes": t["boxes"].detach().cpu(), "labels": t["labels"].detach().cpu(),
                               "masks": t["masks"].detach().cpu().bool()})
        self.map_metric.update(preds_tm, targets_tm)

        for p, t in zip(preds, targets):
            keep = p["scores"].detach().cpu().numpy() >= self.score_thresh
            if len(p["masks"]) > 0:
                masks_pred = (p["masks"][:, 0].detach().cpu().numpy() > self.mask_thresh)[keep]
            else:
                masks_pred = np.zeros((0,) + tuple(t["masks"].shape[-2:]), dtype=bool)
            masks_gt = t["masks"].detach().cpu().numpy().astype(bool)
            pred_list, gt_list = list(masks_pred), list(masks_gt)

            aji = compute_aji(pred_list, gt_list)
            pq = compute_pq(pred_list, gt_list, self.iou_thresh_pq)

            hw = t["masks"].shape[-2:] if len(gt_list) else masks_pred.shape[-2:]
            fg_pred = np.zeros(hw, dtype=bool)
            for m in pred_list:
                fg_pred |= m
            fg_gt = np.zeros(hw, dtype=bool)
            for m in gt_list:
                fg_gt |= m

            self.records.append({
                "image": t.get("image_name", ""), "aji": aji,
                "pq": pq["pq"], "dq": pq["dq"], "sq": pq["sq"],
                "boundary_iou": compute_boundary_iou(fg_pred, fg_gt, self.boundary_width),
                "dice": compute_dice(fg_pred, fg_gt),
                "n_pred": len(pred_list), "n_gt": len(gt_list),
                "count_error": abs(len(pred_list) - len(gt_list)),
            })

    def compute(self):
        """Return ``(summary dict, per-image DataFrame)``."""
        if not self.records:
            summary = {"n_images": 0, **{k: float("nan") for k in (
                "aji_mean", "pq_mean", "dq_mean", "sq_mean", "boundary_iou_mean",
                "dice_mean", "count_mae", "count_corr")}}
            return summary, pd.DataFrame(columns=PER_IMAGE_COLUMNS)

        map_result = self.map_metric.compute()
        map_dict = {k: float(v) for k, v in map_result.items()
                    if torch.is_tensor(v) and v.numel() == 1}
        df = pd.DataFrame(self.records, columns=PER_IMAGE_COLUMNS)
        summary = {
            "n_images": len(df),
            "aji_mean": float(df["aji"].mean()),
            "pq_mean": float(df["pq"].mean()),
            "dq_mean": float(df["dq"].mean()),
            "sq_mean": float(df["sq"].mean()),
            "boundary_iou_mean": float(df["boundary_iou"].mean()),
            "dice_mean": float(df["dice"].mean()),
            "count_mae": float(df["count_error"].mean()),
            "count_corr": (float(df[["n_pred", "n_gt"]].corr().iloc[0, 1])
                           if len(df) > 1 else float("nan")),
            **map_dict,
        }
        return summary, df
