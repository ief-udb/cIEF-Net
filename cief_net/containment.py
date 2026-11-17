"""Containment suppression of tile-border fragments (post-processing).

A conidium crossing a tile border is seen completely by one tile and only
partially by the neighbouring tile. After merging, the partial detection can
remain nested inside the complete one: its IoU with the complete instance
(area ratio a/A, e.g. 0.4) is below any reasonable merging threshold, but its
*containment* ``|A n B| / min(|A|, |B|)`` is close to 1.

Rule (deterministic, never looks at annotations). For every pair of
predictions, larger ``i`` and smaller ``j``, with

    containment(j in i) = |i n j| / |j|  >= tau      (default 0.8)
    area(j) / area(i)                    <= rho      (default 0.7)

``j`` is a fragment of ``i`` and is removed, provided that ``i`` is kept.
Predictions are processed from the largest to the smallest, so the fate of a
parent is decided before that of its children. ``rho`` keeps the rule away
from near-identical pairs, which are the job of IoU-based NMS.

Merge protection (``protect_merges=True``, used by cIEF-Net). If a large
prediction ``i`` contains two or more nearly disjoint smaller predictions (IoU
between them < 0.2) that jointly cover at least 60% of ``i``, then ``i`` is
probably the merge of several touching conidia and its children are the
correct instances: ``i`` is removed and the children are kept.

Pure NumPy; see ``tests/test_containment.py``.
"""

import numpy as np


def _as_bool(m):
    if hasattr(m, "detach"):
        m = m.detach().cpu().numpy()
    m = np.asarray(m)
    if m.ndim == 3:
        m = m.squeeze()
    return m if m.dtype == bool else m > 0.5


def _bbox(m):
    rows = np.flatnonzero(m.any(axis=1))
    if len(rows) == 0:
        return None
    cols = np.flatnonzero(m.any(axis=0))
    return int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1


def pairwise_overlaps(masks):
    """``masks``: list of boolean (H, W) masks. Returns ``(areas, bboxes,
    inter)`` with ``inter[(a, b)] = |m_a n m_b|`` for ``a < b`` whose boxes
    touch and whose intersection is non-empty."""
    areas = np.array([m.sum() for m in masks], dtype=np.float64)
    bbs = [_bbox(m) for m in masks]
    inter = {}
    n = len(masks)
    for a in range(n):
        ba = bbs[a]
        if ba is None:
            continue
        for b in range(a + 1, n):
            bb = bbs[b]
            if bb is None:
                continue
            y0, y1 = max(ba[0], bb[0]), min(ba[1], bb[1])
            x0, x1 = max(ba[2], bb[2]), min(ba[3], bb[3])
            if y0 >= y1 or x0 >= x1:
                continue
            v = int(np.logical_and(masks[a][y0:y1, x0:x1], masks[b][y0:y1, x0:x1]).sum())
            if v:
                inter[(a, b)] = v
    return areas, bbs, inter


def _inter(inter, a, b):
    return inter.get((a, b) if a < b else (b, a), 0)


def decide_suppression(masks, areas, bbs, inter, containment_thresh=0.8, max_area_ratio=0.7,
                       protect_merges=False, merge_coverage=0.6, max_child_iou=0.2):
    """Return ``(keep, info)``: ``keep`` are the kept indices in original
    order; ``info`` holds counts and the reason for every removed index."""
    n = len(masks)
    order = [int(k) for k in np.argsort(-areas, kind="stable")]
    rank = {k: r for r, k in enumerate(order)}

    parents = {j: [] for j in range(n)}     # larger predictions that contain j
    children = {i: [] for i in range(n)}
    for (a, b), v in inter.items():
        i, j = (a, b) if rank[a] < rank[b] else (b, a)     # i is the larger one
        if areas[j] <= 0 or areas[i] <= 0:
            continue
        if v / areas[j] >= containment_thresh and areas[j] <= max_area_ratio * areas[i]:
            parents[j].append(i)
            children[i].append(j)

    removed = {}
    if protect_merges:
        for i in order:
            if len(children[i]) < 2:
                continue
            # nearly disjoint children, from the largest to the smallest
            sel = []
            for j in sorted(children[i], key=lambda k: -areas[k]):
                if all(_inter(inter, j, s) / (areas[j] + areas[s] - _inter(inter, j, s)) < max_child_iou
                       for s in sel):
                    sel.append(j)
            if len(sel) < 2:
                continue
            y0, y1, x0, x1 = bbs[i]
            union_children = np.zeros((y1 - y0, x1 - x0), dtype=bool)
            for j in sel:
                union_children |= masks[j][y0:y1, x0:x1]
            coverage = np.logical_and(union_children, masks[i][y0:y1, x0:x1]).sum() / areas[i]
            if coverage >= merge_coverage:
                removed[i] = "merge"

    for j in order:                         # parents are decided before children
        if j in removed:
            continue
        if any(p not in removed for p in parents[j]):
            removed[j] = "fragment"

    keep = [k for k in range(n) if k not in removed]
    info = {"n_in": n, "n_out": len(keep),
            "n_fragments_removed": sum(1 for v in removed.values() if v == "fragment"),
            "n_merges_removed": sum(1 for v in removed.values() if v == "merge"),
            "reason": removed}
    return keep, info


def suppress_contained(masks, **kw):
    """Masks (lists, NumPy or torch) -> ``(keep, info)``."""
    mb = [_as_bool(m) for m in masks]
    areas, bbs, inter = pairwise_overlaps(mb)
    return decide_suppression(mb, areas, bbs, inter, **kw)


def filter_prediction(pred: dict, keep):
    """Apply ``keep`` to every per-instance entry of a merged prediction dict
    (lists/tuples of length n and arrays/tensors with ``shape[0] == n``).
    Per-pixel maps (H, W) cannot be filtered this way; their keys are returned
    in ``unfiltered_keys`` as a warning."""
    n = len(pred["masks"])
    hw = tuple(np.asarray(_as_bool(pred["masks"][0])).shape) if n else None
    idx = np.asarray(keep, dtype=np.int64)
    out, unfiltered = {}, []
    for k, v in pred.items():
        if isinstance(v, (list, tuple)) and len(v) == n:
            out[k] = type(v)(v[i] for i in keep) if isinstance(v, tuple) else [v[i] for i in keep]
        elif hasattr(v, "shape") and getattr(v, "ndim", 0) >= 1 and v.shape[0] == n \
                and not (v.ndim == 2 and tuple(v.shape) == hw):
            if hasattr(v, "detach"):
                import torch
                out[k] = v[torch.as_tensor(idx, device=v.device)]
            else:
                out[k] = v[idx]
        else:
            out[k] = v
            if hasattr(v, "shape") and hw is not None and tuple(v.shape[-2:]) == hw:
                unfiltered.append(k)
    return out, unfiltered


def apply_containment(pred: dict, **kw):
    """Merged prediction dict -> ``(filtered dict, info)``.

    Keyword arguments are those of :func:`decide_suppression`
    (``containment_thresh``, ``max_area_ratio``, ``protect_merges``, ...)."""
    if not len(pred["masks"]):
        return pred, {"n_in": 0, "n_out": 0, "n_fragments_removed": 0,
                      "n_merges_removed": 0, "reason": {}}
    keep, info = suppress_contained(list(pred["masks"]), **kw)
    out, unfiltered = filter_prediction(pred, keep)
    info["unfiltered_keys"] = unfiltered
    return out, info
