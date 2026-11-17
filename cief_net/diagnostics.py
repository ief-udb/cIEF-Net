"""Error decomposition of instance predictions (NumPy only, no model needed).

Matching uses IoU > 0.5, as PQ, so the SQ recomputed here equals ``sq_mean``
of the evaluation for the same predictions.

For every **matched** annotation (the instances that enter SQ):

``iou``                IoU between prediction and annotation
``log_area_ratio``     log(area_pred / area_gt); < 0 when the prediction is smaller
``frac_missing``       |GT outside pred| / |GT| (under-segmentation)
``frac_excess``        |pred outside GT| / |GT| (over-segmentation)
``rel_displacement``   centroid distance / equivalent radius
``rel_radial_bias``    mean of (r_pred - r_gt) over 64 rays from the GT centroid,
                       / equivalent radius: systematic size error
``rel_radial_noise``   standard deviation of the same: contour jitter
``iou_centred``        IoU after translating the prediction onto the GT centroid
``spec_*``             energy of the radial error by angular harmonic band
                       (k0 size, k1 translation, k2 elongation, k3-5 shape,
                       k>=6 high frequency), in units of the squared equivalent
                       radius (Parseval: they sum to mean(dr^2) / r_eq^2)
``iou_oracle_kK``      polar IoU if harmonics 0..K of the error were corrected
                       exactly

Every **unmatched** annotation gets a status: ``missed`` (max IoU < 0.1),
``merged`` (one prediction covers it and another conidium), ``split`` (two or
more predictions share it) or ``poorly_segmented`` (0.1 <= IoU <= 0.5).
Unmatched predictions are ``spurious``, ``merge``, ``fragment`` or
``poorly_segmented``. Morphology and context of every annotation (area,
equivalent diameter, elongation, solidity, neighbours within 1.5 diameters,
border contact) allow stratification.
"""

import math

import numpy as np

N_DIAG_RAYS = 64
SPECTRAL_BANDS = [("k0_size", 0, 0), ("k1_translation", 1, 1), ("k2_elongation", 2, 2),
                  ("k3_5_shape", 3, 5), ("k6plus_high_freq", 6, 10 ** 6)]
ORACLE_K = (0, 1, 2, 5)


# =============================================================================
# Geometry
# =============================================================================

def as_bool(m):
    if hasattr(m, "detach"):
        m = m.detach().cpu().numpy()
    m = np.asarray(m)
    if m.ndim == 3:
        m = m.squeeze()
    return m > 0.5 if m.dtype != bool else m


def bbox(m):
    rows = np.flatnonzero(m.any(axis=1))
    cols = np.flatnonzero(m.any(axis=0))
    if len(rows) == 0:
        return None
    return int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1


def bbox_union(a, b, pad, h, w):
    return (max(0, min(a[0], b[0]) - pad), min(h, max(a[1], b[1]) + pad),
            max(0, min(a[2], b[2]) - pad), min(w, max(a[3], b[3]) + pad))


def boxes_touch(a, b):
    return not (a[1] <= b[0] or b[1] <= a[0] or a[3] <= b[2] or b[3] <= a[2])


def convex_hull_area(pts):
    pts = np.unique(pts, axis=0)
    if len(pts) < 3:
        return 0.0
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    c = np.array(lower[:-1] + upper[:-1], dtype=np.float64)
    x, y = c[:, 0], c[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def mask_morphology(crop):
    ys, xs = np.nonzero(crop)
    area = len(xs)
    elong = float("nan")
    if area >= 3:
        ev = np.linalg.eigvalsh(np.cov(np.stack([xs, ys])))
        if ev[0] > 1e-9:
            elong = float(np.sqrt(ev[1] / ev[0]))
    # solidity on pixel corners (avoids solidity > 1 for small objects)
    corners = np.concatenate([np.stack([xs + dx, ys + dy], 1) for dx in (0, 1) for dy in (0, 1)])
    hull = convex_hull_area(corners)
    return {"area_px": int(area),
            "eq_diameter_px": float(2.0 * math.sqrt(area / math.pi)) if area else 0.0,
            "elongation": elong,
            "solidity": float(area / hull) if hull > 0 else float("nan"),
            "centroid": (float(ys.mean()), float(xs.mean())) if area else (float("nan"),) * 2}


def ray_radii(mask, cy, cx, n=N_DIAG_RAYS, step=0.5):
    """Distance from ``(cy, cx)`` to the first exit from the mask along n rays."""
    h, w = mask.shape
    ang = 2.0 * np.pi * np.arange(n) / n
    r = np.arange(0.0, math.hypot(h, w) + 1.0, step)
    ys = np.rint(cy + r[None, :] * np.sin(ang)[:, None]).astype(np.int64)
    xs = np.rint(cx + r[None, :] * np.cos(ang)[:, None]).astype(np.int64)
    ok = (ys >= 0) & (ys < h) & (xs >= 0) & (xs < w)
    inside = mask[np.clip(ys, 0, h - 1), np.clip(xs, 0, w - 1)] & ok
    outside = ~inside
    idx = np.where(outside.any(axis=1), outside.argmax(axis=1), len(r) - 1)
    return r[np.maximum(idx, 1) - 1] + step / 2


def ray_radii_subpixel(mask, cy, cx, n=N_DIAG_RAYS, step=0.25):
    """As :func:`ray_radii`, with the boundary at the 0.5 iso-line of the
    bilinear interpolation of the mask (equivalent to marching squares). This
    removes the +-0.5-pixel quantization error that would otherwise
    contaminate the high-frequency band for conidia of about 14 px radius."""
    from scipy.ndimage import map_coordinates
    h, w = mask.shape
    ang = 2.0 * np.pi * np.arange(n) / n
    r = np.arange(0.0, math.hypot(h, w) + 1.0, step)
    ys = cy + r[None, :] * np.sin(ang)[:, None]
    xs = cx + r[None, :] * np.cos(ang)[:, None]
    v = map_coordinates(mask.astype(np.float32), [ys.ravel(), xs.ravel()], order=1,
                        mode="constant", cval=0.0).reshape(ys.shape)
    outside = v < 0.5
    idx = np.where(outside.any(axis=1), outside.argmax(axis=1), len(r) - 1)
    idx = np.maximum(idx, 1)
    row = np.arange(n)
    v0, v1 = v[row, idx - 1], v[row, idx]
    t = np.clip((v0 - 0.5) / np.maximum(v0 - v1, 1e-6), 0.0, 1.0)
    return r[idx - 1] + t * step


def polar_iou(ra, rb):
    """IoU of two star polygons with the same centre and angles (the
    intersection is the polygon of min(ra, rb) ray by ray)."""
    def area(r):
        return 0.5 * np.sin(2 * np.pi / len(r)) * float(np.sum(r * np.roll(r, -1)))
    m = np.minimum(ra, rb)
    ai = area(m)
    u = area(ra) + area(rb) - ai
    return ai / u if u > 0 else 0.0


def spectrum_and_oracle(g, p, cy, cx, r_eq):
    """Fourier decomposition of the radial error dr(theta) = r_pred - r_gt
    (64 rays from the GT centroid, sub-pixel boundary) and the polar IoU of an
    oracle that corrects harmonics 0..K exactly."""
    rg = ray_radii_subpixel(g, cy, cx)
    rp = ray_radii_subpixel(p, cy, cx)
    dr = (rp - rg) / r_eq
    n = len(dr)
    F = np.fft.rfft(dr)
    power = np.abs(F) ** 2 / n ** 2
    power[1:(n + 1) // 2] *= 2.0            # harmonics with a conjugate pair
    row = {f"spec_{name}": float(power[k0:min(k1, len(power) - 1) + 1].sum())
           for name, k0, k1 in SPECTRAL_BANDS}
    row["spec_total"] = float(power.sum())
    row["iou_polar_pred"] = polar_iou(rp, rg)
    for K in ORACLE_K:
        Fk = F.copy()
        Fk[K + 1:] = 0.0
        row[f"iou_oracle_k{K}"] = polar_iou(np.maximum(rp - np.fft.irfft(Fk, n) * r_eq, 0.0), rg)
    return row


def translate(m, dy, dx):
    out = np.zeros_like(m)
    h, w = m.shape
    y0, y1 = max(0, dy), min(h, h + dy)
    x0, x1 = max(0, dx), min(w, w + dx)
    out[y0:y1, x0:x1] = m[y0 - dy:y1 - dy, x0 - dx:x1 - dx]
    return out


# =============================================================================
# One image
# =============================================================================

def annotation_overlaps(name, gt_masks, gt_bbs, gt_areas, dup_thresh, min_thresh=0.5):
    """Pairs of overlapping annotations. ``iou`` ~ 1 only when both annotations
    have nearly the same shape (double annotation); ``containment`` ~ 1 when
    one lies almost entirely inside the other (e.g. a conidium annotated whole
    in one window and cropped in another). Pairs with max(iou, containment)
    >= ``min_thresh`` are returned with a type given by ``dup_thresh``."""
    valid = [i for i in range(len(gt_masks)) if gt_bbs[i] is not None]
    if len(valid) < 2:
        return []
    b = np.array([gt_bbs[i] for i in valid])
    touch = ~((b[:, None, 1] <= b[None, :, 0]) | (b[None, :, 1] <= b[:, None, 0]) |
              (b[:, None, 3] <= b[None, :, 2]) | (b[None, :, 3] <= b[:, None, 2]))
    h, w = gt_masks[valid[0]].shape
    pairs = []
    for a_, b_ in zip(*np.nonzero(np.triu(touch, k=1))):
        i, j = valid[a_], valid[b_]
        y0, y1, x0, x1 = bbox_union(gt_bbs[i], gt_bbs[j], 0, h, w)
        inter = np.logical_and(gt_masks[i][y0:y1, x0:x1], gt_masks[j][y0:y1, x0:x1]).sum()
        if inter == 0:
            continue
        iou = inter / (gt_areas[i] + gt_areas[j] - inter)
        cont = inter / min(gt_areas[i], gt_areas[j])
        if max(iou, cont) < min_thresh:
            continue
        kind = ("probable_duplicate" if iou >= dup_thresh else
                "contained" if cont >= dup_thresh else "moderate_overlap")
        pairs.append({"image": name, "gt_a": int(i), "gt_b": int(j), "iou": float(iou),
                      "containment": float(cont), "area_a": int(gt_areas[i]),
                      "area_b": int(gt_areas[j]), "type": kind})
    return pairs


def analyze_image(name, gt_masks, pred_masks, scores=None, dup_thresh=0.90):
    """Return ``(rows per annotation, rows per prediction, annotation overlap pairs)``."""
    gt_masks = [as_bool(m) for m in gt_masks]
    pred_masks = [as_bool(m) for m in pred_masks]
    if scores is None:
        scores = [float("nan")] * len(pred_masks)
    h, w = (gt_masks[0].shape if gt_masks else pred_masks[0].shape if pred_masks else (0, 0))

    bb_gt = [bbox(m) for m in gt_masks]
    bb_pr = [bbox(m) for m in pred_masks]
    area_gt = np.array([m.sum() for m in gt_masks], dtype=np.float64)
    area_pr = np.array([m.sum() for m in pred_masks], dtype=np.float64)
    ng, npred = len(gt_masks), len(pred_masks)

    inter = np.zeros((ng, npred))      # only for pairs whose boxes touch
    for i in range(ng):
        if bb_gt[i] is None:
            continue
        for j in range(npred):
            if bb_pr[j] is None or not boxes_touch(bb_gt[i], bb_pr[j]):
                continue
            y0, y1, x0, x1 = bbox_union(bb_gt[i], bb_pr[j], 0, h, w)
            inter[i, j] = np.logical_and(gt_masks[i][y0:y1, x0:x1], pred_masks[j][y0:y1, x0:x1]).sum()
    union = area_gt[:, None] + area_pr[None, :] - inter
    iou = np.where(union > 0, inter / np.maximum(union, 1), 0.0)
    cov_gt = inter / np.maximum(area_gt[:, None], 1)     # fraction of GT i covered by pred j
    cov_pr = inter / np.maximum(area_pr[None, :], 1)     # fraction of pred j inside GT i

    # One-to-one matching. With disjoint annotations IoU > 0.5 is already
    # unique, but annotations may overlap; greedy matching by decreasing IoU
    # prevents a prediction from being matched twice.
    match_gt, match_pr = {}, {}
    ci, cj = np.nonzero(iou > 0.5)
    for k in np.argsort(-iou[ci, cj], kind="stable"):
        i, j = int(ci[k]), int(cj[k])
        if i not in match_gt and j not in match_pr:
            match_gt[i], match_pr[j] = j, i

    morph = [mask_morphology(m[b[0]:b[1], b[2]:b[3]]) if b else None for m, b in zip(gt_masks, bb_gt)]
    centres = np.array([(b[0] + mf["centroid"][0], b[2] + mf["centroid"][1]) if mf else (np.nan, np.nan)
                        for mf, b in zip(morph, bb_gt)])
    diam = np.array([mf["eq_diameter_px"] if mf else np.nan for mf in morph])

    gt_pairs = annotation_overlaps(name, gt_masks, bb_gt, area_gt, dup_thresh)
    overlap_of = {}
    for pr in gt_pairs:
        for x, y in ((pr["gt_a"], pr["gt_b"]), (pr["gt_b"], pr["gt_a"])):
            prev = overlap_of.get(x)
            if prev is None or max(pr["iou"], pr["containment"]) > max(prev[1], prev[2]):
                overlap_of[x] = (y, pr["iou"], pr["containment"], pr["type"])

    rows_gt = []
    for i in range(ng):
        if morph[i] is None:
            continue
        mf, b = morph[i], bb_gt[i]
        d_centres = np.hypot(*(centres - centres[i]).T)
        row = {"image": name, "gt_id": i, "area_px": mf["area_px"], "eq_diameter_px": mf["eq_diameter_px"],
               "elongation": mf["elongation"], "solidity": mf["solidity"],
               "n_neighbours": int(np.sum((d_centres < 1.5 * diam[i]) & (d_centres > 0))),
               "touches_border": bool(b[0] == 0 or b[2] == 0 or b[1] >= h or b[3] >= w),
               "iou_max": float(iou[i].max()) if npred else 0.0}
        ov = overlap_of.get(i)
        row["overlaps_gt"] = ov[0] if ov else -1
        row["iou_gt_gt"] = ov[1] if ov else 0.0
        row["containment_gt_gt"] = ov[2] if ov else 0.0
        row["gt_overlap_type"] = ov[3] if ov else ""

        if i in match_gt:
            j = match_gt[i]
            row["status"] = "matched"
            row["iou"] = float(iou[i, j])
            row["log_area_ratio"] = float(np.log(max(area_pr[j], 1) / max(area_gt[i], 1)))
            row["frac_missing"] = float((area_gt[i] - inter[i, j]) / area_gt[i])
            row["frac_excess"] = float((area_pr[j] - inter[i, j]) / area_gt[i])

            r_eq = mf["eq_diameter_px"] / 2.0
            pad = int(math.ceil(r_eq)) + 2
            y0, y1, x0, x1 = bbox_union(b, bb_pr[j], pad, h, w)
            g, p = gt_masks[i][y0:y1, x0:x1], pred_masks[j][y0:y1, x0:x1]
            cy_g, cx_g = centres[i][0] - y0, centres[i][1] - x0
            yp, xp = np.nonzero(p)
            cy_p, cx_p = yp.mean(), xp.mean()
            row["rel_displacement"] = float(math.hypot(cy_p - cy_g, cx_p - cx_g) / r_eq)

            iy, ix = int(round(cy_g)), int(round(cx_g))
            if g[iy, ix] and p[iy, ix]:
                dr = ray_radii(p, cy_g, cx_g) - ray_radii(g, cy_g, cx_g)
                row["rel_radial_bias"] = float(dr.mean() / r_eq)
                row["rel_radial_noise"] = float(dr.std() / r_eq)
                row.update(spectrum_and_oracle(g, p, cy_g, cx_g, r_eq))
            else:
                row["rel_radial_bias"] = float("nan")
                row["rel_radial_noise"] = float("nan")

            p_c = translate(p, int(round(cy_g - cy_p)), int(round(cx_g - cx_p)))
            u = np.logical_or(g, p_c).sum()
            row["iou_centred"] = float(np.logical_and(g, p_c).sum() / u) if u else 0.0
        else:
            covering = np.flatnonzero(cov_gt[i] > 0.2) if npred else []
            merged = [j for j in covering if np.sum(cov_gt[:, j] > 0.3) >= 2 and cov_gt[i, j] > 0.3]
            if row["iou_max"] < 0.1:
                row["status"] = "missed"
            elif merged:
                row["status"] = "merged"
            elif len(covering) >= 2:
                row["status"] = "split"
            else:
                row["status"] = "poorly_segmented"
        rows_gt.append(row)

    rows_pr = []
    for j in range(npred):
        row = {"image": name, "pred_id": j, "score": float(scores[j]) if j < len(scores) else float("nan"),
               "area_px": int(area_pr[j]), "iou_max": float(iou[:, j].max()) if ng else 0.0}
        if j in match_pr:
            row["status"] = "matched"
        elif ng and np.sum(cov_gt[:, j] > 0.3) >= 2:
            row["status"] = "merge"
        elif ng and cov_pr[:, j].max() > 0.7:
            row["status"] = "fragment"
        elif row["iou_max"] >= 0.1:
            row["status"] = "poorly_segmented"
        else:
            row["status"] = "spurious"
        rows_pr.append(row)
    return rows_gt, rows_pr, gt_pairs


# =============================================================================
# Aggregates
# =============================================================================

def stratify(df, column, label):
    matched = df[df["status"] == "matched"]
    allr = df.copy()
    if matched[column].nunique() < 4:
        return {}
    cuts = np.unique(np.nanquantile(allr[column], [0, .25, .5, .75, 1]))
    if len(cuts) < 3:
        return {}
    allr["_q"] = np.digitize(allr[column], cuts[1:-1], right=True)
    out = {}
    for q, g in allr.groupby("_q"):
        m = g[g["status"] == "matched"]
        out[f"{label} [{cuts[q]:.2f}, {cuts[q + 1]:.2f}]"] = {
            "n_gt": int(len(g)), "recall_iou50": float(len(m) / len(g)),
            "sq": float(m["iou"].mean()) if len(m) else None,
            "rel_radial_bias": float(m["rel_radial_bias"].mean()) if len(m) else None,
            "rel_radial_noise": float(m["rel_radial_noise"].mean()) if len(m) else None,
        }
    return out


def summarize_overlaps(df_pairs, df_gt, dup_thresh):
    if df_pairs is None or len(df_pairs) == 0:
        return {"threshold": dup_thresh, "n_pairs": 0}
    bins = [0.5, 0.7, 0.8, 0.9, 0.95, 1.0001]
    labels = ["[0.50,0.70)", "[0.70,0.80)", "[0.80,0.90)", "[0.90,0.95)", "[0.95,1.00]"]
    dist = {col: dict(zip(labels, [int(v) for v in np.histogram(df_pairs[col], bins=bins)[0]]))
            for col in ("iou", "containment")}
    suspect = df_gt[df_gt["gt_overlap_type"].isin(["probable_duplicate", "contained"])]
    unmatched = suspect[suspect["status"] != "matched"]
    return {"threshold": dup_thresh, "n_pairs": int(len(df_pairs)),
            "pairs_by_type": {k: int(v) for k, v in df_pairs["type"].value_counts().items()},
            "pair_iou_distribution": dist["iou"], "pair_containment_distribution": dist["containment"],
            "images_with_suspect_pairs": int(df_pairs.loc[df_pairs["type"] != "moderate_overlap",
                                                          "image"].nunique()),
            "gt_in_suspect_pairs": int(len(suspect)),
            "of_which_unmatched": {k: int(v) for k, v in unmatched["status"].value_counts().items()},
            "n_unmatched_in_suspect_pairs": int(len(unmatched))}


def summarize_spectrum(matched):
    """Fraction of the mean squared radial error per harmonic band and IoU gain
    of an oracle refinement that corrects harmonics 0..K, overall and by
    diameter quartile."""
    if "spec_total" not in matched or matched["spec_total"].notna().sum() == 0:
        return {}
    e = matched[matched["spec_total"].notna()]

    def block(d):
        tot = d["spec_total"].mean()
        return {"n": int(len(d)), "rms_rel_radial_error": float(np.sqrt(tot)),
                "energy_fraction": {n: float(d[f"spec_{n}"].mean() / tot) if tot else None
                                    for n, _, _ in SPECTRAL_BANDS},
                "iou_real_mean": float(d["iou"].mean()),
                "iou_polar_pred_mean": float(d["iou_polar_pred"].mean()),
                "oracle_gain": {f"k<={K}": float((d[f"iou_oracle_k{K}"] - d["iou_polar_pred"]).mean())
                                for K in ORACLE_K}}

    res = {"overall": block(e)}
    cuts = np.nanquantile(e["eq_diameter_px"], [0, .25, .5, .75, 1])
    q = np.digitize(e["eq_diameter_px"], cuts[1:-1], right=True)
    for i in range(4):
        sub = e[q == i]
        if len(sub) >= 10:
            res[f"diameter [{cuts[i]:.1f}, {cuts[i + 1]:.1f}]"] = block(sub)
    return res


def summarize(df_gt, df_pr):
    matched = df_gt[df_gt["status"] == "matched"]
    n_gt, n_pr, n_tp = len(df_gt), len(df_pr), len(matched)
    fp = int((df_pr["status"] != "matched").sum())
    fn = n_gt - n_tp
    dq = n_tp / (n_tp + 0.5 * fp + 0.5 * fn) if (n_tp + fp + fn) else float("nan")
    sq_per_image = matched.groupby("image")["iou"].mean()
    err_px = matched["frac_missing"] + matched["frac_excess"]
    # simulations without retraining (bounds, not substitutes for a real run)
    sq = float(matched["iou"].mean()) if n_tp else float("nan")
    n_frag = int((df_pr["status"] == "fragment").sum())
    dq_no_frag = n_tp / (n_tp + 0.5 * (fp - n_frag) + 0.5 * fn) if n_tp else float("nan")
    sweep = []
    if "score" in df_pr and df_pr["score"].notna().any():
        is_tp = df_pr["status"] == "matched"
        for t in np.round(np.arange(0.30, 0.91, 0.05), 2):
            keep = df_pr["score"] >= t
            tp_t, fp_t = int((keep & is_tp).sum()), int((keep & ~is_tp).sum())
            fn_t = n_gt - tp_t
            dq_t = tp_t / (tp_t + 0.5 * fp_t + 0.5 * fn_t) if (tp_t + fp_t + fn_t) else float("nan")
            sq_t = float(df_pr.loc[keep & is_tp, "iou_max"].mean()) if tp_t else float("nan")
            sweep.append({"prob_thresh": float(t), "tp": tp_t, "fp": fp_t, "fn": fn_t,
                          "dq": dq_t, "pq": dq_t * sq_t})
    return {
        "simulations": {"dq_without_fragments": float(dq_no_frag),
                        "pq_without_fragments": float(dq_no_frag * sq),
                        "prob_thresh_sweep_approx": sweep},
        "n_images": int(df_gt["image"].nunique()), "n_gt": n_gt, "n_pred": n_pr, "n_matched": n_tp,
        "SQ_mean_per_image": float(sq_per_image.mean()) if len(sq_per_image) else None,
        "SQ_global": float(matched["iou"].mean()) if n_tp else None,
        "DQ_global": float(dq),
        "PQ_global": float(dq * matched["iou"].mean()) if n_tp else None,
        "shape_of_matched": {
            "iou_median": float(matched["iou"].median()) if n_tp else None,
            "log_area_ratio_mean": float(matched["log_area_ratio"].mean()) if n_tp else None,
            "frac_smaller_predictions": float((matched["log_area_ratio"] < 0).mean()) if n_tp else None,
            "rel_radial_bias_mean": float(matched["rel_radial_bias"].mean()) if n_tp else None,
            "rel_radial_noise_mean": float(matched["rel_radial_noise"].mean()) if n_tp else None,
            "rel_displacement_mean": float(matched["rel_displacement"].mean()) if n_tp else None,
            "iou_gain_if_centred": float((matched["iou_centred"] - matched["iou"]).mean()) if n_tp else None,
            "error_fraction_missing": float((matched["frac_missing"] / err_px.where(err_px > 0)).mean())
            if n_tp else None,
        },
        "spectrum_and_oracle": summarize_spectrum(matched),
        "unmatched_gt": {k: int(v) for k, v in
                         df_gt.loc[df_gt["status"] != "matched", "status"].value_counts().items()},
        "unmatched_pred": {k: int(v) for k, v in
                           df_pr.loc[df_pr["status"] != "matched", "status"].value_counts().items()},
        "by_size": stratify(df_gt, "eq_diameter_px", "diameter_px"),
        "by_elongation": stratify(df_gt, "elongation", "elongation"),
        "by_neighbours": stratify(df_gt, "n_neighbours", "n_neighbours"),
    }


# =============================================================================
# Sensitivity sweep of containment suppression
# =============================================================================

def subset_metrics(inter, area_gt, area_pr, cols):
    """Metrics of one image using only the predictions ``cols``. Suppression
    only removes predictions, so the GT-prediction intersections computed once
    for all predictions are reused (columns are selected, no pixels are
    recomputed). The AJI here pairs each annotation with its highest-IoU
    prediction without the one-to-one constraint (sensitivity analysis only)."""
    cols = np.asarray(cols, dtype=np.int64)
    ng = len(area_gt)
    it = inter[:, cols] if len(cols) else np.zeros((ng, 0))
    ap_ = area_pr[cols] if len(cols) else np.zeros(0)
    union = area_gt[:, None] + ap_[None, :] - it
    iou = np.where(union > 0, it / np.maximum(union, 1), 0.0)
    used_g, used_p, ious = set(), set(), []
    ci, cj = np.nonzero(iou > 0.5)
    for k in np.argsort(-iou[ci, cj], kind="stable"):
        i, j = int(ci[k]), int(cj[k])
        if i not in used_g and j not in used_p:
            used_g.add(i)
            used_p.add(j)
            ious.append(iou[i, j])
    tp, fp, fn = len(ious), len(cols) - len(ious), ng - len(ious)
    dq = tp / (tp + 0.5 * fp + 0.5 * fn) if (tp + fp + fn) else float("nan")
    sq = float(np.mean(ious)) if tp else float("nan")
    c_acc, u_acc, used = 0.0, 0.0, set()
    for i in range(ng):
        if len(cols) and iou[i].max() > 0:
            j = int(np.argmax(iou[i]))
            c_acc += it[i, j]
            u_acc += area_gt[i] + ap_[j] - it[i, j]
            used.add(j)
        else:
            u_acc += area_gt[i]
    u_acc += sum(ap_[j] for j in range(len(cols)) if j not in used)
    return {"tp": tp, "fp": fp, "fn": fn, "sum_iou": float(np.sum(ious)), "dq": dq, "sq": sq,
            "pq": (dq * sq) if tp else 0.0, "aji": c_acc / u_acc if u_acc else float("nan"),
            "count_error": abs(len(cols) - ng)}


def sweep_containment_image(name, gt_masks, pred_masks, configs, acc, pairwise_overlaps,
                            decide_suppression):
    mg = [as_bool(m) for m in gt_masks]
    mp = [as_bool(m) for m in pred_masks]
    h, w = (mg[0].shape if mg else mp[0].shape if mp else (0, 0))
    bb_g, bb_p = [bbox(m) for m in mg], [bbox(m) for m in mp]
    area_gt = np.array([m.sum() for m in mg], dtype=np.float64)
    inter = np.zeros((len(mg), len(mp)))
    for i in range(len(mg)):
        if bb_g[i] is None:
            continue
        for j in range(len(mp)):
            if bb_p[j] is None or not boxes_touch(bb_g[i], bb_p[j]):
                continue
            y0, y1, x0, x1 = bbox_union(bb_g[i], bb_p[j], 0, h, w)
            inter[i, j] = np.logical_and(mg[i][y0:y1, x0:x1], mp[j][y0:y1, x0:x1]).sum()
    area_pr, bbs, inter_pp = pairwise_overlaps(mp)   # prediction-prediction, once per image
    for c in configs:
        if c is None:
            keep, info = list(range(len(mp))), {"n_fragments_removed": 0, "n_merges_removed": 0}
        else:
            t, r, pm = c
            keep, info = decide_suppression(mp, area_pr, bbs, inter_pp, containment_thresh=t,
                                            max_area_ratio=r, protect_merges=pm)
        m = subset_metrics(inter, area_gt, area_pr, keep)
        m.update(image=name, removed_fragments=info["n_fragments_removed"],
                 removed_merges=info["n_merges_removed"])
        acc[c].append(m)


def sweep_rows(acc):
    rows = []
    for c, lst in acc.items():
        if not lst:
            continue
        tp, fp, fn = (sum(m[k] for m in lst) for k in ("tp", "fp", "fn"))
        sq_g = sum(m["sum_iou"] for m in lst) / tp if tp else float("nan")
        dq_g = tp / (tp + 0.5 * fp + 0.5 * fn) if (tp + fp + fn) else float("nan")
        rows.append({
            "config": "no_suppression" if c is None else
            f"tau={c[0]:.2f} rho={c[1]:.2f} merges={'yes' if c[2] else 'no'}",
            "tau": None if c is None else c[0], "rho": None if c is None else c[1],
            "protect_merges": None if c is None else c[2],
            "removed_fragments": sum(m["removed_fragments"] for m in lst),
            "removed_merges": sum(m["removed_merges"] for m in lst),
            "tp": tp, "fp": fp, "fn": fn,
            "aji_mean": float(np.nanmean([m["aji"] for m in lst])),
            "pq_mean": float(np.nanmean([m["pq"] for m in lst])),
            "dq_mean": float(np.nanmean([m["dq"] for m in lst])),
            "sq_mean": float(np.nanmean([m["sq"] for m in lst])),
            "count_mae": float(np.mean([m["count_error"] for m in lst])),
            "dq_global": dq_g, "pq_global": dq_g * sq_g,
        })
    return rows
