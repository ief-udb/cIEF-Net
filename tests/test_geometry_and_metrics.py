"""Unit tests of the geometry, metrics and post-processing (no model needed)."""

import numpy as np

from cief_net import tiling
from cief_net.containment import apply_containment, suppress_contained
from cief_net.metrics import compute_aji, compute_boundary_iou, compute_dice, compute_pq

H = W = 120
YY, XX = np.mgrid[:H, :W]


def ellipse(cy, cx, ry, rx):
    return ((YY - cy) / ry) ** 2 + ((XX - cx) / rx) ** 2 <= 1


def test_windows_cover_image_and_end_at_border():
    wins = tiling.generate_windows(1300, 700, 512, 0.2)
    xs, ys = sorted({x for x, _ in wins}), sorted({y for _, y in wins})
    assert xs[0] == 0 and xs[-1] == 1300 - 512
    assert ys[0] == 0 and ys[-1] == 700 - 512
    assert tiling.generate_windows(300, 200, 512, 0.2) == [(0, 0)]


def test_crop_instances_discards_small_fragments():
    square = [[0, 0], [40, 0], [40, 40], [0, 40]]
    kept = tiling.crop_instances_to_window([square], 30, 0, 64, min_area_frac=0.3)
    assert kept == []                       # only 25% of the instance inside the tile
    kept = tiling.crop_instances_to_window([square], 10, 0, 64, min_area_frac=0.3)
    assert len(kept) == 1


def test_containment_rules():
    complete = ellipse(30, 30, 12, 20)
    fragment = ellipse(30, 22, 8, 9) & complete
    isolated = ellipse(90, 90, 10, 10)
    a, b = ellipse(90, 30, 10, 12), ellipse(90, 56, 10, 12)
    merge = a | b | ellipse(90, 43, 6, 8)
    masks = [complete, fragment, isolated, a, b, merge]

    keep, _ = suppress_contained(masks)
    assert keep == [0, 2, 5]                # without protection the merge wins
    keep, info = suppress_contained(masks, protect_merges=True)
    assert keep == [0, 2, 3, 4]             # with protection the neighbours are kept
    assert info["n_fragments_removed"] == 1 and info["n_merges_removed"] == 1

    near_copy = ellipse(30, 31, 12, 19) & complete
    assert suppress_contained([complete, near_copy])[0] == [0, 1]   # area ratio > 0.7

    pred = {"masks": np.stack(masks), "scores": np.linspace(.5, 1, 6), "names": list("abcdef")}
    out, _ = apply_containment(pred)
    assert out["masks"].shape[0] == 3 and list(out["names"]) == ["a", "c", "f"]


def test_metrics_perfect_and_empty():
    gt = [ellipse(30, 30, 10, 14), ellipse(80, 80, 12, 9)]
    assert compute_aji(gt, gt) == 1.0
    pq = compute_pq(gt, gt)
    assert pq["pq"] == 1.0 and pq["tp"] == 2
    assert compute_pq([], gt)["fn"] == 2
    fg = gt[0] | gt[1]
    assert compute_dice(fg, fg) == 1.0 and compute_boundary_iou(fg, fg) == 1.0


def test_pq_counts_split_instance():
    gt = [ellipse(60, 60, 20, 30)]
    pieces = [gt[0] & (XX < 50), gt[0] & (XX >= 50) & (XX < 70), gt[0] & (XX >= 70)]
    pq = compute_pq(pieces, gt)             # every piece has IoU < 0.5 with the conidium
    assert pq["tp"] == 0 and pq["fp"] == 3 and pq["fn"] == 1
    assert 0.2 < compute_aji(pieces, gt) < 0.42   # greedy: one piece matched, two penalized
