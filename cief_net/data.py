"""Training data for cIEF-Net.

Images are rescaled to the working scale, augmented on the polygon level and
cut into tiles (see :mod:`cief_net.tiling`). For every tile, the targets of the
star-convex head are generated at the *reduced* resolution of the decoder
output (``tile_size / grid_stride``) with the geometric utilities of the
``stardist`` package (``star_dist`` and ``edt_prob``), so that no geometric
algorithm is re-implemented here:

* ``prob`` (Hr, Wr): object (centre) probability, the normalized Euclidean
  distance transform of each instance, subsampled;
* ``dist`` (n_rays, Hr, Wr): radial distances to the boundary along
  ``n_rays`` equally spaced rays;
* ``density`` and ``offset``: targets of two exploratory heads that are not
  part of the final model (see :mod:`cief_net.model`).

``grid_stride`` must equal the output stride of the decoder. ``star_dist(...,
grid=(s, s))`` and ``edt_prob(...)[::s, ::s]`` are aligned cell by cell.
"""

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter
from stardist import edt_prob, star_dist
from torch.utils.data import DataLoader, Dataset

from . import tiling
from .augmentations import AugConfig, augment_geometric, augment_photometric
from .io import (REF_SIDE_PX, find_image_annotation_pairs, load_image_and_polygons,
                 names_in_folds, read_partition)

TRAIN_FOLDS = ["0", "1", "2", "3"]


# =============================================================================
# Ground truth of a full image (used for evaluation)
# =============================================================================

def full_image_instances(polygons: list, height: int, width: int, min_area: float = 1.0):
    """Masks and boxes of every annotated instance of a full image (no tiling)."""
    masks, boxes = [], []
    for poly in polygons:
        if tiling.polygon_area(poly) < min_area:
            continue
        canvas = Image.new("L", (width, height), 0)
        ImageDraw.Draw(canvas).polygon([tuple(p) for p in poly], outline=1, fill=1)
        mask = np.array(canvas, dtype=bool)
        if mask.sum() == 0:
            continue
        ys, xs = np.where(mask)
        boxes.append([float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)])
        masks.append(mask)
    return masks, boxes


# =============================================================================
# Targets of the star-convex head
# =============================================================================

def instances_to_label_map(instances: list, tile_size: int) -> np.ndarray:
    """Label map (int32, 0 = background, 1..N = instances) from the output of
    :func:`cief_net.tiling.crop_instances_to_window`."""
    label_map = np.zeros((tile_size, tile_size), dtype=np.int32)
    for i, (mask, _bbox) in enumerate(instances, start=1):
        label_map[mask] = i
    return label_map


def build_targets(label_map: np.ndarray, n_rays: int, grid_stride: int,
                  density_sigma: float = 2.0):
    """Return ``(prob, dist, density, offset)`` at the decoder resolution.

    * ``prob`` (Hr, Wr) and ``dist`` (Hr, Wr, n_rays): targets of the head.
    * ``density`` (Hr, Wr): Gaussian count map (sums to about the number of
      instances); exploratory density head only.
    * ``offset`` (2, Hr, Wr): sub-grid correction ``(dy, dx)`` in [-1, 1]
      from each reduced cell inside an instance to the full-resolution peak
      of that instance's probability; exploratory offset head only.
    """
    dist = star_dist(label_map, n_rays=n_rays, grid=(grid_stride, grid_stride))
    prob_full = edt_prob(label_map)
    prob = prob_full[::grid_stride, ::grid_stride]

    h_red, w_red = prob.shape
    density = np.zeros((h_red, w_red), dtype=np.float32)
    offset = np.zeros((2, h_red, w_red), dtype=np.float32)
    half_cell = grid_stride / 2.0
    reduced_labels = label_map[::grid_stride, ::grid_stride]  # same subsampling as `prob`

    for label in np.unique(label_map):
        if label == 0:
            continue
        ys, xs = np.where(label_map == label)
        cy, cx = ys.mean() / grid_stride, xs.mean() / grid_stride
        cy_i, cx_i = int(round(cy)), int(round(cx))
        if 0 <= cy_i < h_red and 0 <= cx_i < w_red:
            density[cy_i, cx_i] += 1.0

        # full-resolution probability peak inside the instance (not the
        # centroid, which can fall outside curved or elongated shapes)
        inside = label_map == label
        prob_obj = np.where(inside, prob_full, -1.0)
        y_peak, x_peak = np.unravel_index(np.argmax(prob_obj), prob_obj.shape)

        cells_y, cells_x = np.where(reduced_labels == label)
        for gy, gx in zip(cells_y, cells_x):
            offset[0, gy, gx] = np.clip((y_peak - gy * grid_stride) / half_cell, -1.0, 1.0)
            offset[1, gy, gx] = np.clip((x_peak - gx * grid_stride) / half_cell, -1.0, 1.0)

    density = gaussian_filter(density, sigma=density_sigma)
    return (prob.astype(np.float32), dist.astype(np.float32),
            density.astype(np.float32), offset.astype(np.float32))


def instance_peak_cells(label_map: np.ndarray, prob_reduced: np.ndarray,
                        grid_stride: int) -> np.ndarray:
    """One reduced-grid cell per complete instance of the tile, the one with the
    highest target probability (the kind of point later selected by polygon
    NMS). These cells anchor the contour-refinement loss. Returns (M, 2) int64
    ``(gy, gx)``.

    Instances touching the tile border are excluded (their target rays are
    truncated by the crop and do not describe the real contour), as are
    instances without any grid cell inside (smaller than about ``grid_stride``
    pixels)."""
    reduced = label_map[::grid_stride, ::grid_stride]
    border = np.unique(np.concatenate([label_map[0], label_map[-1],
                                       label_map[:, 0], label_map[:, -1]]))
    cells = []
    for label in np.unique(reduced):
        if label == 0 or label in border:
            continue
        gy, gx = np.where(reduced == label)
        k = int(np.argmax(prob_reduced[gy, gx]))
        cells.append((int(gy[k]), int(gx[k])))
    return np.asarray(cells, dtype=np.int64).reshape(-1, 2)


# =============================================================================
# Tile-level dataset
# =============================================================================

class ConidiaTileDataset(Dataset):
    """Every item is one tile of one image, with its head targets.

    The list of tiles is fixed at construction time from the image sizes at
    the working scale; augmentation (if enabled) is drawn anew every time a
    tile is read.
    """

    def __init__(self, names: list, pairs: dict, tile_size: int = 512,
                 overlap: float = 0.2, ref_side: int = REF_SIDE_PX,
                 min_area_frac: float = 0.3, n_rays: int = 48, grid_stride: int = 4,
                 density_sigma: float = 2.0, train: bool = False, augment: bool = True,
                 aug_config: AugConfig = None, cache_images: bool = True):
        self.pairs = pairs
        self.tile_size = tile_size
        self.overlap = overlap
        self.ref_side = ref_side
        self.min_area_frac = min_area_frac
        self.n_rays = n_rays
        self.grid_stride = grid_stride
        self.density_sigma = density_sigma
        self.train = train
        self.augment = augment and train
        self.aug_config = aug_config or AugConfig()
        self._cache = {} if cache_images else None

        self.index = []
        for name in names:
            img_path, _ = pairs[name]
            with Image.open(img_path) as im:
                w0, h0 = im.size
            factor = ref_side / max(w0, h0)
            w_s, h_s = max(1, round(w0 * factor)), max(1, round(h0 * factor))
            windows = tiling.generate_windows(w_s, h_s, tile_size, overlap)
            self.index.extend((name, win) for win in windows)

    def __len__(self):
        return len(self.index)

    def _load(self, name):
        if self._cache is not None and name in self._cache:
            return self._cache[name]
        img_path, json_path = self.pairs[name]
        image, polygons, _ = load_image_and_polygons(img_path, json_path, self.ref_side)
        if self._cache is not None:
            self._cache[name] = (image, polygons)
        return image, polygons

    def __getitem__(self, idx):
        name, (x0, y0) = self.index[idx]
        image, polygons = self._load(name)

        if self.augment:
            image, polygons = augment_geometric(image, polygons, self.aug_config)

        crop = tiling.crop_pixels(image, x0, y0, self.tile_size)
        instances = tiling.crop_instances_to_window(polygons, x0, y0, self.tile_size,
                                                    self.min_area_frac)
        if self.augment:
            crop = augment_photometric(crop, self.aug_config)

        label_map = instances_to_label_map(instances, self.tile_size)
        prob, dist, density, offset = build_targets(label_map, self.n_rays,
                                                    self.grid_stride, self.density_sigma)

        image_t = torch.as_tensor(crop, dtype=torch.float32).permute(2, 0, 1) / 255.0
        target = {
            "prob": torch.as_tensor(prob, dtype=torch.float32).unsqueeze(0),            # (1, Hr, Wr)
            "dist": torch.as_tensor(dist, dtype=torch.float32).permute(2, 0, 1),        # (n_rays, Hr, Wr)
            "density": torch.as_tensor(density, dtype=torch.float32).unsqueeze(0),      # (1, Hr, Wr)
            "offset": torch.as_tensor(offset, dtype=torch.float32),                     # (2, Hr, Wr)
            "ref_cells": torch.as_tensor(instance_peak_cells(label_map, prob, self.grid_stride)),
            "image_name": name,
        }
        return image_t, target


def collate_fn(batch):
    images, targets = list(zip(*batch))
    out = {key: torch.stack([t[key] for t in targets], dim=0)
           for key in ("prob", "dist", "density", "offset")}
    out["names"] = [t["image_name"] for t in targets]
    # refinement anchors are flattened together with their batch index
    out["ref_b"] = torch.cat([torch.full((len(t["ref_cells"]),), i, dtype=torch.long)
                              for i, t in enumerate(targets)])
    cells = torch.cat([t["ref_cells"] for t in targets]).reshape(-1, 2)
    out["ref_gy"], out["ref_gx"] = cells[:, 0], cells[:, 1]
    return torch.stack(images, dim=0), out


# =============================================================================
# Data loaders
# =============================================================================

def _train_loader(names, pairs, *, tile_size, overlap, ref_side, min_area_frac, n_rays,
                  grid_stride, density_sigma, batch_size, num_workers, augment, aug_config):
    ds = ConidiaTileDataset(names, pairs, tile_size=tile_size, overlap=overlap,
                            ref_side=ref_side, min_area_frac=min_area_frac, n_rays=n_rays,
                            grid_stride=grid_stride, density_sigma=density_sigma,
                            train=True, augment=augment, aug_config=aug_config)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                      collate_fn=collate_fn)


def build_loaders_cv(images_dir, split_csv, val_fold: str, tile_size=512, overlap=0.2,
                     ref_side=REF_SIDE_PX, min_area_frac=0.3, n_rays=48, grid_stride=4,
                     density_sigma=2.0, batch_size=8, num_workers=2, augment=True,
                     aug_config=None):
    """Training loader on three folds and the image names of the validation fold."""
    pairs = find_image_annotation_pairs(images_dir)
    df = read_partition(split_csv)
    if val_fold not in TRAIN_FOLDS:
        raise ValueError(f"val_fold must be one of {TRAIN_FOLDS}, got {val_fold}")
    names_train = names_in_folds(df, [f for f in TRAIN_FOLDS if f != val_fold])
    names_val = names_in_folds(df, [val_fold])
    dl = _train_loader(names_train, pairs, tile_size=tile_size, overlap=overlap,
                       ref_side=ref_side, min_area_frac=min_area_frac, n_rays=n_rays,
                       grid_stride=grid_stride, density_sigma=density_sigma,
                       batch_size=batch_size, num_workers=num_workers, augment=augment,
                       aug_config=aug_config)
    return dl, names_val, pairs


def build_loaders_holdout(images_dir, split_csv, tile_size=512, overlap=0.2,
                          ref_side=REF_SIDE_PX, min_area_frac=0.3, n_rays=48, grid_stride=4,
                          density_sigma=2.0, batch_size=8, num_workers=2, augment=True,
                          aug_config=None):
    """Training loader on folds 0-3 and the image names of the ``test`` fold."""
    pairs = find_image_annotation_pairs(images_dir)
    df = read_partition(split_csv)
    names_train = names_in_folds(df, TRAIN_FOLDS)
    names_test = names_in_folds(df, ["test"])
    dl = _train_loader(names_train, pairs, tile_size=tile_size, overlap=overlap,
                       ref_side=ref_side, min_area_frac=min_area_frac, n_rays=n_rays,
                       grid_stride=grid_stride, density_sigma=density_sigma,
                       batch_size=batch_size, num_workers=num_workers, augment=augment,
                       aug_config=aug_config)
    return dl, names_test, pairs


def build_eval_tile_loader(names: list, pairs: dict, tile_size=512, overlap=0.2,
                           ref_side=REF_SIDE_PX, min_area_frac=0.3, n_rays=48,
                           grid_stride=4, density_sigma=2.0, batch_size=8,
                           num_workers=2) -> DataLoader:
    """Tile-level loader over evaluation images without augmentation. It is
    used to compute a validation loss on exactly the same scale as the
    training loss (full-image metrics are computed by
    :func:`cief_net.inference.evaluate_images`)."""
    ds = ConidiaTileDataset(names, pairs, tile_size=tile_size, overlap=overlap,
                            ref_side=ref_side, min_area_frac=min_area_frac, n_rays=n_rays,
                            grid_stride=grid_stride, density_sigma=density_sigma,
                            train=False, augment=False)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                      collate_fn=collate_fn)
