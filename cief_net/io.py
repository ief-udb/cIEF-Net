"""Input/output helpers: image-annotation pairs, LabelMe polygons and partition
manifests.

Annotation format
-----------------
Every image ``<name>.<ext>`` is accompanied by a LabelMe-style JSON file
``<name>.json`` in the same folder, with one polygon per conidium::

    {"shapes": [{"label": "conidium", "points": [[x1, y1], [x2, y2], ...]}, ...]}

Partition manifests
-------------------
Two CSV layouts are supported (column names are case-sensitive):

* **Source-domain split** (``image,fold``): ``fold`` is ``0``-``3`` for the four
  cross-validation folds of the training set and ``test`` for the held-out
  test set.
* **Few-shot split** (``image,species,subset``): ``subset`` is ``train`` (the
  support set), ``test`` or ``unused`` (pool images not drawn for this support
  set).

Manifests produced by earlier, Spanish-language versions of the code
(``imagen``, ``especie``, ``conjunto``, ``no_usado``) are read transparently.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
REF_SIDE_PX = 1536  # longest side of the source-domain working scale

_IMAGE_COLUMNS = ("image", "imagen", "nombre", "archivo", "filename", "nombre_imagen", "file")
_LEGACY_COLUMNS = {"especie": "species", "conjunto": "subset"}
_LEGACY_VALUES = {"no_usado": "unused"}


# =============================================================================
# Images and annotations
# =============================================================================

def find_image_annotation_pairs(images_dir) -> dict:
    """Map ``image file name -> (image path, json path)`` for every image under
    ``images_dir`` (searched recursively) that has a JSON file next to it."""
    pairs = {}
    for img_path in Path(images_dir).rglob("*"):
        if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        json_path = img_path.with_suffix(".json")
        if json_path.exists():
            pairs[img_path.name] = (img_path, json_path)
    return pairs


def find_images(images_dir) -> list:
    """All image files under ``images_dir`` (recursive), annotated or not."""
    return sorted(p for p in Path(images_dir).rglob("*")
                  if p.suffix.lower() in IMAGE_EXTENSIONS)


def load_polygons(json_path) -> list:
    """Polygons (lists of ``[x, y]``) with at least three vertices."""
    with open(json_path, "r", encoding="utf-8") as f:
        ann = json.load(f)
    polygons = []
    for shape in ann.get("shapes", []):
        pts = shape.get("points", [])
        if len(pts) >= 3:
            polygons.append([[float(x), float(y)] for x, y in pts])
    return polygons


def load_image_rescaled(img_path, ref_side: int = REF_SIDE_PX):
    """RGB image rescaled (bilinear) so that its longest side equals
    ``ref_side``. Returns ``(image uint8 (H, W, 3), scale factor)``."""
    with Image.open(img_path) as im:
        im = im.convert("RGB")
        w0, h0 = im.size
        factor = ref_side / max(w0, h0)
        nw, nh = max(1, round(w0 * factor)), max(1, round(h0 * factor))
        image = np.array(im.resize((nw, nh), Image.BILINEAR), dtype=np.uint8)
    return image, factor


def load_image_and_polygons(img_path, json_path, ref_side: int = REF_SIDE_PX):
    """Image and polygons at the working scale. Returns
    ``(image, polygons, scale factor)``."""
    image, factor = load_image_rescaled(img_path, ref_side)
    polygons = [[[x * factor, y * factor] for x, y in poly] for poly in load_polygons(json_path)]
    return image, polygons, factor


# =============================================================================
# Partition manifests
# =============================================================================

def read_partition(csv_path) -> pd.DataFrame:
    """Read a partition manifest and normalize it to English column names
    (``image`` plus ``fold`` and/or ``species``/``subset``)."""
    df = pd.read_csv(csv_path)
    image_col = next((c for c in _IMAGE_COLUMNS if c in df.columns), None)
    if image_col is None:
        sys.exit(f"{csv_path}: no image-name column found (tried {list(_IMAGE_COLUMNS)}); "
                 f"columns present: {list(df.columns)}")
    df = df.rename(columns={image_col: "image", **_LEGACY_COLUMNS})
    if "fold" not in df.columns and "subset" not in df.columns:
        sys.exit(f"{csv_path}: a 'fold' or a 'subset' column is required; "
                 f"columns present: {list(df.columns)}")
    if "fold" in df.columns:
        df["fold"] = df["fold"].astype(str)
    if "subset" in df.columns:
        df["subset"] = df["subset"].astype(str).replace(_LEGACY_VALUES)
    return df


def names_in_folds(df: pd.DataFrame, folds: list) -> list:
    """Image names whose ``fold`` is in ``folds`` (source-domain manifests)."""
    if "fold" not in df.columns:
        raise ValueError("the manifest has no 'fold' column")
    return df.loc[df["fold"].isin([str(f) for f in folds]), "image"].tolist()


def names_in_subset(df: pd.DataFrame, subset: str) -> list:
    """Image names of one subset. Uses the ``subset`` column when present
    (few-shot manifests) and the ``fold`` column otherwise (so ``test`` or a
    fold number can be requested from a source-domain manifest)."""
    column = "subset" if "subset" in df.columns else "fold"
    return df.loc[df[column] == str(subset), "image"].tolist()


def fewshot_to_fold_manifest(fewshot_csv, out_path: Path) -> Path:
    """Convert a few-shot manifest (``image,species,subset``) to the fold
    layout used by the training code: support images become fold ``0`` and
    test images fold ``test``; unused pool images are dropped."""
    df = read_partition(fewshot_csv)
    if "subset" not in df.columns:
        sys.exit(f"--fewshot-csv {fewshot_csv} has no 'subset' column")
    fold = df["subset"].map({"train": "0", "test": "test"})
    out = pd.DataFrame({"image": df["image"], "fold": fold}).dropna()
    if out.empty:
        sys.exit(f"--fewshot-csv {fewshot_csv} has no row with subset 'train' or 'test'")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"  few-shot manifest converted to fold layout: {out_path} "
          f"({int((out['fold'] == '0').sum())} train, {int((out['fold'] == 'test').sum())} test)")
    return out_path
