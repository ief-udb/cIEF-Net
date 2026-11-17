"""Support-derived scale calibration for few-shot adaptation.

The radial head learns the range of radii of its training data (about 5-40
pixels inside a 512-pixel tile for *A. alternata* at a longest side of 1536
pixels). Conidia of a different physical or optical size would fall outside
that range at the same working scale. This is a problem of relative object
size, not of learning rate or number of epochs.

Every training run therefore stores the median equivalent diameter
``2 sqrt(area / pi)`` of its training instances next to its checkpoint
(``diameter_calibration.json``). A few-shot run measures the same quantity on
its *support* annotations and sets the effective working side

    s_eff = ref_side * d_ref / d_support,

so that target conidia appear at the size the head was trained on. Only the
support set is used; the test images are never involved. The calibrated side
is stored in the run summary (``ref_side_calibrated``) and reused at
evaluation and inference time.
"""

import json
import sys
from pathlib import Path

import numpy as np

from .io import load_image_and_polygons
from .tiling import polygon_area

CALIBRATION_NAME = "diameter_calibration.json"
LEGACY_CALIBRATION_NAME = "calibracion_diametro.json"
_LEGACY_KEYS = {"mediana_px": "median_px", "media_px": "mean_px",
                "n_instancias": "n_instances", "n_imagenes": "n_images"}


def equivalent_diameter(poly: list) -> float:
    """Diameter of the circle with the polygon's area."""
    area = polygon_area(poly)
    return 2.0 * float(np.sqrt(area / np.pi)) if area > 0 else 0.0


def measure_median_diameter(names: list, pairs: dict, ref_side: int) -> dict:
    """Median (and mean) equivalent diameter of every annotated instance of
    ``names`` at the working scale ``ref_side``."""
    diameters = []
    for name in names:
        img_path, json_path = pairs[name]
        _, polygons, _ = load_image_and_polygons(img_path, json_path, ref_side)
        diameters += [d for d in (equivalent_diameter(p) for p in polygons) if d > 0]
    if not diameters:
        sys.exit("no instance diameter could be measured; check the images and annotations")
    return {"median_px": float(np.median(diameters)), "mean_px": float(np.mean(diameters)),
            "n_instances": len(diameters), "n_images": len(names), "ref_side": ref_side}


def save_calibration(names_train: list, pairs: dict, ref_side: int, out_path: Path) -> dict:
    """Measure and store the calibration of a training run next to its checkpoint."""
    measured = measure_median_diameter(names_train, pairs, ref_side)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(measured, f, indent=2)
    print(f"  diameter calibration saved to {out_path} (median {measured['median_px']:.1f} px, "
          f"{measured['n_instances']} instances)")
    return measured


def read_calibration(path) -> dict:
    with open(path) as f:
        cal = json.load(f)
    cal = {_LEGACY_KEYS.get(k, k): v for k, v in cal.items()}
    for key in ("median_px", "ref_side", "n_instances"):
        if key not in cal:
            sys.exit(f"{path} has no '{key}' entry; is it a diameter calibration file?")
    return cal


def find_calibration(checkpoint_path):
    """Calibration file stored next to a checkpoint (current or legacy name)."""
    folder = Path(checkpoint_path).parent
    for name in (CALIBRATION_NAME, LEGACY_CALIBRATION_NAME):
        if (folder / name).exists():
            return folder / name
    return None


def calibrate_ref_side(ref_side: int, reference_path, names_support: list, pairs: dict) -> int:
    """Effective working side for a few-shot run (see the module docstring)."""
    reference = read_calibration(reference_path)
    d_ref = reference["median_px"]
    if reference["ref_side"] != ref_side:
        print(f"  [warning] the reference was measured at ref_side={reference['ref_side']} but "
              f"--ref-side is {ref_side}; the factor remains valid (diameter and side scale "
              f"linearly), but check that this is intended")
    measured = measure_median_diameter(names_support, pairs, ref_side)
    d_support = measured["median_px"]
    factor = d_ref / d_support
    calibrated = max(1, round(ref_side * factor))
    print("  scale calibration (few-shot support):")
    print(f"    reference median diameter: {d_ref:.1f} px at ref_side={reference['ref_side']} "
          f"({reference['n_instances']} instances)")
    print(f"    support median diameter:   {d_support:.1f} px at ref_side={ref_side} "
          f"({measured['n_images']} images, {measured['n_instances']} instances)")
    print(f"    factor {factor:.3f} -> ref_side {ref_side} -> {calibrated}")
    return calibrated


def ref_side_from_run(checkpoint_path, default: int) -> int:
    """``ref_side`` used to train a checkpoint: the calibrated side of a few-shot
    run or the working side of any run, as stored in its ``summary.json``;
    ``default`` when no summary is found."""
    folder = Path(checkpoint_path).parent
    for name, keys in (("summary.json", ("ref_side_calibrated", "ref_side")),
                       ("resumen.json", ("ref_side_calibrado",))):
        path = folder / name
        if path.exists():
            with open(path) as f:
                summary = json.load(f)
            for key in keys:
                if summary.get(key):
                    return int(round(summary[key]))
    return default
