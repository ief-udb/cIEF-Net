#!/usr/bin/env python3
"""Write the diameter calibration of a source-domain training set.

Every run of ``scripts/train.py`` already writes ``diameter_calibration.json``
next to its checkpoint, and few-shot runs find it automatically. This script
is only needed for checkpoints trained without it (e.g. released weights
copied to a new folder): it measures the median equivalent diameter of the
annotated instances of the training folds at the working scale, in the same
way as the training code.

Example::

    python scripts/measure_reference_diameter.py --images-dir data/alternaria_alternata/images \\
        --split-csv data/splits/alternaria_alternata.csv \\
        --out checkpoints/diameter_calibration.json
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cief_net.calibration import measure_median_diameter  # noqa: E402
from cief_net.io import REF_SIDE_PX, find_image_annotation_pairs, names_in_folds, read_partition  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--split-csv", required=True, help="source-domain manifest (image,fold)")
    ap.add_argument("--folds", default="0,1,2,3", help="folds used to train the checkpoint")
    ap.add_argument("--ref-side", type=int, default=REF_SIDE_PX)
    ap.add_argument("--out", default="diameter_calibration.json")
    args = ap.parse_args()

    folds = [f.strip() for f in args.folds.split(",") if f.strip()]
    names = names_in_folds(read_partition(args.split_csv), folds)
    if not names:
        sys.exit(f"no image in folds {folds}")
    result = measure_median_diameter(names, find_image_annotation_pairs(args.images_dir), args.ref_side)
    result.update(folds=folds, split_csv=str(Path(args.split_csv).resolve()))
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"median equivalent diameter: {result['median_px']:.1f} px ({result['n_instances']} instances, "
          f"{result['n_images']} images) -> {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
