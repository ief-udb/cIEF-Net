#!/usr/bin/env python3
"""Copy partition manifests into the layout expected by the repository.

Source domain (``--source``): a manifest with an image column and a ``fold``
column (``0``-``3`` and ``test``), written to
``data/splits/alternaria_alternata.csv`` as ``image,fold``.

Few-shot (``--fewshot``, repeatable): CSV files or folders of CSV files named
``split_fewshot_<N>_[<species>_]s<seed>.csv``. Each is written to
``data/splits/fewshot/<species>/<N>shot_s<seed>.csv`` with the columns
``image,species,subset`` (subset: ``train``, ``test``, ``unused``). The
collection folder is taken from the file name, from ``--species``, or from a
single-valued ``species``/``especie`` column; a per-image ``species`` column is
kept as it is. Files whose name
does not match the pattern (e.g. ``split_full_*``) are skipped.

Example::

    python scripts/prepare_splits.py --source old/split_5folds.csv \\
        --fewshot old/particiones_tgfc \\
        --fewshot old/particiones_lasiodiplodia --species lasiodiplodia \\
        --out data/splits
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cief_net.io import read_partition  # noqa: E402

PATTERN = re.compile(r"split_fewshot_(\d+)_(?:(.+?)_)?s(\d+)\.csv$")
SPECIES = {"lasiodiplodia": "lasiodiplodia", "colletotrichum_siamense": "colletotrichum_siamense",
           "c_siamense": "colletotrichum_siamense", "neopestalotiopsis": "neopestalotiopsis",
           "hortaea_werneckii": "hortaea_werneckii", "hortaea": "hortaea_werneckii"}


def slug(name: str) -> str:
    key = name.strip().lower().replace(" ", "_").replace(".", "")
    return SPECIES.get(key, key)


def convert_fewshot(path: Path, species_arg: str, out_root: Path):
    m = PATTERN.search(path.name)
    if not m:
        print(f"  [skip] {path.name}: name does not match split_fewshot_<N>_[<species>_]s<seed>.csv")
        return
    n, species_name, seed = m.group(1), m.group(2), m.group(3)
    df = read_partition(path)
    if "subset" not in df.columns:
        sys.exit(f"{path}: no subset/conjunto column")
    if species_name:
        species = slug(species_name)
    elif species_arg:
        species = slug(species_arg)
    elif "species" in df.columns and df["species"].nunique() == 1:
        species = slug(str(df["species"].iloc[0]))
    else:
        sys.exit(f"{path}: species unknown; pass --species")
    # the species column keeps the per-image taxon when the manifest has one
    # (e.g. the four Lasiodiplodia species); the folder uses the collection name
    taxon = df["species"] if "species" in df.columns else species
    out = pd.DataFrame({"image": df["image"], "species": taxon, "subset": df["subset"]})
    bad = set(out["subset"]) - {"train", "test", "unused"}
    if bad:
        sys.exit(f"{path}: unexpected subset values {sorted(bad)}")
    dest = out_root / "fewshot" / species / f"{n}shot_s{seed}.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest, index=False)
    counts = out["subset"].value_counts().to_dict()
    print(f"  {path.name} -> {dest.relative_to(out_root.parent)}  {counts}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", help="source-domain manifest (image,fold)")
    ap.add_argument("--fewshot", action="append", default=[], help="few-shot CSV or folder (repeatable)")
    ap.add_argument("--species", default=None, help="species for few-shot files that do not name it")
    ap.add_argument("--out", default=str(ROOT / "data" / "splits"))
    args = ap.parse_args()
    out_root = Path(args.out)

    if args.source:
        df = read_partition(args.source)
        if "fold" not in df.columns:
            sys.exit(f"{args.source}: no fold column")
        dest = out_root / "alternaria_alternata.csv"
        dest.parent.mkdir(parents=True, exist_ok=True)
        df[["image", "fold"]].to_csv(dest, index=False)
        print(f"  {Path(args.source).name} -> {dest}  {df['fold'].value_counts().to_dict()}")

    for item in args.fewshot:
        p = Path(item)
        files = sorted(p.glob("*.csv")) if p.is_dir() else [p]
        for f in files:
            convert_fewshot(f, args.species, out_root)


if __name__ == "__main__":
    main()
