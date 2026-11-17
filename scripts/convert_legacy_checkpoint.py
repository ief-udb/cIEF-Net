#!/usr/bin/env python3
"""Convert a checkpoint of the original implementation to the current module
names and write ``model_config.json`` next to it.

Loading code converts legacy checkpoints on the fly, so this script is only
needed to publish clean weight files. The architecture flags describe the
legacy run (e.g. ``--use-refiner --refiner-type harmonic --refiner-kmax 24``
for the final cIEF-Net). The conversion is verified by loading the result
into the rebuilt architecture with ``strict=True`` (``--verify``, which needs
the backbone weights).

Example::

    python scripts/convert_legacy_checkpoint.py --legacy old/checkpoint.pt \\
        --out checkpoints/cief_net_alternaria_s42/checkpoint.pt \\
        --use-refiner --refiner-type harmonic --refiner-kmax 24 --verify
"""

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cief_net.checkpoints import convert_legacy_state_dict, is_legacy_state_dict, save_model_config  # noqa: E402
from cief_net.cli import add_architecture_args, arch_from_args  # noqa: E402
from cief_net.model import build_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--legacy", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--verify", action="store_true")
    add_architecture_args(ap)
    args = ap.parse_args()

    state = torch.load(args.legacy, map_location="cpu")
    if not is_legacy_state_dict(state):
        print("note: the checkpoint already uses the current names; it is copied unchanged")
    else:
        state = convert_legacy_state_dict(state)
    arch = arch_from_args(args)
    has_refiner = any(k.startswith("refiner.") for k in state)
    if has_refiner != arch["use_refiner"]:
        sys.exit(f"the checkpoint {'has' if has_refiner else 'has no'} refiner weights; "
                 f"{'add' if has_refiner else 'remove'} --use-refiner")
    if args.verify:
        model = build_model("cief_net", pretrained=True, **arch)
        model.load_state_dict(state, strict=True)
        print("verified: the converted weights load with strict=True")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, out)
    save_model_config(out.parent, arch)
    print(f"saved {out} and {out.parent / 'model_config.json'}")


if __name__ == "__main__":
    main()
