"""Checkpoint helpers.

A cIEF-Net checkpoint is a plain ``state_dict`` saved with ``torch.save``
(``checkpoint.pt``). Training also writes ``model_config.json`` next to it with
the architecture arguments, so that evaluation and inference rebuild the same
architecture without repeating command-line flags.

Checkpoints written by the original (Spanish-language) implementation used
different module names. :func:`convert_legacy_state_dict` maps them to the
current names; :func:`load_checkpoint` applies the conversion automatically.
"""

import json
from pathlib import Path

import torch

CONFIG_NAME = "model_config.json"

# Architecture arguments of build_cief_net that are stored with a checkpoint.
ARCH_KEYS = ("backbone", "patch_size", "decoder_dim", "decoder_stride", "n_rays", "dora_r",
             "dora_alpha", "dora_target_modules", "use_density_head", "use_offset_head",
             "use_refiner", "refiner_type", "refiner_kmax", "refiner_iters", "refiner_profile",
             "refiner_recenter")

# Prefix renames from the original implementation (applied in order; the
# first matching prefix wins).
_LEGACY_PREFIXES = [
    ("decoder.refinar.", "decoder.refine."),
    ("cabeza_stardist.radios.", "star_head.dist."),
    ("cabeza_stardist.prob.", "star_head.prob."),
    ("cabeza_densidad.", "density_head."),
    ("cabeza_offset.", "offset_head."),
    ("refinador.proy_dec.", "refiner.dec_proj."),
    ("refinador.cod_perfil.", "refiner.profile_encoder."),
    ("refinador.norm_circ.", "refiner.circ_norms."),
    ("refinador.circ_rayos.", "refiner.ray_convs."),
    ("refinador.norm_rayos.", "refiner.ray_norms."),
    ("refinador.circ.", "refiner.circ_convs."),
    ("refinador.cabeza.", "refiner.head."),
    ("refinador.residuo.", "refiner.residual."),
    ("refinador.traslacion.", "refiner.translation."),
    ("refinador.", "refiner."),
]

_LEGACY_REFINER_TYPES = {"armonico": "harmonic", "armonico_movil": "harmonic_moving"}


def is_legacy_state_dict(state: dict) -> bool:
    return any(k.startswith(("cabeza_stardist.", "refinador.", "decoder.refinar."))
               for k in state)


def convert_legacy_state_dict(state: dict) -> dict:
    """Rename the keys of an original-implementation state dict."""
    out = {}
    for key, value in state.items():
        new = key
        for old, rep in _LEGACY_PREFIXES:
            if key.startswith(old):
                new = rep + key[len(old):]
                break
        out[new] = value
    return out


def read_state_dict(path, map_location="cpu") -> dict:
    """Load a checkpoint file and convert legacy key names if needed."""
    state = torch.load(path, map_location=map_location)
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    if is_legacy_state_dict(state):
        state = convert_legacy_state_dict(state)
    return state


def load_checkpoint(model, path, map_location="cpu", strict: bool = True):
    """Load ``path`` into ``model`` (legacy names are converted on the fly)."""
    return model.load_state_dict(read_state_dict(path, map_location), strict=strict)


def save_model_config(out_dir, config: dict):
    cfg = {k: config[k] for k in ARCH_KEYS if k in config}
    path = Path(out_dir) / CONFIG_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    return path


def find_model_config(checkpoint_path):
    """``model_config.json`` next to the checkpoint, or ``None``."""
    path = Path(checkpoint_path).parent / CONFIG_NAME
    if not path.exists():
        return None
    with open(path) as f:
        cfg = json.load(f)
    if cfg.get("refiner_type") in _LEGACY_REFINER_TYPES:
        cfg["refiner_type"] = _LEGACY_REFINER_TYPES[cfg["refiner_type"]]
    return cfg
