"""Command-line arguments shared by the scripts in ``scripts/``."""

import argparse

import torch

from .checkpoints import ARCH_KEYS, find_model_config, load_checkpoint
from .inference import InferenceConfig
from .io import REF_SIDE_PX
from .model import build_model

DEFAULT_BACKBONE = "facebook/dinov3-vitl16-pretrain-lvd1689m"


def add_architecture_args(ap: argparse.ArgumentParser):
    g = ap.add_argument_group("architecture (read from model_config.json next to the checkpoint "
                              "when it exists)")
    g.add_argument("--backbone", default=DEFAULT_BACKBONE,
                   help="Hugging Face identifier or local folder of the DINOv3 encoder")
    g.add_argument("--patch-size", type=int, default=16)
    g.add_argument("--decoder-dim", type=int, default=256)
    g.add_argument("--grid-stride", type=int, default=4,
                   help="output stride of the decoder; also the grid of the head targets")
    g.add_argument("--n-rays", type=int, default=48)
    g.add_argument("--dora-r", type=int, default=16)
    g.add_argument("--dora-alpha", type=int, default=None, help="default: 2 * r")
    g.add_argument("--dora-target-modules", default="q_proj,v_proj",
                   help="comma-separated encoder layers that receive DoRA adapters")
    g.add_argument("--use-refiner", action="store_true", help="the model has a contour refiner")
    g.add_argument("--refiner-type", default="harmonic", choices=["harmonic", "pose", "harmonic_moving"],
                   help="harmonic: cIEF-Net; pose: global affine (ablation); "
                        "harmonic_moving: harmonic + voted translation (exploratory)")
    g.add_argument("--refiner-kmax", type=int, default=24,
                   help="highest harmonic of the correction (24 = full band with 48 rays)")
    g.add_argument("--refiner-iters", type=int, default=2, help="refinement iterations (shared weights)")
    g.add_argument("--refiner-profile", type=int, default=8, help="samples per ray profile (0.6r to 1.4r)")
    g.add_argument("--refiner-recenter", action="store_true",
                   help="exploratory: move the polar frame to the contour centroid after every iteration")
    g.add_argument("--use-density-head", action="store_true", help="exploratory density head")
    g.add_argument("--use-offset-head", action="store_true", help="exploratory sub-grid offset head")


def add_inference_args(ap: argparse.ArgumentParser, ref_side_default=REF_SIDE_PX):
    g = ap.add_argument_group("inference")
    g.add_argument("--tile-size", type=int, default=512)
    g.add_argument("--overlap", type=float, default=0.2)
    g.add_argument("--ref-side", type=int, default=ref_side_default,
                   help="longest side of the working scale (1536 for A. alternata; few-shot runs "
                        "use their calibrated side)")
    g.add_argument("--prob-thresh", type=float, default=0.5)
    g.add_argument("--nms-thresh", type=float, default=0.5)
    g.add_argument("--iou-merge", type=float, default=0.5, help="IoU for merging tile duplicates")
    g.add_argument("--max-det", type=int, default=1000)
    g.add_argument("--no-component-filter", action="store_true",
                   help="disable the one-detection-per-connected-component rule")
    g.add_argument("--containment", type=float, default=None,
                   help="containment threshold tau (cIEF-Net: 0.8); omitted = no suppression")
    g.add_argument("--containment-area-ratio", type=float, default=0.7, help="rho")
    g.add_argument("--protect-merges", action="store_true",
                   help="remove a prediction that is covered by >= 2 nearly disjoint smaller ones")
    g.add_argument("--no-refine", action="store_true",
                   help="load the refiner but do not apply it (ablation on the same checkpoint)")
    g.add_argument("--adaptive-nms", action="store_true", help="exploratory (density head)")
    g.add_argument("--density-ref", default="auto")
    g.add_argument("--iou-max-adaptive", type=float, default=0.95)


def add_paper_containment_flag(ap: argparse.ArgumentParser):
    ap.add_argument("--paper-postprocessing", action="store_true",
                    help="shortcut for --containment 0.8 --containment-area-ratio 0.7 --protect-merges")


def apply_paper_postprocessing(args):
    if getattr(args, "paper_postprocessing", False):
        args.containment, args.containment_area_ratio, args.protect_merges = 0.8, 0.7, True


def parse_target_modules(text: str) -> list:
    return [m.strip() for m in text.split(",") if m.strip()]


def arch_from_args(args) -> dict:
    return dict(backbone=args.backbone, patch_size=args.patch_size, decoder_dim=args.decoder_dim,
                decoder_stride=args.grid_stride, n_rays=args.n_rays, dora_r=args.dora_r,
                dora_alpha=args.dora_alpha,
                dora_target_modules=parse_target_modules(args.dora_target_modules)
                if isinstance(args.dora_target_modules, str) else args.dora_target_modules,
                use_density_head=args.use_density_head, use_offset_head=args.use_offset_head,
                use_refiner=args.use_refiner, refiner_type=args.refiner_type,
                refiner_kmax=args.refiner_kmax, refiner_iters=args.refiner_iters,
                refiner_profile=args.refiner_profile, refiner_recenter=args.refiner_recenter)


def load_model_for_inference(args, checkpoint, device):
    """Rebuild the architecture (from ``model_config.json`` when available,
    otherwise from the command line), load the weights and return the model
    in eval mode. ``args.grid_stride`` is synchronized with the checkpoint."""
    arch = arch_from_args(args)
    stored = find_model_config(checkpoint)
    if stored:
        arch.update({k: v for k, v in stored.items() if k in ARCH_KEYS})
        args.grid_stride = arch["decoder_stride"]
        print(f"architecture read from model_config.json next to {checkpoint}")
    model = build_model("cief_net", pretrained=True, **arch)
    load_checkpoint(model, checkpoint, map_location=device)
    model.to(device).eval()
    return model, arch


def inference_config(args) -> InferenceConfig:
    return InferenceConfig(tile_size=args.tile_size, overlap=args.overlap, ref_side=args.ref_side,
                           grid_stride=args.grid_stride, prob_thresh=args.prob_thresh,
                           nms_thresh=args.nms_thresh, iou_merge=args.iou_merge,
                           one_per_component=not args.no_component_filter,
                           refine=not args.no_refine, containment=args.containment,
                           containment_area_ratio=args.containment_area_ratio,
                           protect_merges=args.protect_merges, adaptive_nms=args.adaptive_nms,
                           density_ref=args.density_ref, iou_max_adaptive=args.iou_max_adaptive)


def resolve_device(name):
    if name:
        return name if (name == "cpu" or torch.cuda.is_available()) else "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"
