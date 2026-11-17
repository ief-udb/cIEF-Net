"""Training engine: optimizer, schedule, epoch loop and final evaluation.

The optimizer has two learning-rate groups: a low rate for the DoRA
parameters, which adapt a pretrained encoder that should not move far, and a
higher rate for the newly initialized decoder, head and refiner.

The checkpoint is written *before* the final evaluation, so a failing
evaluation never loses a trained model. Results are always reported for the
weights of the last epoch; ``checkpoint_best.pt`` (best intermediate epoch on
the evaluation images) is optional and never used for reporting.
"""

import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .checkpoints import read_state_dict
from .data import build_eval_tile_loader
from .cli import arch_from_args, inference_config
from .inference import evaluate_images
from .losses import LossConfig, compute_loss
from .model import build_model, count_parameters

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_optimizer(model, lr_backbone: float, lr_new: float, weight_decay: float = 0.05):
    """AdamW with one group for the backbone adapters and one for everything
    else. Empty groups (e.g. when only the refiner is trained) are dropped."""
    backbone = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("backbone")]
    new = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("backbone")]
    groups = [g for g in ({"params": backbone, "lr": lr_backbone}, {"params": new, "lr": lr_new})
              if g["params"]]
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


def build_scheduler(optimizer, scheduler: str, epochs: int, t_max: int = None,
                    step_size: int = 10, gamma: float = 0.1):
    """Cosine annealing with ``T_max = t_max`` (defaults to ``epochs``). A
    period longer than the run (``--epochs 30 --lr-t-max 60`` for cIEF-Net)
    trains on the first half of the cosine, so the learning rate does not
    collapse at the last epoch."""
    if scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=t_max if t_max is not None else epochs, eta_min=0.0)
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)


def train_one_epoch(model, loader, optimizer, device, loss_cfg: LossConfig) -> float:
    model.train()
    total, n_batches, n_skipped = 0.0, 0, 0
    ref_log = {"init": 0.0, "final": 0.0, "n": 0}
    for images, targets in tqdm(loader, desc="  train", unit="batch"):
        output = model(images.to(device))
        loss, log = compute_loss(output, targets, loss_cfg, model)
        if "ref_err_init" in log and np.isfinite(log["ref_err_init"]):
            ref_log["init"] += log["ref_err_init"]
            ref_log["final"] += log["ref_err_final"]
            ref_log["n"] += 1
        # when only the refiner is trained, a batch without complete instances
        # has no term with a gradient
        if not loss.requires_grad:
            continue
        if not torch.isfinite(loss):
            n_skipped += 1
            optimizer.zero_grad()
            continue
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += float(loss.item())
        n_batches += 1

    if n_skipped:
        print(f"  WARNING: {n_skipped} batches skipped because of a non-finite loss")
    if ref_log["n"]:
        print(f"  refiner (predicted contours only): vertex error / radius  initial="
              f"{ref_log['init'] / ref_log['n']:.4f}  refined={ref_log['final'] / ref_log['n']:.4f}  (train)")
    return total / max(n_batches, 1)


@torch.no_grad()
def eval_tile_loss(model, loader, device, loss_cfg: LossConfig) -> float:
    """Validation loss with exactly the training formula (tile by tile, no
    decoding), so that training and validation curves are comparable."""
    model.eval()
    total, n = 0.0, 0
    for images, targets in loader:
        loss, _ = compute_loss(model(images.to(device)), targets, loss_cfg, model)
        if torch.isfinite(loss):
            total += float(loss.item())
            n += 1
    return total / max(n, 1)


def loss_config_from_args(args) -> LossConfig:
    return LossConfig(radial_loss=args.radial_loss, boundary_weighting=not args.no_boundary_weighting,
                      w_prob=args.w_prob, w_radial=args.w_radial, w_density=args.w_density,
                      w_offset=args.w_offset, w_refiner=args.w_refiner, use_refiner=args.use_refiner,
                      grid_stride=args.grid_stride, refiner_p_gt=args.refiner_p_gt,
                      refiner_max_instances=args.refiner_max_instances)


def _load_initial(model, path, device, args):
    print(f"  warm start from {path}")
    state = read_state_dict(path, map_location=device)
    if args.reset_refiner:
        n = sum(k.startswith("refiner.") for k in state)
        state = {k: v for k, v in state.items() if not k.startswith("refiner.")}
        print(f"  --reset-refiner: {n} refiner tensors of the initial checkpoint discarded")
    if args.use_refiner:
        # A base checkpoint has no refiner weights: only those keys may be
        # missing (the refiner then starts as the identity). Any other
        # mismatch is an error.
        missing, unexpected = model.load_state_dict(state, strict=False)
        other = [k for k in missing if not k.startswith("refiner.")]
        if other or unexpected:
            sys.exit(f"the initial checkpoint does not match the architecture: missing {other[:5]}, "
                     f"unexpected {list(unexpected)[:5]}")
        if missing:
            print(f"  refiner initialized from scratch ({len(missing)} tensors; it starts as the "
                  f"identity, so the model reproduces the base)")
    else:
        model.load_state_dict(state)


def _update_best(summary, args, best_value, best_epoch, epoch, model, best_path):
    value = summary.get(args.best_metric)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return best_value, best_epoch
    better = best_value is None or (value > best_value if args.best_mode == "max" else value < best_value)
    if not better:
        return best_value, best_epoch
    if best_path is not None:
        torch.save(model.state_dict(), best_path)
        print(f"    new best epoch ({args.best_metric}={value:.4f}) -> {best_path}")
    return value, epoch


_NOT_IN_HISTORY = {"label", "final_loss", "params_total", "params_trainable", "best_epoch",
                   "best_epoch_value"}


def train_run(loader, eval_names, pairs, args, device, label: str, checkpoint_path=None):
    """Build the model, train it and evaluate the last epoch.

    Returns ``(model, summary, per-image DataFrame, history)``."""
    set_seed(args.seed)
    model = build_model("cief_net", pretrained=not args.random_backbone, **arch_from_args(args))
    model.to(device)

    if args.init_checkpoint:
        _load_initial(model, args.init_checkpoint, device, args)

    if args.freeze_refiner:
        n = 0
        for name, p in model.named_parameters():
            if name.startswith("refiner."):
                p.requires_grad = False
                n += p.numel()
        print(f"  --freeze-refiner: {n:,} refiner parameters frozen (only the base is adapted)")
    if args.refiner_only:
        n = 0
        for name, p in model.named_parameters():
            p.requires_grad = name.startswith("refiner.")
            n += p.numel() if p.requires_grad else 0
        print(f"  --refiner-only: backbone, decoder and head frozen; {n:,} refiner parameters trained")

    params = count_parameters(model)
    print(f"  parameters: total={params['total']:,}  trainable={params['trainable']:,} "
          f"({params['pct_trainable']:.2f}%)")

    loss_cfg = loss_config_from_args(args)
    inf_cfg = inference_config(args)
    optimizer = build_optimizer(model, args.lr_backbone, args.lr_new, args.weight_decay)
    scheduler = build_scheduler(optimizer, args.scheduler, args.epochs, args.lr_t_max,
                                args.lr_step, args.lr_gamma)
    print(f"\n--- {label}: {len(loader.dataset)} training tiles / {len(eval_names)} evaluation images ---")

    best_path = None
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        if args.save_best_checkpoint:
            best_path = checkpoint_path.parent / "checkpoint_best.pt"

    val_loader = None
    if args.eval_every > 0:
        val_loader = build_eval_tile_loader(eval_names, pairs, tile_size=args.tile_size,
                                            overlap=args.overlap, ref_side=args.ref_side,
                                            min_area_frac=args.min_area_frac, n_rays=args.n_rays,
                                            grid_stride=args.grid_stride, batch_size=args.batch_size,
                                            num_workers=args.num_workers)

    best_value, best_epoch, history = None, None, []
    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss = train_one_epoch(model, loader, optimizer, device, loss_cfg)
        scheduler.step()
        dt = time.time() - t0
        entry = {"epoch": epoch + 1, "train_loss": train_loss, "seconds": dt}

        # the last epoch is evaluated below, together with the final checkpoint
        last = (epoch + 1) == args.epochs
        if args.eval_every > 0 and (epoch + 1) % args.eval_every == 0 and not last:
            val_loss = eval_tile_loss(model, val_loader, device, loss_cfg)
            summary_e, _ = evaluate_images(model, eval_names, pairs, device, inf_cfg,
                                           max_det=args.max_det, show_progress=False)
            entry["val_loss"] = val_loss
            entry["eval"] = {k: v for k, v in summary_e.items() if k not in _NOT_IN_HISTORY}
            best_value, best_epoch = _update_best(summary_e, args, best_value, best_epoch,
                                                  epoch + 1, model, best_path)
            print(f"  epoch {epoch + 1}/{args.epochs}: train_loss={train_loss:.4f}  "
                  f"val_loss={val_loss:.4f}  {args.best_metric}="
                  f"{summary_e.get(args.best_metric, float('nan')):.4f}  ({dt:.1f}s)")
        else:
            print(f"  epoch {epoch + 1}/{args.epochs}: train_loss={train_loss:.4f}  ({dt:.1f}s)")
        history.append(entry)

    if checkpoint_path is not None:
        torch.save(model.state_dict(), checkpoint_path)
        print(f"  checkpoint saved (before evaluation): {checkpoint_path}")

    try:
        final_val_loss = eval_tile_loss(model, val_loader, device, loss_cfg) if args.eval_every > 0 else None
        summary, per_image = evaluate_images(model, eval_names, pairs, device, inf_cfg, max_det=args.max_det)
    except Exception:
        if checkpoint_path is not None:
            print(f"  evaluation failed; the trained model is saved at {checkpoint_path} and can be "
                  f"evaluated with scripts/evaluate.py")
        raise

    if args.eval_every > 0:
        history[-1]["val_loss"] = final_val_loss
        history[-1]["eval"] = {k: v for k, v in summary.items() if k not in _NOT_IN_HISTORY}
        best_value, best_epoch = _update_best(summary, args, best_value, best_epoch, args.epochs,
                                              model, best_path)

    summary.update({"label": label, "final_loss": history[-1]["train_loss"] if history else None,
                    "params_total": params["total"], "params_trainable": params["trainable"],
                    "best_epoch": best_epoch, "best_epoch_value": best_value})
    print(f"  {label} -> AJI={summary['aji_mean']:.4f}  PQ={summary['pq_mean']:.4f}  "
          f"mAP50={summary.get('map_50', float('nan')):.4f}  count MAE={summary['count_mae']:.2f}")
    return model, summary, per_image, history
