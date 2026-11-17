"""Training losses.

Stage 1 (base network):

* binary cross-entropy of the centre probability;
* radial distance loss in logarithmic space, ``|log(d_hat + eps) - log(d +
  eps)|``, restricted to object cells (``prob_gt > 0.5``) and weighted by the
  proximity of the cell to the conidial boundary. The ground-truth
  probability (``edt_prob``: 1 at the centre, decreasing towards the border)
  serves as the proximity proxy: ``w = 1 + 2 (1 - |2 p - 1|)``. A relative
  error weighs small and large conidia evenly, whereas the absolute L1 loss of
  StarDist is dominated by large conidia. Both choices are departures from the
  published StarDist formulation (``radial_loss="l1"`` and
  ``boundary_weighting=False`` recover it).

Stage 2 (refiner, :func:`refiner_loss`): smooth-L1 vertex error relative to
the instance size, summed over refinement iterations.

The terms of the exploratory density and offset heads are only computed when
those heads exist.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .model import radii_to_vertices, vertices_to_radii


@dataclass
class LossConfig:
    radial_loss: str = "log"            # "log" (cIEF-Net) or "l1" (StarDist)
    boundary_weighting: bool = True     # weight the radial term by boundary proximity
    w_prob: float = 1.0
    w_radial: float = 1.0
    w_density: float = 0.0              # exploratory density head
    w_offset: float = 0.5               # exploratory offset head
    w_refiner: float = 1.0
    use_refiner: bool = False
    grid_stride: int = 4
    refiner_p_gt: float = 0.3           # fraction of refiner inputs drawn from perturbed GT
    refiner_max_instances: int = 512    # cap per batch (memory)


def refiner_loss(model, output: dict, target: dict, cfg: LossConfig):
    """Loss of the polar contour refiner.

    For every complete instance of the tile (see
    :func:`cief_net.data.instance_peak_cells`), its peak cell ``c = (gy, gx) *
    grid_stride`` anchors:

    * the **initial contour**: the radii *predicted* by the head at ``c``
      (detached, so the refiner learns to correct the actual errors of the
      network). With probability ``refiner_p_gt`` the ground-truth contour
      with a random affine perturbation is used instead, to expose the
      refiner to varied errors;
    * the **target**: the ground-truth vertices along the same rays from ``c``
      (``star_dist`` targets at that cell), so that vertices correspond exactly
      without dynamic matching.

    Loss: smooth L1 (beta 0.05) of the vertex-target distance divided by the
    mean ground-truth radius of the instance (relative error), summed over
    iterations with increasing weight ``k / n_iter``.

    Returns ``(loss, log dict)`` or ``None`` when there is nothing to refine.
    """
    ref = getattr(model, "refiner", None)
    if ref is None or "refiner_maps" not in output or "ref_b" not in target:
        return None
    dev = output["dist"].device
    b, gy, gx = target["ref_b"].to(dev), target["ref_gy"].to(dev), target["ref_gx"].to(dev)
    if len(b) == 0:
        return None
    if len(b) > cfg.refiner_max_instances:
        sel = torch.randperm(len(b), device=dev)[:cfg.refiner_max_instances]
        b, gy, gx = b[sel], gy[sel], gx[sel]

    centres = torch.stack([gy, gx], dim=1).float() * cfg.grid_stride
    r_pred = output["dist"].detach()[b, :, gy, gx]              # (M, K)
    r_gt = target["dist"].to(dev)[b, :, gy, gx]                 # (M, K)
    goal = radii_to_vertices(centres, r_gt)
    initial = radii_to_vertices(centres, r_pred)

    if cfg.refiner_p_gt > 0:
        from_gt = torch.rand(len(b), device=dev) < cfg.refiner_p_gt
        if from_gt.any():
            m = int(from_gt.sum())
            rm = r_gt[from_gt].mean(dim=1, keepdim=True)
            t = torch.randn(m, 2, device=dev) * 0.10 * rm
            A = (torch.exp(torch.randn(m, 1, 1, device=dev) * 0.08) *
                 (torch.eye(2, device=dev)[None] + torch.randn(m, 2, 2, device=dev).clamp(-2, 2) * 0.06))
            rel = goal[from_gt] - centres[from_gt][:, None, :]
            initial = initial.clone()
            perturbed = centres[from_gt][:, None, :] + t[:, None, :] + torch.einsum("mij,mkj->mki", A, rel)
            # back to uniform rays from c (the harmonic refiner assumes the
            # stardist angles; the pose refiner is indifferent)
            initial[from_gt] = radii_to_vertices(
                centres[from_gt], vertices_to_radii(perturbed, centres[from_gt], r_gt.shape[1]))
    else:
        from_gt = torch.zeros(len(b), dtype=torch.bool, device=dev)

    scale = r_gt.mean(dim=1).clamp(min=1.0)[:, None, None]
    iterations = ref(output["refiner_maps"], b, centres, initial)
    n, K = len(iterations), r_gt.shape[1]

    def goal_from(c_k):
        # If the centre moved (moving-centre refiner or recentring), vertex k
        # lies on ray k from c_k; the target is resampled from the same centre
        # to keep the exact vertex correspondence.
        if not getattr(ref, "uniform_rays", False) or c_k is centres:
            return goal
        return radii_to_vertices(c_k, vertices_to_radii(goal, c_k, K))

    total = 0.0
    for k, (c_k, v) in enumerate(iterations, start=1):
        total = total + (k / n) * F.smooth_l1_loss((v - goal_from(c_k)) / scale,
                                                   torch.zeros_like(v), beta=0.05)
    with torch.no_grad():
        # monitoring only on instances that start from the real prediction
        # (perturbed GT inputs are an augmentation and would inflate the gain)
        real = ~from_gt
        if real.any():
            err0 = ((initial[real] - goal[real]).norm(dim=-1) / scale[real][..., 0]).mean()
            goal_last = goal_from(iterations[-1][0])
            err1 = ((iterations[-1][1][real] - goal_last[real]).norm(dim=-1) / scale[real][..., 0]).mean()
        else:
            err0 = err1 = torch.tensor(float("nan"))
    return total, {"refiner": float(total.item()), "ref_err_init": float(err0),
                   "ref_err_final": float(err1)}


def compute_loss(output: dict, target: dict, cfg: LossConfig, model=None):
    """Weighted sum of the loss terms. Returns ``(total, log dict)``."""
    prob_pred, dist_pred = output["prob"], output["dist"]
    device = prob_pred.device
    prob_gt = target["prob"].to(device)
    dist_gt = target["dist"].to(device)

    loss_prob = F.binary_cross_entropy(prob_pred, prob_gt)

    object_mask = (prob_gt > 0.5).float()
    weight = 1.0 + 2.0 * (1.0 - torch.abs(2 * prob_gt - 1)) if cfg.boundary_weighting else 1.0
    if cfg.radial_loss == "log":
        eps = 1e-3   # avoids log(0) without distorting radii of interest (5-40 px)
        diff = (torch.log(dist_pred + eps) - torch.log(dist_gt + eps)).abs()
    else:
        diff = (dist_pred - dist_gt).abs()
    denom = object_mask.sum() * dist_pred.shape[1] + 1e-6
    loss_radial = (diff * object_mask * weight).sum() / denom

    total = cfg.w_prob * loss_prob + cfg.w_radial * loss_radial
    log = {"prob": float(loss_prob.item()), "radial": float(loss_radial.item())}

    if "density" in output:
        loss_density = F.mse_loss(output["density"], target["density"].to(device))
        total = total + cfg.w_density * loss_density
        log["density"] = float(loss_density.item())

    if "offset" in output:
        # supervised only inside objects (the offset is meaningless elsewhere)
        offset_gt = target["offset"].to(device)
        mask = object_mask.expand_as(offset_gt)
        loss_offset = ((output["offset"] - offset_gt).abs() * mask).sum() / (mask.sum() + 1e-6)
        total = total + cfg.w_offset * loss_offset
        log["offset"] = float(loss_offset.item())

    if model is not None and cfg.use_refiner:
        res = refiner_loss(model, output, target, cfg)
        if res is not None:
            total = total + cfg.w_refiner * res[0]
            log.update(res[1])
    return total, log
