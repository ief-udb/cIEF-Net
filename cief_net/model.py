"""cIEF-Net architecture.

    frozen DINOv3 ViT-L/16  +  DoRA adapters (query/value projections)
        -> multi-scale decoder (stride 16 -> 4)
        -> star-convex head (centre probability + n radial distances)
        -> polar contour refiner (harmonic, K = 24 in the final model)

Design rationale (details in the paper):

* **Frozen backbone with DoRA.** The pretrained encoder is adapted without
  overwriting it; fewer than 2% of the parameters are trainable.
* **Decoder.** The 16-pixel patch stride of the ViT is too coarse to regress
  radial distances of conidia of 20-35 pixels, so features are upsampled to
  stride 4 before the dense head.
* **Star-convex head.** Each conidium is a star-convex polygon (as in
  StarDist), and touching conidia are represented by distinct polygons. The
  head uses 48 rays instead of the usual 32.
* **Polar contour refiner.** The radii predicted at the centre are refined
  from image evidence sampled along each ray; the log-ratio correction is
  projected on the first K angular harmonics and applied multiplicatively.

Two auxiliary heads (density and sub-grid offset), the global affine *pose*
refiner and the *moving-centre* harmonic refiner were evaluated during model
development and are kept only to reproduce the supplementary comparisons. They
are disabled by default.

Ray convention (identical to ``stardist.star_dist`` and ``dist_to_coord``):
ray ``k`` has angle ``phi_k = 2 pi k / K`` and direction ``(dy, dx) =
(sin phi_k, cos phi_k)``. Coordinates are tile pixels in ``(y, x)`` order, and
grid cell ``(gy, gx)`` corresponds to pixel ``(gy * stride, gx * stride)``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_REGISTRY = {}


def register_model(name):
    def decorator(func):
        MODEL_REGISTRY[name] = func
        return func
    return decorator


# =============================================================================
# Backbone: any Hugging Face ViT (DINOv3 by default) with DoRA adapters
# =============================================================================

def build_backbone_with_dora(model_name: str, r: int = 16, dora_alpha: int = None,
                             target_modules: list = None, pretrained: bool = True):
    """Load a ViT from the Hugging Face Hub (or a local folder) and insert DoRA
    adapters. The base weights are frozen by ``peft``; only the low-rank
    matrices and the DoRA magnitude vectors are trainable.

    ``target_modules`` defaults to the query and value projections
    (``q_proj``, ``v_proj``), the standard minimal choice for LoRA/DoRA; the
    names exist in ``DINOv3ViTModel``. ``pretrained=False`` builds the model
    from its configuration with random weights (tests only).
    """
    from peft import LoraConfig, get_peft_model
    from transformers import AutoConfig, AutoModel

    if pretrained:
        backbone = AutoModel.from_pretrained(model_name)
    else:
        backbone = AutoModel.from_config(AutoConfig.from_pretrained(model_name))
    config = LoraConfig(r=r, lora_alpha=dora_alpha or (2 * r), use_dora=True,
                        target_modules=target_modules or ["q_proj", "v_proj"])
    return get_peft_model(backbone, config)


def _patch_tokens(hidden_state: torch.Tensor, n_patches: int) -> torch.Tensor:
    """The sequence is ``[CLS, register tokens..., patch tokens...]``; the
    number of register tokens depends on the checkpoint, so the last
    ``n_patches`` tokens are taken."""
    return hidden_state[:, -n_patches:, :]


# =============================================================================
# Decoder
# =============================================================================

class MultiScaleDecoder(nn.Module):
    """ViTDet-style upsampling from the patch stride to the output stride.

    ``stride_in / stride_out`` must be a power of two. Each upsampling block is
    ``ConvTranspose2d(stride 2) + GroupNorm + GELU``; two 3x3 convolutions
    refine the features at the output resolution.
    """

    def __init__(self, dim_in: int, dim_out: int = 256, stride_in: int = 16, stride_out: int = 4):
        super().__init__()
        n_up, s = 0, stride_in
        while s > stride_out:
            s //= 2
            n_up += 1
        if s != stride_out:
            raise ValueError(f"stride_in ({stride_in}) must reduce to stride_out ({stride_out}) "
                             f"by exact powers of two (e.g. 16 -> 4)")

        layers, ch_in = [], dim_in
        for i in range(n_up):
            ch_out = dim_out if i == n_up - 1 else max(dim_out, ch_in // 2)
            layers += [nn.ConvTranspose2d(ch_in, ch_out, kernel_size=2, stride=2),
                       nn.GroupNorm(min(32, ch_out), ch_out), nn.GELU()]
            ch_in = ch_out
        self.upsample = nn.Sequential(*layers)
        self.refine = nn.Sequential(
            nn.Conv2d(ch_in, dim_out, kernel_size=3, padding=1),
            nn.GroupNorm(min(32, dim_out), dim_out), nn.GELU(),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, padding=1),
            nn.GroupNorm(min(32, dim_out), dim_out), nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.refine(self.upsample(x))


# =============================================================================
# Heads
# =============================================================================

class StarConvexHead(nn.Module):
    """Centre probability (sigmoid) and ``n_rays`` non-negative radial
    distances (softplus). Forty-eight rays reduce the angular approximation
    error, which decays as O(1/n^2), at the apical beak of *Alternaria*
    conidia, the region of highest curvature."""

    def __init__(self, dim_in: int, n_rays: int = 48):
        super().__init__()
        self.prob = nn.Conv2d(dim_in, 1, kernel_size=3, padding=1)
        self.dist = nn.Conv2d(dim_in, n_rays, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor):
        return torch.sigmoid(self.prob(x)), F.softplus(self.dist(x))


class DensityHead(nn.Module):
    """Exploratory: non-negative density map supervised with Gaussians at the
    instance centres (its sum approximates the count). Not in the final model."""

    def __init__(self, dim_in: int):
        super().__init__()
        self.conv = nn.Conv2d(dim_in, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor):
        return F.softplus(self.conv(x))


class OffsetHead(nn.Module):
    """Exploratory: sub-grid offset ``(dy, dx)`` in [-1, 1] (tanh) that corrects
    the quantization of the instance anchor to the reduced grid. It is
    de-normalized (times ``grid_stride / 2``) in :mod:`cief_net.inference`.
    Not in the final model."""

    def __init__(self, dim_in: int):
        super().__init__()
        self.conv = nn.Conv2d(dim_in, 2, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor):
        return torch.tanh(self.conv(x))


# =============================================================================
# Polygon geometry in the stardist ray convention
# =============================================================================

def ray_directions(n_rays: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """(n_rays, 2) unit directions ``(dy, dx)``."""
    phi = torch.arange(n_rays, device=device, dtype=dtype) * (2 * torch.pi / n_rays)
    return torch.stack([torch.sin(phi), torch.cos(phi)], dim=-1)


def radii_to_vertices(centres: torch.Tensor, radii: torch.Tensor) -> torch.Tensor:
    """Centres (M, 2) ``yx`` and radii (M, K) -> vertices (M, K, 2) ``yx``."""
    u = ray_directions(radii.shape[1], radii.device, radii.dtype)
    return centres[:, None, :] + radii[:, :, None] * u[None]


def _cross(p, q):
    return p[..., 0] * q[..., 1] - p[..., 1] * q[..., 0]


def vertices_to_radii(vertices: torch.Tensor, centres: torch.Tensor, n_rays: int) -> torch.Tensor:
    """Resample polygons (M, K, 2) as ``n_rays`` distances from ``centres``
    (M, 2) at the stardist angles, by intersecting every ray with the polygon
    edges. This is exact for polygons that are star-shaped with respect to
    the centre (an affine map of a star-shaped polygon about ``c`` is
    star-shaped about the image of ``c``). Rays that hit no edge fall back to
    the mean polygon radius."""
    a = vertices - centres[:, None, :]                       # (M, K, 2)
    d = torch.roll(a, -1, dims=1) - a                        # edges a_k -> a_{k+1}
    u = ray_directions(n_rays, vertices.device, vertices.dtype)
    uu = u[None, :, None, :]                                 # (1, N, 1, 2)
    aa, dd = a[:, None, :, :], d[:, None, :, :]              # (M, 1, K, 2)
    den = _cross(uu, dd)                                     # (M, N, K)
    ok = den.abs() > 1e-9
    den_s = torch.where(ok, den, torch.ones_like(den))
    t = _cross(aa, dd) / den_s
    s = _cross(aa, uu) / den_s
    valid = ok & (s >= -1e-6) & (s <= 1 + 1e-6) & (t > 0)
    t = torch.where(valid, t, torch.full_like(t, float("inf")))
    r = t.min(dim=-1).values                                 # (M, N)
    r_mean = a.norm(dim=-1).mean(dim=1, keepdim=True).expand_as(r)
    return torch.where(torch.isfinite(r), r, r_mean)


def polygon_centroid(vertices: torch.Tensor) -> torch.Tensor:
    """Area centroid (shoelace) of polygons (M, K, 2) ``yx`` -> (M, 2); falls
    back to the vertex mean for degenerate polygons."""
    p, q = vertices, torch.roll(vertices, -1, dims=1)
    cr = _cross(p, q)
    area = cr.sum(dim=1) / 2
    ok = area.abs() > 1e-6
    area_s = torch.where(ok, area, torch.ones_like(area))
    c = ((p + q) * cr[..., None]).sum(dim=1) / (6 * area_s[:, None])
    return torch.where(ok[:, None], c, vertices.mean(dim=1))


def recenter_polygon(vertices: torch.Tensor):
    """Move the polar frame to the polygon centroid and resample the rays from
    there (exploratory option). The polygon barely changes; what changes is
    where the next refinement iteration reads its profiles from. The new
    centre carries no gradient."""
    c = polygon_centroid(vertices).detach()
    return c, radii_to_vertices(c, vertices_to_radii(vertices, c, vertices.shape[1]))


# =============================================================================
# Polar contour refiners
# =============================================================================

class PoseRefiner(nn.Module):
    """Global affine pose correction of the initial contour (ablation).

    For every instance, ``v' = c + t + (I + dA)(v - c)`` and ``c' = c + t``:
    ``t`` covers translation (harmonic 1 of the radial error) and ``dA`` size,
    anisotropic stretch and orientation (harmonics 0 and 2).

    Evidence: for every vertex, a profile of ``n_profile`` points is sampled
    on the centre-vertex segment between 0.6 and 1.4 times its current radius
    (size-normalized coordinates, the polar analogue of RoIAlign) from two
    maps: the decoder features (projected to ``dec_channels``) and a small
    RGB edge stem at stride 2 (< 20k parameters) that supplies the
    high-frequency detail missing from 16-pixel tokens. Profiles are encoded
    per vertex, mixed along the contour with circular convolutions, pooled
    (mean and max) and mapped by an MLP to the six affine parameters. The last
    layer starts at zero, so the untrained refiner is the identity. The update
    is applied ``n_iter`` times with shared weights.

    This class also hosts the profile sampler and encoder shared by the
    harmonic refiners below.
    """

    # recenter: move the polar frame to the contour centroid after every
    # iteration (exploratory; set by build_cief_net).
    recenter = False
    # uniform_rays: True if the output vertices lie on the stardist rays from
    # the returned centre (harmonic refiners). The loss then resamples the
    # ground-truth target from that centre whenever the centre moves.
    uniform_rays = False

    def __init__(self, decoder_dim: int, decoder_stride: int = 4, dec_channels: int = 64,
                 stem_channels: int = 32, n_profile: int = 8, hidden: int = 128,
                 n_iter: int = 2, profile_min: float = 0.6, profile_max: float = 1.4,
                 max_translation: float = 0.5, max_deformation: float = 0.3):
        super().__init__()
        self.decoder_stride = decoder_stride
        self.stem_stride = 2
        self.n_iter = n_iter
        self.max_translation = max_translation
        self.max_deformation = max_deformation
        self.register_buffer("profile_scales", torch.linspace(profile_min, profile_max, n_profile),
                             persistent=False)
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_channels, 3, padding=1), nn.GroupNorm(8, stem_channels), nn.GELU(),
            nn.Conv2d(stem_channels, stem_channels, 3, stride=2, padding=1),
            nn.GroupNorm(8, stem_channels), nn.GELU(),
            nn.Conv2d(stem_channels, stem_channels, 3, padding=1), nn.GroupNorm(8, stem_channels),
            nn.GELU(),
        )
        self.dec_proj = nn.Sequential(nn.Conv2d(decoder_dim, dec_channels, 1),
                                      nn.GroupNorm(8, dec_channels), nn.GELU())
        c = dec_channels + stem_channels
        self.profile_encoder = nn.Sequential(nn.Linear(n_profile * c + 3, hidden), nn.GELU(),
                                             nn.Linear(hidden, hidden), nn.GELU())
        self.circ_convs = nn.ModuleList([
            nn.Conv1d(hidden, hidden, 5, padding=2, padding_mode="circular") for _ in range(2)])
        self.circ_norms = nn.ModuleList([nn.GroupNorm(8, hidden) for _ in range(2)])
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Linear(hidden, 6))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def prepare(self, x: torch.Tensor, decoded: torch.Tensor):
        """Evidence maps of the whole batch (computed once per tile)."""
        return self.dec_proj(decoded), self.stem((x - 0.5) / 0.25)

    @staticmethod
    def _sample(feature_map: torch.Tensor, b: torch.Tensor, pts: torch.Tensor, stride: int):
        """Bilinear sampling. ``feature_map`` (B, C, H, W), ``b`` (M,) batch
        index, ``pts`` (M, Q, 2) ``yx`` pixels -> (M, Q, C). Pixel ``p`` maps
        to index ``p / stride`` (``align_corners=True``), matching the
        ``[::stride]`` targets."""
        B, C, H, W = feature_map.shape
        M, Q, _ = pts.shape
        out = pts.new_zeros(M, Q, C)
        gy = pts[..., 0] / stride / max(H - 1, 1) * 2 - 1
        gx = pts[..., 1] / stride / max(W - 1, 1) * 2 - 1
        grid = torch.stack([gx, gy], dim=-1)                 # grid_sample expects (x, y)
        for bi in torch.unique(b):
            sel = b == bi
            g = grid[sel].reshape(1, -1, 1, 2)
            v = F.grid_sample(feature_map[bi:bi + 1], g, mode="bilinear",
                              padding_mode="zeros", align_corners=True)   # (1, C, m*Q, 1)
            out[sel] = v[0, :, :, 0].t().reshape(-1, Q, C)
        return out

    def _profiles(self, maps, b, centres, vertices):
        """Per-vertex encoded profiles ``h`` (M, hidden, K), plus the vertex
        offsets ``rel`` (M, K, 2) and the mean radius (M, 1)."""
        f_dec, f_stem = maps
        M, K, _ = vertices.shape
        rel = vertices - centres[:, None, :]
        radius = rel.norm(dim=-1).clamp(min=1e-3)
        r_mean = radius.mean(dim=1, keepdim=True)
        pts = centres[:, None, None, :] + rel[:, :, None, :] * self.profile_scales[None, None, :, None]
        P = pts.shape[2]
        pts = pts.reshape(M, K * P, 2)
        feat = torch.cat([self._sample(f_dec, b, pts, self.decoder_stride),
                          self._sample(f_stem, b, pts, self.stem_stride)], dim=-1).reshape(M, K, -1)
        geo = torch.stack([torch.log(radius / r_mean), rel[..., 0] / radius, rel[..., 1] / radius], dim=-1)
        h = self.profile_encoder(torch.cat([feat, geo], dim=-1)).transpose(1, 2)   # (M, hidden, K)
        return rel, r_mean, h

    def step(self, maps, b, centres, vertices):
        rel, r_mean, h = self._profiles(maps, b, centres, vertices)
        M = rel.shape[0]
        for conv, norm in zip(self.circ_convs, self.circ_norms):
            h = h + F.gelu(norm(conv(h)))
        g = torch.cat([h.mean(dim=2), h.amax(dim=2)], dim=1)       # (M, 2*hidden)
        o = self.head(g)
        t = torch.tanh(o[:, :2]) * self.max_translation * r_mean    # (M, 2) pixels
        dA = (torch.tanh(o[:, 2:]) * self.max_deformation).reshape(M, 2, 2)
        A = torch.eye(2, device=o.device, dtype=o.dtype)[None] + dA
        new = centres[:, None, :] + t[:, None, :] + torch.einsum("mij,mkj->mki", A, rel)
        return centres + t, new

    def forward(self, maps, b, centres, vertices):
        """Return the list of ``(centres, vertices)`` after every iteration."""
        outputs = []
        for _ in range(self.n_iter):
            centres, vertices = self.step(maps, b, centres, vertices)
            if self.recenter:
                centres, vertices = recenter_polygon(vertices)
            outputs.append((centres, vertices))
        return outputs


class HarmonicRefiner(PoseRefiner):
    """Polar harmonic contour refiner (the refiner of cIEF-Net).

    Every ray predicts its own log-ratio residual ``delta_k = log r'_k - log
    r_k`` from the profile sampled along that ray (as in the profile search of
    Active Shape Models), after mixing with neighbouring rays by three
    residual circular convolutions with dilations 1, 2 and 4 (receptive field
    of 29 of 48 rays). The residual is projected on the angular harmonics
    ``0..k_max`` of its discrete Fourier expansion and applied as

        r'_k = r_k * exp( max_residual * tanh( Pi_K(delta)_k ) ),

    with ``max_residual = 0.5``. Harmonic 0 is a change of size, 1
    approximately a displacement, 2 an elongation and higher harmonics finer
    shape detail. With 48 rays, ``k_max >= 24`` keeps every resolvable mode
    (no band limitation); the final model uses ``k_max = 24``.

    The centre is not moved (the correction is radial from the NMS point), so
    the contour stays star-convex in the stardist convention.
    """

    uniform_rays = True

    def __init__(self, decoder_dim: int, decoder_stride: int = 4, k_max: int = 24,
                 max_residual: float = 0.5, **kw):
        # The parent builds the shared sampler/encoder and its global pose head;
        # the pose head is then replaced by a per-ray head. The construction
        # order is kept so that a given seed yields the same initial weights as
        # the implementation used for the paper.
        super().__init__(decoder_dim, decoder_stride=decoder_stride, **kw)
        hidden = self.head[0].in_features // 2
        self.k_max = k_max
        self.max_residual = max_residual
        del self.circ_convs, self.circ_norms, self.head
        self.ray_convs = nn.ModuleList([
            nn.Conv1d(hidden, hidden, 5, padding=2 * d, dilation=d, padding_mode="circular")
            for d in (1, 2, 4)])
        self.ray_norms = nn.ModuleList([nn.GroupNorm(8, hidden) for _ in range(3)])
        self.residual = nn.Conv1d(hidden, 1, 1)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def _project(self, d: torch.Tensor) -> torch.Tensor:
        """Keep harmonics ``0..k_max`` of the per-ray residual (M, K)."""
        n = d.shape[-1]
        if self.k_max is None or self.k_max >= n // 2:
            return d
        Fd = torch.fft.rfft(d, dim=-1)
        Fd[..., self.k_max + 1:] = 0
        return torch.fft.irfft(Fd, n=n, dim=-1)

    def _encode(self, maps, b, centres, vertices):
        """Per-ray features mixed along the contour."""
        rel, r_mean, h = self._profiles(maps, b, centres, vertices)
        for conv, norm in zip(self.ray_convs, self.ray_norms):
            h = h + F.gelu(norm(conv(h)))
        return rel, r_mean, h

    def step(self, maps, b, centres, vertices):
        rel, _, h = self._encode(maps, b, centres, vertices)
        d = self._project(self.residual(h)[:, 0])                     # (M, K)
        d = torch.tanh(d) * self.max_residual
        return centres, centres[:, None, :] + rel * torch.exp(d)[..., None]


class MovingCentreHarmonicRefiner(HarmonicRefiner):
    """Exploratory: harmonic refiner plus a rigid translation voted by the rays,

        t = r_mean * max_translation * tanh( (2/K) * sum_k s_k u_k ),

    so that the whole contour moves (``c' = c + t``) and the next iteration
    reads its profiles from the new centre. The translation head starts at
    zero, so with the weights of a :class:`HarmonicRefiner` the model
    reproduces that refiner exactly."""

    def __init__(self, decoder_dim: int, decoder_stride: int = 4, **kw):
        super().__init__(decoder_dim, decoder_stride=decoder_stride, **kw)
        hidden = self.residual.in_channels
        self.translation = nn.Conv1d(hidden, 1, 1)
        nn.init.zeros_(self.translation.weight)
        nn.init.zeros_(self.translation.bias)

    def step(self, maps, b, centres, vertices):
        rel, r_mean, h = self._encode(maps, b, centres, vertices)
        K = rel.shape[1]
        d = torch.tanh(self._project(self.residual(h)[:, 0])) * self.max_residual
        s = self.translation(h)[:, 0]                                   # (M, K)
        u = ray_directions(K, rel.device, rel.dtype)                    # (K, 2) yx
        vote = 2.0 * (s[..., None] * u[None]).mean(dim=1)               # (M, 2)
        t = torch.tanh(vote) * self.max_translation * r_mean            # (M, 2) pixels
        c2 = centres + t
        return c2, c2[:, None, :] + rel * torch.exp(d)[..., None]


REFINERS = {"pose": PoseRefiner, "harmonic": HarmonicRefiner,
            "harmonic_moving": MovingCentreHarmonicRefiner}


# =============================================================================
# Full model
# =============================================================================

class CIEFNet(nn.Module):
    """Backbone + decoder + star-convex head (+ optional refiner).

    ``forward`` returns a dict with ``prob`` (B, 1, Hr, Wr) and ``dist``
    (B, n_rays, Hr, Wr) and, when the corresponding modules exist,
    ``density``, ``offset`` and ``refiner_maps`` (evidence maps consumed by the
    refiner)."""

    def __init__(self, backbone, decoder: MultiScaleDecoder, star_head: StarConvexHead,
                 density_head: DensityHead = None, offset_head: OffsetHead = None,
                 patch_size: int = 16, refiner: PoseRefiner = None):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.star_head = star_head
        self.density_head = density_head
        self.offset_head = offset_head
        self.patch_size = patch_size
        self.refiner = refiner

    def forward(self, x: torch.Tensor) -> dict:
        B, _, H, W = x.shape
        if H % self.patch_size or W % self.patch_size:
            raise ValueError(f"H and W ({H}x{W}) must be multiples of patch_size "
                             f"({self.patch_size}); adjust --tile-size")
        Hp, Wp = H // self.patch_size, W // self.patch_size
        tokens = _patch_tokens(self.backbone(x).last_hidden_state, Hp * Wp)
        feat = tokens.transpose(1, 2).reshape(B, tokens.shape[-1], Hp, Wp)   # token grid

        decoded = self.decoder(feat)
        prob, dist = self.star_head(decoded)
        out = {"prob": prob, "dist": dist}
        if self.density_head is not None:
            out["density"] = self.density_head(decoded)
        if self.offset_head is not None:
            out["offset"] = self.offset_head(decoded)
        if self.refiner is not None:
            out["refiner_maps"] = self.refiner.prepare(x, decoded)
        return out

    @torch.no_grad()
    def refine_inference(self, output: dict, points, dists):
        """Refine the polygons of ONE tile (batch 1). ``points`` (N, 2) ``yx``
        and ``dists`` (N, K) are NumPy arrays in the stardist convention; the
        refined centres and distances, resampled at the stardist angles, are
        returned ready for ``polygons_to_label``."""
        import numpy as np
        if self.refiner is None or len(points) == 0:
            return points, dists
        dev = output["prob"].device
        c = torch.as_tensor(np.asarray(points, dtype=np.float32), device=dev)
        r = torch.as_tensor(np.asarray(dists, dtype=np.float32), device=dev)
        b = torch.zeros(len(c), dtype=torch.long, device=dev)
        c2, v2 = self.refiner(output["refiner_maps"], b, c, radii_to_vertices(c, r))[-1]
        r2 = vertices_to_radii(v2, c2, r.shape[1]).clamp(min=0)
        return c2.cpu().numpy().astype(np.float32), r2.cpu().numpy().astype(np.float32)


@register_model("cief_net")
def build_cief_net(backbone: str = "facebook/dinov3-vitl16-pretrain-lvd1689m",
                   patch_size: int = 16, decoder_dim: int = 256, decoder_stride: int = 4,
                   n_rays: int = 48, dora_r: int = 16, dora_alpha: int = None,
                   dora_target_modules: list = None, pretrained: bool = True,
                   use_density_head: bool = False, use_offset_head: bool = False,
                   use_refiner: bool = False, refiner_type: str = "harmonic",
                   refiner_kmax: int = 24, refiner_iters: int = 2, refiner_profile: int = 8,
                   refiner_recenter: bool = False, **_ignored) -> CIEFNet:
    """Build cIEF-Net. Without ``use_refiner`` the model (and its state dict) is
    the base configuration; the refiner weights live under ``refiner.``.

    ``refiner_type``: ``harmonic`` (final model, ``refiner_kmax=24``),
    ``pose`` (global affine, ablation) or ``harmonic_moving`` (exploratory).
    """
    bb = build_backbone_with_dora(backbone, r=dora_r, dora_alpha=dora_alpha,
                                  target_modules=dora_target_modules, pretrained=pretrained)
    decoder = MultiScaleDecoder(bb.config.hidden_size, decoder_dim, patch_size, decoder_stride)
    star_head = StarConvexHead(decoder_dim, n_rays=n_rays)
    density_head = DensityHead(decoder_dim) if use_density_head else None
    offset_head = OffsetHead(decoder_dim) if use_offset_head else None

    refiner = None
    if use_refiner:
        if refiner_type not in REFINERS:
            raise ValueError(f"unknown refiner_type '{refiner_type}', choose from {list(REFINERS)}")
        extra = {"k_max": refiner_kmax} if refiner_type.startswith("harmonic") else {}
        refiner = REFINERS[refiner_type](decoder_dim, decoder_stride=decoder_stride,
                                         n_iter=refiner_iters, n_profile=refiner_profile, **extra)
        refiner.recenter = bool(refiner_recenter)
    return CIEFNet(bb, decoder, star_head, density_head, offset_head,
                   patch_size=patch_size, refiner=refiner)


def build_model(name: str = "cief_net", **kwargs):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"model '{name}' is not registered; available: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**kwargs)


def count_parameters(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable,
            "pct_trainable": 100 * trainable / total if total else 0.0}
