"""Model tests with a tiny randomly initialized DINOv3 encoder (no download)."""

import pytest
import torch

transformers = pytest.importorskip("transformers")
pytest.importorskip("peft")

from cief_net.checkpoints import convert_legacy_state_dict  # noqa: E402
from cief_net.model import (HarmonicRefiner, build_cief_net, radii_to_vertices,  # noqa: E402
                            vertices_to_radii)


@pytest.fixture(scope="module")
def tiny_backbone(tmp_path_factory):
    cfg = transformers.DINOv3ViTConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                                       num_attention_heads=4, patch_size=16, num_register_tokens=4)
    path = tmp_path_factory.mktemp("tiny_dinov3")
    transformers.DINOv3ViTModel(cfg).save_pretrained(path)
    return str(path)


def test_forward_shapes(tiny_backbone):
    torch.manual_seed(0)
    model = build_cief_net(backbone=tiny_backbone, use_refiner=True).eval()
    out = model(torch.rand(1, 3, 128, 128))
    assert out["prob"].shape == (1, 1, 32, 32)
    assert out["dist"].shape == (1, 48, 32, 32)
    assert "refiner_maps" in out


def test_untrained_refiner_is_identity(tiny_backbone):
    torch.manual_seed(0)
    model = build_cief_net(backbone=tiny_backbone, use_refiner=True).eval()
    out = model(torch.rand(1, 3, 128, 128))
    points = torch.tensor([[40.0, 50.0], [80.0, 64.0]]).numpy()
    dists = (torch.rand(2, 48) * 6 + 8).numpy()
    p2, d2 = model.refine_inference(out, points, dists)
    assert abs(p2 - points).max() < 1e-5
    assert abs(d2 - dists).max() < 1e-3


def test_harmonic_projection_bandlimits():
    ref = HarmonicRefiner(decoder_dim=32, k_max=2)
    theta = torch.arange(48) * (2 * torch.pi / 48)
    d = (0.3 + 0.2 * torch.cos(theta) + 0.1 * torch.cos(5 * theta))[None]
    proj = ref._project(d)
    expected = (0.3 + 0.2 * torch.cos(theta))[None]
    assert torch.allclose(proj, expected, atol=1e-5)
    ref.k_max = 24                          # full band: identity
    assert torch.allclose(ref._project(d), d)


def test_radii_vertices_roundtrip():
    c = torch.tensor([[50.0, 60.0]])
    r = torch.rand(1, 48) * 5 + 10
    assert torch.allclose(vertices_to_radii(radii_to_vertices(c, r), c, 48), r, atol=1e-4)


def test_legacy_key_conversion(tiny_backbone):
    model = build_cief_net(backbone=tiny_backbone, use_refiner=True)
    current = model.state_dict()
    rename = {"star_head.dist.": "cabeza_stardist.radios.", "star_head.prob.": "cabeza_stardist.prob.",
              "decoder.refine.": "decoder.refinar.", "refiner.dec_proj.": "refinador.proy_dec.",
              "refiner.profile_encoder.": "refinador.cod_perfil.", "refiner.ray_convs.": "refinador.circ_rayos.",
              "refiner.ray_norms.": "refinador.norm_rayos.", "refiner.residual.": "refinador.residuo.",
              "refiner.stem.": "refinador.stem."}
    legacy = {}
    for k, v in current.items():
        new = k
        for a, b in rename.items():
            if k.startswith(a):
                new = b + k[len(a):]
                break
        legacy[new] = v
    converted = convert_legacy_state_dict(legacy)
    assert set(converted) == set(current)
    model.load_state_dict(converted, strict=True)
