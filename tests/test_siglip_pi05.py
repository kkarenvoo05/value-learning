"""vl.model siglip_pi05 backbone: shapes / normalisation, and the port against openpi's own JAX SigLIP output
(scripts/export_pi05_siglip.py writes weights.npz + ref.npz; that test skips until they exist)."""

import numpy as np
import pytest
import torch

from vl.model import SIGLIP_PI05, SiglipPi05Features, ValueModel

REF = SIGLIP_PI05.parent / "ref.npz"


def test_value_model_shapes_and_normalisation():
    torch.manual_seed(0)
    m = ValueModel(["agentview", "wrist"], backbone="siglip_pi05", pretrained=False).eval()
    assert torch.allclose(m.mean, torch.full((1, 3, 1, 1), 0.5)) and torch.allclose(m.std, torch.full((1, 3, 1, 1), 0.5))
    x = torch.zeros(1, 3, 224, 224, dtype=torch.uint8)
    assert m.normalize(x).min() == -1.0                            # uint8 0 -> -1, as openpi
    imgs = {c: torch.randint(0, 256, (2, 3, 224, 224), dtype=torch.uint8) for c in m.cams}
    with torch.no_grad():
        assert m(imgs).shape == (2, 201)
    assert m.head[0].in_features == 2 * 1152


@pytest.mark.skipif(not REF.exists(), reason="run scripts/export_pi05_siglip.py first")
def test_port_matches_openpi_jax_encoder():
    with np.load(REF) as z:
        images, ref = z["images"], z["encoded"]
    net = SiglipPi05Features(pretrained=True).eval()
    x = torch.from_numpy(images).float() / 255.0 * 2.0 - 1.0
    with torch.no_grad():
        tok = net.vit.forward_features(x).numpy()
    err, scale = np.abs(tok - ref).max(), np.abs(ref).max()
    assert tok.shape == ref.shape == (4, 256, 1152)
    assert err <= 1e-3 * scale, f"port differs from openpi by {err:.3g} (|encoded| max {scale:.3g})"
    print(f"port vs openpi JAX: max |diff| {err:.3g}, |encoded| max {scale:.3g}")
    with torch.no_grad():  # the pooled feature the value head sees: mean over tokens
        np.testing.assert_allclose(net(x).numpy(), ref.mean(1), atol=1e-3 * scale)
