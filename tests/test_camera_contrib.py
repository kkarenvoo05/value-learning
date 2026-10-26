import torch

from vl.camera_contrib import drop_mask, masked_feats, phase
from vl.model import TokenFusionHead


def test_masked_feats_replaces_only_that_camera():
    f = torch.randn(4, 3 * 5)
    mean = torch.full((5,), 7.0)
    g = masked_feats(f, 1, 3, mean)
    assert torch.equal(g[:, 5:10], mean.expand(4, 5))
    assert torch.equal(g[:, :5], f[:, :5]) and torch.equal(g[:, 10:], f[:, 10:])
    assert not torch.equal(f[:, 5:10], g[:, 5:10])       # input untouched


def test_dropped_token_has_no_effect_on_the_output():
    torch.manual_seed(0)
    head = TokenFusionHead(2, 8, 11, d=16, n_heads=2, n_layers=2, ff=32, hidden=16).eval()
    f = torch.randn(3, 16)
    g = f.clone()
    g[:, 8:] = torch.randn(3, 8)                          # change camera 1 only
    m = drop_mask(3, 1, 2, "cpu")
    assert m.tolist() == [[False, True]] * 3
    assert torch.allclose(head(f, key_padding_mask=m), head(g, key_padding_mask=m), atol=1e-6)
    assert not torch.allclose(head(f), head(g))           # without the mask it does matter


def test_phase_labels():
    import pandas as pd
    d = pd.DataFrame({"T": [300, 300, 300, 300], "frame_index": [10, 150, 260, 10],
                      "outcome": ["success", "success", "success", "timeout"]})
    assert phase(d).tolist() == ["success, first 5 s", "success, middle", "success, last 5 s",
                                 "not success, first 5 s"]
