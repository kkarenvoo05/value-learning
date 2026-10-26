"""vl.attention: per-frame readout attention table of a C3 run."""

import numpy as np


def test_attention_table_columns_and_sums():
    from vl.attention import attention_table
    a = np.random.default_rng(0).random((5, 2, 2))
    a /= a.sum(-1, keepdims=True)                        # (N, layers, tokens), a distribution per layer
    t = attention_table(a, ["wrist_left", "side_1"])
    assert list(t.columns) == ["attn_wrist_left_l0", "attn_side_1_l0", "attn_wrist_left_l1", "attn_side_1_l1"]
    for l in (0, 1):
        assert np.allclose(t[f"attn_wrist_left_l{l}"] + t[f"attn_side_1_l{l}"], 1)
    assert np.allclose(t.attn_side_1_l1, a[:, 1, 1])
