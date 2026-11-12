"""vl.discrete_actions: the DCT action tokenizer and param grouping of the discrete-loss fine-tune [2026-11-06]."""
import numpy as np

from vl import discrete_actions as da


def test_dct_is_orthonormal():
    d = da.dct_matrix(10)
    assert np.allclose(d @ d.T, np.eye(10), atol=1e-5)


def test_tokens_shape_range_and_frequency_order():
    rng = np.random.default_rng(0)
    a = rng.uniform(-1, 1, (4, 10, 32)).astype(np.float32)
    s = da.action_tokens(a, 7)
    assert s.shape == (4, 70) and s.dtype == np.int32
    assert s.min() >= 0 and s.max() < da.N_SYM
    const = np.full((1, 10, 7), 0.5, np.float32)          # constant chunk: only the k = 0 coefficients are nonzero
    sc = da.action_tokens(const, 7).reshape(10, 7)
    assert (sc[0] != da.QMAX).all() and (sc[1:] == da.QMAX).all()


def test_round_trip_is_quantisation_only():
    rng = np.random.default_rng(1)
    a = np.cumsum(rng.normal(0, 0.1, (8, 10, 7)), axis=1).clip(-1, 1).astype(np.float32)
    back = da.decode_tokens(da.action_tokens(a, 7), 10, 7)
    assert np.sqrt(np.mean((back - a) ** 2)) < 0.5 / da.SCALE   # orthonormal DCT: rms error <= max coefficient error


def test_clipping_at_extreme_values():
    s = da.action_tokens(np.full((1, 10, 7), 100.0, np.float32), 7)
    assert s.max() == da.N_SYM - 1


def test_vocab_ids_sit_below_the_special_tokens():
    ids = da.to_vocab(np.arange(da.N_SYM))
    assert ids.max() == da.VOCAB - 1 - da.SKIP and ids.min() > 0
    assert len(set(ids.tolist())) == da.N_SYM


def test_backbone_key_groups():
    assert da.backbone_key("PaliGemma/llm/layers/attn/q_einsum/lora_b")
    assert da.backbone_key("PaliGemma/img/Transformer/encoderblock/MlpBlock_0/Dense_0/kernel")
    assert da.backbone_key("PaliGemma/img/Transformer/encoderblock/MlpBlock_0/Dense_1/kernel")  # SigLIP, not expert
    assert da.backbone_key("PaliGemma/llm/embedder/input_embedding")
    assert not da.backbone_key("PaliGemma/llm/layers/attn/q_einsum_1/lora_b")
    assert not da.backbone_key("PaliGemma/llm/layers/mlp_1/gating_einsum")
    assert not da.backbone_key("action_out_proj/kernel")
    assert not da.backbone_key("time_mlp_in/bias")
