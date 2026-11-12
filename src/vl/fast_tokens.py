"""The real FAST action tokenizer (DCT + scale 10 + round + BPE, vocab 2048) for the discrete action loss [2026-11-07,
item 4; paper: pi*0.6 Sec. V-A "actions discretized via FAST [77]"; FAST = Pertsch et al. 2025].

Why: the first discrete-loss runs used our own 127-symbol DCT code without BPE (vl.discrete_actions), which pi05 has
never seen (CE at init = uniform). pi0.5 is pre-trained on FAST tokens, so with FAST's own vocabulary the backbone
may already know how to read and predict them; --selftest measures that (CE at init vs uniform log 2048).

Tokenizer = physical-intelligence/fast (Hugging Face), files downloaded to FAST_DIR on 2026-11-07 with curl and read
before use: processing_action_tokenizer.py is scipy DCT + a `tokenizers` BPE (no network, no exec). We re-implement
its encoder here instead of loading it with trust_remote_code:
    coeff = round(scale * DCT_ortho(chunk, axis=time)); chars = chr(max(coeff.flatten() - min_token, 0))
    ids = BPE(tokenizer.json).encode("".join(chars))        (flatten = time-major (H, D), as the processor)
min_token = -354 and scale = 10 from processor_config.json. Parity with the official processor is checked in
scripts/pi05_policy_finetune.py --selftest --fast-tokens (it imports the downloaded file as a plain module).
PaliGemma ids = vocab_size - 1 - 128 - id (openpi FASTTokenizer._act_tokens_to_paligemma_tokens), so the softmax runs
over N_FAST = 2048 rows [ours: restricted softmax, as vl.discrete_actions; openpi uses the full vocabulary].
Variable length: padded to F_MAX with a mask; longer sequences are truncated and counted (selftest reports the rate).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

FAST_DIR = Path("/iris/u/ankile/karenvo/pi05/fast_tokenizer")
N_FAST = 2048
F_MAX = 96  # [ours] cap on tokens per chunk (10 x 7 = 70 DCT values; byte-level BPE can exceed 70 in rare chunks)
_BPE = None


def config(fast_dir: Path = FAST_DIR) -> dict:
    c = json.loads((Path(fast_dir) / "processor_config.json").read_text())
    assert c["vocab_size"] == N_FAST, c
    return c


def dct_ortho(x: np.ndarray, axis: int) -> np.ndarray:
    """scipy.fft.dct(x, axis=axis, norm='ortho') without scipy (this repo's env); tested against the matrix form."""
    n = x.shape[axis]
    t, k = np.arange(n)[None, :], np.arange(n)[:, None]
    d = np.sqrt(2.0 / n) * np.cos(np.pi * (t + 0.5) * k / n)
    d[0] /= np.sqrt(2.0)
    return np.moveaxis(np.tensordot(d, np.moveaxis(x, axis, 0), axes=(1, 0)), 0, axis)


def chunk_strings(actions: np.ndarray, scale: float = 10.0, min_token: int = -354) -> list[str]:
    """(b, H, D) normalised actions -> the processor's per-chunk character strings (before BPE)."""
    a = np.asarray(actions)
    assert a.ndim == 3
    try:  # the processor's own op on the input dtype (float32 in training), so rounding matches it exactly
        from scipy.fft import dct
        c = np.around(dct(a, axis=1, norm="ortho") * scale)
    except ImportError:  # this repo's env without scipy (tests)
        c = np.around(dct_ortho(a.astype(np.float64), axis=1) * scale)
    return ["".join(map(chr, np.maximum(e.flatten() - min_token, 0).astype(int))) for e in c]


def bpe(fast_dir: Path = FAST_DIR):
    global _BPE
    if _BPE is None:
        from tokenizers import Tokenizer  # openpi env
        _BPE = Tokenizer.from_file(str(Path(fast_dir) / "tokenizer.json"))
    return _BPE


def encode(actions: np.ndarray, encoder=None, f_max: int = F_MAX) -> tuple[np.ndarray, np.ndarray, int]:
    """(b, H, D) -> (tokens (b, f_max) int32 FAST ids, mask (b, f_max) bool, number of truncated chunks).
    encoder: str -> list[int] (default: the BPE of tokenizer.json)."""
    cfg = config() if encoder is None else {"scale": 10, "min_token": -354}
    enc = encoder or (lambda s: bpe().encode(s).ids)
    tok = np.zeros((len(actions), f_max), np.int32)
    mask = np.zeros((len(actions), f_max), bool)
    cut = 0
    for i, s in enumerate(chunk_strings(actions, cfg["scale"], cfg["min_token"])):
        ids = enc(s)
        assert all(0 <= x < N_FAST for x in ids)
        cut += len(ids) > f_max
        ids = ids[:f_max]
        tok[i, :len(ids)], mask[i, :len(ids)] = ids, True
    return tok, mask, cut
