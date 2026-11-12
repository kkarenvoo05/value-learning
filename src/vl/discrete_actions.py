"""Discrete action-token loss for the pi05 policy fine-tune [2026-11-06] (pi*0.6 / Knowledge Insulation).

Why (reports/2026-11-05-literature-advantage-conditioning/, "How the tag reaches the actions"): pi*0.6 places the
advantage text "after the predicted subtask but before the (discretized and continuous) actions" and trains BOTH a
next-token cross-entropy on discretized action tokens (FAST) and the flow-matching action expert, with a stop
gradient so the expert's loss cannot change the backbone (arXiv 2511.14759 Sec. V-A/V-B, Eqs. 4 and 9). The backbone
then has to carry the tag into its action predictions. openpi's Pi0.compute_loss has the flow loss only, so our
tag reaches the actions only through the expert's attention to the prompt.

What this adds (`discrete_flow_loss`), on top of openpi's Pi0 (src/openpi/models/pi0.py) without editing it:
  * sequence = [images, prompt (with the tag), ..."Action: "] + F discrete action tokens; the prompt block is
    bidirectional as in openpi, the action tokens causal (ar mask 1), so the prompt cannot see them;
  * cross-entropy of each action token given everything before it (teacher forcing) [paper: FAST CE];
  * the flow-matching expert attends to the images + prompt only, never to the discrete tokens [paper: "The action
    expert does not receive these as input"], so inference (openpi sample_actions, no discrete tokens) is unchanged;
  * optional stop gradient on the backbone keys / values the expert reads (`ki=True`) [paper: Knowledge Insulation];
    with ki=False the flow loss also trains the backbone (openpi's behaviour) and the CE is an auxiliary loss.

Tokenizer [ours, not FAST]: FAST = DCT per action dim -> scale 10 -> round -> BPE. The BPE vocabulary is a Hugging
Face tokenizer loaded with remote code, which we did not run; we keep the DCT + quantisation and drop the BPE, so every
chunk is exactly H x D tokens (10 x 7 = 70 for LIBERO), frequency-major with dims interleaved as in FAST. Values are
clipped to +-QMAX (127 symbols) and mapped onto PaliGemma vocab ids just below the last 128 special tokens, the same
region and direction openpi's FASTTokenizer uses (vocab_size - 1 - 128 - token). The CE is a softmax over those 127
ids only [ours: the full 257k-way softmax would cost ~2 GB of logits per micro-batch and we never decode text].
"""

from __future__ import annotations

import numpy as np

VOCAB = 257_152           # openpi gemma.PALIGEMMA_VOCAB_SIZE
SKIP = 128                # openpi FASTTokenizer._fast_skip_tokens
SCALE = 10.0              # FAST default scale
QMAX = 63
N_SYM = 2 * QMAX + 1


def dct_matrix(n: int) -> np.ndarray:
    """Orthonormal DCT-II matrix (n, n): coeffs = D @ x, x = D.T @ coeffs (scipy.fft.dct(norm='ortho'))."""
    t, k = np.arange(n)[None, :], np.arange(n)[:, None]
    d = np.sqrt(2.0 / n) * np.cos(np.pi * (t + 0.5) * k / n)
    d[0] /= np.sqrt(2.0)
    return d.astype(np.float32)


def action_tokens(actions, n_dims: int, xp=np):
    """(b, H, >= n_dims) normalised actions -> (b, H * n_dims) int32 symbols in [0, N_SYM), frequency-major."""
    a = actions[..., :n_dims]
    d = xp.asarray(dct_matrix(a.shape[-2]))
    c = xp.einsum("kt,btd->bkd", d, a.astype(xp.float32))
    q = xp.clip(xp.round(c * SCALE), -QMAX, QMAX).astype(xp.int32)
    return (q + QMAX).reshape(q.shape[0], -1)


def decode_tokens(tok, horizon: int, n_dims: int) -> np.ndarray:
    """Inverse of action_tokens up to quantisation (tests / diagnostics)."""
    q = np.asarray(tok).reshape(-1, horizon, n_dims).astype(np.float32) - QMAX
    return np.einsum("kt,bkd->btd", dct_matrix(horizon), q / SCALE)


def to_vocab(sym, xp=np):
    """Symbol -> PaliGemma token id (openpi FASTTokenizer._act_tokens_to_paligemma_tokens convention)."""
    return VOCAB - 1 - SKIP - sym


def backbone_key(k: str) -> bool:
    """Flattened param path -> True for the PaliGemma side (LLM, SigLIP), False for the action-expert side: the
    expert's gemma weights carry the `_1` suffix (openpi gemma._name), and action in / out projections and the time
    MLP sit outside PaliGemma."""
    import re
    if k.startswith("PaliGemma/img/"):  # SigLIP (its own Dense_1 / MlpBlock_1 names are not the expert)
        return True
    return k.startswith("PaliGemma/llm/") and not re.search(r"(^|/)[A-Za-z_]+_1(/|$)", k[len("PaliGemma/llm/"):])


def discrete_flow_loss(model, rng, observation, actions, *, train: bool, n_dims: int, ki: bool,
                       tokens=None, token_mask=None, n_sym: int = N_SYM):
    """openpi Pi0.compute_loss (same rng use, noise, time and augmentation) plus the discrete action CE.

    tokens / token_mask [2026-11-07]: precomputed (b, F) symbols with a validity mask (vl.fast_tokens, the real FAST
    vocabulary, n_sym = 2048); default: our fixed-length DCT code computed here (n_sym = N_SYM). The CE is the mean
    over each chunk's valid tokens; padded positions are masked out of attention and loss.

    Returns (flow (b, action_horizon), ce (b,)). With the CE ignored and ki=False the flow term equals
    Pi0.compute_loss on the same rng up to bf16 noise (pi05_policy_finetune --selftest check 5).
    """
    import einops
    import jax
    import jax.numpy as jnp
    from openpi.models import model as _model
    from openpi.models.pi0 import make_attn_mask

    preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
    observation = _model.preprocess_observation(preprocess_rng, observation, train=train)
    batch_shape = actions.shape[:-2]
    noise = jax.random.normal(noise_rng, actions.shape)
    time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
    te = time[..., None, None]
    x_t = te * noise + (1 - te) * actions
    u_t = noise - actions
    llm = model.PaliGemma.llm

    # pass 1: backbone over [images, prompt, discrete action tokens]
    prefix_tokens, prefix_mask, prefix_ar = model.embed_prefix(observation)
    b, p = prefix_mask.shape
    if tokens is None:
        sym = jax.lax.stop_gradient(action_tokens(actions, n_dims, jnp))      # (b, F) targets
        tmask = jnp.ones(sym.shape, jnp.bool_)
    else:
        sym, tmask = jnp.asarray(tokens, jnp.int32), jnp.asarray(token_mask, jnp.bool_)
    f = sym.shape[1]
    disc = llm(to_vocab(sym), method="embed")
    tokens1 = jnp.concatenate([prefix_tokens, disc], axis=1)
    mask1 = jnp.concatenate([prefix_mask, tmask], axis=1)
    ar1 = jnp.concatenate([prefix_ar, jnp.ones((f,), jnp.bool_)])
    (h, _), kv = llm([tokens1, None], mask=make_attn_mask(mask1, ar1), positions=jnp.cumsum(mask1, axis=1) - 1)

    # CE: symbol k is predicted from the token before it (the last valid prompt token for k = 0)
    last = p - 1 - jnp.argmax(prefix_mask[:, ::-1], axis=1)
    src = jnp.concatenate([last[:, None], p + jnp.broadcast_to(jnp.arange(f - 1), (b, f - 1))], axis=1)
    hs = jnp.take_along_axis(h, src[..., None], axis=1)                     # (b, F, width)
    rows = jnp.asarray(to_vocab(np.arange(n_sym)))
    logits = llm(hs, method=lambda m, x: jnp.dot(x, m.embedder.input_embedding_table[rows].T)).astype(jnp.float32)
    logp = jax.nn.log_softmax(logits, axis=-1)
    nll = -jnp.take_along_axis(logp, sym[..., None], axis=-1)[..., 0]
    w = tmask.astype(jnp.float32)
    ce = (nll * w).sum(axis=-1) / jnp.maximum(w.sum(axis=-1), 1.0)

    # pass 2: action expert on the cached backbone keys / values of images + prompt only
    if ki:
        kv = jax.lax.stop_gradient(kv)
    suffix_tokens, suffix_mask, suffix_ar, adarms_cond = model.embed_suffix(observation, x_t, time)
    s = suffix_tokens.shape[1]
    seen = jnp.concatenate([prefix_mask, jnp.zeros((b, f), jnp.bool_)], axis=1)   # never the discrete tokens
    full = jnp.concatenate([einops.repeat(seen, "b p -> b s p", s=s), make_attn_mask(suffix_mask, suffix_ar)], -1)
    pos = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
    (_, out), _ = llm([None, suffix_tokens], mask=full, positions=pos, kv_cache=kv, adarms_cond=[None, adarms_cond])
    v_t = model.action_out_proj(out[:, -model.action_horizon:])
    return jnp.mean(jnp.square(v_t - u_t), axis=-1), ce
