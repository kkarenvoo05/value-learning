"""P1c (docs/libero_plan.md 2026-11-03) [ours]: frozen pi05_libero per-layer **hidden states** of the image tokens,
mean-pooled per camera, on P1's frames (all 1,020 fixed + fresh episodes, every 4th frame, P1 meta order).

Why: P1b's per-layer rows were attention values V (a 256-d projection), so the drop from SigLIP (0.122) to V at
layer 1 (0.093) could be the projection. Hidden states remove that confound, and covering the fresh episodes lets the
LLM rows be scored on the 825-episode fit set too.

Per frame, the prefix (both images + prompt + tokenized state) goes through openpi's own LIBERO input transform and
embed_prefix, as scripts/pi05_libero_features.py. The 18 Gemma blocks are then applied one at a time with the
checkpoint's own scanned weights (expert 0 only: the action expert is not run on a prefix pass), recording the
residual stream h_l = input to block l (h_0 = the projected image embeddings, h_18 = the output of the last block)
at --record, plus final_norm(h_18) ("final", what scripts/pi05_libero_features.py saved). Each record is the mean over
each camera's 256 image tokens -> <out>/llm_h<l>.npy and <out>/llm_hfinal.npy, float16 (n, 2, 2048), cameras
[agentview, wrist] as P1. <out>/meta.parquet and state.npy are P1's, so
    PYTHONPATH=scripts:src .venv/bin/python scripts/libero_frozen_encoders.py analyze --out <out> --encoders ...
runs on them directly (vl venv).
Index convention: P1b's llm_v<j> is the V computed inside block j, from h_j, so llm_v<j> and llm_h<j> read the same
point of the network.

--image-only-attn (step 3, ablation): image tokens may not attend to the prompt / state tokens (text still attends to
everything). If the early SNR drop disappears, mixing in the text/state tokens is what adds it.

Self-checks at start (abort on failure):
  - the layer loop reproduces openpi's own llm() final output on the first batch (max rel err < 2e-2, bf16);
  - unless --image-only-attn, llm_hfinal on the first frames matches Stage 3a's saved features
    (features/feats_every4_18163142.npz, agentview / wrist pools) within 3e-2 relative error.
Usage (openpi env, one GPU): see scripts/pi05_layer_hidden.sbatch. Smoke: --max-frames 256.
"""

import argparse
import csv
import shutil
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from openpi.models import gemma as _gemma
from openpi.models import model as _model
from openpi.models.pi0 import make_attn_mask
from openpi.policies import policy_config
from openpi.shared import nnx_utils
from openpi.training import config as _config
from flax import nnx

CKPT = "/iris/u/rbhowmik/cache/openpi/openpi-assets/checkpoints/pi05_libero"
P1 = Path("/iris/u/ankile/karenvo/libero/features/frozen_encoders")
DATASET = Path("/iris/u/ankile/karenvo/libero/tables/moka_fixed_fresh")
STAGE3A = Path("/iris/u/ankile/karenvo/libero/features/feats_every4_18163142.npz")
IMG = 256  # tokens per image slot
DEPTH = 18


def expert0(tree):
    """Drop the action expert's parameters (names ending in _1) from a Block parameter tree."""
    return {k: expert0(v) if isinstance(v, dict) else v for k, v in tree.items() if not k.endswith("_1")}


def swap_paligemma(m, npz, log):
    """Replace every PaliGemma parameter of the loaded pi05 model that the big_vision checkpoint also has (SigLIP,
    its 1152 -> 2048 projector, the Gemma embedder and expert-0 blocks, final norm), as openpi's PaliGemmaWeightLoader
    does when pi0 is first built. Action-expert (*_1) weights are left as they are; the prefix pass never runs them."""
    import flax.traverse_util as tu
    st = nnx.state(m.PaliGemma)
    ref = tu.flatten_dict(st.to_pure_dict(), sep="/")
    with np.load(npz, allow_pickle=False) as z:
        new = {k.removeprefix("params/"): z[k] for k in z.files}
    hit = {k for k in ref if k in new}
    miss = sorted(k for k in ref if k not in new and not any(p.endswith("_1") for p in k.split("/")))
    log(f"PaliGemma swap: {len(hit)} of {len(ref)} tensors replaced; non-expert tensors not in the npz: {miss}")
    assert not miss, "a pi05 backbone tensor has no PaliGemma counterpart: names differ, the swap would be partial"
    bad = [(k, ref[k].shape, new[k].shape) for k in hit if ref[k].shape != new[k].shape]
    assert not bad, f"shape mismatch {bad[:5]}"
    diff = np.mean([float(np.abs(np.asarray(ref[k], np.float32) - new[k].astype(np.float32)).mean()) for k in hit])
    log(f"mean |pi05 - PaliGemma| over replaced tensors {diff:.3e} (must be > 0)")
    assert diff > 0
    merged = {k: (jnp.asarray(new[k], dtype=ref[k].dtype) if k in hit else ref[k]) for k in ref}
    st.replace_by_pure_dict(tu.unflatten_dict(merged, sep="/"))
    nnx.update(m.PaliGemma, st)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--record", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 7, 9, 11, 13, 15, 17, 18])
    ap.add_argument("--image-only-attn", action="store_true")
    ap.add_argument("--max-frames", type=int, default=None, help="smoke: first n frames of P1's order")
    ap.add_argument("--paligemma", default=None, metavar="NPZ",
                    help="Step 0 (VLM gate) [ours]: overwrite pi05's SigLIP + Gemma (expert 0) + embedder with the "
                         "big_vision PaliGemma checkpoint (paligemma-3b-pt-224.npz), i.e. the model before robot "
                         "training. Same prefix (pi05's LIBERO transform, state as text). Stage 3a check skipped.")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:6.0f} s] {m}", flush=True)

    meta = pd.read_parquet(P1 / "meta.parquet").reset_index(drop=True)
    if a.max_frames:
        meta = meta.iloc[: a.max_frames]
    raw_of = {int(r["episode_index"]): r["raw"] for r in csv.DictReader(open(DATASET / "episodes.csv"))}
    n = len(meta)
    meta.to_parquet(out / "meta.parquet")
    st = np.load(P1 / "state.npy")
    np.save(out / "state.npy", st[:n])
    log(f"{n} frames, {meta.episode_index.nunique()} episodes, record {a.record}, "
        f"image-only attn {a.image_only_attn}")

    tc = _config.get_config("pi05_libero")
    policy = policy_config.create_trained_policy(tc, CKPT)
    m = policy._model
    if a.paligemma:
        swap_paligemma(m, a.paligemma, log)
    llm_p = nnx.state(m.PaliGemma.llm).to_pure_dict()
    log(f"backbone loaded; llm params {sorted(llm_p)}")
    layers_p = expert0(llm_p["layers"])
    norm_p = llm_p["final_norm"]
    cfg = _gemma.get_config(tc.model.paligemma_variant)
    block = _gemma.Block(configs=(cfg,))
    norm = _gemma.RMSNorm()
    rec = sorted(set(a.record))
    assert all(0 <= r <= DEPTH for r in rec)

    def prefix(self, obs):  # nnx part (SigLIP + projector + prompt embedding), jitted as a module method
        obs = _model.preprocess_observation(None, obs, train=False)
        tokens, mask, ar = self.embed_prefix(obs)
        return tokens, mask, make_attn_mask(mask, ar)

    def ref_final(self, obs):  # openpi's own prefix pass, for the self-check
        obs = _model.preprocess_observation(None, obs, train=False)
        tokens, mask, ar = self.embed_prefix(obs)
        o, _ = self.PaliGemma.llm([tokens, None], mask=make_attn_mask(mask, ar), positions=jnp.cumsum(mask, 1) - 1)
        return o[0]

    prefix_fn = nnx_utils.module_jit(prefix.__get__(m))
    ref_fn = nnx_utils.module_jit(ref_final.__get__(m))

    def pool(h):
        h = h.astype(jnp.float32)
        return jnp.stack([h[:, :IMG].mean(1), h[:, IMG:2 * IMG].mean(1)], 1)  # (b, 2, w)

    @partial(jax.jit, static_argnums=5)
    def layer_loop(layers_p, norm_p, tokens, mask, am, image_only):
        L = tokens.shape[1]
        if image_only:  # image rows (both live slots) may not see the text columns
            img_row = jnp.arange(L) < 2 * IMG
            text_col = jnp.arange(L) >= 3 * IMG
            am = am & ~(img_row[:, None] & text_col[None, :])[None]
        pos = jnp.cumsum(mask, 1) - 1
        x = tokens.astype(jnp.bfloat16)
        pools = {0: pool(x)}
        for l in range(DEPTH):
            p = jax.tree.map(lambda v: v[l], layers_p)
            (x,), _ = block.apply({"params": p}, [x], None, pos, am[:, None], [None], True)
            pools[l + 1] = pool(x)
        fin, _ = norm.apply({"params": norm_p}, x, None)
        return jnp.stack([pools[r] for r in rec] + [pool(fin)], 1), fin  # (b, len(rec) + 1, 2, w)

    names = [f"llm_h{r}" for r in rec] + ["llm_hfinal"]
    W = cfg.width
    arrs = {k: np.lib.format.open_memmap(out / f"{k}.tmp.npy", mode="w+", dtype=np.float16, shape=(n, 2, W))
            for k in names}
    cache = {}

    def frame(e, t):
        if e not in cache:
            cache.clear()
            with np.load(raw_of[e]) as z:
                cache[e] = (z["image"], z["wrist_image"], z["state"], str(z["prompt"]))
        img, wrist, state, prompt = cache[e]
        return policy._input_transform({"observation/image": img[t], "observation/wrist_image": wrist[t],
                                        "observation/state": state[t], "prompt": prompt})

    E, F = meta.episode_index.values, meta.frame_index.values
    start = 0
    while start < n:
        k = min(a.batch, n - start)
        xs = [frame(int(E[i]), int(F[i])) for i in range(start, start + k)]
        xs += [xs[-1]] * (a.batch - k)  # one compiled shape
        obs = _model.Observation.from_dict(jax.tree.map(lambda *x: jnp.asarray(np.stack(x)), *xs))
        tokens, mask, am = prefix_fn(obs)
        pooled, fin = layer_loop(layers_p, norm_p, tokens, mask, am, a.image_only_attn)
        pooled = np.asarray(pooled)[:k]
        if start == 0:
            if not a.image_only_attn:
                # Compare what is saved: the per-camera pools of the final output (rel L2). The old metric, max
                # single-element |diff| / max |ref| over all tokens, failed at 0.32-0.45 with pi05 weights on every
                # device incl. CPU (2026-11-04, scripts/debug/selfcheck_diag.py): layer 0 matches exactly and the
                # two paths then drift by bf16 rounding (op order of openpi's scanned/remat block vs this unrolled
                # loop; k rel L2 3e-3 at layer 1 -> ~1e-2 at layer 17), while the pools differ by only 0.3-0.5%.
                ref = np.asarray(ref_fn(obs), np.float32)[:k]
                f32 = np.asarray(fin, np.float32)[:k]
                valid = np.asarray(mask)[:k]
                tok = float(np.linalg.norm((f32 - ref)[valid]) / np.linalg.norm(ref[valid]))
                emax = float(np.abs(f32 - ref).max() / np.abs(ref).max())
                pr = np.stack([ref[:, :IMG].mean(1), ref[:, IMG:2 * IMG].mean(1)], 1)
                err = float(np.linalg.norm(pooled[:, -1] - pr) / np.linalg.norm(pr))
                tag = " (PaliGemma weights)" if a.paligemma else ""
                log(f"self-check: layer loop vs openpi llm() final{tag}: pooled rel L2 {err:.2e} (gate 2e-2); "
                    f"valid-token rel L2 {tok:.2e}; max-element rel {emax:.2e} (info only)")
                assert err < 2e-2, err
            if a.paligemma:  # Stage 3a pools are pi05-weight features: a PaliGemma run differs from them by design
                log("PaliGemma weights: Stage 3a check skipped (pools are pi05 features)")
            elif not a.image_only_attn:
                z = np.load(STAGE3A)
                s = pd.DataFrame({c: z[c] for c in ("split", "init_state", "episode", "frame_index")})
                s["src"] = np.arange(len(s))
                j = meta.iloc[:k].reset_index().merge(s, on=["split", "init_state", "episode", "frame_index"])
                assert len(j) > 0, "first batch not in Stage 3a"
                old = z["feats"][j.src.values][:, 1:3].astype(np.float32)
                new = pooled[j["index"].values, -1]
                rel = float(np.linalg.norm(new - old) / np.linalg.norm(old))
                log(f"self-check: llm_hfinal vs Stage 3a pools on {len(j)} frames, rel err {rel:.2e}")
                assert rel < 3e-2, rel
            else:
                log("image-only attention: Stage 3a check skipped (differs by design)")
        for i, name in enumerate(names):
            arrs[name][start:start + k] = pooled[:, i].astype(np.float16)
        start += k
        if (start // a.batch) % 100 == 0 or start == n:
            log(f"{start}/{n} frames, {start / (time.time() - t0):.1f} frames/s")
    for name, arr in arrs.items():
        arr.flush()
        del arr
        (out / f"{name}.tmp.npy").rename(out / f"{name}.npy")
    shutil.copy(Path(__file__), out / Path(__file__).name)
    log(f"done -> {out}")


if __name__ == "__main__":
    main()
