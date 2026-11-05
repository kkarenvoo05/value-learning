"""Stage 3a (docs/libero_plan.md): pooled final-layer prefix features of frozen pi05_libero on Stage 2 frames.

Usage (openpi env, one GPU; see scripts/pi05_libero_features.sbatch):
    .venv/bin/python <value-learning>/scripts/pi05_libero_features.py \
        --run-dir /iris/u/ankile/karenvo/libero/runs/stage2_moka_18155849 --out <features.npz> [--every 4]

Each frame goes through openpi's own LIBERO input transform (as the policy server does) from the raw arrays the
policy saw (scripts/libero_rollouts.py --args.save-raw: 224 x 224 images, 8-d state, prompt). The prefix (both
images + prompt + tokenized state) is run through the full LLM, final norm, as in scripts/pi05_smoke.py. Saved per
frame, float16 [ours]: four mean poolings of the valid prefix tokens, (n, 4, width):
  0 all valid tokens   1 agentview image tokens   2 wrist image tokens   3 text tokens (prompt + state)
plus split, init_state, source episode, frame_index, T and success, to join with frame_targets.parquet.
With --grid G [ours, Stage 3a grid test]: instead of the four means, each image's 16 x 16 tokens are averaged over
(16/G) x (16/G) blocks into G x G cells (row-major), giving (n, 2 G^2 + 1, width): agentview cells, wrist cells,
then the mean of the text tokens.
Frames: every --every-th frame of each episode (t % every == 0) [ours: a probe needs no more].
"""

import argparse
import csv
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from openpi.models import model as _model
from openpi.models.pi0 import make_attn_mask
from openpi.policies import policy_config
from openpi.shared import nnx_utils
from openpi.training import config as _config

CKPT = "/iris/u/rbhowmik/cache/openpi/openpi-assets/checkpoints/pi05_libero"
IMG_TOKENS = 256  # per image slot (SigLIP 224 / 14 = 16 x 16)


def prefix_hidden(self, obs: _model.Observation):
    """Final-layer prefix tokens of the VLM (what the action expert attends to), and their mask."""
    obs = _model.preprocess_observation(None, obs, train=False)
    tokens, mask, ar = self.embed_prefix(obs)
    out, _ = self.PaliGemma.llm([tokens, None], mask=make_attn_mask(mask, ar), positions=jnp.cumsum(mask, 1) - 1)
    return out[0], mask


def pool(tok, mask, n_img_slots):
    """(b, L, w) tokens -> (b, 4, w) means over: all valid, image slot 0, image slot 1, text."""
    tok, m = np.asarray(tok, np.float32), np.asarray(mask, np.float32)
    seg = np.zeros(tok.shape[1], dtype=int) + 3                      # text after the image slots
    seg[:IMG_TOKENS], seg[IMG_TOKENS:2 * IMG_TOKENS] = 1, 2
    seg[2 * IMG_TOKENS:n_img_slots * IMG_TOKENS] = -1                # further image slots (masked in LIBERO)
    out = []
    for k in range(4):
        w = m * (seg != -1) if k == 0 else m * (seg == k)
        out.append((tok * w[..., None]).sum(1) / np.maximum(w.sum(1), 1)[..., None])
    return np.stack(out, 1)


def grid_pool(tok, mask, g):
    """(b, L, w) -> (b, 2 g^2 + 1, w): g x g block means of image slots 0 and 1, then the mean of the text tokens."""
    tok, m = np.asarray(tok, np.float32), np.asarray(mask, np.float32)
    b, _, w = tok.shape
    k = 16 // g
    cells = []
    for slot in (0, 1):
        x = tok[:, slot * IMG_TOKENS:(slot + 1) * IMG_TOKENS].reshape(b, g, k, g, k, w)
        cells.append(x.mean((2, 4)).reshape(b, g * g, w))
    seg_text = np.arange(tok.shape[1]) >= 3 * IMG_TOKENS
    wt = m * seg_text
    cells.append(((tok * wt[..., None]).sum(1) / np.maximum(wt.sum(1), 1)[..., None])[:, None])
    return np.concatenate(cells, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--every", type=int, default=4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-episodes", type=int, default=None, help="smoke: first n episodes per split")
    ap.add_argument("--grid", type=int, default=None, help="G x G cell means per image instead of the 4 pools")
    a = ap.parse_args()
    run = Path(a.run_dir)
    t0 = time.time()
    policy = policy_config.create_trained_policy(_config.get_config("pi05_libero"), CKPT)
    fn = nnx_utils.module_jit(prefix_hidden.__get__(policy._model))
    print(f"loaded in {time.time() - t0:.0f} s", flush=True)

    feats, meta, n_img_slots = [], [], None
    pending, pend_meta = [], []

    def flush():
        nonlocal n_img_slots
        b = _model.Observation.from_dict(jax.tree.map(lambda *xs: jnp.asarray(np.stack(xs)), *pending))
        if n_img_slots is None:
            n_img_slots = len(b.images)
        tok, mask = fn(b)
        if len(feats) == 0:
            print(f"prefix tokens {tok.shape}, valid {int(np.asarray(mask)[0].sum())}, image slots {n_img_slots}",
                  flush=True)
        pooled = grid_pool(tok, mask, a.grid) if a.grid else pool(tok, mask, n_img_slots)
        feats.append(pooled.astype(np.float16))
        meta.extend(pend_meta)
        pending.clear()
        pend_meta.clear()

    for split in ("train", "val"):
        rows = list(csv.DictReader(open(run / split / "episodes.csv")))[: a.max_episodes]
        for r in rows:
            with np.load(run / split / "raw" / r["raw"]) as z:
                img, wrist, state, prompt = z["image"], z["wrist_image"], z["state"], str(z["prompt"])
            T = len(state)
            for t in range(0, T, a.every):
                obs = {"observation/image": img[t], "observation/wrist_image": wrist[t],
                       "observation/state": state[t], "prompt": prompt}
                pending.append(policy._input_transform(obs))
                pend_meta.append((split, int(r["init_state"]), int(r["episode"]), t, T - 1, int(r["success"])))
                if len(pending) == a.batch:
                    flush()
            print(f"{split} ep {r['episode']}: {len(meta) + len(pending)} frames, {time.time() - t0:.0f} s",
                  flush=True)
    if pending:
        flush()
    m = np.array(meta, dtype=object)
    np.savez(a.out, feats=np.concatenate(feats), split=m[:, 0].astype(str), init_state=m[:, 1].astype(int),
             episode=m[:, 2].astype(int), frame_index=m[:, 3].astype(int), T=m[:, 4].astype(int),
             success=m[:, 5].astype(int),
             poolings=np.array([f"grid{a.grid}"] if a.grid else ["all", "agentview", "wrist", "text"]),
             every=a.every, checkpoint=CKPT)
    print(f"wrote {a.out}: {len(m)} frames, feats {np.concatenate(feats).shape}, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
