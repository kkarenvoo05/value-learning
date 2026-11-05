"""Progress probe features (docs/libero_plan.md, 2026-10-30): what the 3b value expert reads, per backbone layer.

Usage (openpi env, one GPU; see scripts/pi05_progress_features.sbatch):
    .venv/bin/python <value-learning>/scripts/pi05_progress_features.py --out <dir> \
        [--expert <run dir>:<width>:<mlp_dim> ...]

Question [ours]: does frozen pi05_libero encode task progress ("a moka pot is on the stove") anywhere the 3b expert
can read it? The 3b expert attends to every layer's prefix keys / values, so the features here are the per-layer
**values** V (one KV head, dim 256) of the prefix tokens, pooled like the 3a grid test (scripts/pi05_libero_features
`grid_pool`, G = 4: per image 4 x 4 cell means, then the text-token mean -> 33 tokens) at --layers. Keys are not
used: they carry RoPE positions, which a mean over cells mixes. Per --expert (a scripts/pi05_value_expert.py run dir
with best.msgpack, and its width / MLP dim), the expert's own final-norm query output h, i.e. the representation its
value and outcome heads read.

Frames: every --every-th frame of every Stage 2 episode, the same rows as scripts/pi05_libero_features.py (split,
init_state, episode, frame_index, T, success), so vl.libero_progress joins them with the 3a features and the image
cache. Writes <out>/meta.npz, <out>/v_layers.npy (n, layers, 33, 256) float16, <out>/h_<name>.npy (n, width) float32.
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization
from openpi.models import gemma as _gemma
from openpi.models import model as _model
from openpi.policies import policy_config
from openpi.shared import nnx_utils
from openpi.training import config as _config

sys.path.insert(0, str(Path(__file__).parent))
from pi05_libero_features import grid_pool  # noqa: E402
from pi05_value_expert import CKPT, ValueExpert, backbone_kv, value_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="/iris/u/ankile/karenvo/libero/runs/stage2_moka_18155849")
    ap.add_argument("--out", required=True)
    ap.add_argument("--every", type=int, default=4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--layers", type=int, nargs="+", default=[1, 5, 9, 13, 17], help="0-based, of 18")
    ap.add_argument("--grid", type=int, default=4)
    ap.add_argument("--expert", nargs="*", default=[], help="<run dir>:<width>:<mlp_dim>")
    ap.add_argument("--max-episodes", type=int, default=None, help="smoke: first n episodes per split")
    a = ap.parse_args()
    run, out = Path(a.run_dir), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:6.0f} s] {m}", flush=True)
    tc = _config.get_config("pi05_libero")
    policy = policy_config.create_trained_policy(tc, CKPT)
    kv_fn = nnx_utils.module_jit(backbone_kv.__get__(policy._model))
    log("backbone loaded")

    rows = []  # (split, init_state, episode, t, T - 1, success, raw path)
    for split in ("train", "val"):
        for r in list(csv.DictReader(open(run / split / "episodes.csv")))[: a.max_episodes]:
            with np.load(run / split / "raw" / r["raw"]) as z:
                T = len(z["state"])
            rows += [(split, int(r["init_state"]), int(r["episode"]), t, T - 1, int(r["success"]),
                      run / split / "raw" / r["raw"]) for t in range(0, T, a.every)]
    n, g = len(rows), a.grid
    log(f"{n} frames")
    np.savez(out / "meta.npz", split=np.array([r[0] for r in rows]), init_state=np.array([r[1] for r in rows]),
             episode=np.array([r[2] for r in rows]), frame_index=np.array([r[3] for r in rows]),
             T=np.array([r[4] for r in rows]), success=np.array([r[5] for r in rows]), layers=np.array(a.layers))
    V = np.lib.format.open_memmap(out / "v_layers.npy", mode="w+", dtype=np.float16,
                                  shape=(n, len(a.layers), 2 * g * g + 1, 256))

    base = _gemma.get_config(tc.model.paligemma_variant)
    experts = []
    for spec in a.expert:
        d, w, m = spec.rsplit(":", 2)
        model = ValueExpert(base, value_config(base, int(w), int(m)))
        experts.append((Path(d).name, model, int(w), d))
    H, loaded, cache = {}, [], {}

    def frame(r):
        if r[6] not in cache:  # rows are grouped by episode: keep one raw file open
            cache.clear()
            with np.load(r[6]) as z:
                cache[r[6]] = (z["image"], z["wrist_image"], z["state"], str(z["prompt"]))
        img, wrist, state, prompt = cache[r[6]]
        t = r[3]
        return policy._input_transform({"observation/image": img[t], "observation/wrist_image": wrist[t],
                                        "observation/state": state[t], "prompt": prompt})

    start, pending = 0, []
    for i, r in enumerate(rows):
        pending.append(frame(r))
        if len(pending) < a.batch and i < n - 1:
            continue
        k = len(pending)
        xs = pending + [pending[-1]] * (a.batch - k)  # pad the last batch to one compiled shape
        obs = _model.Observation.from_dict(jax.tree.map(lambda *x: jnp.asarray(np.stack(x)), *xs))
        kv, mask = kv_fn(obs)
        if start == 0:  # expert parameters need a K/V example for their template
            for name, model, w, d in experts:
                tmpl = model.init(jax.random.PRNGKey(0), kv, mask)["params"]
                params = serialization.from_bytes(jax.device_get(tmpl), (Path(d) / "best.msgpack").read_bytes())
                fwd = jax.jit(lambda p, kv, m, model=model: model.apply({"params": p}, kv, m)[2])
                H[name] = np.lib.format.open_memmap(out / f"h_{name}.npy", mode="w+", dtype=np.float32, shape=(n, w))
                loaded.append((name, params, fwd))
                log(f"expert {name}: width {w}")
        v = kv[1]  # (layers, b, L, 1, 256)
        for j, layer in enumerate(a.layers):
            V[start:start + k, j] = grid_pool(np.asarray(v[layer, :, :, 0], np.float32), mask, g)[:k]
        for name, params, fwd in loaded:
            H[name][start:start + k] = np.asarray(fwd(params, kv, mask), np.float32)[:k]
        start += k
        pending = []
        if (start // a.batch) % 200 == 0:
            log(f"{start}/{n} frames, {start / (time.time() - t0):.1f} frames/s")
    V.flush()
    for h in H.values():
        h.flush()
    log(f"done -> {out}")


if __name__ == "__main__":
    main()
