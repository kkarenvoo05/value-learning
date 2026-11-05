"""Diagnose pi05_layer_hidden.py's self-check failure (rel err 0.32-0.45 with pi05 weights).
Reuses the script's own pieces on a small first batch and reports where layer loop and openpi llm() differ:
by token class (valid image / padded image slot / valid text / padded text) and per layer (k from each block vs the
kv_cache openpi returns)."""
import sys, csv
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jax, jax.numpy as jnp, numpy as np, pandas as pd
import pi05_layer_hidden as H
from openpi.models import gemma as _gemma, model as _model
from openpi.models.pi0 import make_attn_mask
from openpi.policies import policy_config
from openpi.shared import nnx_utils
from openpi.training import config as _config
from flax import nnx

B = int(sys.argv[1]) if len(sys.argv) > 1 else 4
meta = pd.read_parquet(H.P1 / "meta.parquet").reset_index(drop=True).iloc[:B]
raw_of = {int(r["episode_index"]): r["raw"] for r in csv.DictReader(open(H.DATASET / "episodes.csv"))}
tc = _config.get_config("pi05_libero")
policy = policy_config.create_trained_policy(tc, H.CKPT)
m = policy._model
llm_p = nnx.state(m.PaliGemma.llm).to_pure_dict()
print("llm top keys", sorted(llm_p), "layer keys", sorted(llm_p["layers"]), flush=True)
layers_p = H.expert0(llm_p["layers"]); norm_p = llm_p["final_norm"]
print("expert0 keys", sorted(layers_p), flush=True)
lv = jax.tree_util.tree_leaves_with_path(layers_p)
print("dtypes", sorted({str(v.dtype) for _, v in lv}), "embed_dtype", m.PaliGemma.llm.module.embed_dtype if hasattr(m.PaliGemma.llm, "module") else "?", flush=True)
cfg = _gemma.get_config(tc.model.paligemma_variant)
print("cfg", cfg, flush=True)
block = _gemma.Block(configs=(cfg,)); norm = _gemma.RMSNorm()

def prefix(self, obs):
    obs = _model.preprocess_observation(None, obs, train=False)
    tokens, mask, ar = self.embed_prefix(obs)
    return tokens, mask, make_attn_mask(mask, ar)
def ref_final(self, obs):
    obs = _model.preprocess_observation(None, obs, train=False)
    tokens, mask, ar = self.embed_prefix(obs)
    o, kv = self.PaliGemma.llm([tokens, None], mask=make_attn_mask(mask, ar), positions=jnp.cumsum(mask, 1) - 1)
    return o[0], kv
prefix_fn = nnx_utils.module_jit(prefix.__get__(m)); ref_fn = nnx_utils.module_jit(ref_final.__get__(m))

@jax.jit
def loop(layers_p, norm_p, tokens, mask, am):
    pos = jnp.cumsum(mask, 1) - 1
    x = tokens.astype(jnp.bfloat16); ks = []
    for l in range(H.DEPTH):
        p = jax.tree.map(lambda v: v[l], layers_p)
        (x,), (k, v) = block.apply({"params": p}, [x], None, pos, am[:, None], [None], True)
        ks.append(k)
    fin, _ = norm.apply({"params": norm_p}, x, None)
    return fin, jnp.stack(ks)

xs = []
for i in range(B):
    with np.load(raw_of[int(meta.episode_index[i])]) as z:
        t = int(meta.frame_index[i])
        xs.append(policy._input_transform({"observation/image": z["image"][t], "observation/wrist_image": z["wrist_image"][t],
                                           "observation/state": z["state"][t], "prompt": str(z["prompt"])}))
obs = _model.Observation.from_dict(jax.tree.map(lambda *x: jnp.asarray(np.stack(x)), *xs))
tokens, mask, am = prefix_fn(obs)
mask = np.asarray(mask); L = mask.shape[1]
print("L", L, "valid per row", mask.sum(1), "image slot validity", [bool(mask[0, s * 256]) for s in range(3)], flush=True)
fin, ks = loop(layers_p, norm_p, tokens, jnp.asarray(mask), am)
ref, (rk, rv) = ref_fn(obs)
ref2, _ = ref_fn(obs)
fin = np.asarray(fin, np.float32); ref = np.asarray(ref, np.float32); ref2 = np.asarray(ref2, np.float32)
print("ref repeat max|diff|", np.abs(ref - ref2).max(), flush=True)
d = np.abs(fin - ref); scale = np.abs(ref).max()
print("overall max rel err (the self-check)", d.max() / scale, flush=True)
img = np.zeros(L, bool); img[:768] = True
for name, sel in [("valid image", mask & img), ("padded image", ~mask & img), ("valid text", mask & ~img), ("padded text", ~mask & ~img)]:
    if sel.any():
        print(f"{name:13s} n={sel.sum():5d} max|ref|={np.abs(ref[sel]).max():9.3f} max|d|={d[sel].max():9.4f} "
              f"rel(max|d|/max|ref| on class)={d[sel].max() / np.abs(ref[sel]).max():.2e} "
              f"rel L2={np.linalg.norm(fin[sel] - ref[sel]) / np.linalg.norm(ref[sel]):.2e}", flush=True)
ks = np.asarray(ks, np.float32); rk = np.asarray(rk, np.float32)
print("kv shapes", ks.shape, rk.shape, flush=True)
for l in range(H.DEPTH):
    a, b = ks[l][mask], rk[l][mask]
    print(f"layer {l:2d} k rel L2 (valid tokens) {np.linalg.norm(a - b) / np.linalg.norm(b):.2e}", flush=True)
for c in range(2):  # pooled per camera, what the probe saves
    pa, pr = fin[:, c * 256:(c + 1) * 256].mean(1), ref[:, c * 256:(c + 1) * 256].mean(1)
    print(f"camera {c} pooled final rel L2 {np.linalg.norm(pa - pr) / np.linalg.norm(pr):.2e}", flush=True)
