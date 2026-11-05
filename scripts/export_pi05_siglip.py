"""Export pi05_libero's SigLIP image encoder (PaliGemma/img) for the PyTorch value model (docs/libero_plan.md,
2026-11-02 vision-only arm), plus a JAX reference output to check the port against.

Usage (openpi env, CPU is enough):
    JAX_PLATFORMS=cpu .venv/bin/python <value-learning>/scripts/export_pi05_siglip.py --out <dir>

Writes <dir>/weights.npz: float32 arrays under timm `vit_so400m_patch14_siglip_224` names (patch_embed, pos_embed,
27 blocks, final norm), mapped from the big_vision layout openpi uses (scanned encoderblock, q/k/v kernels
(width, heads, head_dim)). The `head` Dense (1152 -> 2048, into the LLM) is not exported [ours: the value model
mean-pools the 1152-d encoder output, the ResNet's pooled-feature analogue]. <dir>/ref.npz: `images` uint8
(4, 3, 224, 224) = agentview + wrist of 2 frames of the moka fixed + fresh cache, and `encoded` float32
(4, 256, 1152) = openpi's own SigLIP module (pi0.py settings: So400m/14, pool none, scan) on them, in [-1, 1]
as openpi's input transform scales uint8 images. vl.model.SiglipPi05Features must reproduce `encoded`.
"""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import traverse_util
from openpi.models import model as _model
from openpi.models import siglip as _siglip

CKPT = "/iris/u/rbhowmik/cache/openpi/openpi-assets/checkpoints/pi05_libero"
FRAMES = "/iris/u/ankile/karenvo/libero/tables/moka_fixed_fresh/frames_every4.npy"
DEPTH, WIDTH = 27, 1152


def to_timm(img: dict) -> dict:
    """big_vision / openpi SigLIP params (flat 'a/b/c' keys) -> timm state-dict names, float32 numpy."""
    f = {k.removesuffix("/value"): np.asarray(v, np.float32) for k, v in img.items()}
    enc = "Transformer/encoderblock/"
    out = {"patch_embed.proj.weight": f["embedding/kernel"].transpose(3, 2, 0, 1),  # HWIO -> OIHW
           "patch_embed.proj.bias": f["embedding/bias"],
           "pos_embed": f["pos_embedding"].reshape(1, -1, WIDTH),
           "norm.weight": f["Transformer/encoder_norm/scale"], "norm.bias": f["Transformer/encoder_norm/bias"]}
    att = enc + "MultiHeadDotProductAttention_0/"
    for i in range(DEPTH):
        b = f"blocks.{i}."
        for ln, name in (("LayerNorm_0", "norm1"), ("LayerNorm_1", "norm2")):
            out[b + name + ".weight"] = f[enc + ln + "/scale"][i]
            out[b + name + ".bias"] = f[enc + ln + "/bias"][i]
        qkv = [f[att + p + "/kernel"][i].reshape(WIDTH, WIDTH).T for p in ("query", "key", "value")]
        out[b + "attn.qkv.weight"] = np.concatenate(qkv, 0)
        out[b + "attn.qkv.bias"] = np.concatenate([f[att + p + "/bias"][i].reshape(-1)
                                                   for p in ("query", "key", "value")])
        out[b + "attn.proj.weight"] = f[att + "out/kernel"][i].reshape(WIDTH, WIDTH).T  # (heads, hd, W) -> (W, W)
        out[b + "attn.proj.bias"] = f[att + "out/bias"][i]
        for d, name in (("Dense_0", "fc1"), ("Dense_1", "fc2")):
            out[b + f"mlp.{name}.weight"] = f[enc + f"MlpBlock_0/{d}/kernel"][i].T
            out[b + f"mlp.{name}.bias"] = f[enc + f"MlpBlock_0/{d}/bias"][i]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    params = _model.restore_params(Path(CKPT) / "params", restore_type=np.ndarray)
    img = traverse_util.flatten_dict(params["PaliGemma"]["img"], sep="/")
    print(f"img params: {len(img)} arrays, {sum(np.asarray(v).size for v in img.values()) / 1e6:.1f} M", flush=True)
    w = to_timm(img)
    np.savez(out / "weights.npz", **w)
    print(f"weights.npz: {len(w)} tensors, {sum(v.size for v in w.values()) / 1e6:.1f} M", flush=True)

    frames = np.load(FRAMES, mmap_mode="r")
    images = np.ascontiguousarray(frames[[0, len(frames) // 2]].reshape(4, 3, 224, 224))  # 2 frames x 2 cams
    x = jnp.asarray(images.transpose(0, 2, 3, 1).astype(np.float32) / 255.0 * 2.0 - 1.0)  # NHWC in [-1, 1]
    mod = _siglip.Module(num_classes=2048, variant="So400m/14", pool_type="none", scan=True, dtype_mm="float32")
    tree = traverse_util.unflatten_dict({k.removesuffix("/value"): jnp.asarray(v, jnp.float32)
                                         for k, v in img.items()}, sep="/")
    _, aux = jax.jit(lambda p, x: mod.apply({"params": p}, x, train=False))(tree, x)
    enc = np.asarray(aux["encoded"], np.float32)
    np.savez(out / "ref.npz", images=images, encoded=enc)
    print(f"ref.npz: images {images.shape}, encoded {enc.shape}, |encoded| max {np.abs(enc).max():.2f}", flush=True)


if __name__ == "__main__":
    main()
