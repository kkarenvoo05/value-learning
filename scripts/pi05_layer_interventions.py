"""Inference-only, between-frame interventions on an existing frozen pi05 value expert.

Run in the openpi environment. --selftest uses a dummy expert on CPU, without a checkpoint.
Actual runs consume pairs.csv and the compact pairs.npz made by libero_layer_diagnosis.py.
No model weights, labels, or splits are changed. See the dated report's protocol.md.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization
from openpi.models import gemma as gemma

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pi05_value_expert import CENTERS, CKPT, ValueExpert, backbone_kv, value_config


def patch_pair(kv, layer_mask, token_mask, component=0):
    """Swap between the two endpoints at identical layer/position. 0=K+V, 1=K, 2=V.

    Prefix masks are deliberately not modified here. The caller checks mask compatibility or
    limits swaps to image tokens, whose positions and visibility masks are unchanged.
    """
    select = layer_mask[:, None, None, None, None] & token_mask[None, None, :, None, None]
    return tuple(jnp.where(select & ((component == 0) | (component == i + 1)), x[:, ::-1], x)
                 for i, x in enumerate(kv))


def readout(model, params, kv, mask):
    logits, outcome, *_ = model.apply({"params": params}, kv, mask)
    probs = jax.nn.softmax(logits.astype(jnp.float32), -1)
    centers = jnp.asarray(CENTERS, jnp.float32)
    return jnp.stack([probs @ centers, probs[:, 101:].sum(-1),
                      jax.nn.softmax(outcome.astype(jnp.float32), -1)[:, 1]], -1)


def selftest():
    base = gemma.get_config("dummy")
    model = ValueExpert(base, value_config(base, width=32, mlp_dim=64), dtype="float32")
    shape = (base.depth, 2, 7, base.num_kv_heads, base.head_dim)
    kv = tuple(jax.random.normal(jax.random.PRNGKey(i), shape) for i in (1, 2))
    mask = jnp.array([[1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 0]], dtype=bool)
    params = model.init(jax.random.PRNGKey(3), kv, mask)["params"]
    fwd = jax.jit(lambda k, m: readout(model, params, k, m))
    initial = fwd(kv, mask)
    all_l, all_t = jnp.ones(base.depth, bool), jnp.ones(7, bool)
    none = patch_pair(kv, ~all_l, all_t)
    np.testing.assert_array_equal(fwd(none, mask), initial)
    full = patch_pair(kv, all_l, all_t)
    np.testing.assert_allclose(fwd(full, mask[::-1]), initial[::-1], rtol=1e-6, atol=1e-6)
    layer_mask = jnp.arange(base.depth) == 0
    token_mask = jnp.arange(7) == 1
    part = patch_pair(kv, layer_mask, token_mask, 2)
    np.testing.assert_array_equal(part[0], kv[0])
    assert not np.array_equal(np.asarray(part[1][0, :, 1]), np.asarray(kv[1][0, :, 1]))
    np.testing.assert_array_equal(part[1][1:], kv[1][1:])
    # Padded tokens must not affect either endpoint; a valid feature change must affect a prediction.
    pad = patch_pair(kv, all_l, jnp.arange(7) == 6)
    np.testing.assert_allclose(fwd(pad, mask), initial, rtol=1e-6, atol=1e-6)
    assert float(jnp.abs(fwd(part, mask) - initial).max()) > 1e-7
    bank = jax.tree.map(lambda x: jnp.stack([x, x]), params)
    ls = jnp.stack([~all_l, all_l, layer_mask, all_l])
    ts = jnp.stack([all_t, all_t, token_mask, jnp.arange(7) == 6])
    cs = jnp.array([0, 0, 2, 0])
    batched = jax.jit(lambda: jax.vmap(lambda lm, tm, c: jax.vmap(
        lambda p: readout(model, p, patch_pair(kv, lm, tm, c), mask))(bank))(ls, ts, cs))()
    expected = jnp.stack([fwd(patch_pair(kv, lm, tm, c), mask) for lm, tm, c in zip(ls, ts, cs)])
    np.testing.assert_allclose(batched[:, 0], expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(batched[:, 1], expected, rtol=1e-5, atol=1e-6)
    print("PASS: identity, full reversal, K/V isolation, padding, valid-token sensitivity, batched seeds and patches", flush=True)


def specifications(depth, length, image_keys, image_masks):
    """Keep camera names tied to the observed prefix order, not dictionary assumptions."""
    assert length >= 256 * len(image_keys)
    groups = {"all": np.ones(length, bool)}
    for i, key in enumerate(image_keys):
        if not np.asarray(image_masks[key]).any():
            continue
        m = np.zeros(length, bool)
        m[256 * i:256 * (i + 1)] = True
        groups[key] = m
    text = np.arange(length) >= 256 * len(image_keys)
    groups["text_state"] = text
    bands = [(f"layer_{i:02d}", [i]) for i in range(depth)]
    bands += [(f"band_{lo:02d}_{hi - 1:02d}", list(range(lo, min(hi, depth))))
              for lo, hi in ((0, 4), (4, 9), (9, 16), (16, 18)) if lo < depth]
    bands += [("all_layers", list(range(depth)))]
    specs = []
    for band, layers in bands:
        lm = np.isin(np.arange(depth), layers)
        for group, tm in groups.items():
            specs.append((band, group, "kv", lm, tm, 0))
        if band.startswith("band_") or band == "all_layers":
            for component, number in (("k", 1), ("v", 2)):
                specs.append((band, "all", component, lm, groups["all"], number))
    return specs


def run(a):
    import pandas as pd
    from openpi.models import model as model_module
    from openpi.policies import policy_config
    from openpi.shared import nnx_utils
    from openpi.training import config

    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log = lambda text: print(f"[{time.time() - t0:7.1f}s] {text}", flush=True)
    manifest = pd.read_csv(a.pairs)
    if a.max_pairs:
        # Pilot covers each cohort, instead of taking only the first jump pairs.
        manifest = manifest.groupby("cohort", sort=False).head(max(1, a.max_pairs // manifest.cohort.nunique())).head(a.max_pairs)
    packed = np.load(a.cache)
    index = {p: i for i, p in enumerate(packed["pair_id"].tolist())}
    image, wrist, state, prompt = [packed[k] for k in ("image", "wrist_image", "state", "prompt")]
    tc = config.get_config("pi05_libero")
    policy = policy_config.create_trained_policy(tc, a.checkpoint)
    kv_fn = nnx_utils.module_jit(backbone_kv.__get__(policy._model))
    base = gemma.get_config(tc.model.paligemma_variant)
    loaded, setups, hashes = [], [], {}
    for run_dir in a.experts:
        setup = json.loads((run_dir / "results.json").read_text())["setup"]
        assert not setup.get("time_input") and not setup.get("blank_state") and not setup.get("subtask_weight", 0)
        assert not setup.get("n_layers"), "This diagnostic holds the full expert fixed."
        setups.append(setup)
        checkpoint_bytes = (run_dir / "best.msgpack").read_bytes()
        hashes[str(run_dir)] = hashlib.sha256(checkpoint_bytes).hexdigest()
        loaded.append(serialization.msgpack_restore(checkpoint_bytes))
    assert len({(s["width"], s["mlp_dim"]) for s in setups}) == 1
    assert len({s["seed"] for s in setups}) == len(setups)
    model = ValueExpert(base, value_config(base, setups[0]["width"], setups[0]["mlp_dim"]))
    params = jax.tree.map(lambda *x: jnp.stack(x), *loaded)
    del loaded
    fwd = jax.jit(lambda kv, mask: jax.vmap(lambda p: readout(model, p, kv, mask))(params))

    @jax.jit
    def interventions(kv, mask, layer_masks, token_masks, components):
        return jax.vmap(lambda lm, tm, c: jax.vmap(
            lambda p: readout(model, p, patch_pair(kv, lm, tm, c), mask))(params))(
                layer_masks, token_masks, components)

    def observation(i, swap=None):
        obs = []
        for side in range(2):
            donor = 1 - side
            x = {"observation/image": image[i, donor if swap in ("agentview", "images") else side].copy(),
                 "observation/wrist_image": wrist[i, donor if swap in ("wrist", "images") else side].copy(),
                 "observation/state": state[i, donor if swap == "state" else side].copy(), "prompt": str(prompt[i])}
            obs.append(policy._input_transform(x))
        return model_module.Observation.from_dict(jax.tree.map(lambda *x: jnp.asarray(np.stack(x)), *obs))

    outputs, baselines, skips = [], [], []
    controls = {"max_saved_value_error": 0., "max_full_swap_error": 0., "max_identity_error": 0.}
    log(f"backbone and {len(setups)} frozen experts loaded; {len(manifest)} pairs")
    for row in manifest.itertuples():
        ix = index[row.pair_id]
        obs = observation(ix)
        kv, mask = kv_fn(obs)
        mask_np = np.asarray(mask)
        compatible = bool(np.array_equal(mask_np[0], mask_np[1]))
        original = np.asarray(fwd(kv, mask))  # seed, endpoint, metric
        flipped = np.asarray(fwd(tuple(x[:, ::-1] for x in kv), mask[::-1]))
        full_error = float(np.abs(flipped - original[:, ::-1]).max())
        assert full_error < 1e-5, (row.pair_id, "full reversal", full_error)
        controls["max_full_swap_error"] = max(controls["max_full_swap_error"], full_error)
        # Jitted intervention path must also reproduce an unmodified readout.
        identity = np.asarray(interventions(kv, mask, jnp.zeros((a.patch_batch, base.depth), bool),
            jnp.ones((a.patch_batch, mask.shape[1]), bool), jnp.zeros(a.patch_batch, jnp.int32)))
        identity_error = float(np.abs(identity - original[None]).max())
        assert identity_error < 1e-5, (row.pair_id, "identity", identity_error)
        controls["max_identity_error"] = max(controls["max_identity_error"], identity_error)
        for si, setup in enumerate(setups):
            seed = setup["seed"]
            expected = np.array([getattr(row, f"va_n18_s{seed}"), getattr(row, f"vb_n18_s{seed}")])
            error = float(np.abs(original[si, :, 0] - expected).max())
            controls["max_saved_value_error"] = max(controls["max_saved_value_error"], error)
            assert error <= a.prediction_tolerance, (row.pair_id, seed, "saved prediction mismatch", error)
            baselines.append({"pair_id": row.pair_id, "cohort": row.cohort, "seed": seed,
                "mask_compatible": compatible, "saved_error": error,
                **{f"{name}_{side}": float(original[si, sj, mi]) for mi, name in enumerate(("value", "success_mass", "outcome"))
                   for sj, side in enumerate(("a", "b"))}})
        image_keys = list(obs.images)
        specs = specifications(base.depth, mask.shape[1], image_keys, obs.image_masks)
        active = []
        for spec in specs:
            if not compatible and spec[1] in ("all", "text_state"):
                skips.append({"pair_id": row.pair_id, "band": spec[0], "tokens": spec[1], "component": spec[2],
                              "reason": "prefix masks differ between endpoints"})
            else:
                active.append(spec)

        def record(pred, band, tokens, component, kind):
            for si, setup in enumerate(setups):
                for sj, direction in enumerate(("b_into_a", "a_into_b")):
                    d = {"pair_id": row.pair_id, "cohort": row.cohort, "episode_index": int(row.episode_index),
                         "seed": setup["seed"], "kind": kind, "band": band, "tokens": tokens,
                         "component": component, "direction": direction}
                    for mi, name in enumerate(("value", "success_mass", "outcome")):
                        d[f"{name}_before"] = float(original[si, sj, mi])
                        d[f"{name}_after"] = float(pred[si, sj, mi])
                        d[f"{name}_donor"] = float(original[si, 1 - sj, mi])
                    outputs.append(d)

        for start in range(0, len(active), a.patch_batch):
            batch = active[start:start + a.patch_batch]
            padded = batch + [batch[-1]] * (a.patch_batch - len(batch))
            got = np.asarray(interventions(kv, mask, jnp.asarray(np.stack([s[3] for s in padded])),
                jnp.asarray(np.stack([s[4] for s in padded])), jnp.asarray([s[5] for s in padded], jnp.int32)))
            for spec, pred in zip(batch, got):
                record(pred, *spec[:3], "cache")
        if not a.no_input_swaps:
            for group in ("agentview", "wrist", "state", "images"):
                mixed_kv, mixed_mask = kv_fn(observation(ix, group))
                record(np.asarray(fwd(mixed_kv, mixed_mask)), "all_layers", group, "input", "input")
        pd.DataFrame(outputs).to_csv(a.out / "interventions.csv", index=False)
        pd.DataFrame(baselines).to_csv(a.out / "baselines.csv", index=False)
        pd.DataFrame(skips, columns=["pair_id", "band", "tokens", "component", "reason"]).to_csv(a.out / "skipped.csv", index=False)
        log(f"{row.pair_id}: baseline delta={original[0, 1, 0] - original[0, 0, 0]:+.4f}; compatible={compatible}; {len(active)} patches")
    (a.out / "run.json").write_text(json.dumps({"completed": True, "pairs": len(manifest), "controls": controls,
        "expert_hashes": hashes, "image_prefix_order": image_keys, "devices": [str(d) for d in jax.devices()],
        "minutes": (time.time() - t0) / 60, "command": sys.argv,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=2))
    log(f"complete; controls {controls}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--pairs", type=Path)
    ap.add_argument("--cache", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--experts", type=Path, nargs="+")
    ap.add_argument("--checkpoint", default=CKPT)
    ap.add_argument("--max-pairs", type=int)
    ap.add_argument("--patch-batch", type=int, default=4)
    ap.add_argument("--prediction-tolerance", type=float, default=.005)
    ap.add_argument("--no-input-swaps", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        assert a.pairs and a.cache and a.out and a.experts
        run(a)


if __name__ == "__main__":
    main()
