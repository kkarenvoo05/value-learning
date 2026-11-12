"""Does the advantage-tagged policy use its tag? (Stage 6 diagnosis, 2026-11-04, after the no-gain eval.)

Eval result: arm C with the positive tag 0.409, with the negative tag 0.430, success-only arm A 0.406 (330 trials
each). The negative tag not hurting suggests the policy ignores the tag. Two offline checks, no simulator [ours]:

(1) Action test, val frames (states never trained on). For each frame and K noise draws z, sample the action chunk
    with prompt tag positive / negative / none using the SAME noise z (sample_actions is deterministic given z).
      tag effect   = RMS( a(pos, z)  - a(neg, z)  )
      noise effect = RMS( a(pos, z1) - a(pos, z2) )     (how much the policy's own sampling moves the action)
    over the executed part of the chunk (first 5 steps, 7 real action dims), in the model's normalised action
    space and in raw action units. Ratio tag / noise ~ 0 => the tag changes nothing the sampling noise doesn't.
(2) Loss test, train frames with their real labels (the labels arm C was trained on). Flow-matching loss of the
    recorded action chunk with the CORRECT tag vs the FLIPPED tag, same noise and time. If the policy learned
    to condition on the tag, correct < flipped. Paired difference with an episode bootstrap CI.
Control: arm A (success-only, never saw a tag) through the same tests = the effect of the prompt text alone.

Usage (openpi env, one GPU; scripts/libero_tag_usage.sbatch):
    .venv/bin/python <vl>/scripts/libero_tag_usage.py --out <dir> [--val-eps 40 --per-ep 8 --k 4 --train-eps 64]
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from flax import nnx

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pi05_policy_finetune import as_inputs, base_policy, batch_arrays, load_finetuned  # noqa: E402
from vl import libero_finetune_data as fd  # noqa: E402

RUNS = Path("/iris/u/ankile/karenvo/libero/runs")
ARMS = {"C_resnet": RUNS / "policy_armC_resnet_s0_18275460", "A_success_only": RUNS / "policy_armA_s0_18288138"}
LABELS = fd.TABLE / "labels" / "oof_resnet_n20_p30" / "labels.parquet"
EXEC, DIMS = 5, 7  # executed actions per replan, real LIBERO action dims
TAGS = {"pos": True, "neg": False, "none": None}
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.0f} s] {msg}", flush=True)


def val_frames(n_eps, per_ep, seed=1):
    eps = pd.read_csv(fd.TABLE / "episodes.csv")
    rows = eps[eps.split == "val"].sort_values("episode_index")
    rows = rows.iloc[np.random.default_rng(seed).permutation(len(rows))[:n_eps]]
    rng, out = np.random.default_rng(seed + 1), []
    for r in rows.itertuples():
        ep = fd.load_episode(r.raw, None, int(r.episode_index))
        for t in np.sort(rng.choice(len(ep["actions"]), per_ep, replace=False)):
            out.append((int(r.episode_index), int(t), ep))
    return out


def train_frames(n_eps, per_ep, seed=2, labels=LABELS):
    """Train frames with their real arm-C label; half the frames positive, half negative where possible."""
    lab = pd.read_parquet(labels)
    eps = pd.read_csv(fd.TABLE / "episodes.csv")
    eps = eps[eps.split == "train"]
    rng = np.random.default_rng(seed)
    pick = eps.iloc[rng.permutation(len(eps))[:n_eps]]
    out = []
    for r in pick.itertuples():
        L = lab[lab.episode_index == r.episode_index].set_index("frame_index").positive
        ep = fd.load_episode(r.raw, None, int(r.episode_index))
        fr = [f for f in L.index if f < len(ep["actions"])]
        pos = [f for f in fr if L[f]]; neg = [f for f in fr if not L[f]]
        h = per_ep // 2
        ts = list(rng.choice(pos, min(h, len(pos)), replace=False)) + list(rng.choice(neg, min(h, len(neg)), replace=False))
        out += [(int(r.episode_index), int(t), ep, bool(L[t])) for t in ts]
    return out


def demo_frames(n_eps, per_ep, labels, seed=2):
    """[2026-11-06] Loss-test frames from the held-out (inner) LIBERO demos with their speed tag, for the speed-tag
    pre-training stage. Speed tags are per episode, so each episode gives frames of one tag; n_eps split evenly
    between positive and negative episodes where possible."""
    lab = pd.read_parquet(labels).drop_duplicates("episode_index").set_index("episode_index").positive
    inner = fd.pretrain_demo_episodes(fd.DEMO_TABLE, ("inner",))
    inner = inner.assign(positive=inner.episode_index.map(lab))
    assert inner.positive.notna().all(), "speed labels missing for inner demos"
    rng = np.random.default_rng(seed)
    pick = pd.concat([g.iloc[rng.permutation(len(g))[:n_eps // 2]] for _, g in inner.groupby("positive")])
    out = []
    for r in pick.itertuples():
        ep = fd.load_episode(r.raw, None, int(r.episode_index))
        ts = rng.choice(len(ep["actions"]), min(per_ep, len(ep["actions"])), replace=False)
        out += [(int(r.episode_index), int(t), ep, bool(r.positive)) for t in ts]
    return out


def rms(x, axis):
    return np.sqrt(np.mean(np.square(x), axis=axis))


def boot_ci(per_ep: pd.Series, n=2000, seed=0):
    """Episode bootstrap of the mean of per-episode means."""
    v = per_ep.values
    rng = np.random.default_rng(seed)
    m = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n)]
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def boot_ratio_ci(num: pd.Series, den: pd.Series, n=2000, seed=0):
    """Episode bootstrap of a ratio of means, sum(num) / sum(den) over the resampled episodes' frames.

    The point estimate is df.tag.mean() / df.noise.mean() (pooled over frames), so the interval must resample
    the same statistic. Takes per-episode SUMS so episodes with more frames weigh more, as in the pooled estimate.
    (Fixed 2026-11-04: the CI used to bootstrap the mean of per-episode ratios, a different statistic, and
    could exclude the point estimate: 18310225 reported 0.021 with CI [0.024, 0.033].)
    """
    a, b = num.values, den.values
    rng = np.random.default_rng(seed)
    m = []
    for _ in range(n):
        i = rng.integers(0, len(a), len(a))
        m.append(a[i].sum() / b[i].sum())
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def sample_variants(policy, frames, k, mb, variants):
    """variants: name -> f(ep, t) -> raw sample. Returns (normalised, raw) actions per variant, (k, n, ah, ad), with
    the SAME noise draws z for every variant (sample_actions is deterministic given z)."""
    ah, ad = policy._model.action_horizon, policy._model.action_dim
    batches = {name: batch_arrays(policy, [f(ep, t) for _, t, ep in frames]) for name, f in variants.items()}
    n = len(frames)
    noise = np.asarray(jax.random.normal(jax.random.PRNGKey(123), (k, n, ah, ad)), np.float32)
    acts = {name: np.zeros((k, n, ah, ad), np.float32) for name in variants}  # normalised model space
    for name in variants:
        b = batches[name]
        for j in range(k):
            for i in range(0, n, mb):
                sl = jax.tree.map(lambda x: x[i:i + mb], b)
                obs, _ = as_inputs(sl)
                m = min(mb, n - i)
                z = jnp.asarray(noise[j, i:i + m])
                if m < mb:  # one compiled shape
                    obs = jax.tree.map(lambda x: jnp.concatenate([x, jnp.repeat(x[-1:], mb - m, 0)]), obs)
                    z = jnp.concatenate([z, jnp.repeat(z[-1:], mb - m, 0)])
                a = policy._sample_actions(jax.random.PRNGKey(0), obs, noise=z)
                acts[name][j, i:i + m] = np.asarray(a, np.float32)[:m]
        log(f"  sampled {name}")
    # raw action units through the policy's output transform (unnormalise; keeps the first DIMS dims)
    state = next(iter(batches.values()))["state"]
    raw = {name: np.stack([np.stack([policy._output_transform({"actions": acts[name][j, i], "state": state[i]})["actions"]
                                     for i in range(n)]) for j in range(k)]) for name in variants}
    return acts, raw


def action_test(policy, frames, k, mb):
    variants = {name: (lambda ep, t, tag=tag: fd.sample(ep, t, tag)) for name, tag in TAGS.items()}
    acts, raw = sample_variants(policy, frames, k, mb, variants)
    ep_ids = np.array([e for e, _, _ in frames])
    res = {}
    for space, A in (("norm", acts), ("raw", raw)):
        ex = {nm: A[nm][:, :, :EXEC, :DIMS] for nm in TAGS}
        tag_pn = rms(ex["pos"] - ex["neg"], (2, 3))          # (k, n)
        tag_pnone = rms(ex["pos"] - ex["none"], (2, 3))
        noise_pp = rms(ex["pos"][:-1] - ex["pos"][1:], (2, 3))  # consecutive noise draws, same tag
        motion = rms(ex["pos"], (2, 3))
        df = pd.DataFrame({"ep": ep_ids, "tag": tag_pn.mean(0), "tag_none": tag_pnone.mean(0),
                           "noise": noise_pp.mean(0), "motion": motion.mean(0)})
        g = df.groupby("ep").sum()
        res[space] = {"tag_pos_vs_neg": float(df.tag.mean()), "tag_pos_vs_none": float(df.tag_none.mean()),
                      "noise_same_tag": float(df.noise.mean()), "action_rms": float(df.motion.mean()),
                      "ratio_tag_over_noise": float(df.tag.mean() / df.noise.mean()),
                      "ratio_ci95": boot_ratio_ci(g.tag, g.noise),
                      "frames_tag_gt_noise": float((df.tag > df.noise).mean())}
    # gripper: does the tag flip open/close in the executed steps?
    g_pos, g_neg = np.sign(raw["pos"][:, :, :EXEC, 6]), np.sign(raw["neg"][:, :, :EXEC, 6])
    g_p2 = np.sign(raw["pos"][1:, :, :EXEC, 6]); g_p1 = np.sign(raw["pos"][:-1, :, :EXEC, 6])
    res["gripper_flip_tag"] = float((g_pos != g_neg).mean())
    res["gripper_flip_noise"] = float((g_p1 != g_p2).mean())
    res["frames"], res["k"] = len(frames), k
    return res


# Prompt test [2026-11-06]: does the policy read its prompt at all? The tag is text in the prompt, so it can only work
# through the language pathway. Same frames and noise; the task instruction replaced (LIBERO-10 bddl names).
PROMPTS = {"near": "turn on the stove and put the moka pot on it",            # same objects, different task
           "far": "put the black bowl in the bottom drawer of the cabinet and close it",
           "empty": ""}


def prompt_test(policy, frames, k, mb):
    def with_prompt(p):
        return lambda ep, t: dict(fd.sample(ep, t, None), prompt=p)
    variants = {"true": lambda ep, t: fd.sample(ep, t, None), **{n: with_prompt(p) for n, p in PROMPTS.items()}}
    acts, raw = sample_variants(policy, frames, k, mb, variants)
    ep_ids = np.array([e for e, _, _ in frames])
    res = {}
    for space, A in (("norm", acts), ("raw", raw)):
        ex = {nm: A[nm][:, :, :EXEC, :DIMS] for nm in variants}
        noise = rms(ex["true"][:-1] - ex["true"][1:], (2, 3)).mean(0)
        r = {"noise_same_prompt": float(noise.mean())}
        for nm in PROMPTS:
            eff = rms(ex["true"] - ex[nm], (2, 3)).mean(0)
            g = pd.DataFrame({"ep": ep_ids, "eff": eff, "noise": noise}).groupby("ep").sum()
            r[nm] = {"effect": float(eff.mean()), "ratio_over_noise": float(eff.mean() / noise.mean()),
                     "ratio_ci95": boot_ratio_ci(g.eff, g.noise), "frames_gt_noise": float((eff > noise).mean())}
        res[space] = r
    g_true = np.sign(raw["true"][:, :, :EXEC, 6])
    res["gripper_flip"] = {nm: float((g_true != np.sign(raw[nm][:, :, :EXEC, 6])).mean()) for nm in PROMPTS}
    res["prompts"], res["frames"], res["k"] = PROMPTS, len(frames), k
    return res


def loss_test(policy, frames, mb, reps=4):
    gd, st = nnx.split(policy._model)
    f = jax.jit(lambda st, obs, act, rng: nnx.merge(gd, st).compute_loss(rng, obs, act, train=False))
    n = len(frames)
    truth = np.array([lab for *_, lab in frames])
    out = {}
    for name, tags in (("correct", truth), ("flipped", ~truth), ("none", None)):
        b = batch_arrays(policy, [fd.sample(ep, t, None if tags is None else bool(tags[i]))
                                  for i, (_, t, ep, _) in enumerate(frames)])
        L = np.zeros(n)
        for r in range(reps):
            for i in range(0, n, mb):
                sl = jax.tree.map(lambda x: x[i:i + mb], b)
                m = min(mb, n - i)
                if m < mb:
                    sl = jax.tree.map(lambda x: np.concatenate([x, np.repeat(x[-1:], mb - m, 0)]), sl)
                obs, act = as_inputs(sl)
                l = np.asarray(f(st, obs, act, jax.random.PRNGKey(1000 * r + i)), np.float32)  # same rng per row/tag
                L[i:i + m] += l.reshape(mb, -1).mean(1)[:m] / reps
        out[name] = L
        log(f"  loss tag {name}: {L.mean():.5f}")
    ep_ids = np.array([e for e, *_ in frames])
    d = pd.DataFrame({"ep": ep_ids, "diff": out["flipped"] - out["correct"], "pos": truth})
    g = d.groupby("ep")["diff"].mean()
    return {"frames": n, "episodes": int(len(g)), "loss_correct": float(out["correct"].mean()),
            "loss_flipped": float(out["flipped"].mean()), "loss_no_tag": float(out["none"].mean()),
            "flipped_minus_correct": float(d["diff"].mean()), "ci95": boot_ci(g),
            "rel_gain": float(d["diff"].mean() / out["correct"].mean()),
            "by_true_label": {str(k): float(v) for k, v in d.groupby("pos")["diff"].mean().items()},
            "frames_correct_lower": float((d["diff"] > 0).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-eps", type=int, default=40)
    ap.add_argument("--per-ep", type=int, default=8)
    ap.add_argument("--k", type=int, default=4, help="noise draws per frame")
    ap.add_argument("--train-eps", type=int, default=64)
    ap.add_argument("--train-per-ep", type=int, default=8)
    ap.add_argument("--mb", type=int, default=8)
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--run", default=None, help="another tagged run to test (name 'tagged'); loss test uses --labels")
    ap.add_argument("--labels", default=str(LABELS), help="labels the tagged run was trained on (loss test)")
    ap.add_argument("--prompt-test", action="store_true", help="also run the prompt test (instruction swapped)")
    ap.add_argument("--base", action="store_true", help="also test plain pi05_libero (name 'base'; prompt test only "
                    "is meaningful: it never saw a tag)")
    ap.add_argument("--steps", nargs="+", default=["final"], help="checkpoints of --run to test (trainable_<step>; "
                    "runs trained with --save-every), e.g. 1000 3000 final: when the tag is learned [2026-11-06]")
    ap.add_argument("--loss-demos", action="store_true", help="[2026-11-06] loss test on the held-out LIBERO demos "
                    "(all tasks) with --labels = speed labels, for the speed-tag pre-training run")
    a = ap.parse_args()
    if a.run:
        ARMS["tagged"] = Path(a.run)
        rj = Path(a.run) / "results.json"  # [2026-11-07] --subtask-prompt runs: prompts carry the frame's subtask
        if rj.exists() and json.loads(rj.read_text())["args"].get("subtask_prompt", False):
            fd.SUBTASK_PROMPT = True
            print("run trained with --subtask-prompt: test prompts include the subtask text", flush=True)
        a.arms = ["tagged"] + [x for x in a.arms if x not in ("C_resnet", "tagged")]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    vf = val_frames(a.val_eps, a.per_ep)
    tf = (demo_frames(a.train_eps, a.train_per_ep, a.labels) if a.loss_demos
          else train_frames(a.train_eps, a.train_per_ep, labels=a.labels))
    log(f"{len(vf)} val frames from {a.val_eps} episodes, {len(tf)} train frames "
        f"({sum(l for *_, l in tf)} positive) from {a.train_eps} episodes")
    res = {"setup": vars(a), "labels": a.labels, "arms": {}}
    todo = [(arm if (arm != "tagged" or len(a.steps) == 1) else f"tagged@{st}", arm, st if arm == "tagged" else "final")
            for arm in a.arms for st in (a.steps if arm == "tagged" else ["final"])]
    if a.base:
        ARMS["base"] = "pi05_libero"
        todo.append(("base", "base", "final"))
    for name, arm, step in todo:
        log(f"arm {name}: {ARMS[arm]} step {step}")
        policy = base_policy()[1] if arm == "base" else load_finetuned(ARMS[arm], step, seed=0)[0]
        r = {"action": action_test(policy, vf, a.k, a.mb)}
        log(f"  action test: {json.dumps(r['action']['norm'])}")
        if a.prompt_test:
            r["prompt"] = prompt_test(policy, vf, a.k, a.mb)
            log(f"  prompt test: {json.dumps(r['prompt']['norm'])}")
        r["loss"] = loss_test(policy, tf, a.mb)
        log(f"  loss test: {json.dumps(r['loss'])}")
        res["arms"][name] = r
        (out / "results.json").write_text(json.dumps(res, indent=2))
        del policy
        gc.collect()
    shutil_copy = Path(__file__).read_text()
    (out / Path(__file__).name).write_text(shutil_copy)
    log(f"done -> {out}")


if __name__ == "__main__":
    main()
