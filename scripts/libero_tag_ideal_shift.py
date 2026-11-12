"""How far SHOULD the action move with the tag, if a policy learned the data exactly? (2026-11-06, tag deep dive)

Why: every tagged fine-tune moves its executed actions by ~0.0008 raw units between the + and - tag (tag/noise ~2%,
the same as a model never trained with a tag; reports/2026-11-04-tag-usage, 18338865, 18352443). Two explanations:
  (a) the labels are too weak: in the data, + and - frames at the same state have almost the same actions, so even
      a perfect learner would barely move;
  (b) the conditioning does not learn: the data supports a clearly larger move than the models show.
Test [ours]: on the 900 train rollouts (every 4th step, 104,049 frames; the label grid), regress the executed action
chunk (first 5 steps x 7 dims, raw units, as libero_tag_usage.py's action test) on
  S = the tag-predictability state (frozen ImageNet ResNet-18 features of both cameras, robot state, t/520, pots on
      the stove, both value functions' V; scripts/libero_tag_predictability.py), plus the tag as +1 / -1,
with an MLP, 5 outer folds by episode, early stopping on 10% of the training episodes. Out of fold:
  ideal shift = RMS over the 35 executed values of f(S, +1) - f(S, -1), mean over frames
     (the analogue of the action test's tag_pos_vs_neg: same state, same "noise", only the tag differs)
  noise ref   = sqrt(2) x RMS of the tag-free residual a - f0(S), mean over frames (analogue of noise_same_tag:
     two draws at the same state); an UPPER bound on the policy's own spread, since S is weaker than pi05's view.
Direction of bias: S misses state detail that pi05 sees, so part of what the tag predicts here is state, not action;
the ideal shift is an upper bound on what a pi05-strength learner should show.

Positive control (also written as a label set): tag = sign of the residual's first principal component over the arm
dims (0-5, standardised; gripper excluded), cut at the median -> 50% positive. It is a deviation of the executed
action from the typical action at this state, so it is (nearly) unpredictable from the state and fully determined by
the action: a policy that can condition on a per-frame text tag at all must learn it. Its action information is
checked with the tag-predictability classifier (S vs S+A). Dose variants flip a random share q of frames.

Usage (one GPU): .venv/bin/python scripts/libero_tag_ideal_shift.py --out reports/2026-11-06-tag-deep-dive

--table mode [2026-11-08, explore collection; ours]: the same test on another episode table, per group of episodes,
so data from different collections can be compared on equal terms (same V, same label rule, same frames per group).
S drops the expert V (only the table's OOF ResNet V exists) and takes pots on the stove from the raw goals; scored:
the given label set, the random 40% floor and the action-residual PC1 positive control (a fully action-determined
tag: it scales with how much the actions vary at a state). Folds are per group.
  .venv/bin/python scripts/libero_tag_ideal_shift.py --table <table> --labels r2_vb40 --layout fixed \
      --group-col round --out <dir>
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import libero_advantage_examples as X  # noqa: E402
import libero_tag_predictability as P  # noqa: E402
from vl.libero_resnet import table_meta  # noqa: E402
from vl.libero_subtask import subtask_labels  # noqa: E402
from vl.metrics import auc  # noqa: E402

EXEC, DIMS, FOLDS, SEED = 5, 7, 5, 0
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:6.0f} s] {m}", flush=True)


class Reg(nn.Module):
    def __init__(self, k, o):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(k, 512), nn.ReLU(), nn.Dropout(0.1), nn.Linear(512, 512), nn.ReLU(),
                               nn.Dropout(0.1), nn.Linear(512, o))

    def forward(self, x):
        return self.f(x)


def fit_reg(Xtr, Ytr, Xva, Yva, Xte_list, device, epochs=60, bs=2048):
    """MSE regression on standardised targets; returns predictions (raw units) for each array in Xte_list."""
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
    ym, ys = Ytr.mean(0, keepdims=True), Ytr.std(0, keepdims=True) + 1e-6
    tx = lambda x: torch.as_tensor((x - mu) / sd, dtype=torch.float32, device=device)
    ty = lambda y: torch.as_tensor((y - ym) / ys, dtype=torch.float32, device=device)
    Xtr_, Ytr_, Xva_, Yva_ = tx(Xtr), ty(Ytr), tx(Xva), ty(Yva)
    torch.manual_seed(SEED)
    m = Reg(Xtr.shape[1], Ytr.shape[1]).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    best, state, bad = 1e9, None, 0
    for _ in range(epochs):
        m.train()
        perm = torch.randperm(len(Xtr_), device=device)
        for s in range(0, len(perm), bs):
            i = perm[s:s + bs]
            opt.zero_grad(); ((m(Xtr_[i]) - Ytr_[i]) ** 2).mean().backward(); opt.step()
        m.eval()
        with torch.no_grad():
            v = float(((m(Xva_) - Yva_) ** 2).mean())
        if v < best - 1e-4:
            best, state, bad = v, {k: x.clone() for k, x in m.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 4:
                break
    m.load_state_dict(state); m.eval()
    with torch.no_grad():
        return [m(tx(x)).cpu().numpy() * ys + ym for x in Xte_list]


def folds(ep_ids):
    rng = np.random.default_rng(SEED)
    ueps = np.unique(ep_ids)
    fold_of = dict(zip(rng.permutation(ueps), np.arange(len(ueps)) % FOLDS))
    fold = np.array([fold_of[x] for x in ep_ids])
    out = []
    for k in range(FOLDS):
        te = fold == k
        tr_eps = np.unique(ep_ids[~te])
        va_eps = np.random.default_rng(SEED + k).choice(tr_eps, max(1, len(tr_eps) // 10), replace=False)
        va = ~te & np.isin(ep_ids, va_eps)
        out.append((~te & ~va, va, te))
    return out


def rms_rows(x):
    return np.sqrt(np.mean(np.square(x), axis=1))


def boot_mean_ci(v, ep_ids, n=2000):
    g = pd.DataFrame({"e": ep_ids, "v": v}).groupby("e").v.agg(["sum", "size"])
    rng = np.random.default_rng(SEED)
    m = []
    for _ in range(n):
        i = rng.integers(0, len(g), len(g))
        m.append(g["sum"].values[i].sum() / g["size"].values[i].sum())
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def predictability(y, S, SA, ep_ids, fl, device):
    """Tag-predictability gain (share of the tag's entropy the action explains beyond the state), as in
    scripts/libero_tag_predictability.py, on the same folds."""
    pS, pSA = np.zeros(len(y)), np.zeros(len(y))
    for tr, va, te in fl:
        pS[te] = P.fit_predict(S[tr], y[tr], S[va], y[va], S[te], device)
        pSA[te] = P.fit_predict(SA[tr], y[tr], SA[va], y[va], SA[te], device)
    b = y.mean()
    H = float(-(b * np.log(b) + (1 - b) * np.log(1 - b)))
    lS, lSA = P.ll(pS, y), P.ll(pSA, y)
    return {"auc_S": float(auc(y.astype(int), pS)), "auc_SA": float(auc(y.astype(int), pSA)),
            "gain_action_over_state": float((lS.mean() - lSA.mean()) / H),
            "gain_ci95": [x / H for x in boot_mean_ci(lS - lSA, ep_ids)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--write-labels", default=str(X.LAB), help="dir for the positive-control label sets")
    ap.add_argument("--candidates", nargs="+", default=None, metavar="NAME",
                    help="[2026-11-07, item 3] score only the current labels, the random control and these label "
                         "sets (labels/<NAME>/labels.parquet); no control label sets are written")
    ap.add_argument("--gate", default=None, metavar="DIR",
                    help="with --candidates: copy the candidate with the largest ideal shift to DIR/labels.parquet "
                         "if its 95%% CI lies above the current labels' CI; otherwise exit 3 (stops the chain)")
    ap.add_argument("--table", default=None, help="--table mode (see the docstring): episode table dir")
    ap.add_argument("--labels", default=None, help="--table: label set <table>/labels/<NAME> (positive and V)")
    ap.add_argument("--v-labels", default=None, help="--table: take V from this label set instead (default --labels)")
    ap.add_argument("--layout", default=None, help="--table: keep only episodes of this layout (fixed / fresh)")
    ap.add_argument("--group-col", default=None, help="--table: episodes.csv column to group by (e.g. round)")
    ap.add_argument("--max-episodes", type=int, default=None, help="--table: per group, for smoke tests")
    ap.add_argument("--curve", type=int, nargs="+", default=None, metavar="N",
                    help="--table: data curve, rerun on N episodes per group, spread evenly over start states")
    ap.add_argument("--curve-seeds", type=int, default=3, help="--curve: subsamples per N (1 at the full group)")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if a.table:
        return table_mode(a, out, device)
    key = ["episode_index", "frame_index"]
    eps = pd.read_csv(X.TABLE / "episodes.csv").set_index("episode_index")
    r = pd.read_parquet(X.LAB / "oof_resnet_n20_p30" / "labels.parquet").sort_values(key).reset_index(drop=True)
    e = pd.read_parquet(X.LAB / "oof_expert_n20_p30" / "labels.parquet").sort_values(key).reset_index(drop=True)
    cal = pd.read_parquet(P.CAL).sort_values(key).reset_index(drop=True)
    rl = pd.read_parquet(P.RULES).sort_values(key).reset_index(drop=True)
    lab = {n: pd.read_parquet(X.LAB / n / "labels.parquet").sort_values(key).reset_index(drop=True)
           for n in ("resnet_valuebinned_p40", "scoped_valuebinned_p40", "oracle_outcome")}
    for t_ in (e, cal, rl, *lab.values()):
        assert (t_[key].values == r[key].values).all()
    meta = table_meta(X.TABLE, 4).reset_index(drop=True)
    meta["row"] = np.arange(len(meta))
    rows = r[key].merge(meta[key + ["row"]], on=key, how="left", validate="one_to_one").row.to_numpy()
    d = r[key].copy()
    img = P.image_features(rows, device).astype(np.float32)
    st, act20, mask20 = P.state_action(d, eps)
    act = act20.reshape(len(d), P.N_ACT, 7)[:, :EXEC, :DIMS].reshape(len(d), -1)  # executed part, raw units
    full = mask20[:, :EXEC].min(1) > 0  # frames with 5 real executed actions (not past the episode end)
    scal = np.stack([d.frame_index.to_numpy() / 520.0, cal.pots.to_numpy(), r.V.to_numpy(), e.V.to_numpy()], 1
                    ).astype(np.float32)
    S = np.concatenate([img, st, scal], 1)
    SA = np.concatenate([S, act20, mask20], 1)
    ep_ids = d.episode_index.to_numpy()
    fl = folds(ep_ids)
    log(f"{len(d)} frames ({full.mean():.3f} with 5 executed actions), {len(np.unique(ep_ids))} episodes")

    # tag-free regression: residual (noise reference) and the positive-control direction
    f0 = np.zeros_like(act)
    for tr, va, te in fl:
        f0[te] = fit_reg(S[tr], act[tr], S[va], act[va], [S[te]], device)[0]
    resid = act - f0
    noise_ref = np.sqrt(2) * rms_rows(resid)
    log(f"tag-free fit: R2 {1 - resid[full].var(0).sum() / act[full].var(0).sum():.3f}, "
        f"noise ref (sqrt2 x resid RMS) {noise_ref[full].mean():.4f}")

    rng = np.random.default_rng(SEED)
    R = resid.reshape(len(d), EXEC, DIMS)[:, :, :6].reshape(len(d), -1)
    Rz = (R - R[full].mean(0)) / (R[full].std(0) + 1e-9)
    _, _, Vt = np.linalg.svd(Rz[full][rng.permutation(full.sum())[:30000]], full_matrices=False)
    score = Rz @ Vt[0]
    ctrl = (score > np.median(score[full])).astype(int)
    cand = {}
    for n in a.candidates or []:
        c = pd.read_parquet(X.LAB / n / "labels.parquet").sort_values(key).reset_index(drop=True)
        assert (c[key].values == r[key].values).all(), n
        cand[n] = c.positive.to_numpy()
    labels = {
        "RECAP faithful (value-binned 40%)": lab["resnet_valuebinned_p40"].positive.to_numpy(),
        "scoped value-binned 40%": lab["scoped_valuebinned_p40"].positive.to_numpy(),
        "oracle outcome": lab["oracle_outcome"].positive.to_numpy(),
        "CONTROL+ stall flag": rl.stall.to_numpy().astype(int),
        "CONTROL- random 40%": (rng.random(len(d)) < 0.4).astype(int),
        "POSITIVE CONTROL: action-residual PC1 (50%)": ctrl,
    }
    for q in (0.25, 0.4):  # dose variants: flip a random share q
        flip = rng.random(len(d)) < q
        labels[f"positive control, {int(q * 100)}% flipped"] = np.where(flip, 1 - ctrl, ctrl)
    if cand:  # candidates mode: current labels, the random floor and the candidates only
        labels = {k: labels[k] for k in ("RECAP faithful (value-binned 40%)", "CONTROL- random 40%")}
        labels.update({f"candidate {n}": y for n, y in cand.items()})

    res = []
    for name, y in labels.items():
        y = y.astype(np.float64)
        sp, sm = np.zeros_like(act), np.zeros_like(act)
        for tr, va, te in fl:
            Xtr = np.concatenate([S[tr], 2 * y[tr, None] - 1], 1)
            Xva = np.concatenate([S[va], 2 * y[va, None] - 1], 1)
            p1, p0 = fit_reg(Xtr, act[tr], Xva, act[va],
                             [np.concatenate([S[te], np.ones((te.sum(), 1), np.float32)], 1),
                              np.concatenate([S[te], -np.ones((te.sum(), 1), np.float32)], 1)], device)
            sp[te], sm[te] = p1, p0
        dlt = (sp - sm)[full]
        shift = rms_rows(dlt)
        d3 = dlt.reshape(-1, EXEC, DIMS)
        row = {"labels": name, "pos_share": float(y.mean()),
               "ideal_shift_raw": float(shift.mean()), "ideal_shift_ci95": boot_mean_ci(shift, ep_ids[full]),
               "ideal_shift_arm": float(np.sqrt((d3[:, :, :6] ** 2).mean(axis=(1, 2))).mean()),
               "ideal_shift_gripper": float(np.sqrt((d3[:, :, 6] ** 2).mean(axis=1)).mean()),
               "noise_ref_raw": float(noise_ref[full].mean()),
               "ideal_ratio_vs_noise_ref": float(shift.mean() / noise_ref[full].mean())}
        if name.startswith(("POSITIVE", "positive", "RECAP", "scoped", "candidate")):
            row.update(predictability(y, S, SA, ep_ids, fl, device))
        res.append(row)
        log(f"{name}: " + json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()
                                       if k != "labels"}))

    # write the positive-control label sets in the policy fine-tune's labels format (same grid as every label set)
    wl = Path(a.write_labels)
    written = {}
    for tag_name, y in (() if cand else
                        (("control_actresid_pc1_p50", labels["POSITIVE CONTROL: action-residual PC1 (50%)"]),
                         ("control_actresid_pc1_p50_flip40", labels["positive control, 40% flipped"]))):
        p = wl / tag_name / "labels.parquet"
        p.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"episode_index": d.episode_index, "frame_index": d.frame_index,
                      "positive": y.astype(np.int64)}).to_parquet(p)
        written[tag_name] = str(p)
        log(f"wrote {p}")
    tab = pd.DataFrame(res)
    tab.to_csv(out / "ideal_shift_table.csv", index=False)
    (out / "ideal_shift_summary.json").write_text(json.dumps(
        {"n_frames": int(len(d)), "full_frames": int(full.sum()), "folds": FOLDS,
         "control_pc1_loading": Vt[0].tolist(), "labels_written": written, "results": res}, indent=2))
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
    print(tab.drop(columns=["ideal_shift_ci95"]).round(4).to_string(index=False))
    if a.gate:
        sys.exit(gate(res, Path(a.gate), out))


def score(name, y, S, SA, act, full, noise_ref, ep_ids, fl, device, pred=True, resid=None) -> dict:
    """Ideal shift of one 0/1 label set (as in main), plus the tag-predictability gain if pred.

    resid (= act - f0(S), out of fold) adds tag_mse_gain [2026-11-08, ours]: 1 - MSE(a, f(S, true tag)) / MSE(a, f0(S))
    on held-out frames. The ideal shift says how much the fitted model USES the tag, which overfitting can inflate on
    small data; this says whether using it is RIGHT (random labels: ~0 or below)."""
    y = y.astype(np.float64)
    sp, sm = np.zeros_like(act), np.zeros_like(act)
    for tr, va, te in fl:
        Xtr = np.concatenate([S[tr], 2 * y[tr, None] - 1], 1)
        Xva = np.concatenate([S[va], 2 * y[va, None] - 1], 1)
        sp[te], sm[te] = fit_reg(Xtr, act[tr], Xva, act[va],
                                 [np.concatenate([S[te], np.ones((te.sum(), 1), np.float32)], 1),
                                  np.concatenate([S[te], -np.ones((te.sum(), 1), np.float32)], 1)], device)
    dlt = (sp - sm)[full]
    shift = rms_rows(dlt)
    d3 = dlt.reshape(-1, EXEC, DIMS)
    row = {"labels": name, "pos_share": float(y.mean()),
           "ideal_shift_raw": float(shift.mean()), "ideal_shift_ci95": boot_mean_ci(shift, ep_ids[full]),
           "ideal_shift_arm": float(np.sqrt((d3[:, :, :6] ** 2).mean(axis=(1, 2))).mean()),
           "ideal_shift_gripper": float(np.sqrt((d3[:, :, 6] ** 2).mean(axis=1)).mean()),
           "noise_ref_raw": float(noise_ref[full].mean()),
           "ideal_ratio_vs_noise_ref": float(shift.mean() / noise_ref[full].mean())}
    if resid is not None:
        e_tag = act - np.where(y[:, None] > 0.5, sp, sm)
        row["tag_mse_gain"] = float(1 - np.square(e_tag[full]).mean() / np.square(resid[full]).mean())
    if pred:
        row.update(predictability(y, S, SA, ep_ids, fl, device))
    return row


def stratified_episodes(ueps: np.ndarray, states: np.ndarray, n: int, seed: int) -> np.ndarray:
    """[2026-11-08, data curve] n of the episodes ueps, as evenly spread over start states as possible: shuffle the
    repeats within each state and the state order, then deal one episode per state in turn."""
    rng = np.random.default_rng(seed)
    by = [rng.permutation(ueps[states == s]) for s in rng.permutation(np.unique(states))]
    order = [b[i] for i in range(max(len(b) for b in by)) for b in by if i < len(b)]
    return np.array(order[:n])


def curve_table(t: pd.DataFrame) -> pd.DataFrame:
    """Mean (and sd over subsamples) per group x n_sub x label set, plus the real-minus-random excess."""
    cols = ["ideal_shift_raw", "ideal_shift_arm", "ideal_shift_gripper", "tag_mse_gain", "gain_action_over_state"]
    t = t.assign(kind=np.where(t.labels.str.startswith("random"), "random",
                               np.where(t.labels.str.startswith("positive control"), "pc1", "labels")))
    c = t.groupby(["group", "n_sub", "kind"])[cols].agg(["mean", "std"])
    c.columns = [f"{x}_{y}" for x, y in c.columns]
    c = c.reset_index().merge(t.groupby(["group", "n_sub"]).sub_seed.nunique().rename("subsamples").reset_index(),
                              on=["group", "n_sub"])
    rnd = c[c.kind == "random"].set_index(["group", "n_sub"]).ideal_shift_raw_mean.rename("random_shift")
    c = c.join(rnd, on=["group", "n_sub"])
    c["shift_over_random"] = c.ideal_shift_raw_mean - c.random_shift
    rm = c[c.kind == "random"].set_index(["group", "n_sub"]).tag_mse_gain_mean.rename("random_mse_gain")
    c = c.join(rm, on=["group", "n_sub"])
    c["mse_gain_over_random"] = c.tag_mse_gain_mean - c.random_mse_gain
    return c


def pots_on_stove(d: pd.DataFrame, eps: pd.DataFrame) -> np.ndarray:
    out = np.zeros(len(d), np.float32)
    for e, idx in d.groupby("episode_index").indices.items():
        with np.load(eps.raw[e]) as z:
            k = subtask_labels(z["goals"])
        out[idx] = k[d.frame_index.to_numpy()[idx]]
    return out


def table_mode(a, out: Path, device) -> None:
    key = ["episode_index", "frame_index"]
    tab = Path(a.table)
    X.TABLE, P.FEAT_CACHE = tab, tab / "feat_r18imagenet_trainframes.npy"  # image features of this table
    eps = pd.read_csv(tab / "episodes.csv", dtype={"goals_end": str}).set_index("episode_index")
    lab = pd.read_parquet(tab / "labels" / a.labels / "labels.parquet").sort_values(key).reset_index(drop=True)
    vl = lab if not a.v_labels else pd.read_parquet(tab / "labels" / a.v_labels / "labels.parquet"
                                                    ).sort_values(key).reset_index(drop=True)
    assert (vl[key].values == lab[key].values).all()
    meta = table_meta(tab, 4).reset_index(drop=True)
    meta["row"] = np.arange(len(meta))
    rows = lab[key].merge(meta[key + ["row"]], on=key, how="left", validate="one_to_one").row.to_numpy()
    img_all = P.image_features(rows, device).astype(np.float32)  # cached for every labelled frame, in lab's order
    keep = np.ones(len(lab), bool)
    if a.layout:
        keep &= lab.episode_index.map(eps.layout).to_numpy() == a.layout
    gcol = lab.episode_index.map(eps[a.group_col]).to_numpy() if a.group_col else np.zeros(len(lab), int)
    results, rng = [], np.random.default_rng(SEED)
    jobs = []
    for g in np.unique(gcol[keep]):
        m = keep & (gcol == g)
        if a.max_episodes:
            m &= np.isin(lab.episode_index.to_numpy(), np.unique(lab.episode_index[m])[:a.max_episodes])
        if not a.curve:
            jobs.append((g, m, None, None))
            continue
        ueps = np.unique(lab.episode_index[m])
        for n in sorted(a.curve):
            if n > len(ueps):
                log(f"group {g}: skip n={n} > {len(ueps)} episodes")
                continue
            for sd in range(1 if n == len(ueps) else a.curve_seeds):
                pick = stratified_episodes(ueps, eps.init_state.reindex(ueps).to_numpy(), n, SEED + 1000 * sd + n)
                jobs.append((g, m & np.isin(lab.episode_index.to_numpy(), pick), n, sd))
    for g, m, n_sub, sub_seed in jobs:
        d = lab.loc[m, key].reset_index(drop=True)
        img = img_all[m]
        st, act20, mask20 = P.state_action(d, eps)
        act = act20.reshape(len(d), P.N_ACT, 7)[:, :EXEC, :DIMS].reshape(len(d), -1)
        full = mask20[:, :EXEC].min(1) > 0
        scal = np.stack([d.frame_index.to_numpy() / 520.0, pots_on_stove(d, eps), vl.V.to_numpy()[m]], 1
                        ).astype(np.float32)
        S = np.concatenate([img, st, scal], 1)
        SA = np.concatenate([S, act20, mask20], 1)
        ep_ids = d.episode_index.to_numpy()
        e_g = eps.loc[np.unique(ep_ids)]
        fl = folds(ep_ids)
        f0 = np.zeros_like(act)
        for tr, va, te in fl:
            f0[te] = fit_reg(S[tr], act[tr], S[va], act[va], [S[te]], device)[0]
        resid = act - f0
        noise_ref = np.sqrt(2) * rms_rows(resid)
        r2 = float(1 - resid[full].var(0).sum() / act[full].var(0).sum())
        R = resid.reshape(len(d), EXEC, DIMS)[:, :, :6].reshape(len(d), -1)
        Rz = (R - R[full].mean(0)) / (R[full].std(0) + 1e-9)
        _, _, Vt = np.linalg.svd(Rz[full][rng.permutation(full.sum())[:30000]], full_matrices=False)
        sc = Rz @ Vt[0]
        info = {"group": str(g), "episodes": int(len(e_g)), "success": float(e_g.success.mean()), "frames": int(len(d)),
                "tag_free_r2": r2, "action_rms": float(rms_rows(act)[full].mean()),
                "states": int(e_g.init_state.nunique()), "n_sub": n_sub, "sub_seed": sub_seed}
        log(f"group {g}: " + json.dumps(info))
        sets = {a.labels: lab.positive.to_numpy()[m], "random 40%": (rng.random(len(d)) < 0.4).astype(int),
                "positive control PC1 (50%)": (sc > np.median(sc[full])).astype(int)}
        for name, y in sets.items():
            row = {**info, **score(name, y, S, SA, act, full, noise_ref, ep_ids, fl, device,
                                   pred=not name.startswith("random"), resid=resid)}
            results.append(row)
            log(f"  {name}: shift {row['ideal_shift_raw']:.4f} {np.round(row['ideal_shift_ci95'], 4).tolist()}, "
                f"ratio {row['ideal_ratio_vs_noise_ref']:.3f}, gain {row.get('gain_action_over_state', float('nan')):.4f}, "
                f"tag mse gain {row['tag_mse_gain']:.4f}")
    t = pd.DataFrame(results)
    if a.curve:
        c = curve_table(t)
        c.to_csv(out / "curve.csv", index=False)
        print(c.round(4).to_string(index=False))
    t.to_csv(out / "ideal_shift_table.csv", index=False)
    (out / "ideal_shift_summary.json").write_text(json.dumps(
        {"table": str(tab), "labels": a.labels, "v_labels": a.v_labels or a.labels, "layout": a.layout,
         "group_col": a.group_col, "max_episodes": a.max_episodes, "curve": a.curve, "curve_seeds": a.curve_seeds,
         "folds": FOLDS, "results": results}, indent=2))
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
    print(t[["group", "episodes", "success", "labels", "pos_share", "ideal_shift_raw", "noise_ref_raw",
             "ideal_ratio_vs_noise_ref", "gain_action_over_state"]].round(4).to_string(index=False))


def gate(res: list[dict], dest: Path, out: Path) -> int:
    """[ours, 2026-11-07] pick the candidate with the largest ideal shift; it passes if its CI is above the current
    labels' CI. Writes <out>/gate.json; on a pass copies the labels to dest/labels.parquet. Returns the exit code."""
    import shutil
    cur = next(r for r in res if r["labels"].startswith("RECAP"))
    cands = [r for r in res if r["labels"].startswith("candidate ")]
    best = max(cands, key=lambda r: r["ideal_shift_raw"])
    ok = best["ideal_shift_ci95"][0] > cur["ideal_shift_ci95"][1]
    name = best["labels"].split(" ", 1)[1]
    g = {"current": cur["ideal_shift_raw"], "current_ci95": cur["ideal_shift_ci95"], "best": name,
         "best_shift": best["ideal_shift_raw"], "best_ci95": best["ideal_shift_ci95"], "pass": bool(ok),
         "rule": "largest ideal shift; pass if its 95% CI lower bound > the current labels' upper bound [ours]"}
    if ok:
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy(X.LAB / name / "labels.parquet", dest / "labels.parquet")
        g["copied_to"] = str(dest / "labels.parquet")
    (out / "gate.json").write_text(json.dumps(g, indent=2))
    log(f"gate: {json.dumps(g)}")
    return 0 if ok else 3


if __name__ == "__main__":
    main()
