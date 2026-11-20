"""AG rounds 0-7 diagnostics for the Week 9 Nov 17 page [2026-11-17].

Inputs (all existing runs, nothing new is run):
  evals        330-trial val evals (10 val states x 33, env seed 7, matched noise seed 1) of base and AG rounds 0-7,
               plus the noise-seed-2 re-evals of rounds 5 and 7
  collections  each round's on-policy collection (round r is collected by the round r-1 policy; 60 train-state + 40
               fresh-layout episodes), with per-step goal predicates and pot positions (--args.save-raw); base =
               the 300 train-state base episodes of the distillation study (distill_collect_base)
  train-state  per-state evals on the 30 train states (10 repeats): base, round 0, round 7
Outputs: reports/2026-11-17-ag-rounds/{fig*.png, collection_episodes.csv, summary.json}

Collection failure classes (one per failed episode, from the goal predicates; a pot is "placed" once it is on the
stove for 10 consecutive steps and "lost" if it is then off for 20 consecutive steps; hand distance = |pot - eef|):
  no_pot         no pot is ever placed
  slid_off       a placed pot is lost within 50 steps of being placed (set down on the stove edge; as the Nov 14
                 analysis: the hand is already away and nothing pushes it)
  knocked_off    a placed pot is lost later (50+ steps after placement, while working on the other pot)
  second_unlifted  the first pot stays on; the second pot never rises 3 cm above its start height
  second_unplaced  the first pot stays on; the second pot is lifted but not placed before the 520-step limit
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RUNS = Path("/iris/u/ankile/karenvo/libero/runs")
OUT = Path(__file__).resolve().parents[1] / "reports" / "2026-11-17-ag-rounds"

EVALS = {  # label -> run dir (noise seed 1 unless named)
    "base": "eval_base_noise1_libero_10_18355478",
    "r0": "eval_ag_residual_vb40_s0_shift099_noise1_libero_10_18438187",
    "r1": "eval_agr_r1_s0_shift099_noise1_libero_10_18460307",
    "r2": "eval_agr_r2_s0_shift099_noise1_libero_10_18460313",
    "r3": "eval_agr_r3_s0_shift099_noise1_libero_10_18460319",
    "r4": "eval_agr_r4_s0_shift099_noise1_libero_10_18463162",
    "r5": "eval_agr_r5_s0_shift099_noise1_libero_10_18484607",
    "r6": "eval_agr_r6_s0_shift099_noise1_libero_10_18509037",
    "r7": "eval_agr_r7_s0_shift099_noise1_libero_10_18524603",
}
EVALS_NOISE2 = {"r5": "eval_agr_r5_s0_shift099_noise2_libero_10_18530240",
                "r7": "eval_agr_r7_s0_shift099_noise2_libero_10_18530241"}
W = {"r0": 9.257, "r1": 8.33, "r2": 8.225, "r3": 5.905, "r4": 5.217, "r5": 9.834, "r6": 6.222, "r7": 8.073}  # server.log
EPISODES = {f"r{r}": 900 + 100 * r for r in range(8)}  # rollouts in the round's training table
# collection c<r> = data collected by policy r-1 (fixed train states, fresh layouts)
COLLECT = {
    "base": ["distill_collect_base_libero_10_18507468"],
    "r0": ["agr_r1_collect_fixed_libero_10_18460302", "agr_r1_collect_fresh_libero_10_18460303"],
    "r1": ["agr_r2_collect_fixed_libero_10_18460308", "agr_r2_collect_fresh_libero_10_18460309"],
    "r2": ["agr_r3_collect_fixed_libero_10_18460314", "agr_r3_collect_fresh_libero_10_18460315"],
    "r3": ["agr_r4_collect_fixed_libero_10_18463157", "agr_r4_collect_fresh_libero_10_18463158"],
    "r4": ["agr_r5_collect_fixed_libero_10_18484602", "agr_r5_collect_fresh_libero_10_18484603"],
    "r5": ["agr_r6_collect_fixed_libero_10_18509032", "agr_r6_collect_fresh_libero_10_18509033"],
    "r6": ["agr_r7_collect_fixed_libero_10_18524598", "agr_r7_collect_fresh_libero_10_18524599"],
}
TRAIN_STATE = {"base": "ps_base_train_noise1_libero_10_18541680", "r0": "ps_r0_train_shift099_noise1_libero_10_18541681",
               "r7": "ps_r7_train_shift099_noise1_libero_10_18541682"}
DSRL = {"s0": {100: 0.548, 200: 0.482, 400: 0.797, 600: 0.830, 800: 0.827, 900: 0.652, 1000: 0.797, 1100: 0.830,
               1200: 0.809, 1300: 0.824, 1800: 0.861},
        "s1": {1300: 0.870, 1800: 0.891}}
ROUNDS = [f"r{r}" for r in range(8)]
CLASSES = ["slid_off", "knocked_off", "second_unlifted", "second_unplaced", "no_pot"]
CLASS_LABEL = {"slid_off": "placed pot slides off (< 50 steps)", "knocked_off": "placed pot lost later (knocked)",
               "second_unlifted": "second pot never lifted", "second_unplaced": "second pot lifted, not placed",
               "no_pot": "no pot ever placed"}
C = {"blue": "#3b6fd8", "orange": "#e07b39", "grey": "#8a8f98", "green": "#3a9a5b", "red": "#c9473f",
     "purple": "#8a5cc2", "teal": "#2a9d9b"}
CLASS_COLOR = {"slid_off": C["red"], "knocked_off": C["orange"], "second_unlifted": C["purple"],
               "second_unplaced": C["teal"], "no_pot": C["grey"]}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def load_eval(name: str) -> pd.DataFrame:
    df = pd.read_csv(RUNS / name / "episodes.csv", dtype={"goals_end": str})
    return df[df["init_state"] >= 0]


def runs_of(mask: np.ndarray, value: bool) -> list[tuple[int, int]]:
    """[start, end) of each maximal run where mask == value."""
    out, start = [], None
    for t, m in enumerate(mask):
        if m == value and start is None:
            start = t
        elif m != value and start is not None:
            out.append((start, t))
            start = None
    if start is not None:
        out.append((start, len(mask)))
    return out


def classify(raw: Path) -> dict:
    z = np.load(raw)
    g = z["goals"][:, :2].astype(bool)  # pot 1 (right start region), pot 2 (left)
    op = z["object_pos"]
    pos = [op[:, 0:3], op[:, 6:9]]
    hand = [np.linalg.norm(op[:, 3:6], axis=1), np.linalg.norm(op[:, 9:12], axis=1)]
    placed, lost = [None, None], [None, None]
    for p in range(2):
        on = [r for r in runs_of(g[:, p], True) if r[1] - r[0] >= 10 or r[1] == len(g)]  # success ends the episode
        if on:
            placed[p] = on[0][0]
            off = [r for r in runs_of(g[:, p], False) if r[0] > placed[p] and r[1] - r[0] >= 20]
            if off:
                lost[p] = off[0][0]
    order = sorted([p for p in range(2) if placed[p] is not None], key=lambda p: placed[p])
    first = order[0] if order else None
    second = order[1] if len(order) > 1 else (1 - first if first is not None else None)
    row = {"first_pot": first, "first_step": placed[first] if first is not None else np.nan,
           "second_step": placed[second] if len(order) > 1 else np.nan, "T": len(g)}
    row["gap"] = row["second_step"] - row["first_step"]
    cls = ""
    loss = [(lost[p], p) for p in range(2) if lost[p] is not None]
    if not order:
        cls = "no_pot"
    elif loss:
        t, p = min(loss)
        cls = "slid_off" if t - placed[p] < 50 else "knocked_off"
        row["lost_after_place"] = t - placed[p]
        row["hand_at_loss"] = hand[p][t]
    elif len(order) == 1:
        z0 = pos[second][0, 2]
        cls = "second_unplaced" if (pos[second][:, 2] - z0).max() > 0.03 else "second_unlifted"
    row["cls"] = cls
    return row


def collections() -> pd.DataFrame:
    rows = []
    for pol, dirs in COLLECT.items():
        for d in dirs:
            ep = pd.read_csv(RUNS / d / "episodes.csv", dtype={"goals_end": str})
            for _, e in ep.iterrows():
                r = classify(RUNS / d / "raw" / e["raw"])
                r.update(policy=pol, run=d, episode=int(e["episode"]), success=int(e["success"]),
                         fresh=int(e["init_state"] < 0), init_state=int(e["init_state"]), steps=int(e["policy_steps"]))
                if r["success"]:
                    r["cls"] = ""
                rows.append(r)
    return pd.DataFrame(rows)


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e5e7eb", lw=0.8)
    ax.set_axisbelow(True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "figure.dpi": 150})
    ev = {k: load_eval(v) for k, v in EVALS.items()}
    ev2 = {k: load_eval(v) for k, v in EVALS_NOISE2.items()}
    col = collections()
    col.to_csv(OUT / "collection_episodes.csv", index=False)
    summ: dict = {"evals": {}, "collections": {}}

    # ---- fig 1: success per round, eval + collection; success vs w
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [1.6, 1]})
    xs = np.arange(8)
    p = np.array([ev[r]["success"].mean() for r in ROUNDS])
    ci = np.array([wilson(int(ev[r]["success"].sum()), len(ev[r])) for r in ROUNDS])
    for r in ["base"] + ROUNDS:
        k, n = int(ev[r]["success"].sum()), len(ev[r])
        summ["evals"][r] = {"success": k / n, "n": n, "ci95": wilson(k, n), "w": W.get(r)}
    a.axhline(ev["base"]["success"].mean(), color=C["grey"], ls="--", lw=1)
    a.text(-0.3, ev["base"]["success"].mean() + 0.01, "base π₀.₅ 52.4%", color=C["grey"], ha="left", fontsize=9)
    a.fill_between(xs, ci[:, 0], ci[:, 1], color=C["blue"], alpha=0.12, lw=0)
    a.plot(xs, p, "o-", color=C["blue"], lw=2, label="held-out eval (330 trials, 95% CI)")
    for i, v in enumerate(p):
        a.annotate(f"{100 * v:.1f}", (i, v), textcoords="offset points", xytext=(-11, 7), ha="center", fontsize=8.5,
                   color=C["blue"])
    for r, d in ev2.items():
        i = ROUNDS.index(r)
        a.plot(i, d["success"].mean(), "o", mfc="white", mec=C["blue"], mew=1.5, ms=7,
               label="same policy, eval noise seed 2" if r == "r5" else None)
        summ["evals"][r]["success_noise2"] = d["success"].mean()
    cs = [col[col.policy == r]["success"].mean() for r in ROUNDS[:7]]
    a.plot(xs[:7], cs, "s:", color=C["green"], lw=1.5, ms=5, label="its own collection (train states + fresh)")
    a.set_xticks(xs, [f"{r}\n{EPISODES[r] / 1000:.1f}k" for r in ROUNDS])
    a.set_xlabel("round  /  rollouts in its training data")
    a.set_ylabel("success")
    a.set_ylim(0.45, 1.05)
    a.set_title("AG rounds: success per round")
    a.legend(loc="upper left", fontsize=8.5, frameon=False)
    style(a)
    ws = np.array([W[r] for r in ROUNDS])
    b.errorbar(ws, p, yerr=[p - ci[:, 0], ci[:, 1] - p], fmt="o", color=C["blue"], ecolor="#b9c8ee", capsize=2)
    for i, r in enumerate(ROUNDS):
        b.annotate(r, (ws[i], p[i]), textcoords="offset points", xytext=(6, -3), fontsize=8.5)
    rho = np.corrcoef(ws, p)[0, 1]
    summ["corr_w_success"] = rho
    b.set_xlabel("guidance weight w (set each round for shift 0.99)")
    b.set_ylabel("success")
    b.set_title(f"Success vs. guidance weight (r = {rho:+.2f})")
    style(b)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_rounds.png")
    plt.close(fig)

    # ---- fig 2: per-state heatmaps (val states, all rounds; train states base / r0 / r7)
    labels = ["base"] + ROUNDS
    vs = sorted(ev["base"]["init_state"].unique())
    M = np.array([[ev[c][ev[c].init_state == s]["success"].mean() for c in labels] for s in vs])
    order = np.argsort(M[:, 0])
    tr = {k: load_eval(v) for k, v in TRAIN_STATE.items()}
    ts = sorted(tr["base"]["init_state"].unique())
    T = np.array([[tr[c][tr[c].init_state == s]["success"].mean() for c in tr] for s in ts])
    torder = np.argsort(T[:, 0])
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 5.6), gridspec_kw={"width_ratios": [9, 3.4]})
    im = a.imshow(M[order], cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    for i in range(len(vs)):
        for j in range(len(labels)):
            v = M[order][i, j]
            a.text(j, i, f"{100 * v:.0f}", ha="center", va="center", fontsize=8, color="black" if 0.25 < v < 0.85 else "white")
    a.set_xticks(range(len(labels)), labels)
    a.set_yticks(range(len(vs)), [f"val {vs[i]}" for i in order])
    a.set_title("Held-out val states (33 trials each)")
    b.imshow(T[torder], cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    for i in range(len(ts)):
        for j in range(3):
            v = T[torder][i, j]
            b.text(j, i, f"{100 * v:.0f}", ha="center", va="center", fontsize=6.5, color="black" if 0.25 < v < 0.85 else "white")
    b.set_xticks(range(3), list(tr))
    b.set_yticks(range(len(ts)), [f"train {ts[i]}" for i in torder], fontsize=6.5)
    b.set_title("Train states (10 trials each)")
    fig.colorbar(im, ax=[a, b], fraction=0.025, pad=0.02, label="success")
    fig.savefig(OUT / "fig2_states.png", bbox_inches="tight")
    plt.close(fig)
    summ["val_states"] = {f"{vs[i]}": dict(zip(labels, M[i].round(3).tolist())) for i in range(len(vs))}
    summ["train_states"] = {"mean": {c: float(T[:, j].mean()) for j, c in enumerate(tr)},
                            "below_50": {c: int((T[:, j] < 0.5).sum()) for j, c in enumerate(tr)},
                            "zero": {c: [int(ts[i]) for i in range(len(ts)) if T[i, j] == 0] for j, c in enumerate(tr)},
                            "spearman_like_corr_base_gain_r7": float(np.corrcoef(T[:, 0], T[:, 2] - T[:, 0])[0, 1])}

    # ---- fig 3: are the same trials failing every round? (matched noise: trial = (state, repeat))
    key = ["init_state", "repeat"]
    F = pd.concat([ev[r].set_index(key)["success"].rename(r) for r in ROUNDS], axis=1)
    nfail = (1 - F).sum(axis=1)
    rate = F.groupby(level=0).mean().mean(axis=1)  # per-state success over the 8 rounds
    rng = np.random.default_rng(0)
    sims = []
    for _ in range(2000):  # null: each trial fails independently in each round at its state's (round-pooled) rate
        q = 1 - rate.reindex(F.index.get_level_values(0)).to_numpy()
        sims.append(np.bincount((rng.random((len(F), 8)) < q[:, None]).sum(1), minlength=9))
    sims = np.array(sims)
    obs = np.bincount(nfail.astype(int), minlength=9)
    fig, a = plt.subplots(figsize=(7.5, 3.6))
    a.bar(np.arange(9) - 0.2, obs, 0.4, color=C["blue"], label="observed (330 trials)")
    a.bar(np.arange(9) + 0.2, sims.mean(0), 0.4, color="#c7cdd6",
          yerr=[sims.mean(0) - np.percentile(sims, 2.5, 0), np.percentile(sims, 97.5, 0) - sims.mean(0)],
          ecolor=C["grey"], capsize=2, label="if failures were random within each state")
    a.set_xticks(range(9))
    a.set_xlabel("rounds (of r0-r7) in which the same trial (state, repeat, noise) failed")
    a.set_ylabel("trials")
    a.set_title("Do the same trials fail every round?")
    a.legend(frameon=False, fontsize=8.5)
    style(a)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_persistence.png")
    plt.close(fig)
    summ["persistence"] = {"observed": obs.tolist(), "null_mean": sims.mean(0).round(1).tolist(),
                           "always_fail_obs": int(obs[8]), "always_fail_null95": float(np.percentile(sims[:, 8], 97.5)),
                           "fail_ge6_obs": int(obs[6:].sum()), "fail_ge6_null": float(sims[:, 6:].sum(1).mean()),
                           "never_fail_obs": int(obs[0]), "never_fail_null": float(sims[:, 0].mean())}

    # ---- fig 4: failure classes per policy (collections), + eval end states by pot
    pols = list(COLLECT)
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.5, 1]})
    bottom = np.zeros(len(pols))
    for c in CLASSES:
        v = np.array([100 * ((col.policy == q) & (col.cls == c)).sum() / (col.policy == q).sum() for q in pols])
        a.bar(range(len(pols)), v, bottom=bottom, color=CLASS_COLOR[c], label=CLASS_LABEL[c], width=0.7)
        bottom += v
    for i, q in enumerate(pols):
        n = (col.policy == q).sum()
        a.text(i, bottom[i] + 0.8, f"{100 - bottom[i]:.0f}%\nn {n}", ha="center", fontsize=7.5, color=C["grey"])
    a.set_xticks(range(len(pols)), [f"{q}" for q in pols])
    a.set_xlabel("policy that collected (its collection feeds the next round)")
    a.set_ylabel("failures per 100 episodes")
    a.set_title("How collection episodes fail (labels: success, n)")
    a.legend(fontsize=7.5, frameon=False, loc="upper right")
    a.set_ylim(0, 52)
    style(a)
    for q in pols:
        s = col[col.policy == q]
        summ["collections"][q] = {"n": int(len(s)), "success": float(s.success.mean()),
                                  **{c: int((s.cls == c).sum()) for c in CLASSES},
                                  "first_pot1_share": float((s.first_pot == 0).mean()),
                                  "first_step_median_success": float(s[s.success == 1].first_step.median()),
                                  "first_step_median_fail": float(s[s.success == 0].first_step.median()),
                                  "gap_median_success": float(s[s.success == 1].gap.median())}
    er = ["r2", "r3", "r4", "r5", "r6", "r7"]  # evals that record the end state
    # collections: the left pot (moka_pot_2) is placed first in ~95% of episodes, so 011 = second pot missing,
    # 101 = the first pot was lost while the second went on
    ends = {"011": ("first pot on, second missing", C["teal"]), "101": ("second pot on, first lost", C["red"]),
            "001": ("neither pot on", C["grey"])}
    bottom = np.zeros(len(er))
    for k, (lab, colr) in ends.items():
        v = np.array([((ev[r].success == 0) & (ev[r].goals_end == k)).sum() for r in er])
        b.bar(range(len(er)), v, bottom=bottom, color=colr, label=lab, width=0.7)
        bottom += v
        for r, x in zip(er, v):
            summ["evals"][r][f"fail_{k}"] = int(x)
    b.set_xticks(range(len(er)), er)
    b.set_ylabel("failed trials (of 330)")
    b.set_title("Held-out eval failures: end state")
    b.set_ylim(0, 115)
    b.legend(fontsize=7.5, frameon=False, loc="upper right")
    style(b)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_failures.png")
    plt.close(fig)

    # ---- fig 5: per val state x end state, rounds r2-r7 pooled
    fig, a = plt.subplots(figsize=(8, 3.6))
    pool = pd.concat([ev[r] for r in er])
    bottom = np.zeros(len(vs))
    sorder = [vs[i] for i in np.argsort([-(1 - pool[pool.init_state == s].success.mean()) for s in vs])]
    for k, (lab, colr) in ends.items():
        v = np.array([100 * ((pool.init_state == s) & (pool.success == 0) & (pool.goals_end == k)).sum()
                      / (pool.init_state == s).sum() for s in sorder])
        a.bar(range(len(vs)), v, bottom=bottom, color=colr, label=lab, width=0.7)
        bottom += v
    a.set_xticks(range(len(vs)), [f"val {s}" for s in sorder])
    a.set_ylabel("failure rate (%)")
    a.set_title("Where held-out failures happen (rounds 2-7 pooled, 198 trials per state)")
    a.legend(fontsize=8, frameon=False)
    style(a)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_state_failures.png")
    plt.close(fig)
    summ["val_state_end"] = {f"{s}": {k: int(((pool.init_state == s) & (pool.success == 0) & (pool.goals_end == k)).sum())
                                      for k in ends} for s in vs}

    # ---- fig 6: timing in collections: first pot step (success vs failure), second-pot gap (successes)
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.8))
    for i, q in enumerate(pols):
        s = col[col.policy == q]
        for off, (m, colr) in zip((-0.18, 0.18), ((1, C["blue"]), (0, C["red"]))):
            d = s[(s.success == m) & s.first_step.notna()].first_step
            if len(d):
                bp = a.boxplot(d, positions=[i + off], widths=0.3, patch_artist=True, showfliers=False)
                for el in bp["boxes"]:
                    el.set(facecolor=colr, alpha=0.35, edgecolor=colr)
                for el in bp["medians"]:
                    el.set(color=colr)
        g = s[s.success == 1].gap.dropna()
        bp = b.boxplot(g, positions=[i], widths=0.5, patch_artist=True, showfliers=False)
        for el in bp["boxes"]:
            el.set(facecolor=C["blue"], alpha=0.35, edgecolor=C["blue"])
        for el in bp["medians"]:
            el.set(color=C["blue"])
    a.plot([], [], "s", color=C["blue"], alpha=0.5, label="successes")
    a.plot([], [], "s", color=C["red"], alpha=0.5, label="failures")
    a.legend(frameon=False, fontsize=8.5)
    a.set_xticks(range(len(pols)), pols)
    a.set_ylabel("step the first pot is placed")
    a.set_title("First pot placement time")
    b.set_xticks(range(len(pols)), pols)
    b.set_ylabel("steps from first to second pot")
    b.set_title("Second pot time (successes)")
    for ax in (a, b):
        style(ax)
        ax.set_xlabel("policy that collected")
    fig.tight_layout()
    fig.savefig(OUT / "fig6_timing.png")
    plt.close(fig)

    (OUT / "summary.json").write_text(json.dumps(summ, indent=1, default=float))
    print(json.dumps({k: summ[k] for k in ("corr_w_success", "persistence", "train_states")}, indent=1, default=float))
    print(json.dumps(summ["collections"], indent=1, default=float))
    print(json.dumps({r: {k: v for k, v in summ["evals"][r].items() if k.startswith("fail")} for r in er}))


if __name__ == "__main__":
    main()
