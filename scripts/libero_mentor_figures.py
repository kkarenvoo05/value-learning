"""Figures for the mentor update (2026-11-14; Karen: "make 1-3 and 5").

    .venv/bin/python scripts/libero_mentor_figures.py --out reports/2026-11-14-figures

1. success vs rollout episodes collected: AG rounds (seed 0, round 0 seed 1 hollow) and DSRL s0 checkpoints (seed 1 at 1300 and 1800 hollow), 330 val
   trials each, Wilson 95% intervals (trial noise only; seeds have differed by ~8 points).
2. success per val start state (10 states x 33 trials) for base, AG rounds 0-5, DSRL checkpoints.
3. failures by simulator end state (one pot on / no pot on, 520-step timeouts) per arm.
5. what the robot does after the first pot is on, in TRAIN-state rollouts with raw state (not the val states, so the
   diagnosis does not tune to the eval set): base pi0.5 (moka_fixed_fresh round 0), AG (round collections with the
   round-4 and round-5 policies), DSRL s0 final agent (moka_distill_dsrl). Per one-pot failure, every step after the
   first pot lands is: second pot lifted (> 3 cm above its resting height) / gripper commanded closed / hovering (open,
   end effector within D of the second pot; D from successful grasps) / elsewhere; knock-offs (the placed pot's goal
   predicate turning false) are marked.
Writes <out>/{fig1_curve,fig2_states,fig3_failures,fig5_after_first_pot}.png, <out>/fig5_episodes.csv,
<out>/summary.json.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

RUNS = Path("/iris/u/ankile/karenvo/libero/runs")
TABLES = Path("/iris/u/ankile/karenvo/libero/tables")
VAL = [9, 12, 13, 17, 25, 32, 40, 42, 44, 48]

# reference palette (dataviz skill, light mode): categorical slots 1-3, blue sequential ramp, text tokens
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
       "#1c5cab", "#184f95", "#104281", "#0d366b"]

AG = {0: "eval_ag_residual_vb40_s0_shift099_noise1_libero_10_18438187",
      1: "eval_agr_r1_s0_shift099_noise1_libero_10_18460307", 2: "eval_agr_r2_s0_shift099_noise1_libero_10_18460313",
      3: "eval_agr_r3_s0_shift099_noise1_libero_10_18460319", 4: "eval_agr_r4_s0_shift099_noise1_libero_10_18463162",
      5: "eval_agr_r5_s0_shift099_noise1_libero_10_18484607"}
AG0_S1 = "eval_ag_residual_vb40_s1_shift099_noise1_libero_10_18477722"
DSRL = {100: "eval_dsrl_s0_agent100_libero_10_18504406", 200: "eval_dsrl_s0_agent200_libero_10_18504407",
        400: "eval_dsrl_s0_agent400_libero_10_18504408", 600: "eval_dsrl_s0_agent600_libero_10_18504409",
        800: "eval_dsrl_s0_agent800_libero_10_18504410", 900: "eval_dsrl_s0_agent900_libero_10_18500663",
        971: "eval_dsrl_s0_ep971_libero_10_18445300", 1000: "eval_dsrl_s0_agent1000_libero_10_18488117",
        1100: "eval_dsrl_s0_agent1100_libero_10_18500678", 1200: "eval_dsrl_s0_agent1200_libero_10_18497492",
        1300: "eval_dsrl_s0_agent1300_libero_10_18500794", 1800: "eval_dsrl_s0_ep1800_libero_10_18452292"}
DSRL_S1 = {1300: "eval_dsrl_s1_agent1300_libero_10_18509147", 1800: "eval_dsrl_s1_agent1800_libero_10_18509148"}  # [2026-11-14]
BASE = "eval_base_noise1_libero_10_18355478"


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.yaxis.label.set_color(INK2), ax.xaxis.label.set_color(INK2)
    ax.set_facecolor(SURF)


def load_eval(name):
    e = pd.read_csv(RUNS / name / "episodes.csv", dtype={"goals_end": str})
    assert len(e) == 330 and sorted(e.init_state.unique()) == VAL, name
    e["pots_end"] = e.goals_end.map(lambda s: s[:2].count("1") if isinstance(s, str) and s else np.nan)
    return e


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def fig1(out, E):
    fig, ax = plt.subplots(figsize=(8, 4.6), facecolor=SURF)
    style(ax)
    ax.grid(axis="y", color=GRID, lw=0.8), ax.set_axisbelow(True)

    def series(keys, xs, color, label, marker="o", hollow=False):
        ys = np.array([E[k].success.mean() for k in keys])
        lo, hi = np.array([wilson(E[k].success.sum(), 330) for k in keys]).T
        ax.errorbar(xs, ys, yerr=[ys - lo, hi - ys], color=color, lw=0 if hollow else 2, marker=marker, ms=8,
                    mfc=SURF if hollow else color, mec=SURF if not hollow else color, mew=2, capsize=0,
                    elinewidth=1.2, label=label, zorder=3)
        return ys

    b = E["base"].success.mean()
    ax.axhline(b, color=INK2, lw=1, ls=(0, (4, 3)))
    ax.text(1890, b - 0.022, f"base π0.5  {b:.1%}", color=INK2, fontsize=9, ha="right")
    ag = series([f"ag{r}" for r in AG], [900 + 100 * r for r in AG], BLUE, "AG rounds (seed 0)")
    series(["ag0_s1"], [900], BLUE, "AG round 0, seed 1", hollow=True)
    ds = series([f"dsrl{n}" for n in DSRL], list(DSRL), ORANGE, "DSRL (seed 0)")
    ds1 = series([f"dsrl_s1_{n}" for n in DSRL_S1], list(DSRL_S1), ORANGE, "DSRL, seed 1", hollow=True)
    for r, y in zip(AG, ag):
        ax.annotate(f"r{r}", (900 + 100 * r, y), xytext=(-9, -14), textcoords="offset points", ha="center",
                    fontsize=8, color=BLUE)
    ax.annotate(f"{ag[-1]:.1%}", (1400, ag[-1]), xytext=(10, 0), textcoords="offset points", va="center",
                fontsize=9, color=INK)
    for y in (ds[-1], ds1[-1]):
        ax.annotate(f"{y:.1%}", (1800, y), xytext=(10, 0), textcoords="offset points", va="center",
                    fontsize=9, color=INK)
    ax.set_xlim(-30, 1960), ax.set_ylim(0.4, 0.95)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
    ax.set_xlabel("rollout episodes collected (AG: 900 base rollouts + 100 per round; AG also uses the demos)")
    ax.set_ylabel("success, 330 held-out trials")
    ax.set_title("Success vs. data collected", loc="left", color=INK, fontsize=12)
    ax.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=INK)
    fig.text(0.01, 0.005, "Bars: 95% Wilson intervals over trials only; separate seeds have differed by ~8 points.",
             fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1)), fig.savefig(out / "fig1_curve.png", dpi=160), plt.close(fig)


ARMS = [("base", "base π0.5")] + [(f"ag{r}", f"AG round {r}") for r in AG] + \
       [(f"dsrl{n}", f"DSRL {n}") for n in (400, 800, 1000, 1300, 1800)]


def fig2(out, E):
    M = np.array([[E[k][E[k].init_state == s].success.mean() for s in VAL] for k, _ in ARMS])
    fig, ax = plt.subplots(figsize=(8, 5.4), facecolor=SURF)
    cmap = ListedColormap(SEQ)
    im = ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j] * 100:.0f}", ha="center", va="center", fontsize=8,
                    color="white" if M[i, j] > 0.45 else INK)
    for i in (0.5, 6.5):
        ax.axhline(i, color=SURF, lw=3)
    ax.set_xticks(range(len(VAL)), [str(s) for s in VAL]), ax.set_yticks(range(len(ARMS)), [n for _, n in ARMS])
    ax.tick_params(length=0, colors=INK2, labelsize=9)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("held-out start state", color=INK2)
    ax.set_title("Success (%) per start state, 33 trials each", loc="left", color=INK, fontsize=12)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, format=matplotlib.ticker.PercentFormatter(1, decimals=0))
    cb.outline.set_visible(False), cb.ax.tick_params(colors=INK2, labelsize=8)
    fig.tight_layout(), fig.savefig(out / "fig2_states.png", dpi=160), plt.close(fig)
    return {n: dict(zip(map(str, VAL), np.round(M[i], 3).tolist())) for i, (_, n) in enumerate(ARMS)}


def fig3(out, E):
    arms = [(k, n) for k, n in ARMS if E[k].pots_end.notna().all()]  # evals before goals_end was saved are left out
    one = np.array([((E[k].success == 0) & (E[k].pots_end == 1)).sum() for k, _ in arms])
    zero = np.array([((E[k].success == 0) & (E[k].pots_end == 0)).sum() for k, _ in arms])
    fig, ax = plt.subplots(figsize=(8, 4.8), facecolor=SURF)
    style(ax)
    ax.grid(axis="x", color=GRID, lw=0.8), ax.set_axisbelow(True)
    y = np.arange(len(arms))[::-1]
    ax.barh(y, one, color=BLUE, height=0.62, label="one pot on the stove", edgecolor=SURF, lw=1)
    ax.barh(y, zero, left=one, color=ORANGE, height=0.62, label="no pot on the stove", edgecolor=SURF, lw=1)
    for yi, a, b in zip(y, one, zero):
        ax.text(a + b + 2, yi, f"{a} + {b}", va="center", fontsize=8, color=INK2)
    ax.set_yticks(y, [n for _, n in arms]), ax.set_xlabel("failed trials out of 330 (all 520-step timeouts)")
    miss = [n for k, n in ARMS if (k, n) not in arms]
    fig.text(0.01, 0.005, f"Not shown (end state not recorded in those evals): {', '.join(miss)}.", fontsize=8,
             color=INK2)
    ax.set_title("How episodes end when they fail", loc="left", color=INK, fontsize=12)
    ax.set_xlim(0, max(one + zero) * 1.35)
    ax.legend(frameon=False, fontsize=9, loc="center right", labelcolor=INK)
    fig.tight_layout(rect=(0, 0.03, 1, 1)), fig.savefig(out / "fig3_failures.png", dpi=160), plt.close(fig)
    return {n: {"one_pot": int(a), "no_pot": int(b)} for (_, n), a, b in zip(arms, one, zero)}


# ---------------------------------------------------------------- fig 5: train-state rollouts with raw state
LIFT = 0.03
SOURCES = [("base π0.5", "moka_agr_r6", [1], 400), ("AG (rounds 4-5)", "moka_agr_r6", [6, 7], None),
           ("DSRL (1.8k agent)", "moka_distill_dsrl", [2], None)]


def episodes(table, rounds, cap):
    e = pd.read_csv(TABLES / table / "episodes.csv", dtype={"goals_end": str})
    e = e[e["round"].isin(rounds) & (e.split == "train")]
    if cap and len(e) > cap:
        e = e.sample(cap, random_state=0)
    return e


def trace(raw):
    with np.load(raw) as z:
        g, s, a, op = z["goals"][:, :2].astype(bool), z["state"], z["actions"], z["object_pos"]
    pz = op[:, [2, 8]]                                   # moka_pot_1 / moka_pot_2 z
    dist = np.linalg.norm(op[:, [3, 4, 5]], axis=1), np.linalg.norm(op[:, [9, 10, 11]], axis=1)
    return g, a[:, 6] > 0, pz, np.stack(dist, 1), s[:, 2]


KNOCK_RUN = 15   # the placed pot counts as knocked off when off the stove >= 15 steps in a row, starting >= 10 steps
SETTLE = 10      # after it first lands (the goal predicate flickers while the pot settles)
DRIFT = 50      # the placed pot leaving the cook region within 50 steps of landing = drifted off after release
CATS = [("drift", "first pot drifted off the cooking area soon after release", AQUA),
        ("knock", "first pot knocked off later, while handling the second", "#eda100"),
        ("never_grasped", "never closed the gripper at the second pot", ORANGE),
        ("missed_grasp", "closed at the second pot but never lifted it", BLUE),
        ("not_placed", "lifted the second pot but never placed it", "#0d366b")]


def first_run(mask, start, n):
    run = 0
    for t in range(start, len(mask)):
        run = run + 1 if mask[t] else 0
        if run >= n:
            return t - n + 1
    return None


def analyse(e, D=None):
    rows, grids = [], []
    for r in e.itertuples():
        g, closed, pz, dist, _ = trace(r.raw)
        on = g.any(1)
        if not on.any():
            rows.append({"episode_index": r.episode_index, "success": r.success, "first_pot": None}); continue
        t1 = int(np.argmax(on)); k1 = int(np.argmax(g[t1])); k2 = 1 - k1
        rest = pz[t1, k2]
        lifted = pz[:, k2] > rest + LIFT
        lifted[:t1] = False
        lift_t = int(np.argmax(lifted)) if lifted.any() else None
        kt = first_run(~g[:, k1], t1 + SETTLE, KNOCK_RUN)
        rec = {"episode_index": r.episode_index, "success": int(r.success), "first_pot": t1,
               "steps_after": len(g) - t1, "second_lifted": lift_t is not None,
               "dist_at_lift": float(dist[lift_t, k2]) if lift_t is not None else None,
               "knocked_first_off": kt is not None, "pot_max_z_carried": float(pz[t1:, k2].max() - rest),
               "end_first_on": bool(g[-1, k1]), "end_second_on": bool(g[-1, k2])}
        if D is not None:
            near_closed = closed & (dist[:, k2] < D)
            rec["ever_closed_near"] = bool(near_closed[t1:].any())
            st = np.where(lifted, 3, np.where(closed, 2, np.where(dist[:, k2] < D, 1, 0)))[t1:]
            rec["hover_frac_after"] = float((st == 1).mean())
            rec["first_off_after"] = None if kt is None else kt - t1
            if not r.success and (rec["end_first_on"] or rec["end_second_on"]):
                rec["category"] = (("drift" if kt - t1 < DRIFT else "knock") if kt is not None and not
                                   rec["end_first_on"] else "not_placed" if rec["second_lifted"] else
                                   "missed_grasp" if rec["ever_closed_near"] else "never_grasped")
            grids.append((r.success, st, None if kt is None else kt - t1, rec.get("category"),
                          (lift_t or 10**6) - t1))
        rows.append(rec)
    return pd.DataFrame(rows), grids


def fig5(out):
    data = {name: episodes(t, rd, cap) for name, t, rd, cap in SOURCES}
    first = pd.concat([analyse(e)[0].assign(src=n) for n, e in data.items()])
    D = float(first[(first.success == 1) & first.dist_at_lift.notna()].dist_at_lift.quantile(0.9) * 1.25)
    res, G = {}, {}
    for n, e in data.items():
        res[n], G[n] = analyse(e, D)
        res[n]["src"] = n
    df = pd.concat(res.values(), ignore_index=True)
    df.to_csv(out / "fig5_episodes.csv", index=False)

    cmap = ListedColormap(["#efeeea", ORANGE, BLUE, "#0d366b"])
    order = {c: i for i, (c, _, _) in enumerate(CATS)}
    fig = plt.figure(figsize=(11, 8.2), facecolor=SURF)
    gs = fig.add_gridspec(2, 3, height_ratios=[3.2, 1.1], hspace=0.5, wspace=0.12)
    summ = {}
    for j, (n, grids) in enumerate(G.items()):
        f = sorted([x for x in grids if x[0] == 0 and x[3] is not None], key=lambda x: (order[x[3]], x[4]))
        d = df[df.src == n]
        one = d[(d.success == 0) & d.first_pot.notna() & (d.end_first_on | d.end_second_on)]
        s = d[(d.success == 1) & d.first_pot.notna()]
        L = 340
        M = np.full((max(len(f), 1), L), np.nan)
        for i, x in enumerate(f):
            M[i, :min(L, len(x[1]))] = x[1][:L]
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(np.ma.masked_invalid(M), cmap=cmap, vmin=-0.5, vmax=3.5, aspect="auto", interpolation="nearest")
        for i, x in enumerate(f):
            if x[2] is not None and x[2] < L:
                ax.plot(x[2], i, marker="x", color=INK, ms=4.5, mew=1.3)
            ax.add_patch(matplotlib.patches.Rectangle((-14, i - 0.5), 9, 1, color=dict(
                (c, col) for c, _, col in CATS)[x[3]], clip_on=False))
        ax.set_xlim(-15, L)
        ax.set_facecolor(SURF)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(colors=INK2, labelsize=8, length=0)
        ax.set_yticks([])
        ax.set_title(f"{n}\n{len(d)} episodes, {d.success.mean():.0%} success, {len(f)} rows", loc="left",
                     fontsize=10, color=INK)
        ax.set_xlabel("steps after the first pot lands", color=INK2, fontsize=9)
        cnt = {c: int((one.category == c).sum()) for c, _, _ in CATS}
        summ[n] = {"episodes": int(len(d)), "success": float(d.success.mean()), "failures_with_a_pot": int(len(one)),
                   "failures_no_pot_ever": int(((d.success == 0) & d.first_pot.isna()).sum()),
                   "failure_category_counts": cnt,
                   "fail_knocked_first_off_ever": float(one.knocked_first_off.mean()) if len(one) else None,
                   "succ_knocked_first_off_ever": float(s.knocked_first_off.mean()),
                   "fail_hover_frac_mean": float(one.hover_frac_after.mean()) if len(one) else None,
                   "succ_hover_frac_mean": float(s.hover_frac_after.mean()),
                   "succ_pot_carried_height_p10_m": float(s.pot_max_z_carried.quantile(0.1)),
                   "fail_lifted_pot_carried_height_median_m": float(
                       one[one.second_lifted.astype(bool)].pot_max_z_carried.median())
                   if one.second_lifted.astype(bool).any() else None}
    axb = fig.add_subplot(gs[1, :])
    style(axb)
    names = list(G)
    y = np.arange(len(names))[::-1]
    for yi, n in zip(y, names):
        c = summ[n]["failure_category_counts"]
        tot = sum(c.values())
        left = 0
        for key, _, col in CATS:
            w = c[key] / tot if tot else 0
            axb.barh(yi, w, left=left, color=col, height=0.62, edgecolor=SURF, lw=1.5)
            if w >= 0.08:
                axb.text(left + w / 2, yi, f"{c[key]}", ha="center", va="center", fontsize=8,
                         color=INK if col == AQUA else "white")
            left += w
    axb.set_yticks(y, names), axb.set_xlim(0, 1)
    axb.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
    axb.set_title("How the failures that end with one pot on the stove end (share; numbers are episodes)",
                  loc="left", fontsize=10, color=INK)
    handles = [matplotlib.patches.Patch(color=c, label=l) for c, l in
               ((ORANGE, f"hovering: gripper open, hand within {D * 100:.0f} cm of the second pot"),
                (BLUE, "gripper closed"), ("#0d366b", "second pot lifted"), ("#efeeea", "elsewhere, gripper open"))]
    handles.append(matplotlib.lines.Line2D([], [], marker="x", color=INK, lw=0,
                                           label=f"first pot leaves the cooking area (>= {KNOCK_RUN} steps)"))
    h2 = [matplotlib.patches.Patch(color=c, label=l) for _, l, c in CATS]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.01, 0.395), ncol=3, frameon=False, fontsize=8.5,
               labelcolor=INK, title="Rows: each step after the first pot lands", title_fontsize=9,
               alignment="left")
    fig.legend(handles=h2, loc="lower left", bbox_to_anchor=(0.01, 0.0), ncol=3, frameon=False, fontsize=8.5,
               labelcolor=INK, title="Left edge of each row and bottom bars: how the failure ends", title_fontsize=9,
               alignment="left")
    fig.suptitle("What the robot does after the first pot is on (training-state rollouts; rows = failures ending "
                 "with one pot on the stove)", x=0.01, ha="left", color=INK, fontsize=12)
    fig.subplots_adjust(left=0.135, right=0.99, top=0.9, bottom=0.1)
    fig.savefig(out / "fig5_after_first_pot.png", dpi=160), plt.close(fig)
    return {"hover_distance_m": D, "lift_threshold_m": LIFT, "knock_run_steps": KNOCK_RUN, "per_source": summ}


def fig5b(out, csv):
    """One example per drift / knock: agentview frames at landing, when the first pot leaves the region, +40, end."""
    df = pd.read_csv(csv)
    picks = []
    for src, cat in (("AG (rounds 4-5)", "drift"), ("DSRL (1.8k agent)", "drift"), ("DSRL (1.8k agent)", "knock")):
        c = df[(df.src == src) & (df.category == cat)].sort_values("episode_index")
        if len(c):
            picks.append((src, cat, c.iloc[0]))
    raws = {}
    for name, t, rd, cap in SOURCES:
        e = episodes(t, rd, cap)
        raws[name] = dict(zip(e.episode_index, e.raw))
    fig, axes = plt.subplots(len(picks), 4, figsize=(9, 2.5 * len(picks)), facecolor=SURF)
    for i, (src, cat, r) in enumerate(picks):
        with np.load(raws[src][r.episode_index]) as z:
            img = z["image"]
        t1, off = int(r.first_pot), int(r.first_pot + r.first_off_after)
        for j, (t, lab) in enumerate(((t1, "first pot lands"), (off, "leaves the cooking area"),
                                      (min(off + 40, len(img) - 1), "+40 steps"), (len(img) - 1, "end (timeout)"))):
            ax = axes[i, j]
            ax.imshow(img[t][::-1, ::-1]), ax.set_xticks([]), ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.set_title(f"{lab}, step {t}", fontsize=8, color=INK2)
        axes[i, 0].set_ylabel(f"{src.split(' (')[0]}: first pot\n{'drifted off' if cat == 'drift' else 'knocked off'}",
                              fontsize=9, color=INK)
    fig.suptitle("Examples (training-state rollouts, agentview camera)", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(), fig.savefig(out / "fig5b_examples.png", dpi=140), plt.close(fig)
    return [{"source": s_, "category": c_, "episode_index": int(r_.episode_index)} for s_, c_, r_ in picks]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip5", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    E = {"base": load_eval(BASE), "ag0_s1": load_eval(AG0_S1)}
    E.update({f"ag{r}": load_eval(n) for r, n in AG.items()})
    E.update({f"dsrl{k}": load_eval(n) for k, n in DSRL.items()})
    E.update({f"dsrl_s1_{k}": load_eval(n) for k, n in DSRL_S1.items()})
    S = {"success": {k: round(float(e.success.mean()), 4) for k, e in E.items()}}
    fig1(out, E)
    S["per_state"] = fig2(out, E)
    S["failures"] = fig3(out, E)
    if a.skip5 and (out / "summary.json").exists():  # keep the fig5 numbers of the last full run
        S.update({k: v for k, v in json.loads((out / "summary.json").read_text()).items() if k.startswith("fig5")})
    if not a.skip5:
        S["fig5"] = fig5(out)
        S["fig5b"] = fig5b(out, out / "fig5_episodes.csv")
    (out / "summary.json").write_text(json.dumps(S, indent=2) + "\n")
    print(json.dumps(S.get("fig5", {}), indent=1))


if __name__ == "__main__":
    main()
