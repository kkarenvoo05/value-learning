"""Action-imagination gate G1 with DSRL's critic [2026-11-13; Karen: "set up the G1 test with DSRL's critic"].

Question: does DSRL's noise-space critic Q(s, w) rank candidate chunks *within a state*? It is the one critic trained
on actions it varied itself (online SAC, scripts/dsrl_pi05.py), and it scores pi0.5 samples directly: each w is a
start noise that frozen pi0.5 turns into a chunk. G1 with offline critics on base candidates failed
(reports/2026-11-12-iql-g1: Spearman -0.07..+0.11, ceiling 0.37). Same gate, same statistics, new critic.

    <openpi>/.venv/bin/python scripts/libero_dsrl_g1.py --probes <run dir> [<run dir> ...] --out reports/2026-11-13-dsrl-g1

Probes: scripts/libero_steer_probe.py --args.cand-source actor|gauss against a DSRL eval server
(libero_steer_probe.sbatch DSRL_AGENT=...): on the 10 held-out val init states, the trunk episode is the DSRL
policy; at each branch step N candidate noise vectors (actor samples, or N(0, I) clipped to [-1, 1]) are turned into
chunks by pi0.5, each candidate's first 5 actions run from the saved MuJoCo state, then the DSRL policy continues
R times -> p_c = success share. DSRL never trained on val states.

Scores per candidate, from the observation the server saw (agentview 224 -> 64 x 64 bilinear, 8-d state;
dsrl_pi05.sac_obs) and the w it used:
  q:<agent>          mean over the 10 Q heads (what SAC's actor maximises) for each agent checkpoint in --agents
  qmin:<agent>       min over heads [ours, diagnostic: pessimistic ensemble]
  logp:<agent>       [ours, diagnostic] log-density of w under that agent's actor (does the policy itself prefer the
                     better candidate?); w clipped to +-(1 - 1e-4) before atanh (gauss candidates sit on the bound)
  random             the null
Check: q:<served agent> must equal the Q the server returned with each candidate (br_q) -> same image, state and w.
Statistics = scripts/libero_iql_g1.py (within-state Spearman over points where p varies, best-of-N pick gain,
bootstrap over the trunk episodes, ceiling from the probe's between-candidate variance with R continuations).
Gate [plan, reports/2026-11-11-advantage-guidance/report.md §5]: within-state rank correlation CI above 0, read next
to the ceiling; reported per probe (actor / gauss), not pooled.
Writes <out>/{table.md, summary.json, scores.parquet}.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dsrl_pi05 import sac_obs  # noqa: E402
from libero_iql_g1 import boot_mean, metrics  # noqa: E402
from vl.dsrl import DSRLAgent  # noqa: E402

RUNS = Path("/iris/u/ankile/karenvo/libero/runs")
DSRL_S0 = RUNS / "dsrl_s0_libero_10_18434524"
AGENTS = {"final": DSRL_S0 / "agent_final_ep1800.pkl",  # 0.861 held-out (18452292); 1357 own episodes
          "ep1000": DSRL_S0 / "agent_1000.pkl", "ep500": DSRL_S0 / "agent_500.pkl", "ep100": DSRL_S0 / "agent_100.pkl"}


def actor_logp(agent: DSRLAgent, pix, st, w, eps=1e-4) -> np.ndarray:
    """log pi(w | s) of the tanh-Gaussian actor (as vl.dsrl.tanh_sample), magnitude 1."""
    mean, log_std = agent.actor.apply_fn({"params": agent.actor.params}, pix, st)
    mean, log_std = np.asarray(mean, np.float64), np.asarray(log_std, np.float64)
    m = agent.cfg.action_magnitude
    u = np.arctanh(np.clip(w / m, -1 + eps, 1 - eps))
    logp_u = -0.5 * (((u - mean) / np.exp(log_std)) ** 2 + 2 * log_std + np.log(2 * np.pi))
    log_det = np.log(np.clip(1 - np.tanh(u) ** 2, 1e-12, None))
    return (logp_u - log_det).sum(-1) - w.shape[-1] * np.log(m)


def load_points(probes: list[Path]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Rows = (probe, episode, point, candidate) with p; pixels (64 x 64), state, w, served Q."""
    rows, P, S, W, Qs = [], [], [], [], []
    for run in probes:
        eps = pd.read_csv(run / "episodes.csv")
        for r in eps.itertuples():
            z = np.load(run / "probe" / r.file)
            src = str(z["cand_source"])
            assert src in ("actor", "gauss"), f"{run}: not a DSRL probe (cand_source {src!r})"
            rp = list(z["rp_t"])
            for i, t in enumerate(z["br_t"]):
                j = rp.index(int(t))
                pix, st = sac_obs({"observation/image": z["rp_image"][j], "observation/state": z["rp_state"][j]}, 64)
                succ = z["br_success"][i]  # (N, R)
                for c in range(succ.shape[0]):
                    P.append(pix), S.append(st), W.append(z["br_w"][i, c]), Qs.append(float(z["br_q"][i, c]))
                    rows.append(dict(probe=src, run=run.name, ep=f"{src}:{r.episode}", scale=src,
                                     init_state=r.init_state, t=int(t), point=f"{src}:{r.episode}:{t}", cand=c,
                                     p=float(succ[c].mean()), R=int(succ.shape[1])))
    return pd.DataFrame(rows), np.stack(P), np.stack(S).astype(np.float32), np.stack(W).astype(np.float32), np.array(Qs)


def ci(per: pd.DataFrame, c: str) -> list:
    """libero_iql_g1.boot_mean, NaN when no point has a finite value (e.g. p never varies)."""
    return boot_mean(per, c) if per[c].notna().any() else [float("nan"), float("nan")]


def ceiling(P: pd.DataFrame) -> dict:
    """As libero_iql_g1: Var(p_hat) = p(1-p)/R, unbiasedly estimated by p_hat(1-p_hat)/(R-1)."""
    R = int(P.R.iloc[0])
    assert (P.R == R).all() and R >= 2
    g = P.groupby("point").p
    var_obs = g.var(ddof=1)
    noise = P.assign(b=P.p * (1 - P.p)).groupby("point").b.mean() / (R - 1)
    var_true = var_obs - noise
    c = float(np.sqrt(max(var_true.mean(), 0) / var_obs.mean())) if var_obs.mean() > 0 else float("nan")
    return {"corr": c, "var_between": float(var_true.mean()), "var_observed": float(var_obs.mean()),
            "points": int(P.point.nunique()), "points_p_varies": int((var_obs > 0).sum()), "R": R,
            "p_mean": float(P.p.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes", nargs="+", required=True, help="DSRL probe run dirs (libero_steer_probe output)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--agents", nargs="+", default=list(AGENTS), help=f"names from {list(AGENTS)} or name=path")
    ap.add_argument("--served", default="final", help="agent the probe server ran (checked against br_q)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    agents = dict(x.split("=", 1) if "=" in x else (x, AGENTS[x]) for x in a.agents)
    P, pix, st, w, served_q = load_points([Path(p) for p in a.probes])
    print(f"{P.point.nunique()} points, {len(P)} candidates, probes {sorted(P.probe.unique())}, "
          f"val states {sorted(P.init_state.unique())}")
    for name, path in agents.items():
        ag = DSRLAgent.load(path)
        qs = np.concatenate([ag.q(pix[s:s + 512], st[s:s + 512], w[s:s + 512]) for s in range(0, len(P), 512)], 1)
        P[f"q:{name}"], P[f"qmin:{name}"] = qs.mean(0), qs.min(0)
        P[f"logp:{name}"] = np.concatenate([actor_logp(ag, pix[s:s + 512], st[s:s + 512], w[s:s + 512])
                                            for s in range(0, len(P), 512)])
        if name == a.served:
            d = float(np.abs(P[f"q:{name}"] - served_q).max())
            print(f"check: offline q:{name} vs served Q, max |diff| = {d:.2e}")
            assert d < 1e-3, "offline Q does not reproduce the server's Q (image / state / w mismatch?)"
    P["random"] = np.random.default_rng(0).random(len(P))
    P.to_parquet(out / "scores.parquet")

    cols = [c for c in P.columns if c.split(":")[0] in ("q", "qmin", "logp")] + ["random"]
    res = {"args": vars(a), "agents": {k: str(v) for k, v in agents.items()}, "probes": {}}
    lines = [f"G1 with DSRL's critic: within-state ranking of candidate noise vectors at held-out branch points "
             f"(val states), DSRL-policy trunk and continuations", ""]
    for probe, Pp in P.groupby("probe"):
        ce = ceiling(Pp)
        n_c = Pp.groupby("point").size().iloc[0]
        lines += [f"## candidates: {probe} ({Pp.run.iloc[0]})",
                  f"{ce['points']} points x {n_c} candidates x {ce['R']} continuations; mean candidate success "
                  f"{ce['p_mean']:.2f}; p varies at {ce['points_p_varies']} points; ceiling corr ~ {ce['corr']:.2f} "
                  f"(var_between {ce['var_between']:.4f} / var_observed {ce['var_observed']:.4f})", "",
                  "| score | within-state Spearman | best-of-N pick gain | Q spread within state / across states |",
                  "|---|---|---|---|"]
        across = Pp.groupby("point")
        arms = []
        for c in cols:
            per = metrics(Pp, c)
            r = {"score": c, "rho": float(per.rho.mean()), "rho_ci95": ci(per, "rho"),
                 "pick": float(per.pick.mean()), "pick_ci95": ci(per, "pick"),
                 "points_rho": int(per.rho.notna().sum()),
                 "spread_ratio": float(per.qsd.mean() / across[c].mean().std()) if c != "random" else float("nan")}
            arms.append(r)
            lines.append(f"| {c} | {r['rho']:+.3f} [{r['rho_ci95'][0]:+.3f}, {r['rho_ci95'][1]:+.3f}] | "
                         f"{r['pick']:+.3f} [{r['pick_ci95'][0]:+.3f}, {r['pick_ci95'][1]:+.3f}] | "
                         f"{r['spread_ratio']:.2f} |")
        g = next((r for r in arms if r["score"] == f"q:{a.served}"), None)
        verdict = ("n/a" if g is None or not np.isfinite(g["rho_ci95"][0])  # no point where p varies
                   else "PASS" if g["rho_ci95"][0] > 0 else "not passed")
        lines += ["", f"gate (q:{a.served} within-state Spearman CI above 0): **{verdict}**", ""]
        res["probes"][probe] = {"ceiling": ce, "arms": arms, "gate": verdict}
    (out / "summary.json").write_text(json.dumps(res, indent=2))
    (out / "table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
