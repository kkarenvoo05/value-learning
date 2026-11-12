"""Gripper-level view of base vs guided rollouts: wrist camera + timeline videos and grasp metrics [2026-11-11].

Question (Karen, Nov 11): is guidance "more decisive" at the grasp? The 224 px agent view is too small to see the
gripper, so this reruns a sample of trials with --args.save-raw (scripts/libero_rollouts.py) and reads, per step:
the wrist image the policy saw, the executed gripper command (action[6]; -1 open, +1 close, as LIBERO's dummy action),
the finger opening (state[6] - state[7], metres), both pots' positions and distances to the gripper, and the BDDL
goal predicates (pot 1 on the stove, pot 2 on the stove).

Per episode metrics:
  closes        gripper close commands (open -> close transitions)
  grasps[p]     closes with the gripper within NEAR m of pot p (its nearest pot)
  lifts[p]      of those, how many lift pot p by > LIFT m within WIN steps (a grasp that worked)
  t_place[p]    first step pot p is on the stove (None if never)
  first/second  which pot was placed first; "second pot" metrics use the other one
  hover2        steps between the first placement and the first successful lift of the second pot
Video per episode: [agent view | wrist view] at 2x, timeline below (gripper command, finger opening, pot heights,
placements) with a moving cursor, at 20 fps (LIBERO's control rate) -> <out>/<label>/epNNNN.mp4
Usage: .venv/bin/python scripts/gripper_viz.py --runs base=<run> guided=<run> --out <dir> [--max-eps N]
"""
import argparse
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

NEAR, LIFT, WIN, GAP, MISS, TIP, STILL = 0.12, 0.05, 200, 8, 0.01, -0.005, 0.01
OUTCOME_COLOR = {"lifted": "#1f8a4c", "missed": "#d03b3b", "held_no_lift": "#c98a00"}
RUNS = Path("/iris/u/ankile/karenvo/libero/runs")


def pot_arrays(z):
    keys = [str(k) for k in z["object_pos_keys"]]
    op = z["object_pos"]
    get = lambda k: op[:, 3 * keys.index(k):3 * keys.index(k) + 3]
    pos = [get("moka_pot_1_pos"), get("moka_pot_2_pos")]
    dist = [np.linalg.norm(get("moka_pot_1_to_robot0_eef_pos"), axis=1), np.linalg.norm(get("moka_pot_2_to_robot0_eef_pos"), axis=1)]
    return pos, dist


def metrics(z):
    """Grasp attempts, from the executed gripper command and the pots' motion (thresholds checked on 4 collect
    episodes, Nov 11: the gripper closes 6-9 cm from a pot's origin when it takes the handle; a pot on the stove sits
    ~3 cm above its start, a tipped pot ends 0.8-2.8 cm below it (upright: > -0.05 cm); real lifts raise the pot 10-15 cm; a close on nothing shuts the fingers to < 1 cm).
      attempt  close-command onset after >= GAP steps of open command, gripper within NEAR of its nearest pot
      lifted   that pot rises > LIFT above its height at the close, within WIN steps
      missed   not lifted and the fingers shut below MISS within 15 steps (closed on air)
    The first pot is the one lifted first; the second pot is the other one."""
    a, s, g = z["actions"], z["state"], z["goals"]
    T = len(a)
    closed = a[:, 6] > 0
    opening = s[:, 6] - s[:, 7]
    pos, dist = pot_arrays(z)
    h = [p[:, 2] - p[0, 2] for p in pos]
    t_place = [int(np.argmax(g[:, i])) if g[:, i].any() else None for i in (0, 1)]
    onsets, last_closed = [], -10 ** 9
    for t in range(T):
        if closed[t]:
            if t - last_closed > GAP:
                onsets.append(t)
            last_closed = t
    att = []  # (t, pot, outcome)
    for i, t in enumerate(onsets):
        p = int(np.argmin([dist[0][t], dist[1][t]]))
        if dist[p][t] > NEAR:
            continue
        end = min(onsets[i + 1] if i + 1 < len(onsets) else T, t + WIN)  # judged until the next close onset
        if (h[p][t:end] - h[p][t]).max() > LIFT:
            out = "lifted"
        elif opening[t:t + 15].min() < MISS:
            out = "missed"
        else:
            out = "held_no_lift"
        att.append((t, p, out))
    lift_t = [next((t for t, p, o in att if p == q and o == "lifted"), None) for q in (0, 1)]  # close onset of the lift
    # step the pot actually leaves its support (> 1 cm above its height at that close): base often closes on the
    # handle and then holds still for 100-200 steps before lifting (base state 48 r0, state 12 r0)
    rise_t = [None if lift_t[q] is None else int(lift_t[q] + np.argmax(h[q][lift_t[q]:] - h[q][lift_t[q]] > 0.01))
              for q in (0, 1)]
    lifted = [q for q in (0, 1) if lift_t[q] is not None]
    first = min(lifted, key=lambda q: lift_t[q]) if lifted else None
    # freezes: the gripper still (< STILL m over the next 10 steps) within NEAR of a pot that is neither lifted
    # (> 1.5 cm) nor on the stove; open = hovering at the handle without closing, closed = holding without lifting
    eef = s[:, :3]
    still = np.r_[np.linalg.norm(eef[10:] - eef[:-10], axis=1) < STILL, np.zeros(min(10, T), bool)]
    near_idle = np.zeros(T, bool)
    for q in (0, 1):
        near_idle |= (dist[q] < NEAR) & (h[q] < 0.015) & ~g[:, q]
    frz_open, frz_closed = still & near_idle & ~closed, still & near_idle & closed
    up = pot_up(z)
    m = dict(T=T, success=bool(g[-1, 0] and g[-1, 1]), close_onsets=len(onsets), attempts=len(att),
             n_placed=sum(x is not None for x in t_place),
             tipped=int(sum((h[q].min() < TIP) or (up is not None and up[q] < 0.5) for q in (0, 1))),
             freeze_open=int(frz_open.sum()), freeze_closed=int(frz_closed.sum()),
             freeze_longest=longest_run(frz_open | frz_closed))
    if first is not None:
        sec = 1 - first
        a1 = [x for x in att if x[1] == first and x[0] <= lift_t[first]]
        a2 = [x for x in att if x[1] == sec and (lift_t[sec] is None or x[0] <= lift_t[sec])]
        m.update(t_lift1=lift_t[first], attempts1=len(a1), t_place1=t_place[first],
                 hold1=rise_t[first] - lift_t[first],
                 attempts2=len(a2), missed2=sum(o == "missed" for _, _, o in a2),
                 lifted2=lift_t[sec] is not None, t_lift2=lift_t[sec], t_place2=t_place[sec],
                 hold2=None if rise_t[sec] is None else rise_t[sec] - lift_t[sec],
                 wait2=None if (lift_t[sec] is None or t_place[first] is None) else int(lift_t[sec] - t_place[first]))
    return m, dict(h=h, closed=closed, open=opening, t_place=t_place, att=att, frz_open=frz_open, frz_closed=frz_closed)


def longest_run(b):
    best = cur = 0
    for v in b:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return int(best)


def pot_up(z):
    """Up-component of each pot's z axis at the episode end (1 upright, ~0.2 lying on its side), from sim_end =
    [time, qpos(24), qvel(22)], qpos = robot 9, pot 1 free joint (pos 3, quat wxyz 4), pot 2 free joint, stove knob
    (layout checked Nov 11 against the end positions in object_pos). None if the layout doesn't match."""
    se = z.get("sim_end")
    if se is None or len(se) != 47:
        return None
    qp = se[1:25]
    pos, _ = pot_arrays(z)
    if not all(np.allclose(qp[9 + 7 * q:12 + 7 * q], pos[q][-1], atol=2e-3) for q in (0, 1)):
        return None
    return [float(1 - 2 * (qp[13 + 7 * q] ** 2 + qp[14 + 7 * q] ** 2)) for q in (0, 1)]


def timeline(info, T, width=896, height=210):
    fig, ax = plt.subplots(3, 1, figsize=(width / 100, height / 100), dpi=100, sharex=True)
    x = np.arange(T)
    ax[0].fill_between(x, 0, info["closed"].astype(float), step="post", color="#8a8a86", alpha=.5, lw=0)
    # freezes as a strip along the bottom: purple hovering open at a pot, amber holding closed without lifting
    ax[0].fill_between(x, 0, 0.18 * info["frz_open"], step="post", color="#7b4bc9", lw=0)
    ax[0].fill_between(x, 0, 0.18 * info["frz_closed"], step="post", color="#c98a00", lw=0)
    ax[0].set_yticks([]); ax[0].set_ylabel("close\ncmd", fontsize=7, rotation=0, ha="right", va="center")
    for t, p, o in info["att"]:  # grasp attempts: green lifted, red closed on air, amber held but no lift
        ax[0].plot([t], [0.5], marker="v", ms=7, color=OUTCOME_COLOR[o], mec="none")
        ax[0].text(t + 3, 0.55, f"pot {p + 1}", fontsize=6, color=OUTCOME_COLOR[o])
    ax[1].plot(x, info["open"] * 100, color="#0b0b0b", lw=1.2)
    ax[1].set_ylabel("finger\n(cm)", fontsize=7, rotation=0, ha="right", va="center"); ax[1].tick_params(labelsize=6)
    for p, col, name in ((0, "#2a78d6", "pot 1"), (1, "#eb6834", "pot 2")):
        ax[2].plot(x, info["h"][p] * 100, color=col, lw=1.4, label=f"{name} height")
        if info["t_place"][p] is not None:
            for a_ in ax:
                a_.axvline(info["t_place"][p], color=col, ls="--", lw=1)
    ax[2].set_ylabel("pot\nheight (cm)", fontsize=7, rotation=0, ha="right", va="center"); ax[2].tick_params(labelsize=6)
    ax[2].legend(fontsize=6, loc="upper left", ncol=2, frameon=False)
    ax[2].set_xlim(0, max(T - 1, 1)); ax[2].set_xlabel("policy step (20 per second)", fontsize=7)
    fig.subplots_adjust(left=0.12, right=0.99, top=0.97, bottom=0.2, hspace=0.15)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    x0 = fig.axes[2].transData.transform((0, 0))[0]
    x1 = fig.axes[2].transData.transform((max(T - 1, 1), 0))[0]
    plt.close(fig)
    return img, x0, x1


def render(z, info, dst: Path, title: str):
    T = len(z["image"])
    bg, x0, x1 = timeline(info, T)
    dst.parent.mkdir(parents=True, exist_ok=True)
    W, H = 896, 448 + 24 + bg.shape[0]
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", "20", "-i", "-",
                          "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "26", "-preset", "veryfast", str(dst)], stdin=subprocess.PIPE)
    # header strips, pre-rendered once (PIL per frame was ~0.2 s/frame); per frame only numpy copies
    hdr = {}
    for state in (False, True):
        im = Image.new("RGB", (W, 24), (249, 249, 247))
        d = ImageDraw.Draw(im)
        d.text((6, 5), f"{title}   gripper {'CLOSE' if state else 'open'}", fill=(208, 59, 59) if state else (11, 11, 11))
        d.text((454, 5), "wrist camera (what the policy sees)", fill=(82, 81, 78))
        hdr[state] = np.asarray(im)
    frame = np.empty((H, W, 3), np.uint8)
    frame[472:] = bg
    for t in range(T):
        frame[:24] = hdr[bool(info["closed"][t])]
        frame[24:472, :448] = z["image"][t].repeat(2, 0).repeat(2, 1)
        frame[24:472, 448:] = z["wrist_image"][t].repeat(2, 0).repeat(2, 1)
        frame[472:] = bg
        cx = int(round(x0 + (x1 - x0) * t / max(T - 1, 1)))
        frame[472:H - 30, max(cx - 1, 0):cx + 1] = (11, 11, 11)
        p.stdin.write(frame.tobytes())
    p.stdin.close()
    p.wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="label=run_dir_name")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-eps", type=int, default=None)
    ap.add_argument("--no-video", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    rows = []
    for spec in a.runs:
        label, run = spec.split("=", 1)
        rd = RUNS / run if not Path(run).is_absolute() else Path(run)
        ep = pd.read_csv(rd / "episodes.csv")
        for r in ep.itertuples():
            if a.max_eps is not None and r.episode >= a.max_eps:
                break
            if not isinstance(r.raw, str):
                continue
            with np.load(rd / "raw" / r.raw) as zf:  # NpzFile re-reads the whole array on every z[key]; load once
                z = {k: zf[k] for k in zf.files}
            m, info = metrics(z)
            m.update(label=label, episode=int(r.episode), init_state=int(r.init_state), repeat=int(r.repeat),
                     csv_success=int(r.success), video=f"{label}/ep{int(r.episode):04d}.mp4")
            rows.append(m)
            if not a.no_video:
                render(z, info, out / m["video"], f"{label} · state {r.init_state} rep {r.repeat} · "
                                                    f"{'success' if r.success else 'failure'}")
    df = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "gripper_metrics.csv", index=False)
    summ = {}
    med = lambda s: None if s.dropna().empty else float(s.dropna().median())
    for label, d in df.groupby("label"):
        d2 = d[d.attempts2.notna()] if "attempts2" in d else d.iloc[:0]  # lifted a first pot
        summ[label] = dict(
            episodes=int(len(d)), success=round(float(d.success.mean()), 3),
            attempts=round(float(d.attempts.mean()), 2), tipped_eps=int((d.tipped > 0).sum()),
            lifted_first=int(len(d2)),
            attempts1_mean=round(float(d2.attempts1.mean()), 2) if len(d2) else None,
            attempts2_mean=round(float(d2.attempts2.mean()), 2) if len(d2) else None,
            missed2_mean=round(float(d2.missed2.mean()), 2) if len(d2) else None,
            lifted2_rate=round(float(d2.lifted2.astype(bool).mean()), 3) if len(d2) else None,
            t_lift1_median=med(d2.t_lift1) if len(d2) else None,
            t_lift2_median=med(d2.t_lift2) if len(d2) else None,
            wait2_median=med(d2.wait2) if len(d2) else None)
    (out / "gripper_summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
