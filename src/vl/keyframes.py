"""Key frames of a rollout for the localized tag-usage test (reports/2026-11-06-tag-hypotheses/, test E4).

The deep dive (reports/2026-11-06-tag-deep-dive/) found that, in the data, positive and negative frames differ mostly
in gripper timing. A tag-usage metric averaged over random frames can hide an effect that only exists where the
gripper is about to switch, or where the robot is stalled (every failure in our data is a timeout). These helpers pick
those frames from a recorded episode (state (T, 8), executed actions (T, 7)) [ours].

Categories:
  grip   the recorded gripper command changes sign within the 5 actions executed from frame t (one frame per switch,
         drawn from the 5 frames whose executed window contains the switch). `grip_dir` = +1 closing (-1 -> +1),
         -1 opening.
  stall  the end effector moves < 1.5 cm over the next 20 steps and the gripper opening changes < 5 mm (the stall
         definition of scripts/libero_label_rules.py), not a grip frame.
  moving every other frame (the comparison: what the old random-frame test mostly saw).
"""

import numpy as np

EXEC = 5            # executed actions per replan
STALL_N = 20        # lookahead of the stall definition (scripts/libero_label_rules.py)
STALL_M, GRIP_M = 0.015, 0.005


def gripper_switches(actions: np.ndarray) -> list[tuple[int, int]]:
    """(frame f, direction) for every sign change of the gripper command: actions[f-1, 6] vs actions[f, 6]."""
    g = np.sign(actions[:, 6])
    out = []
    for f in np.flatnonzero(g[1:] != g[:-1]) + 1:
        if g[f] != 0 and g[f - 1] != 0:
            out.append((int(f), int(g[f])))
    return out


def stall_mask(state: np.ndarray, n: int = STALL_N) -> np.ndarray:
    T = len(state)
    f = np.arange(T)
    fut = np.minimum(f + n, T - 1)
    move = np.linalg.norm(state[fut, :3] - state[f, :3], axis=1)
    grip = np.abs((state[fut, 6] - state[fut, 7]) - (state[f, 6] - state[f, 7]))
    return (move < STALL_M) & (grip < GRIP_M) & (f + n < T)


def key_frames(state: np.ndarray, actions: np.ndarray, rng: np.random.Generator, max_grip: int = 6,
               max_stall: int = 3, n_moving: int = 2) -> list[tuple[int, str, int]]:
    """[(t, category, grip_dir)] for one episode; grip_dir is 0 outside the grip category."""
    T = len(actions)
    out, used = [], np.zeros(T, bool)
    sw = gripper_switches(actions)
    if len(sw) > max_grip:
        sw = [sw[i] for i in np.sort(rng.choice(len(sw), max_grip, replace=False))]
    for f, d in sw:
        lo = max(0, f - EXEC + 1)
        t = int(rng.integers(lo, f + 1))  # t <= f <= t + 4: the switch is inside the executed window
        out.append((t, "grip", d))
    for f, _ in gripper_switches(actions):
        used[max(0, f - EXEC + 1):f + 1] = True
    st = np.flatnonzero(stall_mask(state) & ~used)
    if len(st):
        pick = np.sort(rng.choice(st, min(max_stall, len(st)), replace=False))
        out += [(int(t), "stall", 0) for t in pick]
        used[st] = True
    mv = np.flatnonzero(~used)
    if len(mv):
        out += [(int(t), "moving", 0) for t in np.sort(rng.choice(mv, min(n_moving, len(mv)), replace=False))]
    return out


def earlier_switch(g_pos: np.ndarray, g_neg: np.ndarray, direction: int) -> np.ndarray:
    """Signed gripper shift between tags, oriented so > 0 means the positive tag switches the gripper earlier.

    g_*: (..., EXEC) gripper commands in [-1, 1] over the executed steps. A closing switch (-1 -> +1) happens earlier
    when the command is larger; an opening switch earlier when it is smaller. Returns the mean over steps."""
    return direction * (g_pos - g_neg).mean(-1)
