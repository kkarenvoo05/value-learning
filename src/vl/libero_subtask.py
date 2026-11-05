"""E2 subtask label (docs/libero_plan.md, 2026-11-02 "RECAP vs SeeQ"; Karen: go): per frame, the number of moka pots
on the stove, 0 / 1 / 2, from the recorded BDDL goal predicates. Used only as an auxiliary training target (SeeQ
Table 4 "prediction without conditioning"); the value target is unchanged.

goals (T, 3) bool from the raw episode file: column 0 = On(moka_pot_1, stove), 1 = On(moka_pot_2, stove), evaluated
after action t (vl.libero_milestone), so image t shows the state after action t - 1: label[t] = count(goals[t - 1])
for t >= 1. Image 0 (the initial scene, before any action) gets count(goals[0]) [ours: no predicate is recorded
before action 0]. The count follows the true state, so it can fall if a pot leaves the stove [ours; vl.libero_milestone
additionally asks for HOLD steps, which a per-frame label does not need]. numpy only: imported by the openpi-env
scripts too.
"""

import numpy as np

N_SUBTASK = 3


def subtask_labels(goals: np.ndarray) -> np.ndarray:
    """(T, >= 2) bool goals -> (T,) int64 pots on the stove as seen in image t."""
    k = np.asarray(goals)[:, :2].sum(axis=1).astype(np.int64)
    return np.concatenate([k[:1], k[:-1]])


# B1 (docs/CURRENT.md, Karen: go 2026-11-04): SeeQ-style subtask-scoped value target [SeeQ: subtask scope; ours:
# Monte Carlo instead of TD, two subtasks, the milestone frame as the boundary]. Subtask 0 = "place the first pot",
# ends at the milestone m (vl.libero_milestone: first image with a pot on the stove that stays HOLD steps); subtask
# 1 = "place the second pot" = the rest of the episode, ends at task success. The scope is monotone (once m is
# reached the frame stays in subtask 1). Target of a subtask-0 frame t in an episode that reaches m: -(m - t) / scale
# (the subtask succeeded, no failure penalty, no value carried over from subtask 1); every other frame keeps the
# whole-task target (subtask 1 ends where the task ends; an episode that never reaches m failed subtask 0 exactly
# as it failed the task, so -(T - t) - c_fail is already its subtask-0 target).
def scoped_targets(frame: np.ndarray, value: np.ndarray, m: float, scale: float) -> tuple[np.ndarray, np.ndarray]:
    """One episode: (scope 0/1 per frame, subtask-scoped value). m = milestone frame or nan; scale = horizon + c_fail."""
    frame, value = np.asarray(frame), np.asarray(value, float)
    if np.isnan(m):
        return np.zeros(len(frame), np.int64), value.copy()
    scope = (frame >= m).astype(np.int64)
    return scope, np.where(scope == 0, -(m - frame) / scale, value)
