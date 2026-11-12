"""Stage 6 data (docs/libero_plan.md): what the policy fine-tuning arms train on, without openpi or JAX.

The trainer (scripts/pi05_policy_finetune.py, openpi env) imports this; the tests (tests/test_libero_finetune.py,
this repo's env) check it on synthetic episodes.

Arms (plan Stage 6; same init, steps, preprocessing and recipe for every arm):
  A "imitation"  successful train rollouts only, task prompt unchanged
  C "advantage"  every train rollout, prompt + an advantage tag per frame from a labels file (positive / negative),
                 positive at inference [RECAP / pi*0.6 (arXiv 2511.14759): the improvement indicator enters as text
                 in the prompt; the exact wording here is ours: TAG_FORMAT]
  (B "Rohan's data" is still undefined in the plan; `--episodes` lets it pick any train subset later.)

Training sample = (observation at frame t, the next HORIZON executed actions a_t .. a_{t+H-1}). Near the end the
chunk repeats the last action [ours: LeRobot's delta_timestamps clamp to the episode, which openpi's LIBERO loader
uses, so this matches pi05_libero's own training]. Every frame of an episode is a sample start.

Labels (arm C): a parquet with episode_index, frame_index, positive (0/1), on any frame grid (vl.libero_advantages
labels every 4th frame). Frame t takes the label of the latest labelled frame <= t of its episode [ours]; every
training episode must have a label at frame 0, or the trainer refuses to start.

Held-out init states (configs/libero_moka.yaml): value-val and policy-test states are never trained on; the
selection asserts it. Fresh-layout episodes (init_state -1) are train data.
"""

from __future__ import annotations

import queue
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HORIZON = 10  # pi05_libero action_horizon
SHIFT_DIM = 0  # tag positive control: the action dim the injected shift moves (x translation, raw std ~0.2)
TAG_FORMAT = "{prompt}. Advantage: {tag}"
TABLE = Path("/iris/u/ankile/karenvo/libero/tables/moka_fixed_fresh")
MOKA_CFG = Path(__file__).resolve().parents[2] / "configs" / "libero_moka.yaml"
ARMS = ("imitation", "advantage", "weighted")  # weighted: every rollout, no tag, per-frame loss weights [ours]


# Subtask text before the tag [2026-11-07, item 4; paper: pi*0.6 Sec. V-A/V-B, the model first writes the next subtask
# l_hat, then the indicator, then the actions]. Ours: moka's two subtasks from the BDDL goal predicates (pots on the
# stove 0 -> first, >= 1 -> second; vl.libero_subtask), teacher-forced in the prompt during training and the
# simulator's predicate at inference (an upper bound on a predicted subtask; pi*0.6 predicts it); wording ours.
SUBTASKS = ("put the first moka pot on the stove", "put the second moka pot on the stove")
SUBTASK_FORMAT = "{prompt}. Subtask: {subtask}"
SUBTASK_PROMPT = False  # load_episode default: attach per-frame subtasks (set by the trainer / tag check for such runs)


def tag_prompt(prompt: str, positive: bool | None, subtask: int | None = None) -> str:
    """Task prompt [+ subtask text] + the advantage tag; positive None = no tag (arm A, or a tag-dropout sample),
    subtask None = no subtask text (every run before 2026-11-07; demos, which carry no goal predicates)."""
    if subtask is not None:
        prompt = SUBTASK_FORMAT.format(prompt=prompt, subtask=SUBTASKS[int(subtask)])
    if positive is None:
        return prompt
    return TAG_FORMAT.format(prompt=prompt, tag="positive" if positive else "negative")


def frame_subtasks(goals: np.ndarray) -> np.ndarray:
    """(T,) subtask index of image t (0 = no pot on the stove yet, 1 = at least one), from the recorded goals."""
    from vl.libero_subtask import subtask_labels
    return np.minimum(subtask_labels(goals), 1)


def action_chunk(actions: np.ndarray, t: int, horizon: int = HORIZON) -> np.ndarray:
    """(horizon, dim): actions[t : t + horizon], the last action repeated past the end."""
    idx = np.minimum(np.arange(t, t + horizon), len(actions) - 1)
    return actions[idx]


def held_out_states(cfg_path: Path = MOKA_CFG) -> set[int]:
    split = yaml.safe_load(Path(cfg_path).read_text())["split"]
    return set(split["val"]) | set(split["test"])


def select_episodes(eps: pd.DataFrame, arm: str, episodes: list[int] | None = None,
                    held_out: set[int] | None = None) -> pd.DataFrame:
    """Training episodes of an arm: split == train (or the given subset of it); arm A keeps successes only."""
    assert arm in ARMS, arm
    held_out = held_out_states() if held_out is None else held_out
    sel = eps[eps.split == "train"]
    if episodes is not None:
        missing = set(episodes) - set(sel.episode_index)
        assert not missing, f"not train episodes: {sorted(missing)[:5]}"
        sel = sel[sel.episode_index.isin(episodes)]
    if arm == "imitation":
        sel = sel[sel.success == 1]  # arms C and weighted keep every rollout
    bad = set(sel.init_state) & held_out
    assert not bad, f"held-out init states in the training set: {sorted(bad)}"
    assert len(sel), "no training episodes"
    return sel.sort_values("episode_index").reset_index(drop=True)


DEMO_TABLE = Path("/iris/u/ankile/karenvo/libero/tables/libero_demos")
DEMO_OFFSET = 1_000_000  # demo episode_index = DEMO_OFFSET + demo-table index, so it never collides with a rollout


def demo_episodes(table: Path, task: str) -> pd.DataFrame:
    """The task's LIBERO demonstrations as training rows [RECAP Alg. 1: D_l starts from the task's demos; ours: the
    openpi LeRobot copy holds 29 of the 50 moka demos]. Same columns select_episodes returns; init_state -1 (a demo's
    first frame matches none of the 40 rolled-out init states: nearest 4.1-7.0 mean |px| vs 1.9 within a state)."""
    d = pd.read_csv(Path(table) / "episodes.csv")
    d = d[d.task == task]
    assert len(d), f"no demos for {task!r} in {table}"
    assert d.success.eq(1).all()
    return pd.DataFrame({"episode_index": DEMO_OFFSET + d.episode_index.to_numpy(), "split": "train",
                         "init_state": -1, "success": 1, "raw": d.raw.to_numpy(), "demo": True,
                         "policy_steps": d.policy_steps.to_numpy() if "policy_steps" in d else -1})


def oversample_demos(sel: pd.DataFrame, share: float) -> tuple[pd.DataFrame, int, float]:
    """[2026-11-08, step-1 recipe grid; ours] Repeat every demo row k times so demos are ~`share` of the training
    frames (EpisodeStream draws each listed row's frames once per epoch, so a row listed k times is seen k times).
    Frames counted by policy_steps. Returns (rows, k, realised demo frame share)."""
    assert 0 < share < 1 and "demo" in sel and sel.demo.any() and (~sel.demo).any()
    assert (sel.policy_steps > 0).all(), "frame counts (policy_steps) needed for every row"
    roll, dem = sel.policy_steps[~sel.demo].sum(), sel.policy_steps[sel.demo].sum()
    k = max(1, int(round(share / (1 - share) * roll / dem)))
    out = pd.concat([sel[~sel.demo]] + [sel[sel.demo]] * k, ignore_index=True)
    return out, k, float(k * dem / (k * dem + roll))


def demo_labels(demos: pd.DataFrame) -> pd.DataFrame:
    """Every demo frame tagged positive [RECAP: demonstrations get the positive indicator]: one label at frame 0."""
    return pd.DataFrame({"episode_index": demos.episode_index.to_numpy(), "frame_index": 0, "positive": 1})


def pretrain_demo_episodes(table: Path, roles: tuple[str, ...] = ("fit",)) -> pd.DataFrame:
    """Every task's LIBERO demos (all 40 tasks) as training rows for the speed-tag pre-training stage [2026-11-06;
    RECAP §V-D: the tag is first learned in multi-task pre-training]. Same columns and episode_index offset as
    demo_episodes; `roles` picks the demo table's fit / inner split (inner = held out for the stage's tag check)."""
    d = pd.read_csv(Path(table) / "episodes.csv")
    d = d[d.role.isin(roles)]
    assert len(d) and d.success.eq(1).all()
    return pd.DataFrame({"episode_index": DEMO_OFFSET + d.episode_index.to_numpy(), "split": "train",
                         "init_state": -1, "success": 1, "raw": d.raw.to_numpy(), "demo": True,
                         "task": d.task.to_numpy()})


def speed_labels(demos: pd.DataFrame, positive_share: float = 0.3) -> pd.DataFrame:
    """Episode-level speed tag of each demo: positive = among the fastest `positive_share` of its task's demos.

    [RECAP §IV-C / App. A-F: pre-training values are -(steps to success), and "we select the threshold for each task
    such that approximately 30% of the demonstration data has positive advantage".] Ours: the baseline is the task
    (no learned V, so no V noise), which makes A = -(T - t) - E_task[-(T - t)] constant in t up to the baseline's time
    term, so we tag whole episodes; the share counts episodes, not frames (fast demos are shorter, so the frame share
    is lower; reported by scripts/libero_speed_labels.py). Ties broken by episode_index (deterministic).
    `demos`: the demo table's episodes.csv rows (episode_index, task, policy_steps). Returns one label per episode at
    frame 0 (frame_labels floors it onto every frame), keyed DEMO_OFFSET + episode_index like demo_episodes."""
    assert 0 < positive_share < 1
    out = []
    for task, g in demos.sort_values("episode_index").groupby("task", sort=True):
        k = int(round(positive_share * len(g)))
        assert 0 < k < len(g), f"task {task!r}: {len(g)} demos cannot give {k} positive"
        fast = g.sort_values(["policy_steps", "episode_index"], kind="stable").episode_index.iloc[:k]
        out.append(pd.DataFrame({"episode_index": DEMO_OFFSET + g.episode_index.to_numpy(), "frame_index": 0,
                                 "positive": g.episode_index.isin(fast).astype(int).to_numpy(),
                                 "task": task, "policy_steps": g.policy_steps.to_numpy()}))
    return pd.concat(out, ignore_index=True)


def frame_labels(labels: pd.DataFrame, episode_index: int, n_frames: int) -> np.ndarray:
    """(n_frames,) int8 labels of one episode: each frame takes the latest labelled frame at or before it."""
    g = labels[labels.episode_index == episode_index].sort_values("frame_index")
    assert len(g) and g.frame_index.iloc[0] == 0, f"episode {episode_index}: no label at frame 0"
    assert g.frame_index.iloc[-1] < n_frames, f"episode {episode_index}: label past the last frame"
    assert set(g.positive.unique()) <= {0, 1}
    pos = np.searchsorted(g.frame_index.to_numpy(), np.arange(n_frames), side="right") - 1
    return g.positive.to_numpy(np.int8)[pos]


def frame_weights(weights: pd.DataFrame, episode_index: int, n_frames: int) -> np.ndarray:
    """(n_frames,) float32 loss weights of one episode, floored onto the label grid like frame_labels."""
    g = weights[weights.episode_index == episode_index].sort_values("frame_index")
    assert len(g) and g.frame_index.iloc[0] == 0, f"episode {episode_index}: no weight at frame 0"
    assert g.frame_index.iloc[-1] < n_frames and (g.weight >= 0).all()
    pos = np.searchsorted(g.frame_index.to_numpy(), np.arange(n_frames), side="right") - 1
    return g.weight.to_numpy(np.float32)[pos]


def load_episode(raw: str, labels: pd.DataFrame | None, episode_index: int,
                 weights: pd.DataFrame | None = None, subtask: bool | None = None) -> dict:
    subtask = SUBTASK_PROMPT if subtask is None else subtask
    with np.load(raw) as z:
        ep = {"image": z["image"], "wrist_image": z["wrist_image"], "state": z["state"],
              "actions": z["actions"], "prompt": str(z["prompt"])}
        ep["subtask"] = frame_subtasks(z["goals"]) if subtask and "goals" in z.files else None
    n = len(ep["actions"])
    assert len(ep["image"]) == len(ep["wrist_image"]) == len(ep["state"]) == n
    ep["label"] = None if labels is None else frame_labels(labels, episode_index, n)
    if weights is not None:
        ep["weight"] = frame_weights(weights, episode_index, n)
    return ep


def sample(ep: dict, t: int, tag: bool | None, action_shift: float = 0.0) -> dict:
    """One raw sample in the policy server's input format (+ actions), before openpi's input transform. Episodes
    with loss weights add "weight" (the trainer pops it before the transform).
    action_shift [ours, 2026-11-06, tag positive control]: tagged samples get action dim SHIFT_DIM of the whole chunk
    moved by +shift (positive tag) / -shift (negative tag), raw units; untagged samples are unchanged. With random
    tags this makes the tag informative by construction, independent of the state."""
    chunk = action_chunk(ep["actions"], t)
    if action_shift and tag is not None:
        chunk = chunk.copy()
        chunk[:, SHIFT_DIM] += action_shift if tag else -action_shift
    s = {"observation/image": ep["image"][t], "observation/wrist_image": ep["wrist_image"][t],
            "observation/state": ep["state"][t], "actions": chunk,
            "prompt": tag_prompt(ep["prompt"], tag, None if ep.get("subtask") is None else int(ep["subtask"][t]))}
    if ep.get("weight") is not None:
        s["weight"] = float(ep["weight"][t])
    return s


class EpisodeStream:
    """Endless batches of raw samples. Episodes are read in shuffled groups of `group` (raw files are compressed
    npz of ~50 MB, too slow to read per sample and too big to cache all: 900 episodes ~ 120 GB decoded); inside a
    group every frame is drawn once in random order, then the next group (prefetched in a thread) takes over. One
    pass over all groups = one epoch, so every frame is seen once per epoch [ours].
    tag_dropout: share of arm-C samples whose tag is removed [ours; default 0 = always tagged]."""

    def __init__(self, sel: pd.DataFrame, batch: int, seed: int, group: int = 48, labels: pd.DataFrame | None = None,
                 tag_dropout: float = 0.0, loader=load_episode, weights: pd.DataFrame | None = None,
                 action_shift: float = 0.0):
        self.sel, self.batch, self.group, self.labels, self.weights = sel, batch, group, labels, weights
        self.action_shift = action_shift
        self.tag_dropout, self.loader = tag_dropout, loader
        self.rng = np.random.default_rng(seed)
        self.epoch, self.q = 0, queue.Queue(maxsize=1)
        self._order = []
        threading.Thread(target=self._producer, daemon=True).start()

    def _next_group(self):
        if not self._order:
            self._order = list(self.rng.permutation(len(self.sel)))
            self.epoch += 1
        take, self._order = self._order[:self.group], self._order[self.group:]
        rows = self.sel.iloc[take]
        extra = {} if self.weights is None else {"weights": self.weights}
        return [(int(r.episode_index), self.loader(r.raw, self.labels, int(r.episode_index), **extra))
                for r in rows.itertuples()]

    def _producer(self):
        while True:
            try:
                self.q.put(self._next_group())
            except BaseException as e:  # surface loader errors in the training thread
                self.q.put(e)
                return

    def __iter__(self):
        while True:
            grp = self.q.get()
            if isinstance(grp, BaseException):
                raise grp
            idx = [(k, t) for k, (_, ep) in enumerate(grp) for t in range(len(ep["actions"]))]
            order = self.rng.permutation(len(idx))
            for s in range(0, len(order) - self.batch + 1, self.batch):  # the group's last partial batch is dropped
                out, meta = [], []
                for j in order[s:s + self.batch]:
                    k, t = idx[j]
                    e, ep = grp[k]
                    tag = None
                    if ep["label"] is not None and self.rng.random() >= self.tag_dropout:
                        tag = bool(ep["label"][t])
                    out.append(sample(ep, t, tag, self.action_shift))
                    meta.append((e, t, -1 if tag is None else int(tag)))
                yield out, np.array(meta)
