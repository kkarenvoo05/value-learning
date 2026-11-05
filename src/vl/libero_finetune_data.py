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
TAG_FORMAT = "{prompt}. Advantage: {tag}"
TABLE = Path("/iris/u/ankile/karenvo/libero/tables/moka_fixed_fresh")
MOKA_CFG = Path(__file__).resolve().parents[2] / "configs" / "libero_moka.yaml"
ARMS = ("imitation", "advantage")


def tag_prompt(prompt: str, positive: bool | None) -> str:
    """Task prompt with the advantage tag; None = no tag (arm A, or a tag-dropout sample)."""
    if positive is None:
        return prompt
    return TAG_FORMAT.format(prompt=prompt, tag="positive" if positive else "negative")


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
        sel = sel[sel.success == 1]
    bad = set(sel.init_state) & held_out
    assert not bad, f"held-out init states in the training set: {sorted(bad)}"
    assert len(sel), "no training episodes"
    return sel.sort_values("episode_index").reset_index(drop=True)


def frame_labels(labels: pd.DataFrame, episode_index: int, n_frames: int) -> np.ndarray:
    """(n_frames,) int8 labels of one episode: each frame takes the latest labelled frame at or before it."""
    g = labels[labels.episode_index == episode_index].sort_values("frame_index")
    assert len(g) and g.frame_index.iloc[0] == 0, f"episode {episode_index}: no label at frame 0"
    assert g.frame_index.iloc[-1] < n_frames, f"episode {episode_index}: label past the last frame"
    assert set(g.positive.unique()) <= {0, 1}
    pos = np.searchsorted(g.frame_index.to_numpy(), np.arange(n_frames), side="right") - 1
    return g.positive.to_numpy(np.int8)[pos]


def load_episode(raw: str, labels: pd.DataFrame | None, episode_index: int) -> dict:
    with np.load(raw) as z:
        ep = {"image": z["image"], "wrist_image": z["wrist_image"], "state": z["state"],
              "actions": z["actions"], "prompt": str(z["prompt"])}
    n = len(ep["actions"])
    assert len(ep["image"]) == len(ep["wrist_image"]) == len(ep["state"]) == n
    ep["label"] = None if labels is None else frame_labels(labels, episode_index, n)
    return ep


def sample(ep: dict, t: int, tag: bool | None) -> dict:
    """One raw sample in the policy server's input format (+ actions), before openpi's input transform."""
    return {"observation/image": ep["image"][t], "observation/wrist_image": ep["wrist_image"][t],
            "observation/state": ep["state"][t], "actions": action_chunk(ep["actions"], t),
            "prompt": tag_prompt(ep["prompt"], tag)}


class EpisodeStream:
    """Endless batches of raw samples. Episodes are read in shuffled groups of `group` (raw files are compressed
    npz of ~50 MB, too slow to read per sample and too big to cache all: 900 episodes ~ 120 GB decoded); inside a
    group every frame is drawn once in random order, then the next group (prefetched in a thread) takes over. One
    pass over all groups = one epoch, so every frame is seen once per epoch [ours].
    tag_dropout: share of arm-C samples whose tag is removed [ours; default 0 = always tagged]."""

    def __init__(self, sel: pd.DataFrame, batch: int, seed: int, group: int = 48, labels: pd.DataFrame | None = None,
                 tag_dropout: float = 0.0, loader=load_episode):
        self.sel, self.batch, self.group, self.labels = sel, batch, group, labels
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
        return [(int(r.episode_index), self.loader(r.raw, self.labels, int(r.episode_index)))
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
                    out.append(sample(ep, t, tag))
                    meta.append((e, t, -1 if tag is None else int(tag)))
                yield out, np.array(meta)
