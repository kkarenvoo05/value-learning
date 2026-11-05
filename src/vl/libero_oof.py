"""Out-of-fold split for Stage 5 advantage labels on the train rollouts (numpy / pandas only, so both the torch
ResNet trainer and the openpi value-expert script can import it)."""
import numpy as np
import pandas as pd


def oof_split(meta: pd.DataFrame, fold: int, k_folds: int = 5, n_inner: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold V for Stage 5 labels on the train rollouts (docs/libero_plan.md): (held-out mask, inner mask).
    Folds [ours]: the fixed train init states shuffled (rng 0) and dealt into k_folds groups (whole scenes per fold:
    episodes of one init state start from the identical scene); fresh-layout episodes (init_state -1, each its own
    layout) shuffled (rng 0) and dealt the same way. Held out = the fold's train episodes (predicted, never fit,
    never used for selection). Inner (early stopping) = n_inner fixed states drawn (rng 1 + fold) from the other folds'
    states, so every train episode gets exactly one prediction from a model that neither fit nor selected on it."""
    tr = meta.split == "train"
    eps = meta[tr].groupby("episode_index").init_state.first()
    states = np.sort(eps[eps >= 0].unique())
    fresh = np.sort(eps[eps < 0].index.values)
    rng = np.random.default_rng(0)
    s_fold = dict(zip(rng.permutation(states), np.arange(len(states)) % k_folds))
    f_fold = dict(zip(rng.permutation(fresh), np.arange(len(fresh)) % k_folds))
    ep_fold = {e: (f_fold[e] if s < 0 else s_fold[s]) for e, s in eps.items()}
    held = tr.values & (meta.episode_index.map(ep_fold).values == fold)
    others = np.array([s for s in states if s_fold[s] != fold])
    inner_states = np.random.default_rng(1 + fold).choice(others, size=n_inner, replace=False)
    inner = tr.values & meta.init_state.isin(inner_states).values & ~held
    return held, inner
