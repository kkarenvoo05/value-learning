"""Stage 3 control (docs/libero_plan.md): this repo's ResNet-18 V (vl.model.ValueModel) on the Stage 2 rollouts.

Usage (one GPU):
    .venv/bin/python -m vl.libero_resnet --features <features.npz> --out <run dir> [--steps 8000]

Same frames, targets, inner split and val metrics as the 3a probe (vl.libero_probe), so the numbers compare
directly: the frames listed in the probe's features file (every 4th frame), early stopping on the probe's 5 inner
train states, val = the 10 value-val init states. Model = ValueModel unchanged with cameras [agentview, wrist]
(224 x 224 uint8, exactly as the policy saw them; ImageNet weights, shared backbone, concat -> 256 -> 201).
Recipe = configs/train.yaml `baseline_lowlr` (the repo's best ResNet run): AdamW, backbone lr 1e-5, head 1e-4,
weight decay 0.01, batch 32, 500 warmup steps then cosine; no augmentation. [ours: steps, eval every 500]
Images are packed once into <cache>.npy (uint8, (n, 2, 3, 224, 224)) from the raw episode files.
--history-gap-frames G [G2 ...]: the repo's C2a arm (one gap; C2b with several) (ValueModel history_stack: per camera [now, has_past * past] channel-
stacked into a widened conv1), past = the frame G steps earlier in the same episode (G = 20 is C2a's 2 s at 10 fps),
grey stand-in and has_past = 0 before the episode start.
--subtask-weight W > 0: E2 (docs/libero_plan.md 2026-11-02, SeeQ Table 4 "prediction without conditioning"): a
linear head on the value head's hidden layer predicts the pots on the stove (vl.libero_subtask, 0 / 1 / 2) with
cross-entropy weight W (SeeQ: 0.1) added to the value CE. Value target, selection (inner value CE) and val metrics
unchanged; inner subtask accuracy logged, val predictions gain `subtask` and `p_subtask{0,1,2}`.
--subtask-scoped: B1 (docs/CURRENT.md, SeeQ-style): the target is the value of finishing the *current* subtask
(vl.libero_subtask.scoped_targets: place the first pot, then the second; boundary = the milestone frame) and V is
conditioned on the subtask: the head outputs 2 x 201 logits and the scope's half is read (the true scope from the
sim's goal predicates, train and eval alike; SeeQ predicts it, we have it exactly). Selection, CE and val metrics are
on the scoped target; predictions add `scope` and `value_task` (the whole-task target). [ours: MC, not SeeQ's TD]
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from vl.libero_oof import oof_split  # noqa: F401  (re-exported)
from vl.libero_probe import CENTERS, fit_temperature, inner_split, load, metrics
from vl.libero_subtask import N_SUBTASK, scoped_targets, subtask_labels
from vl.model import ValueModel, value_loss

CAMS = ["agentview", "wrist"]
RUN = Path("/iris/u/ankile/karenvo/libero/runs/stage2_moka_18155849")


def table_meta(dataset: Path, every: int) -> pd.DataFrame:
    """Frames straight from an episode table (scripts/libero_fresh_table.py): every `every`-th frame of
    range(policy_steps), the same grid as the probe's features file (frame 0, 4, ..., T = policy_steps - 1)."""
    eps = pd.read_csv(dataset / "episodes.csv")
    rows = [(r.split, r.init_state, r.source_episode, t, r.policy_steps - 1, r.success, r.episode_index)
            for r in eps.itertuples() for t in range(0, r.policy_steps, every)]
    meta = pd.DataFrame(rows, columns=["split", "init_state", "episode", "frame_index", "T", "success",
                                       "episode_index"])
    ft = pd.read_parquet(dataset / "frame_targets.parquet", columns=["episode_index", "frame_index", "value", "bin"])
    meta = meta.merge(ft, on=["episode_index", "frame_index"], how="left", validate="one_to_one")
    assert meta.bin.notna().all() and (meta.bin >= 0).all()
    if "role" in eps.columns:  # stage-1 demo table (scripts/libero_demo_table.py): fit / inner per episode
        meta["role"] = meta.episode_index.map(eps.set_index("episode_index").role).values
    return meta


def replace_targets(meta: pd.DataFrame, tg: pd.DataFrame) -> pd.DataFrame:
    """--targets [2026-11-07, item 3]: the listed frames take the given value (and its nearest bin); every train frame
    must be listed (val frames keep the MC target, so val metrics stay against the realised return)."""
    key = ["episode_index", "frame_index"]
    assert not tg.duplicated(key).any() and tg.value.between(-1, 0).all()
    m = meta.merge(tg[key + ["value"]].rename(columns={"value": "value_new"}), on=key, how="left",
                   validate="one_to_one")
    tr = (m.split == "train").values
    assert m.value_new.notna().values[tr].all(), "--targets must cover every train frame"
    has = m.value_new.notna().values
    m.loc[has, "value"] = m.value_new[has]
    m.loc[has, "bin"] = np.abs(m.value_new.values[has, None] - CENTERS[None]).argmin(1)
    print(f"--targets: {has.sum()} of {len(m)} frames take the given target (mean |change| "
          f"{np.abs(m.value_new[has] - meta.value.values[has]).mean():.4f})", flush=True)
    return m.drop(columns="value_new")


def add_subtask(meta: pd.DataFrame, dataset: Path) -> pd.DataFrame:
    """meta + `subtask` (vl.libero_subtask) read from each episode's raw `goals`."""
    raw_of = pd.read_csv(dataset / "episodes.csv").set_index("episode_index").raw
    sub = np.full(len(meta), -1, np.int64)
    for e, rows in meta.groupby("episode_index"):
        with np.load(RUN / raw_of[e]) as z:  # RUN / an absolute path = that path
            sub[rows.index.values] = subtask_labels(z["goals"])[rows.frame_index.values]
    assert (sub >= 0).all()
    return meta.assign(subtask=sub)


def add_scope(meta: pd.DataFrame, dataset: Path, scale: float = 1040.0) -> pd.DataFrame:
    """B1: meta + `scope` (0 / 1) and `value_task`; `value` / `bin` replaced by the subtask-scoped target.
    scale = horizon 520 + c_fail 520 (configs/libero_moka.yaml), checked against the table's own targets."""
    from vl.libero_milestone import milestone
    raw_of = pd.read_csv(dataset / "episodes.csv").set_index("episode_index").raw
    ok = meta[meta.frame_index == meta["T"]]
    assert np.allclose(ok.value, np.where(ok.success == 1, 0.0, -520 / scale)), "target scale differs from 1040"
    scope, val = np.zeros(len(meta), np.int64), meta.value.to_numpy(float).copy()
    reached = 0
    for e, rows in meta.groupby("episode_index"):
        with np.load(RUN / raw_of[e]) as z:  # RUN / an absolute path = that path
            m = milestone(z["goals"])
        reached += not np.isnan(m)
        scope[rows.index.values], val[rows.index.values] = scoped_targets(rows.frame_index.values,
                                                                          rows.value.values, m, scale)
    b = np.rint((val + 1.0) * (len(CENTERS) - 1)).astype(np.int64)
    assert (b >= 0).all() and (b < len(CENTERS)).all()
    print(f"subtask-scoped targets: {reached}/{meta.episode_index.nunique()} episodes reach the milestone, "
          f"{np.mean(scope == 0):.3f} of frames in subtask 0, {np.mean(val != meta.value.values):.3f} retargeted",
          flush=True)
    return meta.assign(value_task=meta.value.values, value=val, bin=b, scope=scope)


def pack(meta: pd.DataFrame, cache: Path, dataset: Path | None = None) -> np.ndarray:
    """uint8 (n, 2, 3, 224, 224) for the frames of meta, in meta's order, cached as a .npy memmap.
    Raw files: `raw` of <dataset>/episodes.csv, absolute or relative to Stage 2's run (default: Stage 2's table)."""
    if cache.exists():
        arr = np.load(cache, mmap_mode="r")
        assert len(arr) == len(meta)
        return arr
    arr = np.lib.format.open_memmap(cache.with_suffix(".tmp.npy"), mode="w+", dtype=np.uint8,
                                    shape=(len(meta), 2, 3, 224, 224))
    eps = pd.read_csv((dataset or RUN.parent.parent / "lerobot" / "libero_moka_stage2") / "episodes.csv")
    raw_of = dict(zip(eps.episode_index, eps.raw))
    for e, rows in meta.groupby("episode_index"):
        with np.load(RUN / raw_of[e]) as z:  # RUN / an absolute path = that path
            img, wr = z["image"], z["wrist_image"]
        t = rows.frame_index.values
        arr[rows.index.values, 0] = img[t].transpose(0, 3, 1, 2)
        arr[rows.index.values, 1] = wr[t].transpose(0, 3, 1, 2)
    arr.flush()
    del arr
    cache.with_suffix(".tmp.npy").rename(cache)
    return np.load(cache, mmap_mode="r")


PAST = None  # (src index, has_past) per row when --history-gap-frames is set


def past_index(meta: pd.DataFrame, gaps: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """(n, k) source rows and has-past flags for each gap (frames back, same episode)."""
    key = pd.Index(meta.episode_index.values.astype(np.int64) * 10_000 + meta.frame_index.values)
    srcs, hass = [], []
    for gap in gaps:
        src = key.get_indexer(meta.episode_index.values.astype(np.int64) * 10_000 + meta.frame_index.values - gap)
        has = meta.frame_index.values >= gap
        assert (src[has] >= 0).all(), "gap is not on the frame grid"
        srcs.append(np.maximum(src, 0))
        hass.append(has)
    return np.stack(srcs, 1), np.stack(hass, 1)


def batches(arr, idx, bins, bs, device, shuffle, rng=None):
    order = rng.permutation(idx) if shuffle else idx
    for s in range(0, len(order), bs):
        ix = np.sort(order[s:s + bs])
        x = torch.from_numpy(np.ascontiguousarray(arr[ix])).to(device, non_blocking=True)
        imgs = {c: x[:, k] for k, c in enumerate(CAMS)}
        if PAST is not None:
            src, has = PAST
            xp = torch.from_numpy(np.ascontiguousarray(arr[src[ix].ravel()])).to(device, non_blocking=True)
            xp = xp.view(len(ix), src.shape[1], *xp.shape[1:])                      # (B, k, 2 cams, 3, H, W)
            imgs = (imgs, {"images": {c: xp[:, :, k] for k, c in enumerate(CAMS)},
                           "has_past": torch.as_tensor(has[ix], device=device)})
        yield ix, imgs, torch.as_tensor(bins[ix], device=device)


SCOPE = None  # per-row subtask scope (0 / 1) when --subtask-scoped


def run(model, imgs, ix=None):
    lg = model(imgs[0], past=imgs[1]) if isinstance(imgs, tuple) else model(imgs)
    if SCOPE is not None:  # 2 x 201 logits: read the current subtask's half
        lg = lg.view(len(lg), 2, -1)[torch.arange(len(lg), device=lg.device), torch.as_tensor(SCOPE[ix], device=lg.device)]
    return lg


SUB = None  # per-row subtask label when --subtask-weight > 0


@torch.no_grad()
def evaluate(model, arr, idx, bins, device, return_logits=False, return_aux=False):
    """(value probs or logits, value CE); return_aux (SUB set): also (subtask probs, subtask accuracy)."""
    model.eval()
    out, ce = np.zeros((len(idx), len(CENTERS)), np.float32), 0.0
    aux = np.zeros((len(idx), N_SUBTASK), np.float32) if return_aux else None
    pos = {i: k for k, i in enumerate(idx)}
    for ix, imgs, y in batches(arr, idx, bins, 128, device, False):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if return_aux:
                logits, al = model(imgs, return_aux=True)
                logits = logits.float()
                aux[[pos[i] for i in ix]] = torch.softmax(al.float(), -1).cpu().numpy()
            else:
                logits = run(model, imgs, ix).float()
        ce += float(value_loss(logits, y)) * len(ix)
        out[[pos[i] for i in ix]] = (logits if return_logits else torch.softmax(logits, -1)).cpu().numpy()
    model.train()
    if return_aux:
        return out, ce / len(idx), aux, float((aux.argmax(1) == SUB[idx]).mean())
    return out, ce / len(idx)


def final_eval(model, arr, meta, iv, va, bins, device, out, extra):
    """Val metrics of the loaded model, raw and with a temperature fitted on the inner train states."""
    lg_in, _ = evaluate(model, arr, iv, bins, device, return_logits=True)
    lg_va, _ = evaluate(model, arr, va, bins, device, return_logits=True)
    temp = fit_temperature(lg_in, bins[iv])
    sm = lambda z: np.exp(z - z.max(1, keepdims=True)) / np.exp(z - z.max(1, keepdims=True)).sum(1, keepdims=True)
    p, pt = sm(lg_va), sm(lg_va / temp)
    r, rt = metrics(p, meta.iloc[va]), metrics(pt, meta.iloc[va])
    r.update(extra)
    rt.update(temperature=temp)
    preds = meta.iloc[va][["episode_index", "init_state", "frame_index", "T", "success", "value", "bin"]].copy()
    preds["ev_resnet"], preds["ev_resnet_T"] = p @ CENTERS, pt @ CENTERS
    if SCOPE is not None:
        preds["scope"], preds["value_task"] = SCOPE[va], meta.value_task.values[va]
    if SUB is not None:
        _, _, aux, acc = evaluate(model, arr, va, bins, device, return_aux=True)
        preds["subtask"] = SUB[va]
        for k in range(N_SUBTASK):
            preds[f"p_subtask{k}"] = aux[:, k]
        r["subtask_acc"] = acc
    preds.to_parquet(out / "val_predictions.parquet")
    return r, rt


def write_train_predictions(model, arr, meta, bins, device, out):
    """V on every train frame (fit and inner episodes alike) from the loaded model -> <out>/train_predictions.parquet
    [ours, 2026-11-03 meeting: "train on all, label all" comparison to the out-of-fold labels]."""
    tr = np.flatnonzero((meta.split == "train").values)
    p, _ = evaluate(model, arr, tr, bins, device)
    preds = meta.iloc[tr][["episode_index", "init_state", "frame_index", "T", "success", "value", "bin"]].copy()
    preds["ev_resnet"], preds["fold"] = p @ CENTERS, -1
    if SCOPE is not None:
        preds["scope"], preds["value_task"] = SCOPE[tr], meta.value_task.values[tr]
    preds.to_parquet(out / "train_predictions.parquet")
    print(f"train predictions: {len(preds)} frames of {preds.episode_index.nunique()} episodes, "
          f"MAE {np.abs(preds.ev_resnet - preds.value).mean():.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=None, help="the probe's features file: defines the frames")
    ap.add_argument("--every", type=int, default=None,
                    help="instead of --features: every N-th frame of each episode in --dataset's table "
                         "(4 = the features file's grid); frames cached in <dataset>/frames_every<N>.npy")
    ap.add_argument("--pack-only", action="store_true", help="build the frame cache and exit (CPU)")
    ap.add_argument("--dataset", default="/iris/u/ankile/karenvo/libero/lerobot/libero_moka_stage2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fit-episodes", default=None, help="fit on these episode indices only (one per line, '#')")
    ap.add_argument("--init-from", default=None, metavar="PT", help="start from a stage-1 best.pt (fresh optimiser)")
    ap.add_argument("--eval-only", action="store_true", help="score <out>/best.pt (raw and +T), no training")
    ap.add_argument("--oof-fold", type=int, default=None, help="out-of-fold V: hold out this fold (oof_split), "
                    "early-stop on 5 other-fold states, write oof_predictions.parquet for the held-out episodes")
    ap.add_argument("--oof-folds", type=int, default=5)
    ap.add_argument("--predict-train", action="store_true", help="also write train_predictions.parquet: V on every "
                    "train frame (fit + inner) from best.pt, for train-all / label-all advantage labels")
    ap.add_argument("--backbone", default="resnet18", choices=["resnet18", "siglip_pi05"],
                    help="siglip_pi05: pi05_libero's own SigLIP encoder (vl.model.SiglipPi05Features), same recipe "
                         "as the ResNet (backbone lr 1e-5, head 1e-4) [ours]; gradient checkpointing on")
    ap.add_argument("--backbone-lr", type=float, default=1e-5,
                    help="backbone peak lr (default 1e-5 = the ResNet recipe); SigLIP-direct gentle arm uses 1e-6 [ours]")
    ap.add_argument("--freeze-backbone", action="store_true",
                    help="SigLIP-direct frozen arm: backbone fixed (no grads, eval mode), head only [ours]")
    ap.add_argument("--history-gap-frames", type=int, nargs="+", default=None,
                    help="C2a (one gap) / C2b (several): stack the frames this many steps back")
    ap.add_argument("--subtask-weight", type=float, default=0.0,
                    help="E2: weight of the pots-on-stove auxiliary CE (SeeQ: 0.1); 0 = off (the model has no aux head)")
    ap.add_argument("--subtask-scoped", action="store_true",
                    help="B1: subtask-scoped target, V conditioned on the subtask (2 x 201 head)")
    ap.add_argument("--targets", default=None, metavar="PARQUET",
                    help="[2026-11-07] replace the MC target of the listed frames (episode_index, frame_index, "
                         "value; e.g. vl.libero_td fitted n-step TD targets); bins = nearest of the 201 centres")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    device = "cuda"
    assert (a.features is None) != (a.every is None), "give exactly one of --features / --every"
    t0 = time.time()
    if a.features:
        _, meta = load(Path(a.features), Path(a.dataset))
        arr = pack(meta, Path(a.features).with_suffix(".frames.npy"))
    else:
        meta = table_meta(Path(a.dataset), a.every)
        arr = pack(meta, Path(a.dataset) / f"frames_every{a.every}.npy", Path(a.dataset))
    print(f"frames {arr.shape} ready in {time.time() - t0:.0f} s", flush=True)
    if a.targets:
        assert not (a.subtask_scoped or a.subtask_weight or a.history_gap_frames), "--targets: whole-task V only"
        meta = replace_targets(meta, pd.read_parquet(a.targets))
    if a.pack_only:
        return
    if "role" in meta.columns:  # stage 1 on demos: inner = held-out demos, also reported as "val"
        assert a.fit_episodes is None, "--fit-episodes is for the moka tables"
        inner = (meta.role == "inner").values
        tr = np.flatnonzero((meta.role == "fit").values)
    elif a.oof_fold is not None:
        assert a.fit_episodes is None, "--oof-fold defines its own fit set"
        held, inner = oof_split(meta, a.oof_fold, a.oof_folds)
        tr = np.flatnonzero((meta.split == "train").values & ~inner & ~held)
        ho = np.flatnonzero(held)
        print(f"oof fold {a.oof_fold}/{a.oof_folds}: fit {meta.episode_index.values[tr].size} frames of "
              f"{len(set(meta.episode_index.values[tr]))} episodes, inner {len(set(meta.episode_index[inner]))} "
              f"episodes, held out {len(set(meta.episode_index.values[ho]))} episodes", flush=True)
    else:
        inner = inner_split(meta)
        tr = np.flatnonzero((meta.split == "train").values & ~inner)
    if a.fit_episodes:  # data-size check: a subset of the fit episodes, inner / val unchanged
        keep = {int(x) for x in Path(a.fit_episodes).read_text().split("\n") if x.strip() and x[0] != "#"}
        assert keep <= set(meta.episode_index.values[tr]), "subset episodes outside fit"
        tr = tr[meta.episode_index.isin(keep).values[tr]]
        print(f"fit subset {a.fit_episodes}: {len(keep)} episodes, {len(tr)} frames", flush=True)
    iv = np.flatnonzero(inner)
    va = iv if "role" in meta.columns else np.flatnonzero((meta.split == "val").values)
    bins = meta.bin.values.astype(np.int64)

    global PAST, SUB, SCOPE
    if a.subtask_scoped:
        assert a.every is not None and not a.history_gap_frames and a.subtask_weight == 0 and "role" not in meta.columns \
            and a.backbone == "resnet18", "--subtask-scoped is built for the moka tables (--every), ResNet, no history / aux"
        meta = add_scope(meta, Path(a.dataset))
        bins = meta.bin.values.astype(np.int64)
        SCOPE = meta.scope.values
    if a.subtask_weight > 0:
        assert a.every is not None and not a.history_gap_frames and "role" not in meta.columns, \
            "--subtask-weight is built for the moka episode tables (--every), no history"
        meta = add_subtask(meta, Path(a.dataset))
        SUB = meta.subtask.values
        print("subtask label share (fit frames): "
              + ", ".join(f"{k}: {np.mean(SUB[tr] == k):.3f}" for k in range(N_SUBTASK)), flush=True)
    if a.history_gap_frames:
        PAST = past_index(meta, a.history_gap_frames)
        model = ValueModel(CAMS, history=True, history_stack=True,
                           history_frames=len(a.history_gap_frames)).to(device)
    else:
        model = ValueModel(CAMS, n_bins=len(CENTERS) * (2 if SCOPE is not None else 1), backbone=a.backbone,
                           aux_classes=N_SUBTASK if SUB is not None else 0).to(device)
        if a.freeze_backbone:
            model.freeze_backbone()
        elif a.backbone == "siglip_pi05":
            model.backbone.vit.set_grad_checkpointing(True)  # 27-layer ViT on 64 images per step; fits 24 GB
    if a.eval_only:
        model.load_state_dict(torch.load(out / "best.pt"))
        res = json.loads((out / "results.json").read_text()) if (out / "results.json").exists() else {}
        r, rt = final_eval(model, arr, meta, iv, va, bins, device, out, {})
        res.update(val=r, val_T=rt)
        (out / "results.json").write_text(json.dumps(res, indent=2))
        print(f"resnet18 val: {r}\nresnet18 val +T: {rt}")
        if a.predict_train:
            write_train_predictions(model, arr, meta, bins, device, out)
        return
    if a.init_from:  # stage 2 from a stage-1 best.pt: weights loaded, optimiser and schedule fresh
        model.load_state_dict(torch.load(a.init_from))
        print(f"initialised from {a.init_from}: inner CE at step 0 {evaluate(model, arr, iv, bins, device)[1]:.3f}",
              flush=True)
    head = [p for n, p in model.named_parameters() if not n.startswith("backbone")]
    back = [p for n, p in model.named_parameters() if n.startswith("backbone")]
    assert back and head
    groups = [{"params": head, "lr": 1e-4}]
    if not a.freeze_backbone:
        groups.insert(0, {"params": back, "lr": a.backbone_lr})
    print(f"backbone {a.backbone}: " + ("frozen" if a.freeze_backbone else f"lr {a.backbone_lr:g}") + ", head lr 1e-4",
          flush=True)
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    base = [g["lr"] for g in opt.param_groups]
    rng = np.random.default_rng(a.seed)
    step, best, hist = 0, (np.inf, -1), []
    while step < a.steps:
        for ix, imgs, y in batches(arr, tr, bins, 32, device, True, rng):
            f = min(1, (step + 1) / 500) * 0.5 * (1 + math.cos(math.pi * min(1, step / a.steps)))
            for g, b in zip(opt.param_groups, base):
                g["lr"] = b * f
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if SUB is None:
                    loss = value_loss(run(model, imgs, ix).float(), y)
                else:
                    logits, al = model(imgs, return_aux=True)
                    loss = value_loss(logits.float(), y)  # logged train loss stays the value CE
                    sub_ce = torch.nn.functional.cross_entropy(al.float(), torch.as_tensor(SUB[ix], device=device))
            opt.zero_grad(set_to_none=True)
            (loss if SUB is None else loss + a.subtask_weight * sub_ce).backward()
            opt.step()
            step += 1
            if step % a.eval_every == 0 or step == a.steps:
                if SUB is None:
                    _, ce_in = evaluate(model, arr, iv, bins, device)
                    extra = {}
                else:
                    _, ce_in, _, acc = evaluate(model, arr, iv, bins, device, return_aux=True)
                    extra = {"train_subtask_ce": float(sub_ce.detach()), "inner_subtask_acc": acc}
                hist.append({"step": step, "train_loss": float(loss.detach()), "inner_ce": ce_in, **extra})
                print(f"step {step}: train loss {float(loss.detach()):.3f}, inner-val CE {ce_in:.3f}"
                      + (f", subtask CE {extra['train_subtask_ce']:.3f} inner acc {extra['inner_subtask_acc']:.3f}"
                         if extra else "") + f" ({time.time() - t0:.0f} s)", flush=True)
                if ce_in < best[0]:
                    best = (ce_in, step)
                    torch.save(model.state_dict(), out / "best.pt")
            if step >= a.steps:
                break
    model.load_state_dict(torch.load(out / "best.pt"))
    r, rt = final_eval(model, arr, meta, iv, va, bins, device, out,
                       {"inner_ce": best[0], "best_step": best[1], "steps": a.steps})
    (out / "results.json").write_text(json.dumps({"val": r, "val_T": rt, "history": hist, "setup": vars(a)}, indent=2))
    print(f"resnet18 val: {r}\nresnet18 val +T: {rt}")
    if a.predict_train:
        write_train_predictions(model, arr, meta, bins, device, out)
    if a.oof_fold is not None:
        p, _ = evaluate(model, arr, ho, bins, device)
        preds = meta.iloc[ho][["episode_index", "init_state", "frame_index", "T", "success", "value", "bin"]].copy()
        preds["ev_resnet"], preds["fold"] = p @ CENTERS, a.oof_fold
        if SCOPE is not None:
            preds["scope"], preds["value_task"] = SCOPE[ho], meta.value_task.values[ho]
        preds.to_parquet(out / "oof_predictions.parquet")
        print(f"oof predictions: {len(preds)} frames, MAE {np.abs(preds.ev_resnet - preds.value).mean():.3f}")


if __name__ == "__main__":
    main()
