"""C5 mechanism check: R0 vs camdrop on the wrist-optimism numbers (reuses scripts/wrist_optimism.py)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "scripts")
import wrist_optimism as W
from vl.analysis import annotate, load_trajectories
from vl.config import load_config
from vl.metrics import auc

tc = load_config("configs/train.yaml"); dcfg = load_config(tc["data_config"])
man = pd.read_parquet(load_config(dcfg["dataset_config"])["paths"]["manifest"])
runs = Path(tc["paths"]["runs"]); idx = pd.read_parquet(dcfg["paths"]["frame_index"])
rmed = idx[idx.split == "train"].groupby("repo").value.median().to_dict()
pred = pd.read_parquet(runs / "g0_r0_s0" / "val_predictions.parquet").drop(columns=["V", "ce"])
traj = load_trajectories(Path(dcfg["paths"]["video_root"]), pred, Path("data/analysis/val_trajectory.parquet"))
base = annotate(pred.assign(V=0.0, ce=0.0), man, traj, rmed).drop(columns=["V", "ce", "err", "bias", "cerr"])
base["success"] = base.outcome == "success"
hold = (base.t_s > 5) & (base.to_T_s > 2)
for name, arm in [("R0", "g0_r0"), ("C5", "camdrop")]:
    d = base.merge(W.load_arm(runs, arm, "mean"), on=W.KEYS, validate="1:1")
    d["bias"], d["bias_nw"] = d.V - d.value, d.V_nw - d.value
    for lab, m in [("success", d.success), ("not success", ~d.success)]:
        g = d[m]
        print(f"{name} {lab:12s} bias {g.bias.mean():+.3f}  bias_nw {g.bias_nw.mean():+.3f}  wrist dV {g.wrist_dV.mean():+.3f}  "
              f"wrist dCE {g.wrist_dCE.mean():+.3f}  side dCE {g.side_dCE.mean():+.3f}  hurt share {(g.wrist_dCE > 0).mean():.3f}")
    s = d[hold & d.stall]
    print(f"{name} hold-stall AUC with wrist {auc(s.success, s.V):.3f} without {auc(s.success, s.V_nw):.3f}")
    # per seed: not-success bias
    per = []
    for k in range(3):
        m = pd.read_parquet(runs / f"{arm}_s{k}" / "val_cam_mask.parquet")[W.KEYS + ["V_full"]]
        x = base.merge(m, on=W.KEYS)
        per.append(round((x.V_full - x.value)[~x.success].mean(), 3))
    print(f"{name} not-success bias per seed {per}")
