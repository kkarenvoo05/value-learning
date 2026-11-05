"""Milestone / dip checks (vl.libero_milestone_probe definitions) on the grid and history value models vs the
original probe and ResNet. Needs reports/figures/libero_milestone_probe/events.csv from vl.libero_milestone_probe.
Usage: .venv/bin/python scripts/libero_dip_compare.py   (prints three tables; plan log 2026-10-29)
"""
import numpy as np, pandas as pd
from pathlib import Path
from vl.libero_milestone import jump, episodes, W, HOLD
from vl.libero_milestone_probe import scene_boot
from vl.metrics import auc
F="/iris/u/ankile/karenvo/libero/features/"; R="/iris/u/ankile/karenvo/libero/runs/"
src={ # label: (path, V col, P col or None)
 "probe mean-pool":(F+"feats_every4_18163142_oc","ev_all_mlp","ps_all_mlp"),
 "probe grid4 attn":(F+"feats_grid4_every4_18167104_attn","ev_grid4_attn","ps_grid4_attn"),
 "probe attn on avg (1 token)":(F+"feats_every4_18163142_attnavg","ev_avg1_attn","ps_avg1_attn"),
 "probe attn on avg (4 tokens)":(F+"feats_every4_18163142_attnavg","ev_avg4_attn","ps_avg4_attn"),
 "probe hist 2 s":(F+"feats_every4_18163142_hist20","ev_all_mlp","ps_all_mlp"),
 "probe hist 2/4.8/10 s":(F+"feats_every4_18163142_hist20_48_100","ev_all_mlp","ps_all_mlp"),
 "ResNet":(R+"resnet18_18166232","ev_resnet",None),
 "ResNet c2a 2 s":(R+"resnet18_c2a20_18167078","ev_resnet",None),
 "ResNet hist 2/4.8/10 s":(R+"resnet18_hist20_48_100_18167188","ev_resnet",None)}
ev=pd.read_csv("reports/figures/libero_milestone_probe/events.csv").set_index("episode_index")
val=episodes("val").set_index("episode_index"); T=val.policy_steps; ms=val.milestone
base=pd.read_parquet(F+"feats_every4_18163142_oc/val_predictions.parquet")[["episode_index","init_state","frame_index","success"]]
cols={}
for lab,(p,v,ps) in src.items():
    d=pd.read_parquet(Path(p)/"val_predictions.parquet")
    cols[lab+"|V"]=d.set_index(["episode_index","frame_index"])[v]
    if ps: cols[lab+"|P"]=d.set_index(["episode_index","frame_index"])[ps]
pred=base.set_index(["episode_index","frame_index"]).join(pd.DataFrame(cols),how="inner").reset_index()
assert len(pred)==13944
by={e:g.sort_values("frame_index") for e,g in pred.groupby("episode_index")}
treated=[e for e in by if np.isfinite(ms[e]) and ms[e]>=W and ms[e]+W<=T[e]]
rng=np.random.default_rng(0)
mcols=[c for c in pred.columns if "|" in c]
# jumps: raw, time-matched excess
rows=[]
for e in treated:
    m,g=ms[e],by[e]; f=g.frame_index.values
    ctrl=[c for c in by if c!=e and T[c]>=m+W and not (ms[c]<m+W+HOLD)]
    r={"episode_index":e,"init_state":int(g.init_state.iloc[0]),"success":int(g.success.iloc[0])}
    for c in mcols:
        j=jump(f,g[c].values,m); jc=np.nanmean([jump(by[k].frame_index.values,by[k][c].values,m) for k in ctrl])
        r[c+"/J"],r[c+"/dd"]=j,j-jc
    rows.append(r)
per=pd.DataFrame(rows)
out=[]
for c in mcols:
    f=lambda d:d[d.success==1][c+"/J"].mean()-d[d.success==0][c+"/J"].mean()
    lo,hi=scene_boot(per,f,rng); 
    g2=lambda d:d[c+"/dd"].mean(); lo2,hi2=scene_boot(per,g2,rng)
    out.append({"model":c,"excess jump (old stat)":f"{per[c+'/dd'].mean():+.3f} [{lo2:+.3f},{hi2:+.3f}]",
      "jump|S":per[per.success==1][c+"/J"].mean(),"jump|F":per[per.success==0][c+"/J"].mean(),
      "S−F":f"{f(per):+.3f} [{lo:+.3f},{hi:+.3f}]","AUC(jump→succ)":auc(per.success.values,per[c+"/J"].values)})
print(pd.DataFrame(out).round(3).to_string())
# phase calibration for P(success) models + dip in V
pt=pred[pred.episode_index.isin(treated)].copy(); pt["rel"]=pt.frame_index-pt.episode_index.map(ms)
W_=(("[-10,-5)",-100,-50),("[-5,0)",-50,0),("[0,4)",0,40),("[4,8)",40,80),("[10,15)",100,150))
cal={}
for c in mcols:
    cal[c]={}
    for n,a,b in W_:
        s=pt[(pt.rel>=a)&(pt.rel<b)].groupby("episode_index").agg(x=(c,"mean"),y=("success","first"))
        cal[c][n]=s.x.mean()
cal=pd.DataFrame(cal).T
y={n:pt[(pt.rel>=a)&(pt.rel<b)].groupby("episode_index").success.first().mean() for n,a,b in W_}
print("actual success per window",{k:round(v,3) for k,v in y.items()})
print(cal.round(3).to_string())
# event-aligned dip: mean in [release+1s, near_q) minus [m-5,m) for P/V
d=[]
for c in mcols:
    x=[]
    for e in treated:
        if not np.isfinite(ev.loc[e,"near_q"]): continue
        g=by[e]; f=g.frame_index.values; v=g[c].values; m=ms[e]
        tr=(f>=ev.loc[e,"release"]+10)&(f<ev.loc[e,"near_q"]); pre=(f>=m-50)&(f<m)
        post=(f>=ev.loc[e,"near_q"])&(f<ev.loc[e,"near_q"]+40)
        if tr.any(): x.append((v[tr].mean()-v[pre].mean(), v[post].mean()-v[tr].mean()))
    x=np.array(x); d.append({"model":c,"transit − before":x[:,0].mean(),"reach pot2 − transit":x[:,1].mean(),"n":len(x)})
print(pd.DataFrame(d).round(3).to_string())
