import numpy as np, pandas as pd, json, glob
R="/iris/u/ankile/karenvo/libero/runs/"
A={"base":"eval_base_noise1_libero_10_18355478","r5":"eval_agr_r5_s0_shift099_noise1_*",
 "d1300":"eval_dsrl_s0_agent1300_*","d1357":"eval_dsrl_s0_ep1800_*","s1_1300":"eval_dsrl_s1_agent1300_*"}
D={}
for k,p in A.items():
    g=glob.glob(R+p); assert len(g)==1; d=pd.read_csv(g[0]+"/episodes.csv",dtype={"goals_end":str})
    D[k]=d
    print(k,len(d),d.columns.tolist()[:20], d.repeat.min(), d.repeat.max())
T={}
for k,d in D.items():
    has=d.goals_end.notna().any()
    def code(r):
        if r.success==1: return 2
        if not has: return 0
        return 1 if str(r.goals_end).zfill(3) in ("011","101") else 0
    d["c"]=d.apply(code,axis=1)
    T[k]={str(int(s)):x.sort_values("repeat").c.tolist() for s,x in d.groupby("init_state")}
    T[k+"_steps"]={str(int(s)):x.sort_values("repeat").policy_steps.astype(int).tolist() for s,x in d.groupby("init_state")}
    T[k+"_hasgoals"]=bool(has)
rng=np.random.default_rng(0)
diff={}
for b in ["d1300","d1357","s1_1300"]:
    diff[b]={}
    a=D["r5"].set_index(["init_state","repeat"]).success; c=D[b].set_index(["init_state","repeat"]).success
    for s in sorted(D["r5"].init_state.unique()):
        x=(a.loc[s]-c.loc[s]).values.astype(float)
        m=x[rng.integers(0,len(x),(5000,len(x)))].mean(1)
        diff[b][str(int(s))]=[round(x.mean(),3),round(np.percentile(m,2.5),3),round(np.percentile(m,97.5),3)]
T["diff"]=diff
print(json.dumps(diff))
json.dump(T,open("ps2.json","w"))
