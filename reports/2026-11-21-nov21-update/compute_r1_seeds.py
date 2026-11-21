# Round-1 seed / base-share comparisons for the small AG chains (mentor session 2026-11-21). Paired by start (init_state,repeat or layout,repeat), bootstrap 5000, rng 0. Run: uv run --no-project --with pandas --with numpy python compute_r1_seeds.py
import numpy as np, pandas as pd, json, glob, os
R="/iris/u/ankile/karenvo/libero/runs/"
def L(d):
    df=pd.read_csv(R+d+"/episodes.csv",dtype={"goals_end":str,"error":str})
    err=df.error.notna().sum() if "error" in df else 0
    if d.startswith("ceval"): df["k"]=df.source_raw.str.extract(r"(layout\d+)")[0]+"_"+df["repeat"].astype(str)
    else: df["k"]=df.init_state.astype(str)+"_"+df["repeat"].astype(str)
    return df.set_index("k"), err
E={ # chain: {variant: {split: dir}}
 "b100":{"r0":dict(val="eval_agr_b100_r0_s0_shift099_noise1_libero_10_18544460",moka="ceval_agr_b100_r0_s0_shift099_libero_10_18544461"),
         "s0":dict(val="eval_agr_b100_r1_s0_shift099_noise1_libero_10_18554181",moka="ceval_agr_b100_r1_s0_shift099_libero_10_18554640"),
         "s1":dict(val="eval_agr_b100_r1_s1_shift099_noise1_libero_10_18609329",moka="ceval_agr_b100_r1_s1_shift099_libero_10_18609330")},
 "b50cv":{"r0":dict(val="eval_agr_b50cv_r0_s0_shift099_noise1_libero_10_18554641",moka="ceval_agr_b50cv_r0_s0_shift099_libero_10_18546934"),
         "s0":dict(val="eval_agr_b50cv_r1_s0_shift099_noise1_libero_10_18581090",moka="ceval_agr_b50cv_r1_s0_shift099_libero_10_18581091"),
         "s1":dict(val="eval_agr_b50cv_r1_s1_shift099_noise1_libero_10_18615174",moka="ceval_agr_b50cv_r1_s1_shift099_libero_10_18615175")},
 "mt100cv":{"r0":dict(val="eval_agr_mt100cv_r0_s0_shift099_noise1_libero_10_18552646",moka="ceval_agr_mt100cv_r0_s0_shift099_libero_10_18552647",t64="ceval_agr_mt100cv_r0_s0_shift099_libero_90_18552648"),
         "s0":dict(val="eval_agr_mt100cv_r1_s0_shift099_noise1_libero_10_18604938",moka="ceval_agr_mt100cv_r1_s0_shift099_libero_10_18598986",t64="ceval_agr_mt100cv_r1_s0_shift099_libero_90_18598987"),
         "s1":dict(val="eval_agr_mt100cv_r1_s1_shift099_noise1_libero_10_18615178",moka="ceval_agr_mt100cv_r1_s1_shift099_libero_10_18615179",t64="ceval_agr_mt100cv_r1_s1_shift099_libero_90_18615180"),
         "os91":dict(val="eval_agr_mtos_r1_s0_shift099_noise1_libero_10_18605559",moka="ceval_agr_mtos_r1_s0_shift099_libero_10_18605560",t64="ceval_agr_mtos_r1_s0_shift099_libero_90_18605561"),
         "os77":dict(val="eval_agr_mtos77_r1_s0_shift099_noise1_libero_10_18610989",moka="ceval_agr_mtos77_r1_s0_shift099_libero_10_18610990",t64="ceval_agr_mtos77_r1_s0_shift099_libero_90_18610991")},
 "b100cv":{"r0":dict(val="eval_agr_b100cv_r0_s0_shift099_noise1_libero_10_18546938",moka="ceval_agr_b100cv_r0_s0_shift099_libero_10_18546939"),
         "s0":dict(val="eval_agr_b100cv_r1_s0_shift099_noise1_libero_10_18580609",moka="ceval_agr_b100cv_r1_s0_shift099_libero_10_18580611")},
}
BASE=dict(val="eval_base_noise1_libero_10_18355478",moka="ceval_base_libero_10_18562758",t64="ceval_base_t64_libero_90_18581021")
rng=np.random.default_rng(0)
def pr(a,b):
    i=a.index.intersection(b.index); d=(a.loc[i].success.values-b.loc[i].success.values).astype(float)
    m=d[rng.integers(0,len(d),(5000,len(d)))].mean(1); return [round(100*d.mean(),1),round(100*np.percentile(m,2.5),1),round(100*np.percentile(m,97.5),1),len(d)]
out={}
B={s:L(d)[0] for s,d in BASE.items()}
for c,V in E.items():
    for v,sp in V.items():
        for s,d in sp.items():
            df,err=L(d); r=dict(p=round(100*df.success.mean(),1),n=len(df),err=int(err))
            r["vs_base"]=pr(df,B[s])
            if v!="r0": r["vs_r0"]=pr(df,L(V["r0"][s])[0])
            if v=="s1": r["vs_s0"]=pr(df,L(V["s0"][s])[0]); r["s0_same"]=round(100*L(V["s0"][s])[0].loc[df.index.intersection(L(V["s0"][s])[0].index)].success.mean(),1)
            out[f"{c}|{v}|{s}"]=r; print(c,v,s,r)
for s,b in B.items(): out[f"base|{s}"]=dict(p=round(100*b.success.mean(),1),n=len(b))
json.dump(out,open("r1_seeds.json","w"),indent=0)
