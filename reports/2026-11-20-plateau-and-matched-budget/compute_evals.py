import numpy as np, pandas as pd, json, glob
R="/iris/u/ankile/karenvo/libero/runs/"
def L(p):
    g=sorted(glob.glob(R+p)); assert len(g)>=1,(p,g)
    return pd.read_csv(g[-1]+"/episodes.csv",dtype={"goals_end":str}).set_index(["init_state","repeat"])
A={"base":"eval_base_noise1_libero_10_18355478","r0":"eval_ag_residual_vb40_s0_shift099_noise1_libero_10_18438187",
 **{f"r{i}":f"eval_agr_r{i}_s0_shift099_noise1_libero_10_*" for i in range(1,8)},
 "r5n2":"eval_agr_r5_s0_shift099_noise2_*","r7n2":"eval_agr_r7_s0_shift099_noise2_*","r5cap700":"eval_agr_r5_s0_shift099_noise1_cap700_*",
 **{f"r{r}p{p}":f"eval_agr_r{r}_s0_shift{p}_noise1_*" for r in (5,6,7) for p in ("06","08","12","15","18") },
 **{f"r{r}w{w}":f"eval_agr_r{r}_s0_w{w}_noise1_*" for r in (5,6,7) for w in (1,2,4)},
 **{f"anc_r{r}":f"eval_anchor04_r{r}_shift099_*" for r in (0,5,7)},
 "unc_std":"eval_agr_r7_uncond_s0_std_*", **{f"unc_w{w}":f"eval_agr_r7_uncond_s0_uncond_w{w}_*" for w in (0,2,4)}, "unc_w8":"eval_agr_r7_uncond_s0_uncond_shift099_*",
 **{f"stk_r{r}":f"eval_agr_stk1_on_r{r}_*" for r in (0,5,6)},
 "rs6":"eval_agr_rs6_r6_*","pmix6":"eval_agr_pmix_r6_*","cp15_6":"eval_agr_cp15_r6_*","nz2_6":"eval_agr_nz2_r6_*",
 "iqlr7":"eval_agr_iqlr7_*","r7vb15":"eval_agr_r7_vb15_*","r7vb10":"eval_agr_r7_vb10_*",
 "nd900":"eval_agr_nd900_r0_*","ndb50":"eval_agr_ndb50_r0_*",
 "b50cv_r0":"eval_agr_b50cv_r0_s0_shift099_noise1_libero_10_18554641","b100cv_r0":"eval_agr_b100cv_r0_*","b200cv_r0":"eval_agr_b200cv_r0_*",
 "b100_r0":"eval_agr_b100_r0_*","b100_r1":"eval_agr_b100_r1_*","b100_r2":"eval_agr_b100_r2_s0_shift099_noise1_libero_10_18571091","b100_r2b":"eval_agr_b100_r2_s0_shift099_noise1_libero_10_18596273",
 "b100iql_r0":"eval_agr_b100iql_r0_*","b100iql_r1":"eval_agr_b100iql_r1_*","mt_r0":"eval_agr_mt100cv_r0_*","mt_r1":"eval_agr_mt100cv_r1_*",
 "agjudge":"eval_ag_residual_judge_sym25_*","dj400":"eval_dsrl_judge_sym25_s0_agent400_*","dj600":"eval_dsrl_judge_sym25_s0_agent600_*",
 "d400":"eval_dsrl_s0_agent400_*","d600":"eval_dsrl_s0_agent600_*",
 **{f"d{n}":f"eval_dsrl_s0_agent{n}_*" for n in [100,200,800,900,1000,1100,1200,1300]},"d1357":"eval_dsrl_s0_ep1800_*",
 "s1_1300":"eval_dsrl_s1_agent1300_*","s1_1800":"eval_dsrl_s1_agent1800_*",**{f"s1_{n}":f"eval_dsrl_s1x_agent{n}_*" for n in (2400,3000,3300)},
 **{f"dist_{k}":f"eval_distill_{k}_sampled_s0_atfinal_*" for k in ("base","hybrid","dsrl","ag")},"dist_base_exec":"eval_distill_base_executed_s0_atfinal_*",
 **{f"dist2k_{k}":f"eval_distill_{k}_sampled_s0_at2000_*" for k in ("base","hybrid","dsrl","ag")},
}
D={}
for k,p in A.items():
    try: D[k]=L(p)
    except AssertionError: print("missing",k)
rng=np.random.default_rng(0)
def pr(a,b):
    i=D[a].index.intersection(D[b].index); d=(D[a].loc[i].success-D[b].loc[i].success).values.astype(float)
    m=d[rng.integers(0,len(d),(5000,len(d)))].mean(1); return [round(d.mean(),3),round(np.percentile(m,2.5),3),round(np.percentile(m,97.5),3),len(d)]
def wil(k,n,z=1.96):
    p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d;return [round(c-h,3),round(c+h,3)]
S={k:dict(p=round(D[k].success.mean(),3),n=len(D[k]),k=int(D[k].success.sum()),ci=wil(int(D[k].success.sum()),len(D[k]))) for k in D}
for k,v in S.items(): print(k,v)
P=[("r5n2","r5"),("r7n2","r7"),("r5cap700","r5"),
   ("anc_r0","r0"),("anc_r5","r5"),("anc_r7","r7"),("unc_w0","unc_std"),("unc_w2","unc_std"),("unc_w4","unc_std"),("unc_w8","unc_std"),("unc_std","r7"),
   ("stk_r0","r0"),("stk_r5","r5"),("stk_r6","r6"),("rs6","r6"),("pmix6","r6"),("cp15_6","r6"),("nz2_6","r6"),("pmix6","rs6"),
   ("iqlr7","r7"),("r7vb15","r7"),("r7vb10","r7"),("nd900","r0"),("ndb50","b50cv_r0"),("b100iql_r0","b100_r0"),
   ("b100_r1","b100_r0"),("b100_r2","b100_r0"),("b100_r2b","b100_r2"),("agjudge","r0"),("dj400","d400"),("dj600","d600"),
   ("dist_dsrl","base"),("dist_hybrid","base"),("dist_base","base"),("dist_base_exec","base"),("dist_ag","base"),("dist_ag","r5"),("dist_dsrl","s1_1800"),
   ("s1_2400","r5"),("s1_3300","s1_1800"),("s1_2400","s1_1800")]
PP={f"{a}|{b}":pr(a,b) for a,b in P if a in D and b in D}
for k,v in PP.items(): print(k,v)
for r in (5,6,7):
    for p in ("06","08","12","15","18"):
        k=f"r{r}p{p}"
        if k in D: PP[f"{k}|r{r}"]=pr(k,f"r{r}")
    for w in (1,2,4):
        if f"r{r}w{w}" in D: PP[f"r{r}w{w}|r{r}"]=pr(f"r{r}w{w}",f"r{r}")
json.dump(dict(S=S,P=PP),open("r4.json","w"))
