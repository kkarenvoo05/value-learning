import pandas as pd, numpy as np, sys
from concurrent.futures import ProcessPoolExecutor
GP=np.array([-0.021,0.002,0.066])
def phases(args):
    raw, t0, lift, T = args
    z=np.load(raw); op=z['object_pos']; g=z['goals'][:,:2].astype(bool); st=z['state']
    on=[g[:,p].argmax() if g[:,p].any() else 10**9 for p in (0,1)]; k=int(np.argmax(on))
    h=op[:,6*k+3:6*k+6]-GP; d=np.linalg.norm(h,axis=1); n=len(g)
    sp=np.r_[np.linalg.norm(np.diff(st[:,:3],axis=0),axis=1)*100,0]
    ph=np.zeros(n,int)                       # 0 = first pot
    if not np.isnan(t0):
        t0=int(t0); ph[t0:]=1                # 1 = second-pot approach / grasp
        if not np.isnan(lift): ph[t0+int(lift):]=2   # 2 = carrying / placing the second pot
    stall=(ph==1)&(d<0.08)&(sp<0.2)&(h[:,2]>0.015)
    return ph, stall
if __name__=='__main__':
    f=pd.read_parquet(sys.argv[1]); f=f[f.split=='train']
    L=pd.read_parquet('/iris/u/ankile/karenvo/libero/tables/moka_agr_r7/labels/agr_vb40_r7/labels.parquet')
    with ProcessPoolExecutor(16) as ex: P=list(ex.map(phases, zip(f.raw,f.first_step,f.second_lift_after,f['T']),chunksize=8))
    rows=[]
    for (ph,stl),r in zip(P,f.itertuples()):
        rows.append(pd.DataFrame({'episode_index':r.episode_index,'frame_index':np.arange(len(ph)),'phase':ph,'stall':stl}))
    F=pd.concat(rows); M=L.merge(F,on=['episode_index','frame_index']).merge(f[['episode_index','success','round']],on='episode_index')
    M.to_parquet(sys.argv[2]); print(len(L), len(M))
