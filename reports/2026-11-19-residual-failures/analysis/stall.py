import pandas as pd, numpy as np, sys
from concurrent.futures import ProcessPoolExecutor
GP=np.array([-0.021,0.002,0.066])
def stall(args):
    raw, t0, lift = args
    z=np.load(raw); op=z['object_pos']; g=z['goals'][:,:2].astype(bool); st=z['state']; a=z['actions']
    on=[g[:,p].argmax() if g[:,p].any() else 10**9 for p in (0,1)]; k=int(np.argmax(on))  # second = placed later / never
    h=op[:,6*k+3:6*k+6]-GP; d=np.linalg.norm(h,axis=1); T=len(g)
    end=T if np.isnan(lift) else int(t0+lift)
    seg=np.arange(int(t0),end)
    if len(seg)<30: return dict(stall_steps=0, low_stall=0)
    sp=np.r_[np.linalg.norm(np.diff(st[seg,:3],axis=0),axis=1)*100,0]
    m=(d[seg]<0.08)&(sp<0.2)          # within 8 cm of the grasp pose and nearly still
    low=m&(h[seg,2]>0.015)            # ... with the hand >= 1.5 cm below the grasp pose
    return dict(stall_steps=int(m.sum()), low_stall=int(low.sum()))
if __name__=='__main__':
    f=pd.read_parquet(sys.argv[1]); f=f[(f.split=='train')&f.first_step.notna()&(f.n_placed>=1)].copy()
    with ProcessPoolExecutor(16) as ex: r=list(ex.map(stall, zip(f.raw,f.first_step,f.second_lift_after),chunksize=8))
    f=pd.concat([f.reset_index(drop=True),pd.DataFrame(r)],axis=1); f.to_parquet(sys.argv[2])
