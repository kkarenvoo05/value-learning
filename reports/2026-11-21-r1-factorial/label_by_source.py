import pandas as pd, sys, glob, os
T='/iris/u/ankile/karenvo/libero/tables'
specs=[('900 r1','moka_agr_r1','agr_vb40_r1'),('900 r7','moka_agr_r7','agr_vb40_r7'),
       ('b100 r1','moka_agr_b100_r1','agr_b100_vb40_r1'),('b100iql r1','moka_agr_b100iql_r1','agr_b100iql_iql40_r1'),
       ('b100cv r1','moka_agr_b100cv_r1','agr_b100cv_cv40_r1'),('b50cv r1','moka_agr_b50cv_r1','agr_b50cv_cv40_r1'),
       ('mt100cv r1','mt_agr_mt100cv_r1','agr_mt100cv_cv40_r1')]
rows=[]
for name,t,l in specs:
    ep=pd.read_csv(f'{T}/{t}/episodes.csv'); lab=pd.read_parquet(f'{T}/{t}/labels/{l}/labels.parquet').drop(columns=['round'],errors='ignore')
    d=lab.merge(ep,on='episode_index'); d=d[d.split=='train']
    d['src']=(d['round']>1).map({False:'base',True:'pushed'})
    grp=['task'] if 'task' in d and d.task.nunique()>1 else []
    for keys,g in d.groupby(grp+['src']) if grp else d.groupby('src'):
        e=ep[(ep.split=='train')&(ep.episode_index.isin(g.episode_index))]
        r=dict(chain=name, task=(keys[0] if grp else ''), src=keys[-1] if grp else keys,
               eps=g.episode_index.nunique(), frame_share=len(g)/len(d if not grp else d[d.task==keys[0]]),
               ep_success=e.success.mean(), pos=g.positive.mean(),
               pos_succ=g[g.success==1].positive.mean(), pos_fail=g[g.success==0].positive.mean(), neg_from_succ=(g[(g.positive==0)].success==1).mean(), pos_from_succ=(g[g.positive==1].success==1).mean(),
               A=g.A.mean(), V=g.V.mean())
        rows.append(r)
pd.set_option('display.width',250)
print(pd.DataFrame(rows).round(3).to_string(index=False))
