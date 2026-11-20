# Per-episode failure / support features for every moka_agr_r7 training episode (base 900 + AG collections).
import sys, numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, '/move/u/karenvo/Projects/value-learning/scripts')
T = '/iris/u/ankile/karenvo/libero/tables/moka_agr_r7/'
OUT = sys.argv[1] if len(sys.argv) > 1 else None


def runs(mask, val):
    out, s = [], None
    for i, m in enumerate(mask):
        if m == val and s is None: s = i
        if m != val and s is not None: out.append((s, i)); s = None
    if s is not None: out.append((s, len(mask)))
    return out


def feat(raw):
    try:
        z = np.load(raw)
        if 'object_pos' not in z.files: return {'has_pos': 0}
        g = z['goals'][:, :2].astype(bool); op = z['object_pos']; a = z['actions']
    except Exception as ex:
        return {'has_pos': 0, 'err': str(ex)[:50]}
    T_ = len(g); pos = [op[:, 0:3], op[:, 6:9]]; GP = np.array([-0.021, 0.002, 0.066])  # pot - eef at a successful second lift (median, sd ~1 cm)
    hd = [np.linalg.norm(op[:, 3:6] - GP, axis=1), np.linalg.norm(op[:, 9:12] - GP, axis=1)]  # distance to the grasp pose
    grip = a[:T_, 6] > 0  # close command
    placed, lost, replaced = [None, None], [None, None], [None, None]
    for p in range(2):
        on = [r for r in runs(g[:, p], True) if r[1] - r[0] >= 10 or r[1] == T_]
        if on:
            placed[p] = on[0][0]
            off = [r for r in runs(g[:, p], False) if r[0] > placed[p] and r[1] - r[0] >= 20]
            if off:
                lost[p] = off[0][0]
                back = [r for r in on if r[0] > lost[p]]
                replaced[p] = back[0][0] if back else None
    order = sorted([p for p in range(2) if placed[p] is not None], key=lambda p: placed[p])
    first = order[0] if order else None
    second = (order[1] if len(order) > 1 else 1 - first) if first is not None else None
    r = {'has_pos': 1, 'T': T_, 'first_step': placed[first] if first is not None else np.nan,
         'n_placed': len(order), 'lost_any': int(any(l is not None for l in lost)),
         'recovered': int(any(lost[p] is not None and replaced[p] is not None for p in range(2)))}
    # tipped pot: height of a pot drops well below its start (lying on its side) -> z below start - 2 cm
    r['tipped'] = int(any((pos[p][:, 2] < pos[p][0, 2] - 0.02).any() for p in range(2)))
    if second is not None and not np.isnan(r['first_step']):
        t0 = int(r['first_step']); z0 = pos[second][0, 2]
        lift = np.where(pos[second][t0:, 2] - z0 > 0.03)[0]
        r['second_lift_after'] = int(lift[0]) if len(lift) else np.nan
        seg = slice(t0, t0 + (int(lift[0]) if len(lift) else T_ - t0))
        near = hd[second][seg] < 0.03
        r['hover_steps'] = int((near & ~grip[seg]).sum())           # hand at pot 2, gripper open, before lift
        r['near_steps'] = int(near.sum())
        cl = runs(grip[seg], True)
        r['grasp_closes'] = len(cl)                                 # close commands before the lift (>1 = retries)
        r['closes_near'] = sum(1 for c in cl if hd[second][seg][c[0]:c[1]].min() < 0.03)
        r['min_hand_pot2'] = float(hd[second][seg].min()) if len(hd[second][seg]) else np.nan
    return r


if __name__ == '__main__':
    e = pd.read_csv(T + 'episodes.csv')
    with ProcessPoolExecutor(16) as ex:
        rows = list(ex.map(feat, e.raw, chunksize=8))
    f = pd.concat([e.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    f.to_parquet(OUT)
    print(f.groupby('round').has_pos.mean())
