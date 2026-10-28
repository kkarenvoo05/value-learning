"""vl.c2a_visual: the counterfactual sample respects the groups and spreads over episodes."""

import numpy as np
import pandas as pd

from vl.c2a_visual import GROUPS, sample


def test_sample_takes_each_group_and_round_robins_over_episodes():
    rows = []
    for name, (g, still) in GROUPS.items():
        for ep in range(3):
            n = 20 if ep == 0 else 2                    # one episode dominates each group
            rows += [{"group": g, "still >= 4 s now": still, "episode_uid": f"{name}-{ep}", "frame_index": k}
                     for k in range(n)]
    h = pd.DataFrame(rows)
    s = sample(h, 6, np.random.default_rng(0))
    assert list(s.cf_group.unique()) == list(GROUPS)
    for name, (g, still) in GROUPS.items():
        d = s[s.cf_group == name]
        assert len(d) == 6 and ((d.group == g) & (d["still >= 4 s now"] == still)).all()
        assert d.episode_uid.value_counts().max() == 2   # 3 episodes x 2 before any takes a third
