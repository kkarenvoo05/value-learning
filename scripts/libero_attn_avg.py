"""Grid-test confound check [ours]: the grid run changed both the features (4 x 4 cell means) and the head
(vl.libero_probe.AttnProbe). This runs the same attention head on the averaged features, so only the head changes:
  avg1  attention over one token, the all-token mean (same input as the original MLP probe)
  avg4  attention over the four means (all, agentview, wrist, text)
Everything else is vl.libero_probe unchanged (same split, early stopping, outcome+clock head).
Usage: .venv/bin/python scripts/libero_attn_avg.py --features <feats_every4_*.npz> --pooling avg1 avg4 --heads attn
"""

import vl.libero_probe as lp

_orig = lp.features_for


def features_for(F, pooling):
    if pooling == "avg1":
        return F[:, :1]
    if pooling == "avg4":
        return F
    return _orig(F, pooling)


lp.features_for = features_for

if __name__ == "__main__":
    lp.main()
