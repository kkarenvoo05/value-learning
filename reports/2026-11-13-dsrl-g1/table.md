G1 with DSRL's critic: within-state ranking of candidate noise vectors at held-out branch points (val states), DSRL-policy trunk and continuations

## candidates: actor (actor)
50 points x 8 candidates x 4 continuations; mean candidate success 0.87; p varies at 21 points; ceiling corr ~ 0.64 (var_between 0.0090 / var_observed 0.0220)

| score | within-state Spearman | best-of-N pick gain | Q spread within state / across states |
|---|---|---|---|
| q:final | +0.039 [-0.079, +0.183] | +0.005 [-0.013, +0.022] | 0.00 |
| qmin:final | -0.128 [-0.318, +0.100] | -0.005 [-0.029, +0.018] | 0.01 |
| logp:final | +0.080 [-0.030, +0.228] | -0.005 [-0.033, +0.033] | 0.26 |
| q:ep1000 | +0.085 [-0.024, +0.258] | +0.015 [-0.016, +0.041] | 0.00 |
| qmin:ep1000 | +0.024 [-0.114, +0.164] | +0.000 [-0.049, +0.046] | 0.01 |
| logp:ep1000 | -0.014 [-0.129, +0.098] | +0.010 [-0.014, +0.032] | 0.13 |
| q:ep500 | -0.032 [-0.161, +0.162] | +0.020 [-0.037, +0.072] | 0.00 |
| qmin:ep500 | +0.057 [-0.167, +0.286] | +0.005 [-0.033, +0.054] | 0.01 |
| logp:ep500 | +0.006 [-0.089, +0.126] | +0.005 [-0.022, +0.029] | 0.12 |
| q:ep100 | +0.110 [+0.057, +0.218] | +0.030 [+0.001, +0.067] | 0.01 |
| qmin:ep100 | +0.054 [-0.095, +0.229] | +0.010 [-0.007, +0.028] | 0.02 |
| logp:ep100 | +0.083 [-0.114, +0.304] | +0.025 [-0.003, +0.052] | 0.18 |
| random | -0.115 [-0.271, +0.067] | -0.005 [-0.036, +0.022] | nan |

gate (q:final within-state Spearman CI above 0): **not passed**

## candidates: gauss (gauss)
50 points x 8 candidates x 4 continuations; mean candidate success 0.93; p varies at 23 points; ceiling corr ~ 0.55 (var_between 0.0054 / var_observed 0.0178)

| score | within-state Spearman | best-of-N pick gain | Q spread within state / across states |
|---|---|---|---|
| q:final | +0.040 [-0.064, +0.160] | -0.007 [-0.059, +0.034] | 0.01 |
| qmin:final | -0.004 [-0.199, +0.185] | -0.002 [-0.043, +0.028] | 0.02 |
| logp:final | +0.004 [-0.142, +0.132] | -0.012 [-0.037, +0.009] | 0.30 |
| q:ep1000 | -0.024 [-0.160, +0.115] | +0.023 [-0.001, +0.048] | 0.01 |
| qmin:ep1000 | -0.054 [-0.162, +0.024] | +0.008 [-0.016, +0.032] | 0.02 |
| logp:ep1000 | -0.007 [-0.215, +0.148] | -0.007 [-0.031, +0.014] | 0.28 |
| q:ep500 | -0.118 [-0.281, +0.056] | -0.027 [-0.062, +0.003] | 0.01 |
| qmin:ep500 | -0.069 [-0.258, +0.077] | -0.012 [-0.043, +0.019] | 0.02 |
| logp:ep500 | -0.099 [-0.278, +0.016] | -0.017 [-0.048, +0.006] | 0.35 |
| q:ep100 | +0.037 [-0.125, +0.203] | -0.017 [-0.059, +0.017] | 0.02 |
| qmin:ep100 | +0.013 [-0.145, +0.157] | -0.012 [-0.048, +0.019] | 0.03 |
| logp:ep100 | -0.017 [-0.241, +0.169] | -0.002 [-0.039, +0.029] | 0.58 |
| random | +0.058 [-0.111, +0.248] | -0.002 [-0.039, +0.024] | nan |

gate (q:final within-state Spearman CI above 0): **not passed**

