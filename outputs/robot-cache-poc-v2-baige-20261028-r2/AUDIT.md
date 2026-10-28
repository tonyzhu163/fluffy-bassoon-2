# Cached robot A1 v2 audit

Checked 2026-10-28T18:02:30+08:00; Baige `job-61p1rncjnr1x`.

Source `c3e9cca68b93b6e777cbe5314e130bcaf85c8e14`; cache SHA256 `74910dc755b959fe86ec0a3d4cf1c05957ba57510c64bbb1bb2c85ba56e41575`.

Each seed targets 500 updates with a fixed 900-second cap. Scores use 256 paired validation trials.

| Run | Status | Updates | Elapsed s | Wall s/update | Greedy | Wiped | Reward blind | Sampled |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| persistent_ctm-seed0 | clean_budget_stop | 353 | 904.4 | 2.562 | 56.64% | 49.95% | 37.65% | 55.03% |
| persistent_ctm-seed1 | clean_budget_stop | 350 | 902.1 | 2.577 | 57.67% | 50.00% | 37.06% | 56.54% |
| persistent_ctm-seed2 | clean_budget_stop | 358 | 903.7 | 2.524 | 55.27% | 46.34% | 36.38% | 55.66% |
| transformer_xl-seed0 | completed | 500 | 246.5 | 0.493 | 53.61% | 42.63% | 37.45% | 52.93% |
| transformer_xl-seed1 | completed | 500 | 243.4 | 0.487 | 57.57% | 45.46% | 36.47% | 57.23% |
| transformer_xl-seed2 | completed | 500 | 245.9 | 0.492 | 53.96% | 47.17% | 38.57% | 54.05% |
| gru-seed0 | completed | 500 | 65.2 | 0.130 | 51.56% | 42.38% | 35.99% | 51.12% |
| gru-seed1 | completed | 500 | 63.6 | 0.127 | 55.18% | 42.38% | 37.16% | 55.13% |
| gru-seed2 | completed | 500 | 66.5 | 0.133 | 56.05% | 42.38% | 35.94% | 55.27% |

Paired references: zero 42.38%, oracle 74.76%, thompson 45.36%.

| Core | Final evaluated seeds | Completed 500 | Achieved updates | Greedy mean ± seed SD | Sampled mean | Mean elapsed s | Wall s/update |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| persistent_ctm | 3 | 0 | 353,350,358 | 56.53% ± 1.20 pp SD | 55.75% | 903.4 | 2.555 |
| transformer_xl | 3 | 3 | 500,500,500 | 55.05% ± 2.19 pp SD | 54.74% | 245.3 | 0.491 |
| gru | 3 | 3 | 500,500,500 | 54.26% ± 2.38 pp SD | 53.84% | 65.1 | 0.130 |

CTM seed 0 stopped cleanly after update 353. The atomic checkpoint file exists (1,281,837 bytes), has no remaining `.partial` file, and its SHA256 matches the saved summary: `bf8c6ba1fcccf6f9cc2219d7616e040b063ad1f8385e42a4f476dffb699c083c`. It was not deserialized on the development host. That checkpoint verification was recorded at 17:01; the current table now includes final evaluations for all three CTM seeds.

The mirror contains configurations, saved summaries, baseline arrays and initial/final evaluation arrays; checkpoints and large action traces remain on the server.

This is a cached-table plumbing result. Different achieved updates, CTM tick cost and model budgets prevent a matched architecture claim. The visual pi0.5 bridge is a separate live experiment.

## Evidence limits

- Existing single-repeat headroom cache; no new simulator evidence.
- Same objects and angle support across training and validation; no held-out-object or held-out-angle claim.
- Training seeds are independent; paired conditions reuse fixed evaluation seeds.
- CTM T4 versus XL/GRU T1; width equality does not match parameters, state or compute.
- 900-second seed caps are immutable; compare achieved updates and completed flags.
- Wall seconds per update include evaluation/checkpoint overhead and concurrent resource contention.
