# Frozen native anchor OOD controls

Each point is success in query episode j following j-1 actual completed episodes. Both frozen native controls lack learned history conditioning; these curves are controls, not adaptation evidence. Cumulative ever-success remains separate. Paired contrasts are descriptive and do not admit source policies or choose checkpoints.

Counts are successes out of50 in the actual query; columns denote prior completed episodes.

| Task / frozen control | 0 | 1 | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- | --- |
| CloseFridge / original SFT | 33 | 32 | 36 | 34 | 36 | 36 |
| CloseFridge / fixed130 PPO | 31 | 33 | 34 | 39 | 35 | 41 |
| TurnOnMicrowave / original SFT | 6 | 9 | 11 | 5 | 9 | 8 |
| TurnOnMicrowave / fixed130 PPO | 6 | 12 | 15 | 10 | 11 | 9 |

Both groups passed independent audits of every raw draw, mapped physical command, feedback, native success receipt and decoded video frame. Full paired descriptive statistics and hashed provenance are in the sibling JSON.
