# Frozen native anchor ID controls

Each point is success in query episode j following j-1 actual completed episodes. Both frozen native controls lack learned history conditioning; these curves are controls, not adaptation evidence. Cumulative ever-success remains separate. Paired contrasts are descriptive and do not admit source policies or choose checkpoints.

Counts are successes out of50 in the actual query; columns denote prior completed episodes.

| Task / frozen control | 0 | 1 | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- | --- |
| OpenDrawer / original SFT | 35 | 32 | 31 | 28 | 30 | 34 |
| OpenDrawer / fixed130 PPO | 37 | 34 | 34 | 29 | 36 | 32 |
| CloseToasterOvenDoor / original SFT | 15 | 18 | 14 | 21 | 18 | 21 |
| CloseToasterOvenDoor / fixed130 PPO | 23 | 24 | 27 | 25 | 23 | 26 |
| CoffeeSetupMug / original SFT | 21 | 13 | 15 | 18 | 21 | 15 |
| CoffeeSetupMug / fixed130 PPO | 24 | 20 | 17 | 16 | 13 | 19 |

Both groups passed independent audits of every raw draw, mapped physical command, feedback, native success receipt and decoded video frame. Full paired descriptive statistics and hashed provenance are in the sibling JSON.
