# Contact feedback on a held-out command and grasp combination

On the LIBERO-Pro2.0 Goal 2 bottle task, a frozen π0.5 motor succeeded on seven of eight fresh query poses with reversed X and the D17 grasp fault. A reader using contact from a previous attempt and a reader using contact from the current grasp chose the same corrections and had the same outcome on every pose. A ten-control free-space probe with the fitting-selected fixed grip succeeded on four. The preregistered criterion for a previous-attempt advantage was unmet. [Plan](PLAN.md) · [Native scores and paired outcomes](reader02-analysis.json)

| Information available to the controller | Held-out combination | All four conditions |
| --- | ---: | ---: |
| Free-space probe and fixed grip | 4/8 | 25/32 |
| Previous attempt, factored reader | 7/8 | 30/32 |
| Previous attempt, joint reader | 7/8 | 30/32 |
| Current attempt alone, live contact decision | 7/8 | 30/32 |
| Supplied command and grasp factors | 7/8 | 30/32 |

Each reader fit used the same three training conditions and eight source pose pairs per condition. Both previous-attempt readers used the same nine public features, 38 parameters and source supervision. For each of seeds 41, 42 and 43, both readers recovered both factors on all eight held-out pose pairs and all 32 evaluation assignments. A joint reader given these engineered command and motion features therefore generalized to the unseen pairing. These three fits share the same eight held-out query poses. [Factor accuracy by fit and condition](reader02-transfer.json) · [Reader interface](INTERFACE.md)

Compared with the free-space controller, using contact recovered query poses 36, 38 and 42 with zero losses on the held-out combination. Both grip choices failed on pose 39 with the correct command convention. Giving the factored reader a prior attempt from the opposite grasp condition reduced success to 4/8. Swapping the command convention reduced it to 0/8. Across all four conditions, the contact controller had six wins and one loss against the probe. The live current-only controller and the previous-attempt reader had zero discordant outcomes across all 32 assignments. [Per-pose comparisons](reader02-analysis.json) · [Native grasp choices](reader02-transfer.json)

With a common practice episode, six of eight held-out supports succeeded. The previous-attempt and current-only arms both succeeded at least once in 8/8 support–query pairs. Standalone current-only control used 1,575 physical controls across the eight queries. The previous-attempt reader used the same 1,575 plus 1,653 support controls. The standalone probe with fixed grip used 1,994 total. Each query had the same 300-control limit. [Success and interaction counts](reader02-analysis.json)

This pilot establishes when each factor became observable on one simulated task. Motion exposed the action sign in free space. Current grasp feedback arrived in time to choose a useful regrasp, leaving no measured benefit for prior contact or explicit factorization. A stronger memory test would require a task in which the useful action must be chosen before the current attempt reveals the fault. Different arm kinematics, camera mounts and tools remain separate transfer questions. [Motor qualification](../libero-command-grasp-20261119-r1/RESULT.md)

## Reproduction

The Baige run used 72 fitting episodes and 192 fresh evaluation episodes. Each evaluation assignment included one support, four independently executed static query settings and one live current-only query. Reader choices were scored against those native returns. A separate Linux CPU pass checked the tapes and reproduced the analysis byte for byte. The completed panel had zero differences in the paired initial camera checks or first post-probe model-input checks. The fitted weights were frozen before the first evaluation attempt and verified on reuse. [Execution and retry record](EXECUTION.md) · [Completion receipt](reader02-complete.json) · [Weights provenance](reader02-fit-source.json)
