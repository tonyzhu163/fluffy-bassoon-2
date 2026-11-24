# Native arm transfer on two Panda fitting tasks

**All 40 episodes succeeded.** On Kinova3, stock, policy-only, adapter-only, joint and joint with history zeroed each earned native return 1 on all four starts. Each also succeeded on the same four starts on Panda, preserving source-arm performance on this panel. Stock reached the reward ceiling on both arms. Joint training produced **zero rescues and zero new failures** against each comparator. The joint policy did respond to execution history, including when observations, weights, diffusion noise and sampler implementation were held fixed. That response yielded no observed gain in terminal reward. [Preregistration](PLAN.md) · [Verified endpoints, interventions and original artifact hashes](evidence/native-gpu01.json)

## Paired terminal outcomes

Each cell gives issued controls **to native success**. All 40 reward tapes contain zeros followed by one terminal reward of 1. Every episode terminated on success before its official horizon. Each row fixes the robot, task, released layout, reset seed and diffusion bank across all five settings. The two robots share the released scene and noise for each task/layout. [Original episode tapes and job summary](evidence/native-gpu01-tapes.tar.gz)

| Arm | Task and released row | Stock | Policy only | Adapter only | Joint | Joint, zero history |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Kinova3 | Object 48 | 195 | 209 | 196 | 260 | 225 |
| Panda | Object 48 | 146 | 147 | 141 | 147 | 137 |
| Kinova3 | Object 49 | 141 | 145 | 152 | 147 | 146 |
| Panda | Object 49 | 133 | 143 | 135 | 133 | 133 |
| Kinova3 | Spatial 48 | 106 | 180 | 105 | 105 | 105 |
| Panda | Spatial 48 | 104 | 103 | 104 | 104 | 104 |
| Kinova3 | Spatial 49 | 79 | 76 | 78 | 79 | 80 |
| Panda | Spatial 49 | 69 | 69 | 68 | 68 | 70 |

Object 0 asks the robot to place alphabet soup in the basket. Spatial 0 asks for the black bowl between the plate and ramekin to be placed on the plate. Both tasks appeared in the actual Panda fitting draws. The four evaluation layouts differed from all 400 original demonstration starts across the eight source tasks. The Object and Spatial horizons were 280 and 220 controls, respectively. Timing varied within successful episodes. On Kinova3 Object 48, for example, joint used 65 more controls than stock and 35 more than joint with zero history. [Fitting draw identities](evidence/fit-inputs.json) · [Scene comparison](evidence/scene-comparisons.json)

For **each** joint-minus-comparator comparison (stock, policy only, adapter only or zero history), there were four paired successes on Kinova3 and four on Panda. The success-rate difference and mean-return difference were zero on each arm. The Kinova3-minus-Panda difference in these treatment effects was zero on each of Object 48, Object 49, Spatial 48 and Spatial 49. Dropping any one shared start left all three mean effects at zero. This leave-one-start-out check describes sensitivity within four selected starts. With two familiar tasks, two layouts per task, one noise bank per layout and no stock failures, these data give little information about rescue rates on more difficult starts. [Per-comparator paired estimates and leave-one-out values](evidence/native-gpu01.json)

## History and action divergence

On **each of the eight arm/start cases**, joint first differed from stock, policy-only and adapter-only at the opening proposal and at issued control 0. Joint and joint-zero had identical openings and first differed in both proposal and issued command at control 5, when the first completed replan entered history. Paired observations matched through the replan that produced each first differing command. Indices start at zero.

At control 10 of each joint trajectory, the runner recomputed two unissued proposals on the **same observation and noise with the same joint weights**. The table gives the maximum absolute coordinate change among the first five proposed actions, in normalized action units. The lagged input was the preceding replan's nonempty history. Both aligned and lagged inputs used the same compiled history sampler, and changing their temporal content changed the proposal on **8/8 cases**.

| Task and row | Kinova3, zeroed | Kinova3, lagged | Panda, zeroed | Panda, lagged |
| --- | ---: | ---: | ---: | ---: |
| Object 48 | 0.005721 | 0.002005 | 0.005096 | 0.002898 |
| Object 49 | 0.004635 | 0.001308 | 0.005829 | 0.003126 |
| Spatial 48 | 0.006298 | 0.002053 | 0.006263 | 0.002128 |
| Spatial 49 | 0.006900 | 0.001748 | 0.007525 | 0.002044 |

The terminal joint-versus-zero contrast tests the **deployed program's response to masking all 52 history entries**. The recovered sampler routes an all-zero history through its plain compiled path and a nonempty history through its offset path. The zeroed column therefore includes any numerical effect of switching paths. The lagged column isolates sensitivity to history on one path. Those lagged proposals were never issued, so their terminal consequences are unknown. The history input includes earlier commands, observed motion and jaw-width information together. The data do not isolate arm dynamics as the cause of the response. [Per-case first divergences, input hashes and counterfactual proposal changes](evidence/native-gpu01.json)

## Execution and integrity

Baige Distributed Training job `job-8la2t917zkhi` exited 0 on 2026-11-24 at **12:13:29 Shanghai**. The recorded job step ran 741 seconds on one RTX 4090, or 0.206 GPU-hours. Its in-job probe passed at 12:02:07. At 12:08:07 the first terminal tape verified against the registered Kinova3 Object row 48, seed 42820048, CPU fixture scene, released horizon, noise bank and NVIDIA renderer. All 40 resets were accepted on the first attempt. Stock and the independently materialized no-op kernel gave bitwise-identical opening chunks in all eight cases. [Monitoring record](evidence/native-gpu01-monitor.tsv) · [Preregistered starts and seeds](contract.json)

The native executable was `c51e767770bc48827a1d6d3837cddefe1b7a1d6c`, with the sealed `model/gpu02` fit from `29e9e5e5209db26ce0cbf695387c64cdf79b98fa`. Its receipt SHA-256 was `4855546591381f149d7509a6af3d45db32e9e82a732788c8376d504f19417dbc`. Before and after evaluation the frozen receipts were byte-identical, both SHA-256 `24557d1397635f49a9c6508993bc60197ed679aeb27b97f82212702955dc13e9`. They cover the resident stock parameters, fitted head arrays, effective kernels, normalization and on-disk fit. The installed kernels and fitted arrays also matched the trainer's receipt.

A separate two-core CPU readback reconstructed all 5,067 controls from the 40 trajectories, verified the issued actions, rewards, causal histories, noise and terminal flags, and reproduced the saved native summary (SHA-256 `5845a7f8eff4464f946ef7007e7bd7cd10c3442526df71bb05038fd09700ae5c`). The original tapes remain under `/mowenjie/fluffy-bassoon-2/outputs/frozen-vla-cotraining-20261124-r1/matched-task/gpu01`. All 46 files in the portable archive were checked against their original digests. Its SHA-256 is `819d237b91504b845ff385fafe95e9d66059dd6bb7609510178701ca8dde0b3f`. The compact readback includes SHA-256s for all 40 original episode files and 14 run, fit and source artifacts. [Readback implementation](native_readback.py) · [Readback and per-file hashes](evidence/native-gpu01.json) · [Portable archive](evidence/native-gpu01-tapes.tar.gz)
