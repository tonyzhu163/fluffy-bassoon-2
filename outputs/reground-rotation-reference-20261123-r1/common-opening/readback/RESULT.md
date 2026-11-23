# Independent CPU readback of the shared-opening sensitivity

**The two preassigned shards cover every original opening disagreement.** At 17:55:46 Shanghai on November 23, the Baige reporter finished all 90 released ReGround shards and their 1,400 paired assignments. The three arms agreed bit for bit on every initial 25-component robot state. Seven assignments differed in the first ten proposed actions, all in seed 0 shard 004 (`10_3`, starts 44–49) or seed 1 shard 015 (`spatial_0`, start 48). Additional shard requests numbered zero. All four supplemental producer jobs exited successfully. [Final request ledger](REQUESTS.json) · [Producer receipts](shard-receipts.json)

| Original arm pair | Identical starts | Identical first ten proposals | Pass registered opening check |
| --- | ---: | ---: | ---: |
| SymMeta / Base | 1,400 | 1,393 | 1,393 |
| ReGround / Base | 1,400 | 1,393 | 1,393 |
| SymMeta / ReGround | 1,400 | 1,395 | 1,395 |

Every assignment supplied ten proposals. Every discordant pair differed at all ten control indices. The registered SymMeta/Base check also compares the first 11 robot states, ten issued actions and ten speeds. The reporter loaded the two comparison functions directly from the verified revision of `native_grid.py`. [Original case rows](original-openings.jsonl.gz) · [Predicate and source revisions](provenance.json)

## Outcomes on the 32 controlled assignments

Each cell gives the original count → shared-opening count on the same assignments. The 25 companions had matching original proposals across all three arms.

| Original opening | Assignments | Base success | SymMeta success | ReGround success | Base unsafe | SymMeta unsafe | ReGround unsafe |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Discordant | 7 | 0 → 0 | 1 → 1 | 6 → 6 | 7 → 7 | 1 → 1 | 3 → 3 |
| Concordant companions | 25 | 0 → 0 | 12 → 11 | 25 → 25 | 24 → 24 | 3 → 3 | 6 → 6 |
| **Whole selected shards** | **32** | **0 → 0** | **13 → 12** | **31 → 31** | **31 → 31** | **4 → 4** | **9 → 9** |

ReGround lost its original success on drawer start 44 and gained one on start 45. SymMeta lost its original success on companion `goal_1` start 21, where the original and controlled full 50-action opening proposals matched. Its first different proposed and issued action came at zero-based control 20. Companion start 24 also diverged for SymMeta, at control 130, and succeeded both times. All 25 companion trajectories matched their originals across all five saved arrays for Base and ReGround, as did 23 of 25 for SymMeta. The case rows also identify differences in the CSV controller-deviation diagnostic where the saved trajectories agree. All 96 native unsafe flags reproduced their original values assignment by assignment. [Paired case rows and first differing indices](controlled-cases.jsonl.gz) · [Counts by shard and subset](summary.json)

The controlled counts cover the complete proposal-selected subset. The changed ReGround successes show sensitivity at individual drawer starts. The SymMeta loss on a concordant companion shows that later rollout variation can also change success. Fresh worker resets and a shared first proposal were applied together, so these reruns measure their combined effect.

## What the independent checks establish

- **Full opening and reset.** All eight raw batch files reproduced the SHA-256 recorded by capture and each of the three scored arms. Readback independently hashed each of the 32 float32 50×7 lane proposals. Every digest matched the controlled SymMeta full-query log, and the first ten vectors matched the Base, SymMeta and ReGround traces for all 96 episodes. Base and SymMeta also passed the full registered state/action/speed prefix check on all 32 assignments. [Case evidence](controlled-cases.jsonl.gz) · [Batch and arm hashes](shard-receipts.json)
- **Pose, cameras, state and noise.** The saved public end-effector pose, jaw and joints reconstructed every fresh 25-vector. Each matched the opening state in all three original and all three controlled traces. Public EEF position and jaw also matched both streaming calibration records at their saved precision. Readback hashed the raw agent and wrist images, all nine public observation leaves and all eight simulator-state leaves per lane. Each scored arm's receipt binds its live reset checks to that same tape. An independent CPU generator reproduced every float in all 32 saved 50×32 noise tensors from the registered nonces and chunk seeds. [Verification code](controlled.py) · [CPU noise generator](noise.cc)
- **Historical evidence.** For 26 of 32 assignments, the original streaming agent-view image and full 50×7 proposal matched the fresh capture. Both differed on all six drawer starts. The original Base and ReGround stores retained the initial robot state and traced proposals. Independent historical wrist images and full simulator snapshots are unavailable. Across the fresh arms, the raw inputs are retained once in capture, and per-arm receipts record the live comparisons. [Original-to-controlled evidence](controlled-cases.jsonl.gz)
- **Native outcomes.** The audit checked CSV keys, native termination, lengths and shapes of all five trace arrays. All 43 controlled successes ended before their task horizon, and all 53 failures reached it. The success field is the native task predicate recorded by the evaluator. The archive contains one native success flag per episode. Per-step rewards were not archived. Readback recomputed speed violations from every saved speed sample using the released π₀.₅ limit and reproduced every unsafe flag from those violations and the recorded displacement, fall and non-target contact counts. The fitted arm's four weight-file hashes matched each seed's model files. [Source and runtime provenance](provenance.json) · [Model receipts](shard-receipts.json)

The reporter ran on one Baige CPU core and the controlled verification on a second core, with zero audit GPUs. It polled unpublished shard paths at nine-second intervals, cached each sealed original triple once and atomically updated the local request file. Only selected episodes with differing full-trajectory digests needed a targeted second read to locate the first changed sample. [Incremental reporter](reporter.py) · [Local request publisher](receive.py) · [Comparison code](summarize.py)
