# Native ReGround rotation panel at 15, 100 and 1,000 target transitions

**At 15 transitions, neither fit recovered a success. At 100, seed 3 recovered five. At 1,000, the two fits recovered 28 of 64 assignments, compared with 0 for frozen Base and 58 for the 10,000-transition ReGround control.** The 1,000-transition controller gained 28 successes over Base. Against the 10,000-transition controller it won two assignments and lost 32, a net deficit of 46.9 percentage points (task-stratified paired-start 95% interval 37.5–56.2 points for the deficit). [Assignment records](evidence/native-panel/assignments.csv) · [Intervals and pair counts](evidence/native-panel/summary.json)

## Success and unsafe incidence

The denominator for each seed is 32. Each combined rate uses 64 assignments on **32 distinct task–start pairs**, four per task across eight tasks. The same starts, replica and policy-noise keys recur across both seeds, all budgets and both archived controls. The table gives arithmetic assignment rates and task-stratified paired-start 95% intervals in brackets.

| Target transitions | Success, seeds 0 / 3 | Combined success | Success % [95% interval] | Unsafe, seeds 0 / 3 | Combined unsafe | Unsafe % [95% interval] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen Base | 0 / 0 | 0/64 | 0.0 [0.0, 0.0] | 22 / 22 | 44/64 | 68.8 [56.2, 81.2] |
| 15 | 0 / 0 | 0/64 | 0.0 [0.0, 0.0] | 6 / 0 | 6/64 | 9.4 [6.2, 12.5] |
| 100 | 0 / 5 | 5/64 | 7.8 [3.1, 12.5] | 0 / 20 | 20/64 | 31.2 [25.0, 37.5] |
| 1,000 | 9 / 19 | 28/64 | 43.8 [37.5, 50.0] | 18 / 15 | 33/64 | 51.6 [42.2, 60.9] |
| 10,000 | 32 / 26 | 58/64 | 90.6 [85.9, 95.3] | 15 / 15 | 30/64 | 46.9 [39.1, 54.7] |

All 64 budget-15 episodes reached their native time limit. The six unsafe episodes at that budget each crossed the speed limit, with zero contact, fallen-object or displacement flags. At 1,000, the controller was unsafe on 33 assignments versus 30 at 10,000. Fourteen pairs were unsafe only at 1,000, and eleven only at 10,000. The paired interval for this 4.7-point increase spans −7.8 to +17.2 points. A smaller budget therefore did not yield a consistent ordering of unsafe incidence across fits or controls. [Episode steps and unsafe predicates](evidence/native-panel/assignments.csv)

## Within-start comparisons

Differences are percentage points, with the first arm minus the second. Each discordance column counts assignments with the event only in the first arm / only in the second arm. For unsafe, a positive difference means more unsafe episodes in the first arm.

| Comparison | Success only, first / second | Success difference [95% interval] | Unsafe only, first / second | Unsafe difference [95% interval] |
| --- | ---: | ---: | ---: | ---: |
| 15 vs Base | 0 / 0 | 0.0 [0.0, 0.0] | 1 / 39 | −59.4 [−71.9, −46.9] |
| 100 vs Base | 5 / 0 | +7.8 [+3.1, +12.5] | 8 / 32 | −37.5 [−51.6, −23.4] |
| 1,000 vs Base | 28 / 0 | +43.8 [+37.5, +50.0] | 13 / 24 | −17.2 [−32.8, 0.0] |
| 15 vs 10,000 | 0 / 58 | −90.6 [−95.3, −85.9] | 6 / 30 | −37.5 [−46.9, −29.7] |
| 100 vs 10,000 | 0 / 53 | −82.8 [−89.1, −76.6] | 9 / 19 | −15.6 [−26.6, −6.2] |
| 1,000 vs 10,000 | 2 / 32 | −46.9 [−56.2, −37.5] | 14 / 11 | +4.7 [−7.8, +17.2] |
| 1,000 vs 100 | 24 / 1 | +35.9 [+28.1, +43.8] | 23 / 10 | +20.3 [+9.4, +31.2] |

At 100 transitions, the five successes were three on `goal_2`, one on `object_3` and one on `spatial_0`, all at seed 3. At 1,000, both seeds succeeded on all four `goal_1` starts, and both failed on all four `10_3` starts. For budgets 15 and 100, seed 0's training transitions came from the policy-driven trajectory and seed 3's from babbling. Collection, initialization and split all vary with seed, so these two fits cannot isolate the effect of acquisition source. Each fit includes its internal validation transitions in the stated budget. Across these nested prefixes the two seeds used 2,000 distinct archived target transitions and scored 60,689 new native steps. [Paired counts by seed and task](evidence/native-panel/paired-by-task-seed.csv) · [Calibration access and splits](RESULT.md)

## Pairing, uncertainty and sensitivity

For each of the six cells, the pinned `panel.py` on Baige CPU verified the published low-budget shards and joined all keys to the archived Base and 10,000-transition rows. The supplemental readback checked the ordered four-lane checkpoint identities, scene and policy-noise seeds, the post-settle 25-dimensional robot state and the first ten actions in the frozen policy’s opening proposal. **Zero mismatches** occurred in the 192 comparisons with Base and the 192 with the 10,000-transition control. Across all five arms, the same checks passed in 640/640 pairwise comparisons, including every budget-to-budget pair. Scene seed was zero throughout. These traces establish agreement of the recorded opening fields. The archives do not supply a pair of reset-camera images for a pixel comparison. [Opening checks and maximum deviations](evidence/native-panel/openings.csv)

Restricting to exact openings retained all assignments. Allowing arbitrary binary outcomes on mismatched openings leaves each paired point estimate unchanged because there are zero such rows. For 1,000 versus Base, resampling from the pooled 32 starts gives a 95% success-gain interval of +31.2 to +56.2 points. Keeping the observed task mix fixed gives +37.5 to +50.0. Both calculations carry both seeds and all arms together for every sampled start. The primary calculation draws four starts with replacement within each task, with 10,000 paired bootstrap replicates, percentile endpoints and RNG seed 20261123. Four observed starts per task and two fixed fits limit extrapolation. A zero-width bootstrap interval at zero successes describes the empirical panel. [Both interval constructions, each seed, all contrasts and mismatch bounds](evidence/native-panel/summary.json) · [CPU readback source](read_native_panel.py)

## Jobs and coverage

| Attempt | Baige Distributed Training job | Seed | Batches / new episodes | Elapsed GPU-seconds | Elapsed GPU-hours |
| --- | --- | ---: | ---: | ---: | ---: |
| `panel-s0-01` | `job-oba1tdxissci` | 0 | 24 / 96 | 1,998 | 0.5550 |
| `panel-s3-01` | `job-sc23pialxxia` | 3 | 24 / 96 | 2,000 | 0.5556 |
| **Total** | | | **48 / 192** | **3,998** | **1.1106** |

Each attempt passed its one-card RTX 4090 probe, strictly loaded the frozen policy, fitted six target models and exited 0. Both launchers started at 10:27:55 UTC on 23 November 2026. At 10:36 the budget-15 fits were on disk, and both first batch checkpoints appeared at 10:46:59. At 10:57 each attempt had published 64 episodes. At the final 11:17 check each had published all 96. The table uses elapsed launcher time from the exit receipts. The combined reservation ceiling was 2 GPU-hours. All six subsequent scores and the statistical readback ran on Baige CPU with CUDA hidden and affinity restricted to cores 0 and 1. [GPU and exit receipts](evidence/native-panel/attempts/) · [Intermediate check](evidence/native-panel/check-1057-utc.jsonl) · [Final check](evidence/native-panel/check-1117-utc.jsonl) · [Job ledger and evaluation contracts](evidence/native-panel/jobs.json)

The launcher and six `panel.py` invocations used commit `200647539605980ceb81bb10d32cc003a6f736dd`. The supplemental CPU reader used `748586646ef91dc72c8679befd5869c883c7f52e`. The six compact `panel-s*-b*.json` records retain the pinned reader's summaries and episode rows. Full traces, fitted weights and resumable batch checkpoints are under `baige-dev:/mowenjie/fluffy-bassoon-2/outputs/reground-low-budget-20261123-r1/production/`. [Six scored panels and compact records](evidence/native-panel/) · [Execution contract](HANDOFF.md)

**The full 1,400-assignment low-budget evaluations were untouched.** Each of the three conditions has 64 panel assignments available for reuse and 1,336 unscored assignments. No full-grid expansion or additional job was submitted. The 1,400-assignment Base and 10,000-transition controls were already archived.
