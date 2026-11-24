# Native confirmation of IIWA approach transfer to Kinova3

The direct source prefix rescued one fresh drawer layout over continuous tuned control, with four successes against three and no paired losses. On the prespecified moka task it lost both tuned successes and finished zero of six layouts. Applying the target inverse to the same moka demonstration raised success from zero to three. These six fresh starts per task show a drawer rescue and a reversal on moka. [Registration](PLAN.md) · [Frozen source and release identities](registration.json) · [Native results](evidence/native01/result.json)

## Task outcomes

All four conditions used the same frozen π0.5 and task-specific PR7 seed-41 calibration from a separate 15-control Kinova3 support on layout 11. **Tuned** used it throughout. **Switch** ran raw on-policy target control for 105 controls, then switched to tuned. **Source** issued the original IIWA prefix directly. **Corrected** issued that same prefix through the target inverse. All three handbacks used standard tuned target control thereafter. Each task-layout had its own noise bank, shared across all four conditions. [Support, model and layout receipts](evidence/native01/origin.json)

| Task | Tuned | Switch | Source | Corrected | Source minus Tuned, percentage points (95% paired interval) |
|---|---:|---:|---:|---:|---:|
| Drawer | 3/6 | 0/6 | 4/6 | 3/6 | +16.7 [0, +50.0] |
| Moka | 2/6 | 1/6 | 0/6 | 3/6 | −33.3 [−66.7, 0] |

Each cell below gives the native endpoint and number of actual controls. Success ended the episode immediately. A horizon endpoint took 300 drawer controls or 520 moka controls. All 12 matched quartets are included. [Original episode records](evidence/native01/episodes.tar.gz)

### Drawer

| Layout | Tuned | Switch | Source | Corrected |
|---:|---|---|---|---|
| 20 | Horizon · 300 | Horizon · 300 | Horizon · 300 | Horizon · 300 |
| 21 | Success · 189 | Horizon · 300 | Success · 214 | Success · 173 |
| 22 | Horizon · 300 | Horizon · 300 | Horizon · 300 | Success · 208 |
| 23 | Success · 196 | Horizon · 300 | Success · 178 | Horizon · 300 |
| 24 | Success · 186 | Horizon · 300 | Success · 174 | Success · 187 |
| 25 | Horizon · 300 | Horizon · 300 | Success · 285 | Horizon · 300 |

The direct approach preserved all three tuned successes and rescued layout 25. Switch failed on all six, so merely scheduling a raw-to-tuned transition did not reproduce the rescue. Correcting the identical source tape rescued layout 22 relative to direct replay and lost 23 and 25. That exchange leaves a one-layout net advantage for bypass on this panel.

| Comparison | Rescue layouts | Loss layouts | Ties | Net difference, pp | 95% paired interval, pp |
|---|---|---|---:|---:|---:|
| Source minus Tuned | 25 | None | 5 | +16.7 | [0, +50.0] |
| Source minus Switch | 21, 23, 24, 25 | None | 2 | +66.7 | [+33.3, +100.0] |
| Source minus Corrected | 23, 25 | 22 | 3 | +16.7 | [−33.3, +66.7] |
| Switch minus Tuned | None | 21, 23, 24 | 3 | −50.0 | [−83.3, −16.7] |
| Corrected minus Tuned | 22 | 23 | 4 | 0 | [−50.0, +50.0] |
| Corrected minus Switch | 21, 22, 24 | None | 3 | +50.0 | [+16.7, +83.3] |

### Moka

| Layout | Tuned | Switch | Source | Corrected |
|---:|---|---|---|---|
| 20 | Horizon · 520 | Horizon · 520 | Horizon · 520 | Horizon · 520 |
| 21 | Success · 489 | Horizon · 520 | Horizon · 520 | Horizon · 520 |
| 22 | Success · 498 | Horizon · 520 | Horizon · 520 | Success · 430 |
| 23 | Horizon · 520 | Horizon · 520 | Horizon · 520 | Success · 453 |
| 24 | Horizon · 520 | Success · 382 | Horizon · 520 | Success · 447 |
| 25 | Horizon · 520 | Horizon · 520 | Horizon · 520 | Horizon · 520 |

The direct approach lost both tuned successes, on 21 and 22. Correcting the same demonstrated commands recovered 22 and added successes on 23 and 24 relative to direct replay. The released reward requires both pots on the stove and the stove on. The stage at which each unsuccessful trajectory failed remains undetermined by its terminal reward.

| Comparison | Rescue layouts | Loss layouts | Ties | Net difference, pp | 95% paired interval, pp |
|---|---|---|---:|---:|---:|
| Source minus Tuned | None | 21, 22 | 4 | −33.3 | [−66.7, 0] |
| Source minus Switch | None | 24 | 5 | −16.7 | [−50.0, 0] |
| Source minus Corrected | None | 22, 23, 24 | 3 | −50.0 | [−83.3, −16.7] |
| Switch minus Tuned | 24 | 21, 22 | 3 | −16.7 | [−66.7, +33.3] |
| Corrected minus Tuned | 23, 24 | 21 | 3 | +16.7 | [−33.3, +66.7] |
| Corrected minus Switch | 22, 23 | None | 4 | +33.3 | [0, +66.7] |

The net difference in percentage points is `100 × (rescues − losses) / 6`. Intervals are the 2.5th and 97.5th percentiles of all 46,656 ordered six-layout paired resamples per task. In particular, a sample with no observed losses cannot produce a loss in its empirical resamples. These intervals describe the observed layouts. Each task uses one prespecified donor and one noise bank per layout, limiting inference about other tasks, donors or noise draws.

## Secondary pooled result

Across the 12 paired starts, Tuned succeeded on 5, Switch on 1, Source on 4 and Corrected on 6. Source minus Tuned had one rescue and two losses, a net −8.3 percentage points. Its task-stratified paired interval was [−33.3, +16.7]. Source minus Switch had four rescues and one loss, +25.0 points [0, +50.0]. Source minus Corrected had two rescues and four losses, −16.7 points [−50.0, +16.7]. Each resample retained six layouts per task. [All secondary contrasts](evidence/native01/result.json)

## Command verification and compute

The exact raw successful IIWA tapes first commanded closure at zero-based indices 106 on drawer layout 34 and 107 on moka layout 31. Both therefore contributed the first 105 controls. All six drawer and six moka Source episodes issued exactly those 7D commands. Corrected used the same source information. On each task the inverse changed all 105 XYZ triplets, including every individual translation coordinate, and left rotation and grip equal on all 105. Clipping erased zero component differences. The largest absolute changes were 0.108302 on drawer and 0.025996 on moka. [Source registration](registration.json) · [Prefix comparisons](evidence/native01/result.json)

The scheduler recorded all 18,249 native controls and 3,656 genuine frozen queries, including queries discarded during replay. Drawer contributed 6,190 controls and 1,242 queries. Moka contributed 12,059 controls and 2,414 queries. All 48 episodes passed the cut. Every one of the 36 scheduled handbacks used the current target observation at global replan 21. Its tuned state differed from the raw alternative, and all five issued commands differed from the unconverted proposal at every handback. All four openings matched within each quartet on the first accepted reset, including target physics, anchor, observations, inputs, opening proposal and noise. [Native step and query archive](evidence/native01/episodes.tar.gz) · [Scheduler readback](evidence/native01/readback.json)

Baige Distributed Training job `job-zyhf39n6wf5f` used one RTX 4090 at commit `22cc4bd4caeb9a07bb2d69c745a1bf6aff522061`. The original launcher ran from 12:42:42 to 12:59:58 Shanghai on November 24 and exited zero, 1,036 seconds or 0.288 launcher GPU-hours. The scheduler read back all 48 episodes. Post-transfer verification matched all 56 Linux file hashes, re-derived each endpoint from the underlying native steps, and reproduced every task-level interval by explicit enumeration. [Launch](evidence/native01/launch.txt) · [GPU probe](evidence/native01/gpu.json) · [Exit](evidence/native01/exit.txt) · [Linux checksums](evidence/native01/BAIGE-SHA256SUMS) · [CPU qualification and exact command](HANDOFF.md)
