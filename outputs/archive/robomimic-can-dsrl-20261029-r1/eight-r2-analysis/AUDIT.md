# Eight-r2 descriptive result audit

All eight arms completed 28,800 training primitive steps, with training seeds
1 and 2 each applied to all four methods. RL² normal scores 41/64 recorded
successful episodes, versus 40/64 with its memory reset and 39/64 for the
separately trained reset controller. The memory effects remain small: no
normal-versus-reset difference for seed 1, and one or two additional successful
attempts for seed 2. These results do not establish a substantial benefit from
cross-attempt memory or consistent learned reward-history use.

The design crosses two training seeds with the same eight initial scenes,
2000000–2000007, each retried four times. The 64 episode records per condition
are not 64 independent trials. Shared scenes also make the 16 seed-scene cells
dependent across seeds. We report descriptive paired differences, without
significance tests or broad method rankings.

## Outcomes and retry curves

Counts in the first three numeric columns are **any-step / release-logger**
successes. Seed totals each have denominator 32; the combined denominator is
64 repeated episode records. Coverage is the number of scenes with any
success, out of eight, separately for seeds 1 and 2.

| Configuration | Seed 1 any/release | Seed 2 any/release | Combined any/release | Scene coverage, seeds 1 / 2 |
|---|---:|---:|---:|---|
| DSRL-NA frozen | 15 / 15 | 16 / 16 | 31 / 31 | 5 / 5 |
| Retry DSRL-NA frozen | 14 / 13 | 15 / 15 | 29 / 28 | 5 / 7 |
| RL² normal | 20 / 20 | 21 / 21 | 41 / 41 | 6 / 7 |
| RL² memory reset | 20 / 20 | 20 / 20 | 40 / 40 | 6 / 7 |
| RL² reward blind | 21 / 21 | 18 / 18 | 39 / 39 | 7 / 6 |
| Trained reset normal | 20 / 20 | 19 / 19 | 39 / 39 | 6 / 7 |
| Trained reset memory reset | 20 / 20 | 19 / 19 | 39 / 39 | 6 / 7 |
| Trained reset reward blind | 20 / 20 | 18 / 18 | 38 / 38 | 6 / 6 |

The only difference between metrics is retry DSRL-NA, seed 1, scene 2000004,
attempt 1: success occurs at some primitive step but no complete four-action
chunk meets the release logger's criterion. Its any-step attempt counts are
[5, 4, 3, 2], while its release counts are [4, 4, 3, 2]. Every other trained
configuration agrees between metrics at every scene and attempt.

Any-step successes by attempt 1–4, each out of eight scenes:

| Configuration | Seed 1 | Seed 2 |
|---|---|---|
| DSRL-NA frozen | 4, 4, 4, 3 | 4, 4, 4, 4 |
| Retry DSRL-NA frozen | 5, 4, 3, 2 | 4, 4, 4, 3 |
| RL² normal | 5, 5, 5, 5 | 6, 4, 6, 5 |
| RL² memory reset | 5, 5, 5, 5 | 6, 4, 5, 5 |
| RL² reward blind | 5, 4, 6, 6 | 4, 5, 5, 4 |
| Trained reset normal | 5, 5, 5, 5 | 6, 4, 5, 4 |
| Trained reset memory reset | 5, 5, 5, 5 | 6, 4, 5, 4 |
| Trained reset reward blind | 5, 4, 5, 6 | 4, 5, 5, 4 |

Normal RL² has no increasing retry-success curve: its first/fourth attempt
counts are 5/5 for seed 1 and 6/5 for seed 2. Resetting trained-reset memory at
evaluation is an identity control and yields identical recorded outcomes.

## Paired effects

Each vector is **RL² normal minus the comparator**, in successful attempts per
scene, ordered 2000000 through 2000007. Complete binary matrices and both
metrics are in `RESULTS.json`.

| Comparator | Seed 1 scene differences | Seed 2 scene differences | Total difference, seeds 1 / 2 |
|---|---|---|---|
| RL² memory reset | 0, 0, 0, 0, 0, 0, 0, 0 | 0, 0, 0, 0, 1, 0, 0, 0 | 0 / +1 |
| Trained reset normal | 0, 0, 0, 0, 0, 0, 0, 0 | 0, 0, 1, 0, 1, 0, 0, 0 | 0 / +2 |
| RL² reward blind | 0, 1, 0, 0, -1, -1, 0, 0 | 0, 0, 2, -1, 1, 0, 0, 1 | -1 / +3 |
| DSRL-NA frozen | 1, 1, 1, 4, -3, -3, 0, 4 | 0, 1, 3, 2, -1, -1, 0, 1 | +5 / +5 |
| Retry DSRL-NA frozen | 0, 4, 3, 3, -3, -1, 0, 0 | 2, 0, 2, 0, 0, 2, -1, 1 | +6 / +6 |

The memory lesion's sole difference is seed 2, scene 2000004, attempt 3.
The trained-reset comparison additionally differs on seed 2, scene 2000002,
attempt 4. Descriptively these are +1/64 and +2/64, or +1.5625 and +3.125
percentage points in mean scene success rate. Neither comparison improves
scene coverage in either seed. Reward blinding helps by one recorded episode
in seed 1 and hurts by three in seed 2, a mixed result rather than consistent
evidence that reward feedback is being used beneficially.

RL² normal exceeds frozen DSRL-NA by five recorded successes in each seed,
and retry DSRL-NA by six. This supports a favorable short-run score comparison,
while the reset controls show that the score difference cannot be attributed
to cross-attempt memory from these results alone.

The old native reference is retained separately: qualification-r4, seed 1,
on RTX 4090, scores 15/32 any-step and 14/32 release-logger, with attempt
counts [5, 3, 2, 5] and [5, 3, 2, 4], and coverage 8/8. There is no newly
measured seed-2 native baseline; that historical reference is not duplicated
or pooled into this campaign. The earlier 4090 pilot also repeats training
seed 1, with a different source/hardware execution. It is not an independent
third training seed. The changed seed-1 counts are not presented as a bitwise
reproducibility result.

## Budgets, provenance and integrity

| Per-arm quantity | Each DSRL arm | Each GTrXL arm |
|---|---:|---:|
| Training primitive steps | 28,800 | 28,800 |
| Initial warmup steps | 24,016 | 0 |
| Trainable parameters | 42,623,037 | 833,465 |
| Optimizer invocations | 20,930 | 24 |
| Training motor samples | 2,303,520 | 7,200 |
| Evaluation primitive steps, including controls | 9,600 | 28,800 |

Each DSRL arm performs 5,980 actor, 5,980 critic, 5,980 entropy and 2,990
noise-critic optimizer invocations. Each GTrXL arm has only six full-trial PPO
updates, four optimizer epochs each. Training simulator interactions match;
model size, optimizer and decoder-query compute do not. Total method times
also include three evaluation conditions for GTrXL and one for DSRL.
No convergence, VLA, cross-task meta-learning or online-DSRL result is claimed.

Across eight arms, recorded work totals 230,400 training and 153,600 evaluation
primitive steps, 83,816 optimizer invocations and 9,242,880 training motor
samples. The campaign admission reports a driver elapsed clock of 808.603016
seconds. Multiplying that clock by eight GPUs gives **1.796896 driver-based
GPU-hours**. The final stage event records a distinct driver clock of
808.608892 seconds. Neither establishes the exact scheduler allocation or
billing interval; its terminal timestamp was unavailable. Including the
prior 16-GPU-minute budget reserve gives a recorded driver-plus-reserve value
of 2.063562 hours, not an exact billed charge.

Source: `b9829df25eeee7cd59d0345fbc4e114b68e0b49f`.
Baige job: `job-lrdwz9x2s3kk`. The eight attested devices report the raw SKU
`NVIDIA RPBZZZ6` in the user-approved Pro6K queue. The audit preserves that
identifier without asserting a decoded marketing model.

All **441** metadata/hash/count validations passed. The 54 mirrored raw files
match independently obtained remote hashes. Eight distinct selected-device
worker receipts match the complete parent UUID/PCI allocation report and
own-PID evidence. The recorded CPU fixture log reports 55 passed tests and 27
passed subtests; its hash and source hashes match the plan. The permitted
GPU-probe admission migration is recorded as leaving scientific code unchanged.

All contracts agree on source, runtime, frozen motor, checkpoint/normalization
identities, declared geometry and intended training seed. Initial-scene hashes
and retry resets match across seeds, methods and conditions. Recurrent reset
contracts are correct. All four GTrXL runs have six finite updates, nonzero
gradients and zero old-policy likelihood replay error. Final summaries and
launcher records agree on complete budgets. All 16 recovery payloads were
independently streamed for SHA256 on Baige and match the recorded manifests;
the frozen checkpoint and normalization hashes also match. No checkpoint was
downloaded or deserialized.

This audit executed only stdlib artifact reads, hashing and counts. It ran no
model, tensor, test, optimizer or simulator computation. `derive_results.py`
reproduces `RESULTS.json`, which includes per-seed matrices, both metrics,
pooled descriptive counts, paired effects, budgets and validation evidence.
`raw/remote-manifest.json` records raw/source/recovery/input hashes. No prior
4090 pilot samples are included in the new two-seed combined counts.
