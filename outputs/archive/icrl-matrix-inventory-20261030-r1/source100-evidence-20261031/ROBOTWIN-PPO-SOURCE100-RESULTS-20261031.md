# RoboTwin PPO source100 and final paired development results

Recorded 2026-10-31 07:03 Asia/Shanghai. Both PPO learners completed the original 100 source rounds and all declared iteration10/25/50/100 evaluations. Baige job-fji2xx8oq53h exited zero and reached scheduler and pod Success at06:43:32. This completes two source learners; the third, SAC, and the final-bank offline pipeline remain separate unfinished work.

| Learner | Source actions | Complete K4 trials | Source episodes | PPO updates | Adam steps |
| --- | ---: | ---: | ---: | ---: | ---: |
| Persistent GTrXL PPO | 964,875 | 600 | 2,400 | 100 | 500 |
| Independently trained episode-reset GTrXL PPO | 963,759 | 600 | 2,400 | 100 | 500 |

Both retain the original native600/500/400 horizons, full1024D H32 noise/bound1.5, frozen Motus motor, seed20261030, six complete K4 trials per global update, five epochs and LR1e-6. Scientific db0184089fa88a5f96c153f9b80029bb91b77def, runtime57016ba08eed79043fd67bf8b085bd6d42e6d766 and supervisor0c16c2b794f8de5b78603206917e72195083c265 remain pinned. The original crash recoveries and all source prefixes/costs remain preserved.

## Stage100 robot results

| Controller | Bottle adjustment | Stapler placement |
| --- | ---: | ---: |
| Native frozen motor, earlier matched-case panel | 6/8 | 0/8 |
| Persistent PPO, full history | 7/8 | 1/8 |
| Same weights, episode memory reset | 6/8 | 1/8 |
| Same weights, reward blind | 8/8 | 1/8 |
| Independently trained episode-reset PPO | 8/8 | 0/8 |

Each cell is one K8 trial on a development task. Episodes1/2/4/8 have0/1/3/7 prior episodes; the eight episodes are not eight independent trials. All16 target cases match actual seeds and initial raw-observation/feature hashes across conditions. Episode-reset PPO retains recurrence within an episode. Source validation was2/24 persistent and3/24 independently trained reset.

The full-history results establish no reliable memory advantage: reward blindness matches or exceeds normal on this panel, stapler success is sparse, and source validation remains weak. One training seed, two development tasks and one K8 trial per cell do not establish broad method rankings or teacher expertise. The stage-native before/after probes use a different21161030 seed bank and cannot serve as paired native controls. The earlier native panel provides a cross-run matched initial-case control:6/8 bottle and0/8 stapler. Native uses unclipped standard Gaussian noise; noise draws are not shared with learned policies. Its2fd history wrapper and final570 PPO executor are distinct, while underlying capture/motor/norm/backend configuration and actual initial cases match. This is not a contemporaneous randomized replication. At07:06 the coordinator independently rehashed all nine binding inputs/results/unit receipts and compared48 persistent-PPO plus16 trained-reset target records against16 native actual seeds/raw-observation/feature hashes: all matched. [Owner binding](ROBOTWIN-NATIVE16-OWNER-BINDING-20261031.json), [coordinator static audit](ROBOTWIN-NATIVE16-COORDINATOR-AUDIT-20261031.json). The reused native episodes cost5,051 actions in the already-counted old panel. No new actions/GPU-seconds were spent and old panel leftovers stay sealed.

Persistent stage100 evaluation used25,822 actions:10,259 source validation,13,963 target and1,600 unpaired native reference actions. Reset used14,639:10,010 source validation and4,629 target. These evaluations add zero source actions or SGD. The final cumulative original ledger charges are1,068,449 persistent and1,021,785 reset, both with zero pending/uncertain actions under unchanged caps.

## Evidence and accounting

The execution owner's report is committed at2e2eb26e8e82771969739f632dc4213cb2369128. Its [persistent audit](ROBOTWIN-PPO-STAGE100-OWNER-AUDIT-20261031.json) and [reset audit](ROBOTWIN-RESET-STAGE100-OWNER-AUDIT-20261031.json) bind outcomes, case pairing, unchanged source counters/RNG/motor and176 clean owned teardown receipts. Their retained SHAs are in [the copy receipt](ROBOTWIN-STAGE100-RETAINED-REPORTS-20261031.json).

The coordinator independently read saved terminal pointers and rehashed all four source100/final coordinator recoveries at07:01:52. All expected SHAs matched; each263-member ZIP passed CRC. Both arms have complete stages10/25/50/100 and next_input=null. [Static audit](ROBOTWIN-PPO-SOURCE100-INDEPENDENT-ARTIFACTS-20261031.json). This was read-only artifact access, with no scheduler query, model deserialization, inference, tests or new physical actions.

The owner verified actual scheduler/pod Success. [Terminal receipt](ROBOTWIN-PPO-SCHEDULER-CLOSURE-20261031.json):02:26:01–06:43:32,15,451 actual GPU-seconds. Pod15,455s is a separate clock. Known terminal historical costs become77,333+15,451=92,784 GPU-seconds, excluding still-unclosed SAC and an unverified historical630s. A reservation ceiling is not actual consumption or reusable credit.

At the owner's06:56 saved report, SAC had93 complete rounds/935,644 source actions/594,460 steps per actor, critic and temperature optimizer; later in-flight updates were excluded from the closed source bank. This is timestamped owner evidence, not current coordinator scheduler verification. Source100 completion and its final evaluation still require their own receipts.
