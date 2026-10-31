# Frozen pi0.5 DSRL baseline

Human priority, 2026-10-31: **DSRL first; native PPO afterward.** Establish
ordinary frozen-motor steering before another meta-learning fit. This is a
pi0.5 adaptation of the authors' LIBERO90 task57 experiment, not a literal
reproduction of their pi0 numbers or a full-benchmark result.

## Experiment

- Frozen motor: official OpenPI LIBERO40 pi0.5, receipt
  `ec064cce32f5003e2e3979c5712d7d6385018f6bfcc59cb2161105881e70d170`.
  Native H10/execute10; steer with a32D coordinate tiled over10 slots.
- Task57, cream cheese to tray, chosen by the published DSRL experiment.
  Fresh PixelSAC seed0 and replay; training cycles physical states0..39.
  Matched query states40..49 have recorded initial hashes and fixed RNG seeds.
- Reuse the qualified authors' PixelSAC:10 critics, batch256, UTD20,
  actorLR1e-4, critic/temperatureLR3e-4. Gaussian behavior until replay>500
  query chunks and the first completed update. Fixed behavior weights within
  each episode; update afterward. Reward -1/query, final success0, bootstrap
  timeout, discount .999 raised to actual executed duration.
- Initial/final native10-case panels use full-horizon independent Gaussian
  noise. Stochastic SAC10-case panels follow the first complete update burst
  crossing each10k optimizer-update threshold. Report actual update counts.
  Query transitions never enter replay or advance the training RNG.
- Stop after the first completed episode/update burst reaching500k optimizer
  updates, or earlier at the cumulative time/action/storage ceiling. A paused
  execution slice resumes the same learner, replay, cursor and resource charge.

## Resource ceiling and launch

One seed, up to10 cumulative GPU-hours including startup, qualification,
failures, retries and cleanup; up to1M actual simulator actions including
settling and queries; up to64GiB new outputs. The standing human <=10GPUh
instruction authorizes this agreed-scope phase. It does not fund the pending
25h source40 expansion or borrow another experiment's unused reservations.

The conservative nominal bound is723,980 actions:510,780 training/settling,
209,100 initial-native plus learned queries, and4,100 final-native. This is an
upper bound, not a measured dataset size. Old task0/40 timing suggests about
4.1-4.6h for500k updates before the denser new query schedule; completion within
10h is unproven. Preserve full recovery if a ceiling is reached.

The implementation reuses TargetServer, OfficialSAC, OnlineSourceSimulator,
run_episode, frozen_query and JournaledSimulator. Old protected task0/40/six
drivers and immutable runs stay intact. The new population/update-stop/cursor
code gets focused allocated-Linux checks, followed by the actual allocation
GPU probe and real simulator/optimizer advancement. No new logger, teacher
label generation or meta-learning training is a launch prerequisite.

Reviewed main: `f7dde2c94baee3ee59c4450a0ed30543d1cba67f`; relevant qualified
motor/SAC/collector code is unchanged from the previously reviewed main.
Implementation source and exact grant hash are bound before dispatch.
Sole Baige launcher/runtime owner: `/root/finish_assembly`.

## Interpretation

Primary evidence is the matched physical-case success curve versus native,
with actual query/chunk/primitive/action/optimizer/GPU costs. One seed and10
queries per checkpoint are a learning pilot, not statistical reproduction.
Record actual actor updates and finite losses; files and optimizer counts alone
do not establish useful learning. Assess any observed gain against stochastic
variation, the changed noise parameterization and contemporaneous controls.

All task57 records have role `target-online-baseline`; none enter source40
meta-pretraining. Native action-expert PPO remains queued. Its later authorized
25 TOTAL iterations retain the same128 matched starting-model cases.

References: [published DSRL](https://arxiv.org/html/2506.15799v2),
[static fidelity/readiness](../dsrl-pi05-task57-readiness-20261031-r1/READINESS.md),
[existing capture](../dsrl-frozen-pi05-data-readiness-20261031-r1/READINESS.md).
