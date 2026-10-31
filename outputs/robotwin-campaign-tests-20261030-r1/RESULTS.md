# Staged source-learning integration checks

Baige **job-mjbpmea60ptq**, source
`525d669e8fff3a995de444aee940d5e0db505197`, exited0.
**86 tests passed in18.73seconds**, with one existing pytest-hook deprecation
warning. The real allocated CUDA/GPU-count probe passed. Scheduler runtime was
1m10s; it finished2026-10-30 00:17:54 Shanghai. Conservative initial accounting
is43m20s of45minutes, leaving100seconds. No real robot-control actions were
executed by this test job.

The added integration test runs the complete staged driver with a simulator
fixture and all three learner interfaces. It checks exact action barriers,
SAC warmup/update dispatch with unchanged real500/256/20 configuration,
truthful stage-budget cuts, immutable snapshots and pending PPO preservation.
Interrupted and uninterrupted runs retain the same source RNG/action stream,
source weights/replay/update counters, native Gaussian samples, completed
units and physical budget totals. Frozen and online target SAC clones leave
source training untouched. Existing real learner tests cover optimizer math.

These checks validate code behavior, not robot learning or success rates.
The last real source run remains32episodes/11,475actions and zero optimizer
updates; the faster observation path's real parity gate remains the separately
recorded two-scene qualification.

The two-hour staged training allocation is prepared but **unsubmitted**,
awaiting the larger resource decision. See the [staged protocol](../robotwin-staged-20261030-r1/PROTOCOL.md).
