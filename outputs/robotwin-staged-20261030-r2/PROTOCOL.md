# Revised two-hour RoboTwin comparison proposal

2026-10-30 Asia/Shanghai. **Unlaunched; larger resource approval is pending.**
This replaces the earlier prepared525d669/3e-4PPO command. The common motor,
all six source tasks, K4 training/K8 evaluation, two fixed target tasks,
32k/64k/128k matched source-action milestones and1M shared benchmark cap remain
as in the [original staged protocol](../robotwin-staged-20261030-r1/PROTOCOL.md).
The maximum three-scale cost remains603,426actions including13,026prior;
all new diagnostics used0robot actions and receive0fresh source-data credit.

## Measured reason for revision

One real batch3/five-epoch PPO update completed5optimizer steps in5.58seconds
with exact original likelihood/value replay and full audited recovery. It did
not advance the batch8 main comparison. The missing process-global high TF32
setting was recovered from the pinned motor implementation and restored.

The3e-4update was unacceptable: reported mean approximate KL8.86e19. A second
allocated inspection measured the exact conditional Gaussian KL(old||new) as
12.8442, with100%of the saved joint ratios clipped. Its variance component was
only0.000841; the movement was predominantly in means across1024noise dimensions,
not a collapsed standard deviation. Recurrent reconstruction was repeatable.
The huge ratio-based estimator is not the exact KL value.

| Independent fork LR, same140decisions/fiveepochs | Analytic post KL | Post clipped fraction | Declared gate |
| --- | ---: | ---: | --- |
|3e-4|12.8442|1.000|fail|
|3e-5|0.43548|0.9714|fail|
|3e-6|0.01065|0.750|fail on clip fraction|

All forks have full independent recoveries; these are not three sequential
meta-training updates or robot-success results. Both lower-rate forks repeated
empty-history recurrent reconstruction with mean/log-probability error0.0.
The3e-6rate reduces distribution drift substantially but does not pass the
predeclared conservative gate. See [diagnostic evidence](../robotwin-saved-ppo-lr-20261030-r1/RESULTS.md).

## Initial allocation and source admission

Request the same pending **oneGPU,120-minute hard scheduler slice**. Its first
steps are the focused allocated Linux checks, including seven newly added
rollback tests, then a source-only optimizer calibration at1e-6and3e-7. Each
candidate uses the original unchanged weights/optimizer/RNG and same three
complete K4 trials; keep five epochs and the same PPO objective. No simulator
or VLA is loaded for calibration. Gate remains mean analytic joint post KL<=.02,
post clip fraction<=.2, finite nonzero gradient, changed weights, exact old
likelihood replay, repeatable reconstruction and complete recovery.

Select the largest passing predeclared rate and save the selection receipt.
If neither passes, stop before any new robot action. The main three learners
start from fresh independent initial weights; no diagnostic-trained weights
are transferred. Persistent and trained-reset PPO use the same qualified LR,
five epochs and8trial batches. SAC keeps500decision warmup,256batch,UTD20 and
its original optimizer. Only source data select this setting; no target result
is used. This is an explicitly recorded optimizer revision, not a change to
existing run contracts or a claim that calibration establishes learning.

## Operational KL guard and resumability

The new optional source PPO guard rejects nonfinite reported metrics or mean
reported approximate KL>.1. It preserves a complete pre-update coordinator on
disk, restores learner/optimizer/sampler/global RNG on rejection, retains all
pending on-policy rollouts, and saves the rejected state separately for
forensics. Halt the run rather than continuing with an unsafe update. This is
an execution guard, not an analytic trust-region guarantee or a new PPO loss.
Old defaults remain unchanged when the guard is disabled.

On routine wall-clock continuation, reuse the original calibration result and
qualified recipe without recalibration. Keep the pinned source, shared action
ledger, all source recoveries, pending PPO, raw corpus, stage snapshots and
evaluation cursors. Source and target evaluations clone the appropriate source
checkpoint; target SAC updates cannot alter source training.

The first scale and evaluations extrapolate to1.70collection hours from the
qualified26.96actions/s two-scene gate, before raw I/O/optimizers/all-task
variance. The measured small PPO update suggests roughly40ms per stored chunk
for its five epochs, but is not a full training-throughput measurement. SAC
UTD20 throughput is still unmeasured. A two-hour slice may stop before a full
first-scale evaluation. All source/target outcomes and optimizer counters must
be measured explicitly. No strong robot-performance result exists yet.

The original45GPU-minute envelope is almost exhausted; see the final measured
accounting in the [LR diagnostic](../robotwin-saved-ppo-lr-20261030-r1/COMPUTE-ACCOUNTING.json).
[AGENTS.md](../../AGENTS.md) requires approval for a “materially larger resource
budget.” This proposal and its conditional launcher are prepared; it is not
an allocation authorization. New guard/configuration changes have AST/shell
checks only; their Linux tests run first inside the proposed approved slice.

## Steering interface across benchmarks

RoboTwin's existing arm has**1,024 steering coordinates**: a full native
`(H32,32)` initial-noise tensor. Both its SAC and PPO controllers use that
same interface. The quoted0.01065analytic KL is summed across all1,024
coordinates for each stored decision, then averaged over140decisions. It is
not an average per coordinate and not a complete-trajectory divergence.

The current LIBERO implementation instead takes one32D steering vector and
repeats it across the motor's nativeH10 noise tensor (`expand_noise` in
vla_libero.py). Its trained steering coordinates are clipped to[-1,1];
RoboTwin's current bound is[-1.5,1.5]. The native Gaussian references use full
independent horizon noise. Motor/SFT/runtime and warmup conventions also differ.
Do not treat results across these two benchmarks as a dimension-matched test
of the same steering interface, or call the full1024 RoboTwin baseline an
exact canonical DSRL reproduction.

A prospective32D tied-noise RoboTwin ablation is a concrete interface question:
apply the same tied interface to SAC, persistent PPO and trained-reset PPO,
with fresh collection, declared bounds and an unlearned tied-Gaussian reference
alongside the native independent-noise reference. It may reduce the policy's
search dimension; this has not been tested here. Existing1024D actions,
likelihoods and outcomes cannot be sliced, averaged or relabeled as fresh
on-policy32D data. This ablation is **not implemented, substituted or launched**
by the current proposal; discuss and freeze its contract and budget separately.

Prepared source pin:`c3b6281c032e2c712a62191b03219b1e70e12947`, verified
clean on the Baige development host. The runtime output and shared ledger do
not yet exist. The scheduler form remains unsubmitted at120minutes/oneGPU,
automatic retries disabled. Any prospective tied-noise arm shares the existing
1M RoboTwin ceiling and requires a revised allocation; it does not create an
additional1M allowance.

[Preparation-only tied32 implementation and fresh-data costs](TIED32-PREPARATION.md)
records a separate possible arm; the current full1024 proposal is unchanged.
