# RoboTwin steering at three source-data scales

Prepared 2026-10-30 Asia/Shanghai. This is the next proposed source-learning
run, with early evaluations instead of waiting for a 272k-action-per-arm run.
No larger allocation has been launched. The previous proposal and all original
artifacts remain intact.

Train feature-conditioned noise SAC, persistent GTrXL/PPO and independently
trained episode-reset GTrXL/PPO against the same frozen Motus pi0.5 H32 motor.
All six VLRLA source tasks retain uniform shuffled task blocks and nominal K4
trials. SAC retains warmup500chunks, batch256, UTD20. PPO retains batches of
eight trials and five epochs. Neither controller updates the motor.

## Scales and measurements

Pause all three source learners at exactly **32,000, 64,000 and 128,000
executed source actions each**. Cap each physical call before execution; a
milestone can externally truncate a K4 trial. Label this cut and bootstrap its
value target. Retain its sampled PPO data in the pending batch; an intermediate
milestone does not itself cause an optimizer update. This makes the source
interaction counts equal at every evaluation while preserving the declared
optimizer schedule. Original runs have unchanged contracts.

Save a full coordinator snapshot at each scale, including all learners,
optimizers, SAC replay, samplers, RNG and pending PPO trials. Evaluate cloned
learners, then restore the source RNG. Keep durable per-unit evaluation cursors
so a resumed allocation skips completed units. No new source data is collected
until all arms reach the current scale and its evaluation completes.

Report actual update counters, finite losses and gradient norms, source actions
by task, completed episodes and successes, and the raw-record/policy manifest.
At 32k actions, an arm needs at least1,000 H32 decisions, beyond SAC's500-decision
warmup. PPO needs at least20 nominal K4 trials, hence at least two normal
eight-trial updates. These are opportunity bounds, not measured results or a
claim of adequate training. Subsequent scales must report their actual counters.

At every scale, evaluate all six source tasks on K4 trials with fixed seeds
at or above3,000,000,000, disjoint from the source sampler. Evaluate the first
two tasks in the pre-existing canonical target list: `adjust_bottle` and
`move_stapler_pad`. This choice precedes robot outcome inspection. Use K8 and
episode indices1/2/4/8. Conditions are persistent PPO normal/memory-reset/
reward-blind, frozen/online source SAC, and trained-reset PPO normal. Online
SAC gets a fresh target replay for each trial and carries the source optimizer.
Native Gaussian references run before and after target evaluation.

Report all three scales; do not select the best checkpoint using monitored
target outcomes. These two target tasks and one seed trial per condition are
development measurements, with limited precision. The motor's SFT already
covered all50 RoboTwin tasks; this tests transfer of steering, not unseen motor
tasks. Episodes5–8 exceed the K4 training-trial length.

## One shared physical-action cap

| Component | Maximum executed actions |
| --- | ---: |
| Source:128k each for three arms |384,000|
| Source validation:6tasks × K4 ×3arms ×3scales ×400 |86,400|
| Two targets:K8 ×6conditions ×3scales ×400 |115,200|
| Native:2tasks ×2passes ×3scales ×400 |4,800|
| Already executed source r4 and camera qualification |13,026|
| **Planned maximum including prior actions** |**603,426**|
| **Unallocated remainder of the1M benchmark cap** |**396,574**|

Early success lowers actual cost. All phases, retries, unresolved simulator
intents and explicit settling/noops count in the same benchmark ledger. There
are no explicit reset-policy actions in the qualified RoboTwin wrapper; native
internal physics ticks are disclosed separately from executed robot-control
actions. The full allowance is a cap, not an instruction to spend the remainder.
Keep the global ledger at
`/mowenjie/fluffy-bassoon-2/campaigns/robotwin-one-million-20261029-r1`.
The old unlaunched272k proposal must not initialize a competing ledger or run
alongside this proposal. Further source-cap/contract migration must preserve
this ledger and all existing charges.

## Resource decision and execution

The two-scene qualification measured26.96 executed actions/s with exact
reference parity. Applying that measured rate mechanically gives about6.08
GPU-hours for the planned590,400new actions, and about1.70hours to collect and
evaluate the first scale. These estimates exclude optimizer time, raw-data I/O,
all-task variance, startup and checkpointing. The completed r4 run measured
12.70actions/s before the camera optimization. Neither number measures the new
three-arm end-to-end training workload.

Propose the same pending **two-hour, one-GPU resumable slice**, with a hard
scheduler limit and time reserved for recovery. It should establish real SAC
and PPO updates and measure learning/I/O throughput; it may stop before the
first complete evaluation. Continue later only within an explicitly approved
resource envelope. Keep60GB storage headroom; the current shared filesystem
has1.4TB available. Record actual byte counts and archive/checkpoint time.

The original45 GPU-minute envelope has42m10s conservatively accounted before
the staged tests;2m50s remains. One new test allocation capped at two minutes
fits that envelope and executes no real robot actions. A two-hour training
allocation exceeds it. [AGENTS.md](../../AGENTS.md) says, “Ask before expanding
the scientific scope or a materially larger resource budget.” Approval for
that larger time budget is still pending; implementation and bounded
verification continue meanwhile.

Current data remain32episodes,11,475source actions,374decisions, zero optimizer
updates. The camera gate adds1,551 charged qualification actions, with no
source-learning credit. The old feature/hash artifacts cannot reconstruct raw
images. The new collector records raw RGB/proprioception, actual noise and
motor targets, rewards and termination/duration flags, policy snapshots and
fresh PPO behavior metadata. No new data or learning results are claimed here.
