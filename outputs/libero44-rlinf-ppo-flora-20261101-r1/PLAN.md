# LIBERO44: standard RLinf PPO versus Flora-reward PPO

**SUPERSEDED PROPOSAL — DO NOT EXECUTE.** After this proposal was prepared, the
human directed the source owner to pull and follow coordination.md. Main
`a0f5acbd06a40877a340416a0ae48f407c397c69` points to
`outputs/flora-ppo-comparison-20261101-r1/PLAN.md`, which is the authoritative
draft. It uses RLinf130-SFT, full-horizon/no-success-stopping training, a
64-episode calibration stage and binary Flora labels, rather than the
OpenPI40/continuous-score/onset design below. Preserve this file as proposal
history only. The historical OpenPI40 6/20 baseline is not a measurement of the
new draft's initialization. Both old GPU allocations are released and all
three old launch-capable automations are paused; no comparison is launched.

Prepared 2026-11-01 Asia/Shanghai from main `c729a989021ceadceed71f6fb80782c9905f4db4`.
The human selected LIBERO task44 and NVIDIA Flora-Rewarder, and paused all other
project experiments. This document is preparation, not a runtime qualification
or a claim that either comparison arm has launched.

## Question and common initialization

Does replacing simulator training rewards with a fixed Flora outcome score
change native pi0.5 PPO learning on task44, "turn on the stove"?

Both arms start from the SAME existing released OpenPI pi05_libero checkpoint,
already fine-tuned on LIBERO40. Do not train another LIBERO40 stage, load a DSRL
controller/replay, or continue an assessed TRAIN actor. The intended motor has
receipt `ec064cce32f5003e2e3979c5712d7d6385018f6bfcc59cb2161105881e70d170`
and JAX parameter signature
`c565dd3f0c4005be728cf7597698cba937c4011cd5bd813700a89b789f9d69e9`.
Its historical native task44 result is6/20 on ten reset states with two noise
draws each. That is development evidence, not the new RLinf-loaded baseline.

The existing native-PPO CPU274-second qualification is RoboTwin/ALOHA H50/14D.
It does not qualify this LIBERO H10/7D path. Bind an explicit official conversion
and normalization/tokenizer/camera/state/action mapping, test identical-input
and fixed-noise parity, and measure the actual native RLinf initialization.
Do not silently substitute the RLinf130 or RoboTwin initializer. Conversion
failure is a readiness failure, not permission to change the comparison.

## Two independent online learners

| Arm | Training reward | Actor, critic and optimizer |
| --- | --- | --- |
| Native PPO | Simulator terminal success0/1 | Same RLinf implementation and initialization |
| Flora PPO | Frozen Flora outcome `progress=(expected_score-1)/4` | Same RLinf implementation and initialization |

Keep the value head, GAE, clipped policy/value objectives, actor/value learning
rates, trainable parameter set, global loss weighting, policy seed, rollout
schedule, horizons, termination/bootstrap masks and evaluation schedule matched.
Initialize each learner's critic/Adam/samplers identically and independently.
This is PPO in both arms, not GRPO and not a replacement value critic.

For every policy version, collect fresh on-policy complete episodes, finalize
all reward labels, compute GAE, update PPO, then collect the next batch. Do not
optimize first and label later, reuse unrelated historical trajectories, drop a
labeling failure, or substitute a simulator reward in the Flora arm. Put each
terminal scalar on the final valid transition once. Retain original simulator
reward/success out of band for auditing and common evaluation.

Use train states0..39 at scene seed123. The existing states40..49 with two
draws are a viewed development/validation bank; they are not a blind final test.
Do not select checkpoints or tune either reward mapping from those outcomes.
Any final test bank must be fixed before its first outcome. Task44 itself was
selected through earlier development screening, so no unseen-task claim follows.
Keep success-based episode termination matched and disclose its oracle access.

## Flora identity and inputs

- Official code: `NVIDIA/Flora@129a2ada74c526ee4692f3d1b111bea6ca19d5b7`.
- Model: `nvidia/Flora-Rewarder@031299e46e452fc0cedb4e0975eb9fa95fc37783`.
- Use the official terminal scorer and its fixed normalized expected-score
  output. Preserve score probabilities, expected score, progress and raw JSON.
- Supply the actual instruction and clean synchronized native-clock videos.
  Disable `info_on_video`; no reward/success/done text or overlays reach Flora.
  Chunk-boundary frames must not be mislabeled as primitive-rate video.
- Pin prompts, frame sampling, camera order, generation settings and actual
  staged weight/config/tokenizer hashes before outcomes. Progress is not a
  calibrated success probability.
- Keep the rewarder environment separate from the qualified PPO environment.
  Official terminal inference is validated on H10080GB; this does not certify
  RTX96GB co-residence. Capacity and label latency need actual qualification.

## Bounded first execution and readiness

Prepare a two-iteration matched onset pilot first: eight logical environments,
one rollout round, natural H400/execute10, global32 chunk minibatch, microbatch1
and one optimization epoch. Each arm has20 development evaluations before and
20 after; the pair has at most45,920 physical actions including ten settling
actions per episode, before any separately declared qualification allowance.
These are explicit small-pilot adaptations, not a published-recipe reproduction
or evidence of convergence. Keep effective minibatch weighting fixed if tuning
execution microbatches. The primary learning profile must retain the official
pinned LIBERO PPO loss/GAE/optimizer settings and declare its training schedule
before outcomes; onset results cannot select a reward mapping or a favorable
checkpoint. Do not silently treat changed training schedules as exact resume.

The existing LIBERO chunk implementation can physically execute actions after
the first done flag and then mask them. Charge every actually executed action;
use the exact existing valid masks for loss and terminal placement, and end the
rewarder video at the last valid frame. Never infer physical cost from loss masks.

Readiness requires focused allocated-Linux CPU checks, checkpoint conversion
and native parity, actual step0 evaluation, video-label identity/placement
checks, complete optimizer/sampler/RNG recovery, and measured PPO/Flora memory
and throughput. Mac work is static only. Target70–80GiB peak on96GiB devices;
do not assume that both models can remain resident together.

The execution owner must issue a fresh finite comparison resource/action/output
plan. All startup, conversion, evaluation, label generation, failed attempts,
idle reservations and cleanup count. Paused jobs' unused caps are not funding
for this comparison, and existing run contracts/ledgers stay immutable.

## Pause state at handoff

The sole execution owner reports that primary SAC `job-2qn8dpw3yvlb` stopped
and released its allocation at12:59:03; assessment R2 `job-bc2ooyztqsfh` finished
and released at12:57:11. Full checkpoint/index reconciliation and the formal
cost/release receipts remain with that owner. Other queued successors are held;
the LIBERO90 launch heartbeat has been paused. No comparison job is launched.
