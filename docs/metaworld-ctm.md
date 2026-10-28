# State-based CTM meta-RL qualification

The first objective is to establish continuous-control learning, dependence on
interaction history and measured compute cost before expanding VLA integration.
This is a new implementation informed by [METTA](https://arxiv.org/abs/2610.31656),
not a reproduction of its unreleased implementation.

## Environment and splits

Pin Meta-World 3.1.1 and MuJoCo 3.3.0. ML1 uses `push-v3` with hidden goal
coordinates and the same sampled instance throughout a trial. ML10 uses its
ten training families and five held-out families. Goal coordinates are visible
on ML10 to follow METTA's Appendix C rather than upstream ML10 defaults. The
family stays fixed within a trial; ML10 instances are resampled at episode
boundaries. Benchmark seeds and ordered task-pool hashes are recorded.

Train on the benchmark's native training tasks. Evaluate train-family success
on independently generated tasks from the training families, using benchmark
seed plus 1,000,000; reject exact task overlap. Also evaluate the benchmark's
test tasks. Seeds and task identifiers never enter policy observations.

An episode has 500 decisions. It runs to its declared horizon even after
success. Each episode resets physical state; recurrent state persists across
the trial. Trial reset clears recurrent state. Previous executed action, scaled
reward, boundary bit and normalized public clocks accompany the 39-dimensional
state. Terminal reward and boundary feedback survive inner resets. V3's
`reset(seed=...)` ignores the argument; explicitly call `seed` before reset.

Use native dense rewards multiplied by a fixed 0.1 for PPO and feedback.
Latch `info['success']` across each episode and score it independently of
reward. Average success equally over episodes and task families. Preserve raw
per-lane outcomes, family labels and native returns. Report success by episode
as well as aggregate success; the last-episode metric is a separate quantity.

## Learners and optimization

Compare persistent robot CTM T1, GRU T1 and Transformer-XL T1 with a 64-step
cache. CTM uses bounded synchronization and decay once per observation. Choose
the largest width divisible by four under a common 250,000-parameter total
policy cap, requiring at least 85% of the cap. Record parameters, retained
state bytes and CUDA peak allocation. This does not match compute or memory
capacity; the XL baseline has a finite cache and is not a full-history baseline.

All learners have separate 64,64 tanh actor and critic heads, each receiving
the current normalized state alongside the memory output. Running state
normalization is frozen during collection and PPO replay, then updated after
the complete optimizer update. Its buffers belong in the checkpoint.

CTM defaults to one internal iteration. `--ctm-ticks` explicitly selects another
count. The earlier immutable r2 pilot used T4; its measurements remain T4 results.

The existing full-trial PPO replays complete chronological trials with gradients
across episode boundaries. Use Adam 3e-4, gamma .99, GAE lambda .95, clip .2,
value coefficient .5, gradient norm .5, and no entropy bonus. Continuous actions
use the existing tanh Gaussian and corrected likelihood. This differs from
METTA's sequential segment optimizer and its running-return reward scaling.
Pilot batch size, PPO epochs and evaluation length also differ and must remain
visible in manifests and chart labels. Do not call a pilot curve a METTA reproduction.

## PPO replay execution and optional truncation

The implementation follows the recurrence/head separation and saved starting
states in [Sakana CTM RL's trainer](https://github.com/SakanaAI/continuous-thought-machines/blob/4a6c9c3a7fb5dc4bca6381cc7883a3b9252c6466/tasks/rl/train.py).
Its `Agent.get_states` scans recurrence, then evaluates actor/critic heads on
the entire time/batch tensor. Its default rollout is 100 steps, initialized
from a state snapshot. It does not implement a fused full-trial recurrence.

Here the heads and distributions are batched across time, and all-false reset
rows skip constructing unused initial states. Full-trial replay remains the
default (`--replay-chunk-length 0`) and keeps cross-episode gradients.

Optional `--replay-chunk-length 100` records detached incoming behavior states
at each chunk boundary, then batches windows as independent recurrent lanes.
This adapts upstream state-snapshot replay to the existing complete-trial
collector. It is **truncated BPTT**, with protocol
`metaworld_state_cached_chunk_ppo_v2`; it changes the gradient estimator and
minibatch membership. It must not be reported as an equivalent full-trial
optimization. Frozen forward history is retained across chunks and physical
episode resets. The first window of each trial reinserts the current learned
initial state so its parameters still receive gradients.

Trial length must divide exactly by chunk length. Keep full-trial GAE, global
advantage normalization, every transition, loss weighting, PPO epochs and
optimizer steps fixed. For 10 lanes x 2,500 steps, a 2-trial minibatch becomes
50 windows x 100 steps, still 5,000 transitions and five optimizer steps per
epoch. This retains our full-trial return estimator rather than importing
upstream's rollout-end bootstrap or its unrelated hyperparameters. Cached
states are ephemeral rollout data; checkpoints remain between complete trials
with policy, optimizer, normalizer, sampler and all RNG states.

Qualification compares frozen likelihoods/values, full-trial outputs and
parameter gradients against a serial reference, a full-length-window PPO
update against full-trial PPO, explicit truncated credit boundaries, and
exact next-update recovery for CTM/GRU/XL. Time the original serial replay,
batched-head full-trial replay and cached-window replay on the same real ML10
batch and the same CTM T1 weights. Report collection and optimization separately.
One timed update is a throughput check, not a learning result.

## Admission, recovery and accounting

Model work and tests run only in a scheduled Linux allocation. A GPU probe
must pass inside the actual job. Check fake-environment boundaries, hidden-goal
masking, exact seeded recovery through a subsequent PPO update for all three
cores, and real simulator trials for both benchmark wrappers. Before training's
first optimizer step, reconstruct a complete behavior trial and verify saved
log probabilities and values within 2e-5.

Checkpoint every complete update, including optimizer, task sampler, normalizer
and all RNG states. Time limits and signals stop between complete updates.
Partial optimizer failures retain the preceding complete checkpoint and write
a failure receipt. Resume only under the identical source and scientific
contract, into a fresh output identity. No active run is modified.

For one allocated GPU, GPU hours are active process wall seconds / 3,600,
including CPU simulation and evaluation. Log collection and optimization time
separately and provide a training-only time axis as a companion. Queue time is
excluded. Include hardware, CPU allocation, dependency versions, parameter
counts and source revision. Resumes accumulate prior runtime. A scheduler's
startup/idle time is additional allocation overhead, reported by the launcher.

Final diagnostics evaluate normal, memory-reset-at-episode-boundary and
reward-blind policies on paired seeds with fixed weights. Memory reset still
receives one-step feedback. These interventions probe different sources of
evidence; neither alone proves a learned reward-driven RL algorithm.

## Progression

First qualify execution and measure complete 2,500-step training trials with
one seed and bounded GPU time. Then select a feasible immutable training
configuration and run three screening seeds. Use five seeds for final curves.
The first VLA admission evidence should include reliable continuous-control
learning, adaptation across episodes, useful-history interventions and measured
cost relative to the recurrent baselines. Train-family success alone is
insufficient. PointMaze is the subsequent long-context exploration benchmark.
