# Can DSRL / RL² evaluation

Pilot-r3 completed all four methods at 28,800 training primitive steps each.
The comparison is executable and interaction-budget matched. It does not yet
establish a substantial memory benefit: RL² normal gains one successful
attempt over both its memory-reset lesion and the separately trained reset
controller, on one of eight paired scenes.

## Verified pilot results

Training seed 1 was used for all four arms. Evaluation used the same eight
initial scenes, seeds 2000000–2000007, with four retries per scene and frozen
weights. The statistical unit is the paired initial scene: eight scenes and
one training seed, not 32 independent trials. Counts below are descriptive;
no significance claim is made.

| Configuration | Any-step successes by attempt, each /8 | Any-step total /32 | Release-logger total /32 | Scenes with any success /8 |
|---|---|---:|---:|---:|
| Native frozen motor | 5, 3, 2, 5 | 15 | 14 | 8 |
| DSRL-NA frozen | 5, 5, 3, 4 | 17 | 17 | 6 |
| Retry DSRL-NA frozen | 4, 3, 2, 3 | 12 | 12 | 5 |
| RL² normal | 5, 6, 5, 5 | 21 | 21 | 7 |
| RL² memory reset | 5, 5, 5, 5 | 20 | 20 | 6 |
| RL² reward blind | 6, 4, 6, 6 | 22 | 22 | 7 |
| Trained reset normal | 5, 5, 5, 5 | 20 | 20 | 6 |
| Trained reset memory reset | 5, 5, 5, 5 | 20 | 20 | 6 |
| Trained reset reward blind | 5, 4, 5, 6 | 20 | 20 | 6 |

Any-step success means success at any primitive step. The released logger
requires a complete four-action chunk of success reward. The metrics agree
at every trained-method scene/attempt; native release counts are 5, 3, 2, 4,
differing only on scene 2000006, attempt 4.

RL² normal exceeds each reset comparator only on scene 2000004, attempt 2;
the other seven scene totals tie. This is a descriptive +1/32, or +3.125
percentage points in mean scene success rate. First and fourth attempts both
score 5/8, so the pilot shows no increasing retry-success curve. Reward-blind
evaluation reaches 22/32 versus normal's 21/32, providing no positive evidence
of benefit from reward history. Native covers all eight scenes at least once,
versus seven for RL² normal. Online weight adaptation was not evaluated here.

## Qualification and budgets

Qualification-r4, source `c1032117b953acd94592336629a1d28c4f563d62`, passed
69 focused CPU tests in 132.51 seconds on allocated Baige RTX 4090 resources,
plus CUDA allocation and real tensor/own-PID attestation. Released motor
sampler parity and all primitive observations, actions, rewards and horizons
in the 1,800-step wrapper parity check matched exactly. PPO likelihood replay
error was zero. PPO and DSRL recovery reproduced the next sampled actions,
simulator data and optimizer updates; the frozen motor stayed unchanged.

The actual four-lane/K4/four-epoch GTrXL rollout/replay/update took 71.907
seconds: 60.583 rollout, 0.855 replay and 10.469 optimization, at 1.548 GiB peak
allocated CUDA memory. A released DSRL 20+10 update at batch 256 took 0.461
seconds. These timings qualified the full history and update geometry.

| Arm | Training primitive steps | Warmup steps | Trainable parameters | Optimizer invocations | Training motor samples |
|---|---:|---:|---:|---:|---:|
| Each DSRL arm | 28,800 | 24,016 | 42,623,037 | 20,930 | 2,303,520 |
| Each GTrXL arm | 28,800 | 0 | 833,465 | 24 | 7,200 |

Each DSRL arm performed 5,980 actor, 5,980 critic, 5,980 entropy and 2,990
noise-critic optimizer invocations. Each GTrXL arm performed six full-trial
PPO updates with four epochs each. Environment interactions are matched;
parameter, optimizer and decoder-query budgets are not. These short results
do not reproduce a complete paper learning curve.

Pilot source `103e0f5a1a58d2b7f09756ea74e81605d72cbcf6` completed on Baige
`job-6j2y16w9u33o`, using two RTX 4090 GPUs for 1,528.92 seconds elapsed,
or 50.964 allocated GPU-minutes. It recorded 115,200 training and 76,800
evaluation primitive steps across all arms and controls. All six PPO updates
per arm had zero likelihood replay error, finite metrics and nonzero
gradients. Initial-scene hashes, reset contracts, frozen-motor/input identities,
source and final budget counters passed the independent metadata audit.

The earlier qualification-r2 stopped before Can episodes because RoboMimic's
legacy `mujoco_py` import was missing. The isolated runtime uses upstream
RoboMimic `8df6e8e482458befaf1decfcd9d635d68a7afcae`, which makes that import
optional. Qualification-r4 verified the repaired runtime. Serial pilot-r2
declined training on its conservative budget estimate; pilot-r3 then used the
recorded balanced two-GPU plan. The source migration between qualification
and pilot changed GPU attestation only; scientific-file identities were
recorded unchanged.

Evidence: [pilot-r3 audit](../../outputs/robomimic-can-dsrl-20261029-r1/pilot-r3-analysis/AUDIT.md),
[derived results and validations](../../outputs/robomimic-can-dsrl-20261029-r1/pilot-r3-analysis/RESULTS.json),
and [mirrored raw JSON/hash manifest](../../outputs/robomimic-can-dsrl-20261029-r1/pilot-r3-analysis/raw/remote-manifest.json).
The audit passed 277 metadata/hash/count validations over 33 mirrored files;
it did not load checkpoint tensors or run models, tests or simulation.

## Two-scene online-adaptation control

The optional DSRL weight-adaptation control completed both predeclared scenes,
2000000 and 2000001, with four attempts each. Compare it only with those same
two scenes from the frozen evaluations:

| Configuration | Any-step successes /8 | Release-logger successes /8 |
|---|---:|---:|
| Native frozen motor | 5 | 5 |
| DSRL-NA frozen | 7 | 7 |
| Retry DSRL-NA frozen | 1 | 1 |
| RL² normal | 8 | 8 |
| RL² memory reset | 8 | 8 |
| RL² reward blind | 7 | 7 |
| Trained reset normal | 8 | 8 |
| DSRL-NA online adaptation | 4 | 4 |

Online successes were [1, 1, 0, 0] and [1, 0, 1, 0] in scene order. This
two-scene control shows no descriptive improvement over frozen DSRL; it is
too small to establish a reliable effect or rank the methods generally.
Learner, optimizer and replay state started afresh from the same training
reference for each scene; updates persisted within its four attempts.

The control used 2,400 new primitive interactions and 600 adaptation calls:
12,000 actor, critic and entropy optimizer invocations each, plus 6,000
noise-critic invocations (42,000 total). Its unchanged 20+10 update schedule
ran after each single-lane chunk, giving four times the updates per new
transition of the original four-lane training schedule. Deployment
interactions are matched to the paired frozen evaluations; compute is not.
The source reference, frozen motor and input hashes remained unchanged.

Baige `job-060f4odvylld`, source
`2a02e6c7b165e37199df81ba46308d82caffdde2`, completed with exit 0 in 523.569
driver seconds (8.726 allocated GPU-minutes). The two-scene learner payload
reported 436.977 seconds and 0.793 GiB peak allocated CUDA memory.
All 131 metadata/hash/count audit checks passed, including independently
streamed post-run reference and input hashes. No checkpoint tensors were
loaded or copied to macOS for this audit.
Evidence: [online control audit](../../outputs/robomimic-can-dsrl-20261029-r1/online-r1-analysis/AUDIT.md)
and [derived paired results](../../outputs/robomimic-can-dsrl-20261029-r1/online-r1-analysis/RESULTS.json).
