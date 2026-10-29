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
