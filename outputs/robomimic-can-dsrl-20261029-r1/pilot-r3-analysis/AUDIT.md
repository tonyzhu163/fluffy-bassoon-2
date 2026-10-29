# Pilot-r3 descriptive results audit

The four arms completed 28,800 training primitive steps each, using training
seed 1 and the same eight evaluation scenes, each retried four times. This
pilot supports execution and comparison feasibility. It does not establish a
substantial memory benefit: RL² normal has one extra successful attempt versus
both its memory-reset lesion and the separately trained reset controller.
That difference occurs only on scene 2000004, attempt 2; the other seven scenes
tie. Attempt 1 and attempt 4 both score 5/8 for RL² normal.

The inference unit is the initial scene: eight paired scenes and one training
seed. The 32 recorded episodes in each row are repeated observations within
those scenes, not 32 independent trials. No significance test is reported.

## Outcomes

Each attempt column below counts successes out of eight scenes. Totals count
recorded successful episodes out of 32 and are descriptive.

| Configuration | Attempt 1 | Attempt 2 | Attempt 3 | Attempt 4 | Any-step success total | Release-logger total | Scenes with any success / 8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Native frozen motor | 5 | 3 | 2 | 5 | 15 | 14 | 8 |
| DSRL-NA frozen | 5 | 5 | 3 | 4 | 17 | 17 | 6 |
| Retry DSRL-NA frozen | 4 | 3 | 2 | 3 | 12 | 12 | 5 |
| RL² normal | 5 | 6 | 5 | 5 | 21 | 21 | 7 |
| RL² memory reset | 5 | 5 | 5 | 5 | 20 | 20 | 6 |
| RL² reward blind | 6 | 4 | 6 | 6 | 22 | 22 | 7 |
| Trained reset normal | 5 | 5 | 5 | 5 | 20 | 20 | 6 |
| Trained reset memory reset | 5 | 5 | 5 | 5 | 20 | 20 | 6 |
| Trained reset reward blind | 5 | 4 | 5 | 6 | 20 | 20 | 6 |

For every trained configuration, the two success metrics agree at every scene
and attempt. Native release-logger counts are [5, 3, 2, 4], differing from
any-step success only on scene 2000006, attempt 4. The release criterion
requires an entire four-action chunk to have success reward, rather than a
single successful primitive step.

RL² normal minus each comparator, in successful attempts per scene:

| Scene | RL² memory reset | Trained reset normal | RL² reward blind | DSRL-NA | Retry DSRL-NA |
|---|---:|---:|---:|---:|---:|
| 2000000 | 0 | 0 | 0 | 0 | 3 |
| 2000001 | 0 | 0 | 1 | 1 | 4 |
| 2000002 | 0 | 0 | 0 | 0 | 2 |
| 2000003 | 0 | 0 | 0 | 2 | 0 |
| 2000004 | 1 | 1 | -1 | 1 | -2 |
| 2000005 | 0 | 0 | -1 | -3 | 1 |
| 2000006 | 0 | 0 | 0 | 0 | 0 |
| 2000007 | 0 | 0 | 0 | 3 | 1 |

The memory comparison is a descriptive +1/32, or +3.125 percentage points in
mean scene success rate. Reward blinding produces 22/32 versus 21/32, so this
pilot also supplies no positive evidence that reward history is necessary.
Native achieves at least one success in all eight scenes, while RL² normal
does so in seven; higher mean episode success here does not imply broader
scene coverage. Online DSRL weight adaptation was outside this pilot.

## Budgets and integrity

| Arm | Training steps | Warmup steps | Trainable parameters | Optimizer invocations | Training motor samples | Evaluation steps |
|---|---:|---:|---:|---:|---:|---:|
| DSRL-NA | 28,800 | 24,016 | 42,623,037 | 20,930 | 2,303,520 | 9,600 |
| Retry DSRL-NA | 28,800 | 24,016 | 42,623,037 | 20,930 | 2,303,520 | 9,600 |
| RL² | 28,800 | 0 | 833,465 | 24 | 7,200 | 28,800 |
| Trained reset | 28,800 | 0 | 833,465 | 24 | 7,200 | 28,800 |

Each DSRL arm performs 5,980 actor, 5,980 critic, 5,980 entropy and 2,990
noise-critic optimizer invocations. Each GTrXL arm performs six full-trial PPO
updates with four optimizer epochs each. Environment training interactions are
matched; model size, optimization and decoder-query budgets are not. These are
short pilot results, not reproductions of a complete paper learning curve.

The pilot accounts for 115,200 training and 76,800 evaluation primitive steps.
It ran on two RTX 4090 GPUs for 1,528.92 seconds elapsed, recording 50.964
allocated GPU-minutes. Each GTrXL arm's total method time includes three frozen
evaluation conditions, versus one for each DSRL arm, so those total times do
not directly compare training speed.

All 277 JSON/hash/count validations in `RESULTS.json` passed. The 33 mirrored
raw files match independently obtained remote SHA256 values. All four trained
arms have the same source commit, seed, motor/checkpoint/normalization identities and
eight initial-scene hashes, with identical retry resets. Every recurrent
evaluation has the intended reset contract. All logged metrics are finite;
all six PPO updates in each arm have exactly zero likelihood replay error and
nonzero gradients. Final summaries report the full optimizer budgets. DSRL's
three progress records stop before its final update; the completed summary,
not that last progress row, establishes its final count.

Pilot source: `103e0f5a1a58d2b7f09756ea74e81605d72cbcf6`.
Qualification source: `c1032117b953acd94592336629a1d28c4f563d62`.
The plan records a GPU-helper-only migration between them; scientific-file
hashes agree with the parent allocation report. Qualification records exact
environment/sampler parity, unchanged motor weights, complete DSRL/PPO
recovery equivalence and zero full-trial likelihood replay error. Its actual
four-lane/four-attempt/four-epoch PPO measurement was 71.907 seconds total
(60.583 rollout, 0.855 replay, 10.469 update), with 1.548 GiB peak allocated
CUDA memory.

This audit uses only stdlib JSON, hashing and counts. It does not load model,
replay or optimizer tensors, or run any simulator/test. Checkpoint payloads
remain on Baige; recovery evidence here consists of the launcher's recorded
hashes. Frozen-motor identity is recorded in all contracts and qualification;
successful driver completion is workflow evidence for its final frozen-weight
check. No independent tensor inspection is claimed.

`derive_results.py` reproduces the descriptive summaries from `raw/`.
`RESULTS.json` contains both success metrics, all scene/attempt matrices,
paired differences, budgets and validation results. `raw/remote-manifest.json`
records the remote paths and SHA256 values of all mirrored evidence.
