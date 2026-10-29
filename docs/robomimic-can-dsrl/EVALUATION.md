# Can DSRL / RL² evaluation

The completed two-seed eight-card comparison scored 41/64 any-step successes
for RL² and 31/64 for frozen DSRL-NA at equal training interactions. Clearing
RL²'s memory between attempts still scored 40/64, and a controller trained
with that reset scored 39/64. The higher GTrXL/PPO controller score therefore
does not establish a substantial benefit from learning across attempts.

## Final two-seed comparison

All eight arms completed 28,800 primitive training steps: seeds 1 and 2 for
DSRL-NA, retry DSRL-NA, persistent RL² and trained episode-reset GTrXL. Each
evaluation used the same eight held-out initial scenes, seeds
2000000–2000007, with four attempts per scene and frozen weights. Normal,
memory-reset and reward-blind recurrent evaluations all completed. These are
two training seeds crossed with eight shared scenes, with attempts nested
within each cell; the 64 episode records are not 64 independent tasks.
Counts and differences below are descriptive, without a significance claim.

| Configuration | Seed 1 any /32 | Seed 2 any /32 | Any total /64 | Release total /64 | Scenes with any success, seed 1 / seed 2, each /8 |
|---|---:|---:|---:|---:|---|
| DSRL-NA frozen | 15 | 16 | 31 | 31 | 5 / 5 |
| Retry DSRL-NA frozen | 14 | 15 | 29 | 28 | 5 / 7 |
| RL² normal | 20 | 21 | 41 | 41 | 6 / 7 |
| RL² memory reset | 20 | 20 | 40 | 40 | 6 / 7 |
| RL² reward blind | 21 | 18 | 39 | 39 | 7 / 6 |
| Trained reset normal | 20 | 19 | 39 | 39 | 6 / 7 |
| Trained reset memory reset | 20 | 19 | 39 | 39 | 6 / 7 |
| Trained reset reward blind | 20 | 18 | 38 | 38 | 6 / 6 |

Any-step success means success at any primitive step. The released logger
requires a complete four-action chunk of success reward. Only seed 1 retry
DSRL differs between metrics: scene 2000004, attempt 1 succeeds by the first
definition but not the second. Both metrics give the same main comparison.

RL² exceeds DSRL by five successful attempts in each seed, a descriptive
15.625 percentage-point difference overall. The memory lesion has no effect
in seed 1 and removes one success in seed 2, on scene 2000004, attempt 3.
The trained-reset comparison ties in seed 1; in seed 2 RL² adds successes on
scene 2000002, attempt 4 and scene 2000004, attempt 3. Thus the memory gap is
1/64 and the trained-reset gap is 2/64. Reward history has mixed effects:
normal minus reward-blind is −1 success in seed 1 and +3 in seed 2.

RL² normal successes by attempt are [5, 5, 5, 5]/8 in seed 1 and
[6, 4, 6, 5]/8 in seed 2, or [11, 9, 11, 10]/16 when counted across seeds.
There is no increasing retry-success curve. Together with the reset controls,
this provides little evidence that the score gain comes from meta-learning
across attempts. It remains consistent with a better controller under this
short training protocol. Parameter counts, optimizers and decoder queries
also differ, as recorded below; interactions alone are matched. Only six PPO
updates per recurrent arm were run, and DSRL spent 24,016 of its 28,800 steps
in its released warmup. This is neither a convergence comparison nor a VLA
or cross-task generalization result.

The earlier native motor result remains a separate 4090 qualification
reference: 15/32 any-step and 14/32 released successes, covering all eight
scenes. No new seed-2 native evaluation was measured. It is not duplicated
into a /64 baseline. The earlier 4090 pilot below is also kept separate;
its repeated seed 1 is not an independent third training seed, and no
bitwise equivalence across these hardware executions is claimed.

Baige `job-lrdwz9x2s3kk`, source
`b9829df25eeee7cd59d0345fbc4e114b68e0b49f`, completed with driver exit 0
at 16:55:47 Shanghai on 2026-10-29. Eight independent GPU lanes ran in one
exclusive scheduler container, using the user-approved raw device identifier
`NVIDIA RPBZZZ6`, with no DDP. The campaign recorded 230,400 training and
153,600 evaluation primitive steps. Its 808.603 driver seconds correspond
to 1.797 GPU-hours at eight cards; adding the conservative two-minute,
eight-card reserve for the earlier preflight failure gives 2.064 GPU-hours
on this driver-duration accounting. The exact scheduler release timestamp
was not independently observed. The enforced 58-minute retry cap plus the
two-minute reserve bounded this campaign by the authorized eight GPU-hours.

All training/evaluation counts, paired scenes, both success metrics and
controls passed 441 metadata/hash/count checks. An independent integrity
audit passed 139 checks and streamed hashes over 2,528,001,706 recovery
payload bytes. All four recurrent arms recorded six finite PPO
updates with zero likelihood replay error. Final complete policy, optimizer,
sampler and RNG recovery payloads remain on Baige, with independently
streamed SHA256 receipts; they were not downloaded or deserialized on macOS.
Runtime/source/input/frozen-motor identities and actual GPU worker receipts
were checked against the qualification migrations. Evidence:
[derived results and count audit](../../outputs/robomimic-can-dsrl-20261029-r1/eight-r2-analysis/AUDIT.md),
[full per-scene results](../../outputs/robomimic-can-dsrl-20261029-r1/eight-r2-analysis/RESULTS.json),
and [independent integrity audit](../../outputs/robomimic-can-dsrl-20261029-r1/eight-r2-integrity/audit.json).

## Earlier single-seed 4090 pilot

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

## First eight-card allocation

Baige `job-8blx7x2ib4uu`, source
`77d72f748d948c7b744cad39de20582ee37cab56`, failed its parent GPU admission
before any training worker or arm started. All eight allocated cards reported
100% utilization with 0–3 MiB of memory in use. The probe rejected this
utilization reading before its real tensor and own-PID checks; those checks
therefore remain unqualified for this allocation. No new learning result was
produced. The scheduler reported 69 seconds of running time; the driver
recorded 30.725 seconds. The failure evidence is retained in
[the allocation audit](../../outputs/robomimic-can-dsrl-20261029-r1/eight-r1-failure-audit/audit.json).

The fresh allocation `job-lrdwz9x2s3kk`, source
`b9829df25eeee7cd59d0345fbc4e114b68e0b49f`, passed 55 CPU tests and 27
subtests, followed by the full eight-card CUDA/PCI/own-PID probe. Its explicit
Baige training admission records utilization as telemetry while retaining
allocation, memory, process-ownership and real tensor checks. All eight
actual worker GPU/PID proofs and frozen-model contracts passed an independent
33-check metadata audit over 20 hash-verified files. This establishes runtime
qualification. The completed matched comparison and final audits are
reported above.
Evidence: [retry launch audit](../../outputs/robomimic-can-dsrl-20261029-r1/eight-r2-launch-check/audit.json).
