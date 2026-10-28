# CTM in-context RL POC launch record

2026-10-28, Asia/Shanghai. The user authorized implementation and launches as
ready, requested maximal parallel subagents, then prioritized the POC over a
duplicate CMM calibration. The existing thu2 calibration remains untouched.
The single live coordination file remains in fluffy-bassoon until its pinned
campaigns report. This is the dated experiment record, not a second live queue.

## Executed admission and compact pilot

- Baige `job-fqv47701dkg3`, source `27d7b96`, failed admission before training:
  one reward-blind test fixture selected only unrewarded actions; an older CUDA
  test hardcoded Slurm. Its immutable evidence is preserved. Both were fixed.
- Baige `job-uomh9zm3mpnd`, source
  `a3f18a8b576e1a81f7a926b5b7424b2a7ef5a55a`, launched 16:11 Shanghai,
  4 RPBZZZ6 / 32 CPUs / 64 GiB RAM / 8 GiB SHM, 60-minute cap, no retries or
  automatic deletion. Allocation-local four-GPU arithmetic/PCI/PID probe passed.
  The full suite passed 256 tests, 13 skips, 27 subtests in 15.92 seconds.
- All 14 cells completed: bounded CTM T1/T4/T8, CMM, Transformer-XL, GRU and
  looped Transformer; K64/K128 six-arm Bernoulli; seed 1000; three updates.
  Total trainable parameters are 95,715 / 99,035 / 96,007 / 97,187 / 96,223
  for CTM / CMM / XL / GRU / looped Transformer. Each cell completed twelve
  frozen greedy/sampled normal/memory-wiped/reward-blind conditions.
  This is execution and cost qualification, not a trained architecture ranking.
- The same allocation waited for three CPU scalar robot-cache fits to finish:
  CTM, XL and GRU, 500 updates each, one seed, protocol v1, theta/phi ±30°.
  All greedy normal returns equal zero correction, 0.6640625. These fits show
  no adaptation. Preserve their complete checkpoints and lesion arrays.

Remote records:
`/mowenjie/fluffy-bassoon-2/outputs/ctm-screen-pilot-baige-20261028-r2/`
and `/mowenjie/fluffy-bassoon-2/outputs/robot-cache-poc-baige-20261028-r1/`.

## Wider-angle scalar control

The existing 704-row frozen-motor headroom cache also contains exact net angles
{-90,-45,0,45,90}. A distinct v2 protocol uses theta and phi in {-45,0,45},
without interpolation or new simulator data. Scene splits remain disjoint and
objects remain shared. Exact cache SHA256:
`74910dc755b959fe86ec0a3d4cf1c05957ba57510c64bbb1bb2c85ba56e41575`.
On the validation table, fixed-zero expected success is 0.4167 versus 0.7500
for angle cancellation, compared with 0.6458 versus 0.7500 for v1.

V2 is a separate bounded CPU POC: CTM T4, XL T1, GRU T1; seeds 0/1/2;
500 updates, 64 trial lanes, K8, 4 PPO epochs, minibatch 16, learning rate
3e-4, reward offset .5, value coefficient .01, entropy coefficient .01.
It is an equal-width plumbing comparison, not a matched architecture claim.
The main paper needs independent caches, additional task/angle splits, and live
evaluation. Do not expand to 70 full screen fits based on admission alone.

## Main robot model: frozen pi0.5 feature bridge

The main robot direction reads native frozen pi0.5 KV features and public
action/reward/boundary history. Persistent CTM and Transformer-XL use the same
native inputs and zero-initialized flow-velocity residual interface. Memory
advances once per action chunk and persists across physical attempts. The
motor remains frozen. The scalar cache controller is a reward-only control.
The current residual interface is VLRLA-style, not an exact reproduction of
VLRLA's cache intervention.

Native zero-bridge qualification: Baige `job-sr0t8ifv2fls`, source
`44c5da3cd30c35daa3473f843d032d0a486ee238`, 1 RPBZZZ6 / 16 CPUs /
64 GiB RAM / 8 GiB SHM, 20-minute cap, no retries or automatic deletion.
Eight synthetic bridge tests passed. The live gate compares native, CTM and XL
on the same observations and cloned noise at all 16 chunks of one 80-step
episode. The gate passed exact model/decoded/prepared-action and noise-stream parity,
actual renderer binding, and unchanged motor/bridge fingerprints. One common
trajectory succeeded; elapsed 158.76 seconds and peak CUDA allocation 6.68 GiB.
No trained feature-bridge adaptation result exists yet.

Visual features can reveal dynamics without reward. Include both same-checkpoint
reward-blind lesions and separately trained reward-blind visual learners; do not
assume their adaptation must be flat.

## Live native bridge learner launch

Train CTM T4 and Transformer-XL T1 concurrently on two allocated RPBZZZ6 GPUs,
32 CPUs, 128 GiB RAM, 16 GiB SHM, 180-minute scheduler cap; no automatic retry
or deletion. Use the latest clean main commit, two complete K4 trials per update,
100 target updates, 9,000-second driver cap, AdamW 1e-4, four PPO epochs maximum,
explicit joint target KL .05. The exact joint Gaussian auxiliary flow likelihood
is replayed chronologically over complete trials; no dimensional averaging.
Initial two updates require numerical replay equality and frozen native motor
fingerprints. Save complete optimizer/sampler/global and motor RNG state after
every update. Evaluate held-out rotations/overlays at updates 0/10/50/100 under
normal memory, memory wiping and reward blinding. Two evaluation lanes are a
bounded diagnostic, not a reliable architecture comparison.

Run the three scalar v2 CPU workers concurrently on the same allocation, with
seeds 0/1/2 per core. The earlier standalone v2 draft was never submitted: the
CPU queue rejects distributed-training jobs. Use a fresh r2 output identity.

Submitted as **Baige `job-61p1rncjnr1x`** at 16:45:46 Shanghai, running
16:45:57. Exact source `c3e9cca68b93b6e777cbe5314e130bcaf85c8e14`.
Parent two-GPU probe passed; all 32 new CPU tests passed in 3.71 seconds inside
the allocation. Both live workers loaded the same frozen motor fingerprint and
passed actual renderer binding on their distinct assigned GPUs. By 17:01 Shanghai both workers passed the first two real live PPO updates
(eight optimizer steps each), with zero likelihood/value replay error and
unchanged frozen motor fingerprints. Native collection takes 153–162 seconds
per update; PPO takes about eight seconds for CTM and two for XL. At this
cadence the immutable 9,000-second cap is expected to stop before the target
100 updates, with complete resumable checkpoints. Held-out adaptation is
still unproven. [Launch receipt](../pi05-live-bridge-poc-baige-20261028-r1/LAUNCH.md).
See
`docs/pi05-live-bridge-ppo.md` for the durable method contract.

## Paper comparison and task priorities

Architectures: matched Transformer-XL (primary robot bridge), GRU (RL²-style
recurrent learner), looped Transformer (internal compute control), and calibrated
CMM (CTM-family alternative). Match inputs/data/optimization; report parameters,
retained state, environment samples, wall time and GPU memory rather than claiming
all are matched simultaneously. Task references: zero correction, random/fixed
actions, Thompson sampling, a structure-aware posterior learner and oracle.
Training comparisons: distillation versus trial PPO, and warm start plus PPO;
AD and DPT are training schemes, not independent memory architectures.

After rotation, prioritize hidden action gains/axis transforms, hidden-goal
navigation, and within-trial changes in hidden dynamics/reward. A robot task with
an unobserved preferred target/bin would separate reward learning from visual
system identification, but first qualify that the frozen motor can execute each
candidate action/task. These are next-task proposals, not launches in this run.

Primary references checked: [VLRLA](https://arxiv.org/abs/2610.19517),
[RL²](https://arxiv.org/abs/1611.02779),
[Algorithm Distillation](https://arxiv.org/abs/2210.14215), and
[DPT](https://arxiv.org/abs/2306.14892).

## Monitoring and preservation

The `CTM POC progress` heartbeat checks every twenty minutes and stays quiet
for unchanged healthy state. It reads saved artifacts through the configured
Baige development endpoint, performs no model computation on dev, and preserves
all active pins, checkpoints and the failed admission record. Notify verified
results, failures or needed action. Record fresh job identities in this dated
folder and the single live coordination file.
