# Twelve-process packed PPO test

Baige `job-230io8i5cmpd`, source
`0a116e745cdbd004b3ab4b7cca3f8fed40bd9604`. Completed 2026-10-30 at
21:56:59 Asia/Shanghai, exit 0. Human authorized two independent groups of six
simulators on one GPU, targeting memory headroom. This is a fresh one-round
systems qualification; it does not replace trained production recoveries.

## Result

| Measure | Observed |
| --- | ---: |
| Concurrent simulator processes | 12, six per learner |
| Peak GPU memory | 65.552 GiB = 70.386 GB |
| Reported device memory | 95.593 GiB |
| Headroom at peak | 30.041 GiB |
| Sampled mean / peak GPU utilization | 24.62% / 100% |
| Source actions retained and physically charged | 21,310 |
| Complete source trials / episodes | 12 K4 / 48 |
| Driver wall time | 412.824 s |
| Scheduler GPU time, including probe/setup/shutdown | 453 s = 7m33s |
| Combined driver actions/s | 51.620 |
| Combined whole-allocation actions/s | 47.042 |

Twelve fits comfortably. Memory is below the internal 70–80 **GiB** planning
band; it is approximately 70.4 decimal **GB**. The units are shown explicitly
because the human described the target in GB. The GPU was not continuously
busy: 100% was a sampled peak, and the full-envelope sampled mean was 24.62%.
Do not equate memory occupancy with compute utilization or claim a matched
scalar speedup from this test.

| Learner | Actions | Complete trials / episodes | PPO updates / SGD steps | Approximate KL |
| --- | ---: | ---: | ---: | ---: |
| Persistent PPO | 10,510 | 6 K4 / 24 | 1 / 5 | 0.001555 |
| Separately trained episode-reset PPO | 10,800 | 6 K4 / 24 | 1 / 5 | 0.001425 |

Every one of the twelve trials replayed with exactly zero maximum joint
log-probability and value error at the original 2e-4 / 1e-5 tolerances. Both
original optimizer gates passed; LR remained 1e-6, with five epochs, fixed
global batch six and unchanged loss weighting. Adam state was allocated during
these updates and is included in the memory envelope. Frozen motor hashes
matched before and after.

The per-group native collection/corpus phases took 235.058 / 235.649 seconds,
44.712 / 45.831 actions/s respectively. These overlap on the same GPU and stop
before replay/checkpoint acknowledgment and optimizer work. Use driver or
scheduler totals above for an end-to-end rate. No paired scalar baseline ran.

## Integrity and scope

All twelve workers attested renderer PCI `0000:a2:00.0`, the allocated GPU.
Idle scalar preflight scenes were closed before parallel collection. Both
learners used the same six task/case plans, four complete episodes per trial,
native task cutoffs 600/500/400 and H32 motor actions. Actual reset retries are
retained: one bottle-case requested seed1499316795 became actual1499316799 in
both learners, following the pinned native retry order.

Corpus totals (339 / 348 completed chunks), group receipts, source counters and
independent physical ledgers agree. There are zero pending, unresolved or
uncertain actions and zero external budget cuts. One persistent-PPO bottle
episode succeeded at110 rather than400 actions, explaining the290-action
difference between learners. Each corpus also contains its zero-action scalar
preflight reset, separate from the48 source episodes.

Both groups reached checkpointed status; their ordinary-resume blockers were
removed only after durable coordinator acknowledgment. Read-only final checks
of the coordinators and canonical campaign fences passed. Both parent processes
exited0 and the final GPU-memory sample was0; the launcher also checks and
cleans its owned worker sessions after native parent failure.

The run logged existing OIDN errors (1,431 / 1,467 invalid handles and25
unsupported-CUDA errors per learner), also seen in scalar production. Three
unmodified representative RGB views are preserved in `representative-frame/`;
they contain plausible scene/object content but visible rendering grain.
Nonblank-image checks and these examples do not establish complete image
equivalence. There was no GIL/fatal-Python/segmentation fault in this short run.
The previously observed production planner/GIL failure around1000 resets was
not tested: each simulator here performed only four source episodes.

The source controllers were initialized fresh at seed20261030. These one-round
checkpoints are not substitutes for the active100-round experiment's trained
source states, and no existing production run or worktree was modified. A
fixed-policy K4 trial is not itself a changing-policy AD learning history.

## Validation, budget and artifacts

Before launch,171 Linux CPU tests passed in32.55s on the complete runtime at
`7d37565`. Its runtime/backend/driver files are byte-identical to the final
executable. Four final launcher ownership/orphan-cleanup tests passed in1.49s
at `0a116e7`. macOS was used for edits/static checks and report inspection.

The dedicated new allowance was21,600 actions /2,700 GPU-seconds. Actual spending
was21,310 /453. The canonical reservation is settled; unused290 actions and
2,247 seconds are sealed, unreleased and unavailable as an automatic retry.
The older12k-action RoboTwin qualification hold remains separate and unchanged.

`TERMINAL-AUDIT.json`, `settled-reservation.json`, the two `packed/*/summary.json`
files, GPU trace and independent process audit preserve thin verified evidence.
Large corpora, optimizer/sampler/RNG recoveries and native group journals remain
on project storage under this run ID. `scheduler-complete.png` records the
terminal allocation and cost. The scalar production adapter is byte-unchanged
from the owner's `db0184089fa88a5f96c153f9b80029bb91b77def` baseline.
