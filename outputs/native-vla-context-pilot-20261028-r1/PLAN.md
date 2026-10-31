# Native pi0.5 context pilot

**Superseded by [r4](../native-vla-context-pilot-20261028-r4/RESULTS.md), the completed run; this plan's scope still applies.**

Date: 2026-10-28 (Asia/Shanghai). Status: implementation ready for allocated
Linux qualification; no successful model computation claimed by this plan.

The user requested trying the native-context proposal. Start from reviewed
GitHub/local main `5746cbb0db126ff01bb721da6b7560ffc47169b1`; the new candidate
commit is recorded in the launch receipt. The primary clone stays on `main`.
Use a separate detached worktree and fresh output identity for each attempt.

## Bounded scope

One Baige GPU, 16 CPU cores and 64 GiB RAM, at most 60 allocation minutes;
driver cap 2,700 seconds with scheduler margin. No larger sweep, new simulator,
new backbone, latent-search sweep or real robot. Before submission, verify
current queue capacity and the GPU type through the scheduler console.

The first gates are exact native empty-history parity on a real observation,
12 tiny-set fitting updates with nonzero history-input gradients and decreasing
loss, unchanged base weights, and joint stochastic-velocity replay within .002
absolute log-probability error. The gate restores the initial adapter weights
before collecting any training data; its fit is not a learned initializer.

If admitted, collect one four-attempt trial at each of -45, 0 and 45 degrees
for training and separate trials at -30/30 for flow-loss validation. Seeded
scene plans separate training overlays 0-11 from validation overlays 12-15.
Also execute the inverse-rotation label policy on the same 45-degree trial,
with the paired initial-noise stream: require at least .25 success and no less
than the base collector. If this small competence gate fails, save data and
stop before supervised training.

Warm-start at most 20 optimizer updates, each averaging two uniformly sampled
query losses over the three training trials. Recompute native attention from
frozen image/language input embeddings and the complete scalar feedback history.
Train rank-four LoRA on expert-0 Q/K/V/O and a linear feedback embedding. Base
motor and native action expert weights remain frozen. Supervise only five
executed steps and seven active channels of the 8x32 model action.

Collect one stochastic four-attempt trial and take one full-batch PPO optimizer
step after likelihood replay qualification. The Gaussian mean is the native
history-conditioned flow velocity. All four passes, eight slots and 32 channels
are stochastic, giving a valid joint auxiliary-action likelihood; this is not
the external bridge's residual likelihood. Use sigma .05, gamma/lambda one,
clip .2, value coefficient .01 and gradient norm cap one. This is a transition
smoke test, not a completed meta-RL training run.

Finally evaluate the frozen candidate at both held-out theta values, four
attempts per trial, under normal, wiped-between-attempts and reward-blind
history, with paired scenes/noise across conditions. These tiny cohorts cannot
support a generalization or architectural claim. Rotation can be inferred
visually; reward-blind improvement is not automatically a failure.

## Reproducibility and recovery

The new history v2 stores frozen prefix **input embeddings**, state, actual
requested commands, delayed public feedback and separate action labels. It
does not reuse old final-layer K/V/suffix records when native attention changes.
Save each complete history atomically and hash its bytes in recovery snapshots.
Snapshots include trainable weights, optimizer, global RNG, private sampler and
motor RNG, data paths/hashes and complete update counts. Resume only into a
fresh output directory with the same source/policy/data contract.

Run the new interface/likelihood tests in the allocated CPU development
container before GPU submission, then repeat them in the target allocation.
The target allocation runs the GPU probe for training and verifies renderer
binding. Record source/runtime/motor hashes, GPU identity, token/interface
settings, peak memory, phase timing, validation loss, PPO replay and per-attempt
outcomes. Preserve failure evidence and diagnose it before a fresh run ID.

School SSH aliases are not configured in this session's shell; the availability
script therefore reports both school clusters unreachable. The authenticated
Baige console and existing allocated CPU development container are available.
Installed pi0.py, modules/gemma.py and env_io.py match the inspected public
RLinf e1a82eb source byte for byte. The Baige canonical clone currently fetches
from an older local Git bundle; transfer the reviewed candidate as a new Git
bundle rather than overwriting or retargeting that clone.

Live status belongs in the single coordination file referenced by
[the repository pointer](../coordination.md); this plan does not change cutover.
