# Native history experiment execution

Authorized November 17: run the small native RoboCasa experiment to determine
whether history helps. This is a separate phase with an **8 GPU-hour working
budget**, including qualification, collection and small reader fits. It does
not reopen the earlier 230 GPU-hour campaign. Account for the existing gate
from its process start at 11:09:33 Shanghai.

The allocated Baige development container has one complete RTX 4090 with a
100% compute allocation. The gate's actual CUDA probe passed; no other GPU
process was present before it began. The project mount has about 93 GiB free,
and a write/read/delete probe passed. No quota utility is available. Collection
keeps compact action tapes and public-state/reward records, rather than full
camera streams. Shared environments and checkpoints stay in place.

## Immutable qualification already running

- Source: `38eaf99bcc17d3930dab99aea4170fc0842e9c67`.
- Checkout: `/mowenjie/fluffy-bassoon-2/sources/feedback-advantage-38eaf99b`.
- Output: `/mowenjie/fluffy-bassoon-2/outputs/robocasa-feedback-advantage-20261117-r1/gate01`.
- Development hostname: `aibox-rffe453e3884-5f96c78b88-6ssrn`.
- Controller PID 319421; collector PID 319472, verified against exact commands.
- Gate completed at 11:26:35 Shanghai: all 18 episodes, both signs on both
  scenes qualified; exact tape and fresh closed-loop repeats passed.

Do not mutate this checkout or reuse the output identity. Inspect the native
qualification before launching the sixteen-scene panel. The two gate scenes
are development cases; if the panel proceeds, they remain in its training
portion. Failure of this calibration intervention is a negative qualification
result, not a negative result for every history learner.

## Changes before any later learner fit

Review found that NumPy advanced-index increments undercount repeated samples
inside a minibatch. Use `np.add.at` for the actual exposure inventory. All
methods now receive the same sampler seed within a fit seed, matching the
declared comparison. These changes do not affect the active collector.

The implementation evaluates three possible query corrections (-1, 0, +1),
so the probed suite contains 768 constructed training examples and the drift
suite 384, assembled from a much smaller set of measured native branches.
They are not hundreds of independent native trials. Final update 1024 is the
fixed endpoint; intermediate held-out curves do not select the winner.

The native branch table exhaustively measures each candidate correction at
the paired decision state. Scoring a frozen reader against it is a finite
action-selection evaluation using native outcomes, not a new closed-loop
VLA rollout or evidence that the previous large KV adapter now learns.
The opposite-calibration donor is a controlled contradictory-history
intervention, not a random unrelated-history baseline. The mask is an
evaluation-only input intervention and may be out of the training distribution.
Small scene counts and shared support/query scenes limit statistical precision;
pairing dependencies must be retained in the final readout.

Before fitting, score Stage 1 separately. Bootstrap blocks join every scene
that appears as a query, support or donor in a shared trial. The primary
eight-scene cross-scene panel has only two independent four-scene blocks;
its intervals are descriptive and coarse. The all-to-all panel has one
connected block and reports no confidence interval. This avoids treating
recombined native branch outcomes as independent episodes. Save complete
actor/optimizer/sampler/RNG snapshots at each declared training milestone.

## Continuation and monitoring

The pinned continuation at `02d6071efe0107f43943c54911ad56f23d918c53`
(PID 320466) waits for the immutable gate receipt, verifies the native contrasts,
then runs `panel01`, a separate `reference01`, and conditionally `fit01` plus its
audit. Only the training-split reference controls whether fitting proceeds.
The numerical training code matches the CPU-qualified `4d5ce29e` byte for byte.
The continuation bounds its own child process groups by the eight-hour phase
deadline, 19:09:33 Shanghai. It archives completed raw evidence with readback
hashes. Errors stop the continuation for diagnosis; it does not retry silently.

Heartbeat `monitor-robocasa-history-pilot` checks this thread every twenty
minutes and stops after delivering the result, a diagnosed stop, or budget
expiry. It must not launch duplicate jobs or modify active worktrees. After
completion it retrieves compact evidence and publishes the verified readout.
As of 11:29 the gate has passed and the panel is advancing; the first screened
scene failed nominally and is retained in the admission ledger.

The fixed decision trigger uses simulator handle geometry. The small history
reader selects a correction at that trigger; it is not a new full-trajectory
VLA policy. Native branch-table scoring and the conditional scene selection
must remain explicit in the final interpretation.

## Explicit migration to four RTX workers

The user requested multiple RTX 6000 GPUs. The Pro6K queue showed 5–8 free
RPBZZZ6 GPUs, distributed across nodes, with 16 of the personal 32-GPU quota
available. Use four one-GPU instances in one PyTorchJob, each 16 CPU cores,
128 GiB RAM, 16 GiB shared memory, maximum 75 minutes, retries disabled.
Each actual instance must pass the repository CUDA probe and native simulator
rollouts. Reuse the previous RoboCasa CUDA 12.8 container and PFS/BOS mounts.

`parallel_run.py handoff` verifies the three owned process commands, pauses
only the old continuation, finishes the in-flight scene, then stops collection
at the next scene boundary. Every completed nominal and perturbed branch is
retained and hashed. The frozen motor has no optimizer state. A just-entered
scene is restarted under a fresh output identity; its partial evidence remains.
Neither active source checkout is edited.

The remaining original seeds are divided into four fixed, disjoint round-robin
shards. Each GPU collects all nine branches of an eligible scene locally.
The native rollout function and its settings are unchanged. `--screen-all`
changes only shard termination. Merge into `panel02` takes the first sixteen
nominally eligible scenes in the original order, first eight for training and
next eight held out. Completion order and perturbed outcomes never select the
panel. Extra collected scenes are retained but excluded from the fixed readout.
Hardware migration is part of the provenance: early scenes use RTX 4090,
remaining scenes use actual Pro6K devices reported by the probes. We do not
claim cross-hardware bit-identical motor trajectories; within-scene pairing
still requires exact reset, prefix and tape checks.

Charge the full **5 GPU-hour scheduler reservation** for parallel collection,
including startup and any barrier idle time, plus the original 4090 phase up
to handoff. The subsequent `reference02` and conditional `fit02` keep the same
models, losses, seeds, milestones and train-only qualification. The fit may use
only the balance of the original eight GPU-hours, and must finish no later than
19:09:33 Shanghai. `PARALLEL-CONTINUATION.json` supersedes the intentionally
stopped old continuation after `MIGRATION.json` is written. Do not restart the
old driver or add workers to occupied slots.
