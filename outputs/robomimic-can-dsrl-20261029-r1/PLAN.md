# Original Can DSRL / RL² pilot

2026-10-29, user approved: start on the released low-dimensional RoboMimic Can
benchmark on Baige. Keep the released physics. Method contract:
[docs/robomimic-can-dsrl.md](../../docs/robomimic-can-dsrl.md).

The user selected GTrXL after the initial GRU implementation. The GRU job
`job-glohz29446yg` was stopped at14:19:03 Shanghai without ever receiving a
GPU or producing training results. Use fresh qualification-r4/pilot-r2 IDs.

First qualify on one exclusive RTX4090,16 CPUs,64GiB RAM. The combined job
has a110-minute scheduler cap; a qualification exceeding20minutes refuses
the90-minute pilot. Together with the prior failed qualification's conservative
six-minute charge, allocated usage stays below the original two-GPU-hour cap.
No automatic retry. Run focused CPU tests inside the allocation, real Can
sampler/environment parity, nonzero PPO and original DSRL updates, complete
optimizer/sampler/RNG continuation and the actual GTrXL four-lane/K4/four-epoch
PPO update with peak CUDA memory and timing. Evaluate32 native episodes.

Admit all four methods only if qualification, native headroom and measured
cost fit: original iid DSRL-NA, retry-sampled DSRL-NA, persistent RL²/GTrXL and
separately trained episode-reset GTrXL. Choose48,000 or28,800 primitive training
steps using qualification timing only, identically for all methods, including
DSRL warmup. Evaluate the same eight held-out scenes and K4 attempts, including
memory and feedback controls. A deadline yields a complete recovery and a
partial result. This short pilot establishes feasibility and learning signal;
it cannot establish a method win or a VLA/cross-task meta-learning claim.

The user separately approved the follow-on: eight RTX6000-family GPUs for
parallel matched seeds, at most one hour elapsed/eight allocated GPU-hours.
Launch only after the complete4090pilot and its artifacts qualify. Use four
training seeds with a paired DSRL lane and GTrXL lane per seed; each lane runs
the corresponding persistent/retry or trained-reset arms serially. Fix the
common step target from measured throughput before launch, preserving complete
boundary checkpoints and time for frozen evaluation. Do not silently switch to
joint distributed optimization or expand time/cards.

Review current main before each launch, commit and record its exact source,
and deploy through Git into fresh immutable run worktrees. Preserve concurrent
edits and unrelated jobs. Live status belongs in outputs/coordination.md;
verified findings belong in docs/robomimic-can-dsrl/EVALUATION.md.

Qualification-r4 passed on4090 and native success was15/32. The serial pilot-r2
held training on timing admission. Fresh pilot-r3 therefore uses2RTX4090s in
one exclusive container for at most50minutes (100GPU-minutes), with balanced
lanes: GPU0 original DSRL→persistentGTrXL; GPU1 retryDSRL→trained-resetGTrXL.
All four arms retain seed1, equal28,800 training interactions, eight held-out
scenes/K4 and all recurrent evaluation conditions. No optional online phase.
The1.25timing factor uses actual same-hardware qualification costs and saves
complete episode/trial boundaries. GPU-attestation helper migration adds
support for a complete two-GPU parent probe; scientific source stays identical
to qualification-r4 and helper fixtures plus real per-worker CUDA checks run
inside the new allocation before model construction. Record this migration.

The eight-card follow-on predeclares two execution geometries under the same
one-hour/eight-GPU-hour cap: prefer four matched training seeds with16arms
on eight balanced serial lanes; if their complete conservative workload does
not fit, use two matched seeds with8arms, one independent method per GPU.
All four methods, the complete verified common step target (at least28,800)
and every8-scene/K4 evaluation condition remain required. Select the greatest
feasible seed count from measured completed pilot phase costs, with the2xRTX
timing-transfer reserve. Record both candidates and the selected design before
training, and notify the actual selected seed count at launch. The user
specified parallel matched seeds and resource/time limits; four seeds was an
execution proposal. Do not drop trained-reset/retry controls or truncate arms
to force four repetitions into the elapsed-time cap.
