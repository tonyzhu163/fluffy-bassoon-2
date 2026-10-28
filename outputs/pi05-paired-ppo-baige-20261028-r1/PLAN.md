# Matched balanced DPT → PPO versus direct PPO

Prepared 2026-10-28 on reviewed base `decabec00981734d9fe2dfe8d42cd0d705238efa`.
This is a conditional follow-up plan. It launches no job, changes no active run,
and does not promote a warm start from its offline flow loss.

The question is whether a balanced DPT initialization improves subsequent
online learning at the same online interaction budget. This comparison does
not isolate a memory architecture advantage: both routes use persistent CTM,
width 64, T4, the same frozen native pi0.5 motor and public input interface.

## Admission phases

1. Finish the actual interface qualification. Separately fitted per-θ teachers
   must execute the bounded bridge through deterministic four-Euler
   `StochasticPi05BridgePolicy.sample_chunk` on fresh scenes. Save paired bridge,
   zero-bridge and external-oracle outcomes at θ = −45°, 0°, +45°. A reviewed
   completed live pass with pinned evidence is required. External angle
   cancellation, zero-residual parity, and lower flow-matching loss alone do
   not satisfy this gate. Until that decision exists, PPO remains gated.
2. Pin the balanced portable DPT checkpoint, its full native interface, source,
   history manifest and pretraining counters. Its own live success is reported
   separately from the privileged teacher gate. A candidate may be a useful
   initialization even if it has not established in-context improvement.
3. Run allocated Linux qualification for the new planner, public-history
   reader, recovery paths and final trainer hook. Review and pin the latest
   clean `main` commit before creating fresh run worktrees/output identities.
4. Start the paired online pilot only after those conditions hold. Measure
   capacity and elapsed time over the first two updates, verify exact unchanged
   likelihood replay, finite gradients, frozen native weights and matching
   plan/noise receipts. Continue toward the bounded target only while healthy
   and within the reviewed cap.

The planner validates an explicit reviewed receipt; it does not invent a
success threshold after inspecting the results. The reviewer records the pass
criterion and rationale alongside the actual live outcomes and evidence hash.

## Shared online contract

| Quantity | Both routes |
| --- | --- |
| Core | Persistent CTM, width 64, T4 |
| Native motor | Frozen pi0.5, four Euler steps, horizon 8, execute first 5 |
| Residual | Active 7 dimensions, bounded mean scale 1.0, fixed σ = 0.05 |
| Trial | K4 attempts, 80 simulator steps per attempt; memory reset per trial |
| Global batch | Six complete trials per update |
| Cells in every update | `{−45°, 0°, +45°} × {object 0, object 1}` |
| PPO | Up to four whole-trial epochs, clip 0.2, target joint KL 0.05 |
| AdamW | Learning rate 1e−4, weight decay 0, betas (0.9, 0.999), ε = 1e−8 |
| Loss | Value coefficient 0.01, entropy coefficient 0.01, gradient norm cap 1 |
| Returns | Equal attempt weights; γ = λ = 1; terminal boundary only at trial end |
| Initial target | 20 updates per route |

The direct route starts from a freshly seeded core with zero residual output
and zero critic. The DPT route loads compatible portable balanced DPT weights
and zeroes its critic. Both use fresh optimizers, the same new phase seed and
fresh RNG streams; no parent optimizer or recurrent trial state is transferred.
Fixed-σ Gaussian entropy is constant. The PPO actor retains the summed joint
auxiliary flow likelihood; no division by the 224 coordinates is introduced.

`PairedPPOPlanProvider(infos, seed=..., updates=20)` is callable with the number
of already completed updates. It returns a deep copy of the private plan for
that update. Both routes use the same scene/reset plan and the same explicit
`motor_residual_rng_seed`. Their physical trajectories can differ because their
actions differ. The provider uses no mutable sampler, policy or route identity.
Scenes are distinct within an update; seeded reuse across updates is allowed.

The new trainer hook must save the returned plan, its canonical `plan_sha256`
and per-update noise seed, and reset the private motor/residual generator to
that seed immediately before collection. It must not feed private plan metadata
or θ to the student. Existing active trainer contracts remain unchanged.

## Budgets and recovery

One update is `6 × 4 × 80 = 1,920` simulator steps and 384 policy chunks.
Twenty updates are **38,400 online simulator steps per route**, 120 complete
trials, 480 attempts and 7,680 policy chunks. The pair plans 76,800 online
simulator steps. Eighty optimizer steps per route is a maximum; KL stopping can
reduce the realized count. There is no claim that 100 updates are required.

Resource proposal for root review: two GPUs, at most 90 minutes of scheduler
wall time, with at most 3,600 driver seconds per route. Allocation size and final
caps remain pending measured batch-six throughput. A nominal target is not a
completion promise. Save full resumable checkpoints at complete update
boundaries with policy, optimizer, sampler, explicit motor generator and global
RNG state. A preempted partial trial is discarded; a partial PPO update is not
promoted. Report the elapsed cap and final complete update count for each route.

Compare routes at common completed online budgets. If one route completes
fewer than 20 updates, report matching saved checkpoints and the partial status.
Keep DPT collection/fitting costs, frozen motor calls, online PPO and evaluation
costs separate. Reusing teacher histories across candidates does not create new
unique simulator exposure, although their replay compute must be counted.

## Evaluation and causal interpretation

Use matched held-out ±30° conditions and fresh scene overlays with fixed
evaluation plans/noise seeds. Report per-attempt success and attempt-four fresh
probes after three supports. Cloned probe replicas share their original support
history cluster. Report initialization behavior before online learning as well
as the common completed online budgets selected in advance.

Crossed evaluation uses common public histories and reader-owned recurrent
replay. The public history contains only native KV/mask, issued pre-intervention
command, terminal reward, boundary and clocks. It excludes source memory,
actor likelihood, value estimates and private θ. Hash the shared public input
contract independently of reader weights/core and record source collector and
checkpoint provenance separately.

A reader may encounter unfamiliar histories from another collector. Therefore
a crossed score combines source experience and reader compatibility; it is not
a pure collection-quality score. Preserve that familiarity flag and compare
both own-history and common-history performance. Favorable live differences are
`success_after − success_reference`; favorable error reductions are
`error_reference − error_after`. Offline flow loss remains a separate diagnostic.
No adaptation claim follows from retry totals or an imitation-loss reduction.
