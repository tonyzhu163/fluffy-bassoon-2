# RoboCasa and RoboTwin project monitoring

Human request, 2026-11-05 Asia/Shanghai: babysit the existing project jobs.
The user renewed this request at about23:10 on Nov5 for the next eight hours,
through about07:10 on Nov6. Resolve routine blockers using reasonable
assumptions within existing scientific scope and human budgets. Keep checking
in this active task; notify on meaningful changes, failures or required action.
The human also explicitly authorized direct coordination with the three owner
chats listed below. This monitoring work does not allocate accelerator time.

## Direction and ownership

- RoboTwin: continue the common broad-SFT Sidney initializer and RLinf
  qualification, PPO source histories, GRPO comparison, jointly trained AD,
  DPT, DIT and IC-IQL, then source-task and held-out-task evaluations with
  memory controls. Owner: `Bring back RoboTwin`, chat
  `01a1a9d3-d2ed-7163-8dbf-a55d4f009ea2`.
- Existing RoboCasa source/evaluation work continues under `Monitor GRPO PPO
  jobs and evals`, chat `01a1a717-4b4f-7173-88a6-4a3ffef3259e`.
- The eight RoboCasa native-expert steering fits remain owned by `Launch joint
  three-task training`, chat `01a1a62c-f32b-7ce0-a78f-ed43f8254bb1`.

Each owner retains its job, immutable learner contract, checkpoint, scientific
gates and budget. Existing human holds remain controlling. Do not duplicate
launches or combine the separate budget pools. The monitor chat is
`01a1ab47-e0cb-7890-bbf8-088a1be17240`.

## Monitoring and recovery

Check owner status plus the actual progress, error and complete-recovery
artifacts at least every 20 minutes while allocations run. React earlier to
failures or admission changes. Track scientific optimizer progress separately
from scratch throughput updates, and training loss separately from fresh
simulator success. Validate complete paired banks before reporting test counts.

Coordinate recoveries with the existing owner, preserve completed cases and
full optimizer/sampler/RNG state, and require a diagnosis before each retry.
Follow existing approved limits: RoboCasa source/readiness campaign 70 GPUh,
the eight-fit campaign 150 GPUh, and RoboTwin at most 50 cumulative GPUh without
further human authorization. Check each owner's current ledger before launch.

## Shared storage admission

The source-member migration is complete: 37,440 payloads / 127.068 GiB, with
original hashes and full PFS readbacks verified. Fifteen PFS-only real reads
passed with BOS access disabled. The cache is an active dependency.

The Counter-to-Cabinet TEST512 stopped at its 200 GiB free-space guard with
376 complete paired cases and one unmatched panel. Preserve both; exclude the
unmatched panel from the primary count. The meta owner retains its original
220 GiB launch/readout admission floor. For RoboTwin's original-format
two-iteration pilot, reserve 125 GiB (134,217,728,000 bytes), including transient
peaks. Canonical image references avoid duplicate sidecar images while
preserving complete chronological data and recovery.

Before resuming large writers, reconcile all three owners' incremental output
reservations against fresh shared capacity, including the strictest active
guard. Freeing space means verified archival of the responsible owner's closed
artifacts; protect active caches, source archives and complete recoveries.
Do not lower a guard or move an active dependency to admit another job.

The coordinated fit-only/evaluation reservations are Counter recovery 5 GiB,
Faucet TEST512 15 GiB, and eight checkpoint-only fits 6 GiB. SOURCE5 requires
its 200 GiB floor plus 20 GiB extra headroom and its own 20 GiB future output,
so tests plus fits require at least 246 GiB actual free before launch. Hold
student ID/OOD rollout writes for separate admission. Count archival capacity
only after byte/hash/readback verification and actual block release, and do
not add previous reclamation to a free-space reading that already includes it.

Schedule phases if all writers cannot fit concurrently: complete the pending
Counter/Faucet tests and resume fit-only jobs when admitted; keep RoboTwin's
large pilot held until its measured peak output and combined free space pass.
Assess retention of newly closed artifacts only after their dependencies end.

The bounded RoboTwin archive qualification temporarily reserves another 8 GiB
of shared scratch, making the tests-plus-fits threshold 254 GiB until cleanup.
Its separate 16 GiB node-local scratch does not consume that PFS reservation.
The initial zero-GPU console form created no job; the CPU serialization checks
use one charged GPU allocation with a 25-minute ceiling.

After qualification cleanup, simultaneous tests, fit-only growth and the
125 GiB RoboTwin pilot require at least 371 GiB actual fresh free. Compression
can remain off if that original-format admission passes; an exact prefix
round trip alone does not bound a complete production archive's size.

Live status belongs in the monitor's own row in `outputs/coordination.md`.
Scientific results remain in their owners' evidence folders and FINDINGS.

## User-requested host-memory investigation

The scientific RoboTwin retry is held while its owner runs a bounded native
simulator memory diagnostic. Existing256-scene deployment exhausted host RAM
before newSGD; the intermediate-render path is disabled. Measure per-scene
creation, repeated resets and allocator release before selecting a fix or
justifying a larger allocation. Preserve the original loss/batch recipe and
all prior compute/action charges. See RAM-TRACE-20261105-2306.json.

## Nov6 overnight phase

Source70 is closed at51.632222/70GPUh, with endpoint tests
7.045556/8GPUh. Fresh Faucet passes the original Holm3 family; Counter and
Coffee fail, leaving three effective tasks including prior OpenDrawer and
Toaster. All31.060575GiB post-test allocated-block reclamation is verified;
canonical TRAIN histories, inference weights and member cache remain hot.
There is no new five-source admission or Coffee run. Source future growth is0.

The exploratory eight-fit campaign remains capped at150GPUh. Seven fits
converged; KV DIT continues. Fitted interfaces and packed archive restoration
are qualified. Full24k evaluation needs the explicit150-to190GPUh human
budget decision requested at00:10, which is pending. Keep the published
411GiB admission guard, including its conservative20GiB source field.

RoboTwin memory01 identified glibc retention of freed scene buffers; batch-end
trimming alone peaks106.2GiB at64scenes. Memory02 timed out during the last
per-scene comparison, so it cannot admit PPO. Its complete data and charges
are retained. Fresh memory03 qualifies only the paired trim comparison with
a40-minute bound; the existing scientific source remains held until it passes.
