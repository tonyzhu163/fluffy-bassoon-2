# RoboCasa and RoboTwin project monitoring

Human request, 2026-11-05 Asia/Shanghai: babysit the existing project jobs.
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
220 GiB launch/readout admission floor and holds all eight resumes. RoboTwin's
two-iteration output estimate is approximately 125-170 GB, pending its owner's
validation of whether duplicate rollout sidecars can be omitted safely.

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

Live status belongs in the monitor's own row in `outputs/coordination.md`.
Scientific results remain in their owners' evidence folders and FINDINGS.
