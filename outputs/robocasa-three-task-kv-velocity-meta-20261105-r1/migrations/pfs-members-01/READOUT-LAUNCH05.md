# Prepared launch; approval pending

Do not submit the main benchmark until a direct human budget decision is
recorded. The current hard ceiling remains **150 GPU-hours**.

The reviewable proposal is `READOUT-BUDGET-PROPOSAL05.json`: the unchanged
eight-model, 24,000-episode, 15,120,000-policy-control matrix; approximately
55.84 measured/modelled GPU-hours including setup, a 57-hour readout allowance,
and a proposed total campaign ceiling of190 GPU-hours. This is a proposal,
not authorization.

## Final admission

After direct approval, the coordinating owner records the exact human reply,
updates the canonical `META-ALLOCATION-LEDGER.json` ceiling, and atomically
creates the run-root `READOUT-ADMISSION.json`. It must contain `approved: true`,
`human_budget_authorization`, `approved_GPU_hours`, `implementation_commit`,
`scientific_contract_sha256`, `fitted_qualification_sha256`,
`packed_storage_qualification_sha256`, and `selected_candidates` from the
proposal. Lock the seven converged candidates now. KV DIT's hash remains null
until its validation stopping rule is met; never choose it using task success.

The **readout/packed-watcher implementation stays pinned** to
`87984718c9199797d31a158c47225cc695623afe` at
`/mowenjie/fluffy-bassoon-2/sources/steering-packed-87984718`.
The fitted observer and native motor are unchanged from the actual GPU
qualification. CPU-only packed archive, explicit full raw restoration and an
independent restored-wave audit passed. The packed CPU watcher is already
running; use its current receipt/heartbeat, not its dev-container PID inside a
GPU container.

## Scheduler settings

Check current queue capacity before submission. Use Baige Pro6K / 算法Z组
(`aihcq-rh0wwk9muei4`), one instance, one actual allocated GPU,16 CPU cores,
64GiB RAM,8GiB shared memory, **450-minute** maximum runtime, automatic fault
tolerance/retry off, and no idle-resource/preemption option. The previously
qualified image is
`ccr-2ff5lkkz-vpc.cnc.bj.baidubce.com/db/strong-plant:v2.0-cu12.8-cudnn9-ubuntu22.04_mowenjie_20261015152301`.
Mount the existing private PFS at `/mowenjie` read/write and the existing private
BOS read-only. Preserve scheduler GPU visibility. Each new allocation must be
entered in the canonical ledger with its actual creation timestamp and a
7.5-GPU-hour reservation; reconcile releases using exact scheduler times.

The eight commands follow. They do not construct an optimizer. **Submit only
validation-converged candidates.** Each runs ID before OOD, with the original
fixed cases/draws, six complete episodes and full/reset memory modes.

```bash
bash /mowenjie/fluffy-bassoon-2/sources/steering-packed-87984718/outputs/robocasa-three-task-kv-velocity-meta-20261105-r1/EVAL-JOINT.sh kv_bridge ad readout-pfs05-kv-ad
bash /mowenjie/fluffy-bassoon-2/sources/steering-packed-87984718/outputs/robocasa-three-task-kv-velocity-meta-20261105-r1/EVAL-JOINT.sh kv_bridge dpt readout-pfs05-kv-dpt
bash /mowenjie/fluffy-bassoon-2/sources/steering-packed-87984718/outputs/robocasa-three-task-kv-velocity-meta-20261105-r1/EVAL-JOINT.sh kv_bridge dit readout-pfs05-kv-dit
bash /mowenjie/fluffy-bassoon-2/sources/steering-packed-87984718/outputs/robocasa-three-task-kv-velocity-meta-20261105-r1/EVAL-JOINT.sh kv_bridge ic_iql readout-pfs05-kv-ic-iql
bash /mowenjie/fluffy-bassoon-2/sources/steering-packed-87984718/outputs/robocasa-three-task-kv-velocity-meta-20261105-r1/EVAL-JOINT.sh velocity_bridge ad readout-pfs05-vel-ad
bash /mowenjie/fluffy-bassoon-2/sources/steering-packed-87984718/outputs/robocasa-three-task-kv-velocity-meta-20261105-r1/EVAL-JOINT.sh velocity_bridge dpt readout-pfs05-vel-dpt
bash /mowenjie/fluffy-bassoon-2/sources/steering-packed-87984718/outputs/robocasa-three-task-kv-velocity-meta-20261105-r1/EVAL-JOINT.sh velocity_bridge dit readout-pfs05-vel-dit
bash /mowenjie/fluffy-bassoon-2/sources/steering-packed-87984718/outputs/robocasa-three-task-kv-velocity-meta-20261105-r1/EVAL-JOINT.sh velocity_bridge ic_iql readout-pfs05-vel-ic-iql
```

All eight7.5-hour slots plus the current fit/qualification maximum reserve
187.7944 GPU-hours. The actual point forecast is lower; retain the remainder
for fitting/complete-wave closeout and stop before the approved hard cap.
Never silently shorten cases, episodes, horizons or comparisons.

## Storage and recovery

Retain the agreed shared protected floor and reservations:220+125+20+6+40 =
**411GiB minimum free** before admission. The40GiB readout allowance includes
eight in-flight waves, the four packing/stage-back workers, and hot metadata.
Stop before another wave when this run's hot readouts+packing spool reach12GiB
or the packed-watcher heartbeat is older than600 seconds. Exit the GPU
allocation at an operational checkpoint rather than wait for archival.

`WATCH.py` with `STEERING_PACKED_ARCHIVE=1` independently audits closed waves,
packs all raw bytes, verifies full cold-object SHA and every restored member,
then keeps the original `RESULT.json`/`AUDIT.json` metadata on PFS. Raw inspection
requires the exact command in the bank's `PACKED-ARCHIVE.json`; this reconstructs
all original absolute raw paths. Never archive incomplete waves or relocate the
37,440-member active PFS source cache, model/environment, source histories or
resumable learner checkpoints.

Use fresh attempt names for any operational resume. Reuse the exact pinned
readout implementation and immutable selected candidate binding. Preserve
incomplete evidence and charge all repeated physical work.
