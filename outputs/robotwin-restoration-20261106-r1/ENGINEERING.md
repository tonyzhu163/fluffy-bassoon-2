# RoboCasa engineering carried into RoboTwin

This is an implementation audit, not a claim that planned ports already ran.
Every item applies to training and evaluation unless the cell states otherwise.

| Existing implementation/evidence | RoboTwin disposition | Required evidence |
| --- | --- | --- |
| `meta_rl/robocasa_residual.py` NativeRoboCasaMotor.infer_batch and parallel rollout collector | Reuse native predict_action_batch; one shared motor per GPU; batch current public observations with explicit per-case noise. Legacy ProductionRobotwinVectorEnv's per-lane scalar encode/decode is not the new training/eval path. | Native-call batch telemetry, full lane coverage, action/pixel/state boundary parity, throughput including resets. |
| `meta_rl/robocasa_interface.py` prepare_batch/execute_batch | Reuse one frozen pre-action prefix per observation and all denoising steps/candidates. Invalidate after execution or new observation; bind checkpoint/normalization for persistent caches. Keep trainable KV gradients for student fitting. | One prefix build/decision, eager parity, no stale cache, frozen expert weights and finite bridge gradients. |
| `meta_rl/robocasa_residual_parallel.py` draw_schedule and episode_partitions | Keep episode/chunk noise independent of worker arrival; complete episode/trial ownership; preserve cross-episode memory and train minibatch semantics. | Same case/noise manifest for train/eval batch profiles, no padding/drop/duplicate, equivalent paired cases. |
| `rl/robotwin_noise.py` allocated renderer and synchronous scene lifecycle; RoboCasa process isolation | Bind rendering/inference to allocation. Keep each SAPIEN scene on its owning process/thread; use native vectors/subprocesses rather than putting one scene across arbitrary threads. | GPU UUID/PCI records and real nonblank cameras; close/reset stress, host RAM and SHM peaks. |
| RoboCasa rollout-capacity3248 PERFORMANCE-DIAGNOSIS | Profile 1..64+ inference lanes and actor microbatches independently. Preserve total environments/rounds/global2048; include tail/reset/I/O times. | End-to-end timing, useful GPU memory, largest safe and fastest profiles; actual hardware and reserve. |
| Sidney throughput03 micro64/128/256/512 and REPLAY evidence | Reuse measured candidate grid; largest safe256 on historical ~95GiB device, fastest64. Remeasure restored model/protocol. Recompute old probabilities for changed native batching before all PPO epochs. | Ratio1/logprob and critic-value replay under actual BF16 partition; include reference overhead. |
| `rl/sidney.py` frozen vision/train and native dtype boundary; `rlinf_robocasa_flow_ppo.py` keep_frozen_image_encoder_in_eval | Preserve each chosen checkpoint's native precision/activations, frozen VLM/vision eval and optimizer exclusion. Never carry Sidney-specific dtype/activation changes into Motus unchecked. | Native model numerical reference, frozen parameter hashes, every accumulation-boundary gradient audit. |
| RoboCasa fast/final-only rendering and counted native factories | Suppress redundant intermediate RGB readbacks only where controller/lighting/observation semantics are qualified. Add passive fast-controller execution/duration metadata independently of native learner masks. | Native physical/RNG/reward/termination parity, exact delivered prefix+partial/no-op counts and physics duration; image parity measured separately. |
| `rlinf_robocasa_flow_ppo_runtime.py` gzip level1 chunk archives, async I/O and actor_batch_drop | Use bounded background writes with owned CPU snapshots, explicit backpressure and drained/checksummed boundaries. Deduplicate repeated observations through immutable references; drop actor-copy raw RGB only after its canonical source is durable. | Queue depth/bytes, exception propagation, no mutated tensors, all actions/observations recoverable, storage throughput/cap. |
| RoboCasa checkpoint_weight_export, native PilotActor/PilotRollout/PilotRunner recovery | Export plain inference weights only at required eval/final boundaries; keep required resumable DCP plus optimizer/sampler/RNG sidecars. Strict full-tensor readback/calibration for eval exports. | Independent actor/all-worker restore and next case; finite inference calibration; zero SGD in evaluation. |
| `meta_rl/robocasa_pairing.py` canonical_case/native_fingerprint | Pair actual task/prompt/public observations/exposed articulations; adapt to SAPIEN rather than reuse MuJoCo fingerprints. Separate named policy RNG from constructor randomness. | Same actual initial identity across methods; declare unexposed physics variability rather than promise exact simulation replay. |
| RoboCasa global action journals, canonical banks and matched evaluations | Keep conservative request escrows plus measured delivered/physics sidecars; precommit common DEV/TEST cases; rotated/paired evaluation order; save all outcomes. | No outcome-based case filtering, full/reset/reward controls, episode/action/cost coverage, final independent audit. |
| Native optional torch_compile and bounded worker/ffmpeg threads | Compile only qualified stable hot shapes if warmup amortizes; keep eager fallback. Limit BLAS/model/encoder/video threads and avoid loading one motor per environment. | Same explicit sampler inputs, numerical tolerance and peak memory; compile+warmup+runtime wall cost. |

Changes that alter science stay separate: shorter executed chunks, Flow-SDE
step count/noise, reward shaping/filtering, PPO epoch count, critic interface,
observation/camera convention, horizon and source-data exposure. RoboCasa's
H10/K5 or four-epoch result is evidence for that condition, not a transparent
speed optimization to transplant into RLinf's H50/five-epoch reference.

Do not duplicate environment-specific RoboCasa MuJoCo reset constructors or
hash rules in SAPIEN. Port the determinism/recoverability requirement through
RoboTwin's native reset and physical identity APIs.

Implementation checkpoint: `rl/robotwin_motor.py` now provides the shared frozen
ODE motor for residual train/eval, one native transform/prefix, explicit
case-addressed initial noise, consumed observation tokens, device-resident
features and frozen parameter/buffer checks. `tests/test_robotwin_motor.py`
protects parity and gradient isolation; 63 distinct allocated-Linux core checks passed (CPUMOTOR01.json); real-native GPU
qualification is next. The real SAPIEN execution-accounting port passed
`EXECUTION01-COMPLETE.json`; it preserves the original fast learner masks.
This does not yet qualify image equivalence, the Flow-SDE teacher sampler,
parallel scene throughput or corpus I/O.

The frozen motor now accepts native NumPy worker observations and enforces
process/thread ownership. Thirteen motor checks pass at f8d13cdb (65 distinct
core checks with the prior52). The first real Motus episode passed exact native
action parity, residual gradient isolation and succeeded at160 requested targets;
a reset instrumentation harness error stopped motor02 before batching.
`meta_rl/background_io.py` adds bounded jobs/bytes, owned CPU snapshots,
backpressure and error propagation. Native rollout and environment workers
share its opt-in `native_pilot.bounded_io` path in both train and eval; Linux
checkpoint/trace checks and future production enablement remain pending.

CPUIO01 passed10 allocated-Linux checkpoint/trace cases at771f16a5,
including both native training/evaluation writes. Motor03 completed3 real
Motus episodes and the full1..256 sweep. Every size fit with the operating
reserve, but B2..256 violated the precommitted1e-3 singleton action tolerance
(maxabs0.00747..0.09819). B1 remains selected. B256 measured56.78GiB reserved
and33.95GiB free; this is the largest tested size, not a capacity ceiling.
Do not call larger native batching a transparent equivalent optimization.
BATCHMATH01 compares native batch/repeat/adapter and all-row preprocessing
before any new numerical protocol or execution profile is selected.

BATCHMATH01 completed at3a8e8ccc. Native batch and adapter actions are exact
atB1,2,8,32,64; native repeats are exact. Every preprocessing field of all64
rows matches its singleton reference. Native batch versus singleton drift
therefore arises after preprocessing inside native execution; BF16 has not
been isolated as the cause. Retain the original rejected equivalence gate.
New residual train/eval explicitly record actual batch/row as their numerical
condition. `rl/robotwin_batch.py` composes the frozen motor with owned spawned
simulators, full requested actions and measured physical transitions. This
shared runtime still needs Linux CPU and real concurrent-scene qualification.

CPUBATCH01 passed19 allocated-Linux checks at1248f550: six new runtime/process
cases plus thirteen motor cases (81 distinct core cases including prior
accounting/I/O checks). Selected lane order, complete proposal lifecycle,
immutable base/features, measured terminal partial targets/duration, missing
measurement failure, process ownership and retained replies after peer failure
passed. BATCH01-CONTRACT.json defines the next real two-scene qualification;
these CPU checks alone do not qualify parallel SAPIEN execution.

BATCH02 completed on actual Pro6K at0fcf7841 with two process-owned scenes,
two resets per lane and one shared frozen motor. Four full H192 episodes
(all timeouts) produced24 transitions/768 delivered targets/12000 physics
steps. Native B2 comparison was exact,12prefix/12sample calls served24rows,
zero residual preserved commands and head gradients stayed outside the motor.
Training-head and evaluation-head checks used the same runtime. Fresh-process
readback verified every pre-dispatch intent, transition and consecutive public
observation. Owned workers closed; I/O drained at2jobs/11.10MB peak payload.
GPU-wide free stayed78.71GiB; parent Torch reserved6.99GiB, host peak17.80GiB.
This qualifies the shared residual runtime, not a trained learner, native
Flow-SDE actor capacity, larger scene count, real unequal-length tail or
success improvement. BATCH01's literal-timestep harness failure remains
recorded and charged; actual native timestep is0.004000000189989805.
