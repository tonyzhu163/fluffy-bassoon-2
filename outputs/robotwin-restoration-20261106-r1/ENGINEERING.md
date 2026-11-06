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
protects parity and gradient isolation; allocated Linux and real-native GPU
qualification are pending. The real SAPIEN execution-accounting port passed
`EXECUTION01-COMPLETE.json`; it preserves the original fast learner masks.
This does not yet qualify image equivalence, the Flow-SDE teacher sampler,
parallel scene throughput or corpus I/O.
