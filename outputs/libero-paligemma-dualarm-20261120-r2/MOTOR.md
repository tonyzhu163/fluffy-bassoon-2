# Dual-arm PaliGemma source motor

Fit one motor on successful Panda and Sawyer demonstrations of BASE `libero_goal/5`. The selected motor maps external RGB, wrist RGB, the eight-value hand state and the instruction to five native seven-axis controls. The same checkpoint must pass both arms on development and confirmation. [Run plan](PLAN.md)

| Stage | Released layouts per arm | Decision |
| --- | --- | --- |
| SFT | 0–27 | Fit the action head and final two Gemma layers' rank-eight LoRA on successful controls. |
| Offline development | 28–31 | Rank retained updates by equal-arm action MAE. Advance the registered shortlist. |
| Online development | 32–35 | Run all four fresh starts per arm for every shortlisted update. Require at least 3/4 on each arm. |
| Sealed confirmation | 40–47 | Run all eight fresh starts per arm for the selected checkpoint. Require at least 6/8 on each arm. |

The motor takes the next observation after executing each five-control chunk. A fresh reset starts every episode. Cameras and hand state supply the causal progress information. Keeping this interface stateless lets the existing frozen motor and history adapter use exactly the same proposals. Reset labels, expert phase, elapsed demonstration length, future frames, motion, and reward stay outside the motor input. Each episode records the raw proposals and issued controls so their equality can be checked. [Motor](../libero-paligemma-crossarm-20261120-r1/motor.py) · [Native rollout](motor_eval.py)

## Linux CLI

Use the run's launcher in an allocated single-GPU Baige Distributed Training job. Set `PIN` to the deployed implementation revision and the two bank variables to collector directories on project NVMe, each containing an `index.json`. A single combined bank can also be supplied. The launcher sets the verified MIX snapshot, simulator interpreter, EGL device, and code revision. Submit the three commands as successive stages, with fresh scheduler attempt names. `fit` and `seal` must point to the same persistent paths across attempts. [Launcher](run.sh)

```bash
set -eu
run=outputs/libero-paligemma-dualarm-20261120-r2
campaign=/mowenjie/fluffy-bassoon-2/$run
fit="$campaign/sft01/fit"
seal="$campaign/source-confirm-seal"

bash "$run/run.sh" "$PIN" sft01 vlm motor_train.py train \
    --output "$fit" --demos "$PANDA_BANK" "$SAWYER_BANK" \
    --steps 800 --candidate-steps 200 400 800 --shortlist 3 \
    --microbatch 4 --seed 20261120 --anchors-per-pose 32

bash "$run/run.sh" "$PIN" dev01 vlm motor_train.py dev --output "$fit"

bash "$run/run.sh" "$PIN" confirm01 vlm motor_train.py confirm \
    --output "$fit" --seal-dir "$seal"
```

The plain entry point in the qualified policy environment is `python -Bu motor_train.py {train,dev,confirm}`. Pass `--weights PATH --revision REV` when the launcher environment variables are absent. Supply an explicit `--scratch DIR` for camera staging on another project NVMe volume. The defaults above are an initial implementation recipe. Alter a schedule or add new corrective training trajectories in a fresh fit directory before opening confirmation.

## Demonstration bank and sampling

The bank index uses the prior collector's `suite`, `task_id`, `case` and `attempts`. An accepted row names the relative NPZ in `file` and carries `body`, `pose`, `seed`, `steps`, `done`, `native_success`, `terminal_reward` and `reset`. The tape uses the original `save_tape` schema, including pre-action camera pairs, state and issued controls, motion, terminal state, prompt, body, pose, seed, reward and success. Filenames may include a seed, reference ID, repeat or expert variant. Pass multiple roots to `--demos` in a stable order. Failed rows supply no training frames. The loader checks the native reset identities, release pin, BDDL, sparse success at the final action, finite numerics, camera dimensions, quaternion norms and agreement between the tape and receipt. Donor layouts must respect the split. For older Panda replay indices with only a release demo number, `DUAL_ARCHIVE` supplies the pinned Goal 5 HDF5 used to resolve that donor's layout. [Bank reader](motor_bank.py) · [Tape writer](../libero-paligemma-crossarm-20261120-r1/demos.py)

The manifest reports coverage separately for each arm and split. It counts released layout IDs, distinct released rows, pose/seed pairs, native initial states and distinct training trajectories. Equal camera, state, instruction and action streams within an arm share one sampling entry even when their archive or seed differs. The loader rejects an exact trajectory or native released row crossing the offline holdout. Each arm needs successful data on multiple training poses and at least one offline validation pose. The manifest lists the exact successful subset of IDs.

Each update takes four anchors per arm. Within an arm it samples a successful pose uniformly, a distinct tape for that pose uniformly, then a pre-action frame uniformly. The five-control target is masked at the episode boundary. State normalization uses only training frames. AdamW uses a head learning rate of 0.0003, a LoRA learning rate of 0.0001, weight decay 0.01, norm clipping at 1, and Huber beta 0.1. The base snapshot is the pinned general `google/paligemma-3b-mix-224`; the original loader verifies the release files before attaching a fresh action head and LoRA. [Upstream weight identities](../libero-paligemma-crossarm-20261120-r1/motor_weights.py) · [Trainer](motor_train.py)

Both cameras are inflated once per fit into a single temporary memory mapping. Allow up to 0.375 MiB of scratch per accepted frame pair. Linux reclaims the raw file on worker exit. The global batch remains eight at microbatch sizes 1, 2, 4 and 8. The training receipt records per-update time, throughput and peak GPU allocation. Validation uses fixed anchors spread across available layouts, variants and time, with equal weight per layout within arm. The shortlist sorts by equal-arm MAE, then earlier update. Online development breaks ties by minimum arm success count, total successes, offline MAE, then earlier update. A worker loads the original VLM once per stage and swaps only the retained LoRA, head and normalization tensors between evaluated candidates.

## Retries and evidence

- For an interrupted fit, use the same training arguments and fit directory with `--resume` in a fresh scheduler attempt. The registered data, recipe, code, runtime, seed and microbatch must match. `training.pt` atomically records model, optimizer, sampler, CPU/CUDA and Python/NumPy RNG state at update 1, every 25 updates, each candidate and the final update. If setup stopped before `manifest.json` was published, rerun the original fit command against its empty or lock-only directory.
- Repeat an interrupted `dev` or `confirm` command with the same inputs and a fresh attempt name. The stage contract is written before its first reset. Each completed episode has one atomic tape containing its receipt. A retry verifies existing tapes and runs missing starts. A failed native rollout remains in the denominator. A completed below-threshold panel writes `status: complete`, `source_gate: false` and exits 2. Its counts distinguish this outcome from a runtime or CLI error.
- A passing development run writes `selected/checkpoint.pt` and `selected/motor.lock`. Confirmation re-derives the winner from all development tapes and verifies the selected bytes before the first confirmation reset. Use one campaign-wide `--seal-dir` across every fit and retry. Its `selection.json` binds the winning hash, development evidence, confirmation panel and fit location. A different winner cannot enter that seal.
- `manifest.json`, `curve.json` and `candidates.json` hold configuration, source hashes, coverage, timings, validation metrics and candidate digests. `dev/report.json` and `confirm/report.json` hold counts, opening fingerprints and paths and hashes for the native traces. The frozen transfer interface is `motor.Motor.load(FIT / "selected", weights)` followed by `motor.act(obs)`. Require a completed, passing `FIT/confirm/report.json` whose `motor_checkpoint_sha256` matches the selected file. [Transfer gate](transfer_gate.py)

## Implementation evidence

The implementation was reviewed by reading the collector's accepted-row construction, original tape writer and model loader, the simulator reset and step path, and the transfer gate's checkpoint, chunk and native-tape checks. The development and confirmation success denominators are fixed at four and eight per arm. The implementation-only assignment ended before any Linux execution. Model loading, numerical behavior, stage retries, resource measurements and native success outcomes remain unverified. [Collector](expert.py) · [Native reset and step](../libero-crossarm-20261120-r1/arm_env.py) · [Gate implementation](motor_eval.py)
