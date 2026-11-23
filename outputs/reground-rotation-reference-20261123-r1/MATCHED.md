# Frozen SymMeta on the released ReGround rotation

The matched adapter passed Linux CPU qualification on the released Panda simulator with the archived seed-41 reader. On spatial task 0, test start 15, the native replay reproduced all 281 robot states, 280 issued actions, 280 speeds, and all success and safety readouts exactly. This was a wiring test with a noise-driven fixture policy. Scored π₀.₅ rollouts and their first-pair comparisons belong to the parent's scheduled evaluation. [Qualification receipt](evidence/symmeta-cpu.json) · [Qualification program](symmeta_cpu.py)

## Cases, controls and outputs

The native-format jobs contain the same 1,400 assignments as the reference, over eight tasks, 35 starts (15–49), rep 0 and five adaptation seeds (0–4). The adapter validates every job against the released grid. Both runners use the release's 16-episode shards, lexicographic episode order and four environments per production batch. There are 18 shards per seed. For SymMeta, the five seed labels index repeated evaluations of the same frozen reader and 280 task–start/noise pairs. For ReGround they index different fits. [Matched jobs](evidence/symmeta-jobs.json.gz) · [Reference jobs](evidence/rotation-jobs.json.gz) · [Reference protocol](REFERENCE-PROTOCOL.md)

The run uses upstream `jobs.run_eval_job`, `rollout.run_batch`, the spawn-based `EnvPool`, `TaskEnv`, `VLA.predict_chunk`, and `SafetyTracker`. The harness replaces the controller and requests an additional endpoint render after the first five controls of each ten-control policy window while the reader is learning. Each policy query retains the native task text, batched preprocessing, 50-action proposal, ten-control execution window, and PyTorch noise from the released episode and chunk seeds. Scene seed 0, stored LIBERO initial state, ten settle controls, task horizons, success detection and deployed action clipping all come from the native loop. [Runner](symmeta_adapter.py) · [Pinned release](https://github.com/umersiddique94/ReGround/tree/16a98486f560676b9036d1a6d067b9dd51a5917a)

| Interface | Values used |
| --- | --- |
| Reader update | Consecutive raw agent-camera frames, current and preceding public EEF positions and gripper joint positions, five commands actually issued before the update. One independent estimator per live episode. |
| Camera orientation | LeRobot's raw `pixels.image` equals robosuite `agentview_image`. The archived reader flips height and width once and area-pools each 4 × 4 block to 64 × 64 RGB. The policy's own processor flips its raw agent and wrist images. |
| Motor observation | The initial public EEF position anchors the archived yaw correction of subsequent EEF displacements. Only the EEF position is replaced in the native policy input. The quaternion and grip go through native preprocessing. The reader continues to observe the original public coordinates. |
| Action correction | Identity uses native Base pose clipping and gripper pass-through. An estimated adapter applies the archived inverse to five clipped proposals. It rotates XY translation by inverse estimated yaw, rescales XYZ translation, and applies the estimated gripper sign. Wrist coordinates keep their native bounded proposal. The environment rotates **both** translational and angular command triplets. |
| Timing and costs | Start at identity, update after each completed nonterminal five-control chunk, apply an estimate after two chunks, and stop updating after 24 chunks (120 scored controls). Requery the policy at 0, 10, 20, … . Preserve the last estimate for subsequent controls. One native start, zero extra resets, zero probe controls and zero offline target-fit transitions per case. |

Only public sensor slices and completed control blocks cross the reader pipe. Episode identifiers, rotation, simulator snapshots, native 25-D evaluation traces, outcome and safety fields stay with the evaluator. Policy generation and rendering use the assigned accelerator. A fresh executable running the archived Python 3.11/JAX environment hosts the reader on CPU, outside the native torch process and its spawned simulator children. There is one such process per lane and one OpenCV thread. [Reader transport](symmeta_reader.py) · [Seed-41 update](../symmeta-original-pro-20261123-r1/eval_reader.py)

The shard writer preserves native `results.csv`, `manifest.json` and zipped traces (`X`, `A`, `A_vla`, `ref`, `speed`). As for Base, the absent reference trajectory is represented by NaNs. The `unsafe` flag uses contact with a non-target object, non-target displacement or fall, or speed above the released π₀.₅ envelope. The separate `table_steps` count is also retained. Timing columns keep the native measurement boundaries. `wall_s` includes observations and updates, `ms_vla` measures policy queries, and `ms_per_step` measures the controller's action call. Reader compute and request round-trip seconds appear on each update in `calibration.json`.

Each calibration record includes its case/noise key, scored controls, online target transitions used, reset and probe costs, extra render requests, zero-based first changed control, completed updates, public sensor readbacks, issued means and policy queries. Online transitions are part of the scored attempt. Compare them separately with ReGround's generated target data and its 10,000-transition target fit, shared across cases for a fitted seed. [Reference budgets](REFERENCE-PROTOCOL.md)

## Source pins and CPU receipts

| Component | Verified identity |
| --- | --- |
| ReGround release | `16a98486f560676b9036d1a6d067b9dd51a5917a` |
| Frozen policy | `lerobot/pi05_libero_finetuned` at `7152f13e34c6c115d808aea0e5d08aca74eac14b`. Scoring requires the reference runner's successful strict key-load check. The authors' historical checkpoint hash is unreported. |
| Tokenizer and assets | PaliGemma `35e4f46485b4d07967e7e9935bc3786aad50687c`, LIBERO assets `0b3ea86be5fe169d0fd036ae63d1070ec09e90f6`. Both methods use the run's shared environment, BDDL and initial-state banks. |
| Reader and inverse | `eval_reader.py` from `fd909227c2013cc58be1ced519b03ea1328199ea`, SHA-256 `f83b1c0b9ada76c2126f366294785222e121703e02977c66bf530b1fb731b1e5`. Historical estimator, transforms and JIT at `ff2ddf0ab03e6fbe243979556d8d8f2ebb9bc29c`. |
| Weights | `xemb-icml-campaign-20261121-r1/gpu03v2/stage1/checkpoints/equiv_v2_dis_dz-passive_attempt1-seed41.msgpack`, SHA-256 `c12bace5cb391499d6abd7cc5767f94b1519de7d77940efb0cfcb71ca0f686a4`. |
| Runtime | Native Python 3.12.15, LeRobot 0.6.1, hf-libero 0.1.5, robosuite 1.4.0, MuJoCo 3.8.1, torch 2.11.0+cu128, NumPy 2.2.6. Reader Python 3.11.17, JAX 0.5.3 CPU, NumPy 1.26.4, OpenCV 5.1.0. |

Baige CPU qualification used Mesa llvmpipe and zero GPU-hours. Test start 15 used episode noise seed 709751874. The first two release chunk seeds were 1283716440 and 584941787. All 28 policy queries followed the ten-control cadence. There were 12 extra midpoint renders, 24 updates and 120 in-attempt calibration transitions. The first ten commands matched native Base clipping. Control index 10 was the first intervention. All 270 subsequent commands equaled the archived five-control inverse, including the wrist-command pass-through.

Independent raw captures at controls 0, 5, 10 and 15 matched the actual reader's before/after hashes before and after image reduction. Public EEF and gripper joint positions matched the raw sensor values at both ends of each checked chunk. At the second policy query only the processed EEF-position coordinates changed. Replaying the full action tape with the stock ten-control render schedule gave the same native outcome, 28 speed violations, and an unsafe flag of 1. These values characterize the CPU fixture. The scheduled comparison must establish the policy outcomes and first-pair agreement on its accelerator. [CPU evidence](evidence/symmeta-cpu.json) · [Reference runtime receipts](RUNTIME-ACTIVE.md)

The CPU candidate was `285f3635d8ed53a8d119da769e7624e47e5c8333`. The Python programs and assignment bytes in the handoff match that detached candidate. The scored shell entry also includes the reference runner's second system-library directory for the simulator's PCRE dependency, from commit `23a66415`. Raw CSV, NPZ traces, manifests, calibration ledger and log are in the shared run under `symmeta-cpu.OdjNx8UD/q02/` and `symmeta-cpu.OdjNx8UD/q02.log`. The final commit is also deployed as a detached checkout at `$root/symmeta-src-<full-commit>`.

## Scheduling handoff

Use the final committed checkout containing this run. A sparse checkout also needs `outputs/symmeta-original-pro-20261123-r1/eval_reader.py`, `scripts/[train]gpu_probe.py` and `src/fluffy_bassoon/baige_gpu.py`. The launcher reads runtime assets from the shared Baige run and the archived reader locations. The following example runs seed 0, shard 0 inside a one-GPU parent allocation on the same accelerator type used by the reference. A parent running multiple lanes probes the allocation once and gives each lane a distinct `--rank`. The scheduler owns the wall limit and cumulative budget.

```bash
run=outputs/reground-rotation-reference-20261123-r1
root=/mowenjie/fluffy-bassoon-2/$run
export SYMMETA_EXPECTED_CODE=$(git rev-parse HEAD)
attempt=$root/symmeta-native/attempts/$HOSTNAME
mkdir -p "$root/symmeta-native/attempts"
mkdir "$attempt"
bash "$run/symmeta_adapter.sh" probe "$attempt" 1
bash "$run/symmeta_adapter.sh" lane --attempt "$attempt" --rank 0 --seeds 0 --shards 0
```

Omit the seed and shard filters for the full grid. Multiple ranks can claim from the same grid. Each validated shard publishes atomically under `$root/symmeta-native/shards/seed<seed>/rot_z/90/symmeta/shard<NNN>/`. A fresh attempt reuses completed shards and reruns interrupted shards. For CPU qualification, pass `cpu --attempt FRESH_OUTPUT_DIRECTORY` to the same script. The `manifest` command emits the native-format assignment list.

After the corresponding reference Base and ReGround shards exist, run the pairing command. It checks native row keys and noise, Base's common opening (state, proposed and issued actions, and speed), and ReGround's reset and first policy proposal. A partial comparison can supply the same `--seeds` and `--shards` filters. The full paired table feeds the shared three-method analysis. [Comparison code](compare.py)

```bash
bash "$run/symmeta_adapter.sh" pair --attempt "$attempt"
source "$root/runtime/activate.sh"
"$root/env/bin/python" "$run/compare.py" "$attempt/paired-cases.json" "$attempt/comparison.json"
```
