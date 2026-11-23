# Baige runtime for original LIBERO-PRO

**CPU-qualified on November 23, 2026.** The requested simulator and original published states are installed on Baige. Linux restored a locked Panda state exactly, rendered both cameras in software, verified the complete frozen π₀.₅ checkpoint, and loaded the existing seed-41 streaming reader. Preparation used **0 GPU-hours**. The parent owns the deployment pin, GPU admission and submission.

## Runtime and evaluator interface

The run-specific runtime is `/mowenjie/fluffy-bassoon-2/setups/symmeta-original-pro-20261123-r1`, abbreviated as `$runtime` below. [prepare_runtime.sh](prepare_runtime.sh) provides the following paths. Both application interpreters and all pre-existing environments are reused read-only.

| Export | Verified selection |
| --- | --- |
| `ORIGINAL_PRO_ROOT` | `$runtime/release`, requested hongyuxiyohung/LIBERO-PRO at `e235a970758ecd78ecbb585fca6b915a8d44d18d`. Import `from liberopro.liberopro.envs import OffScreenRenderEnv`. |
| `ORIGINAL_PRO_ASSETS` | `$runtime/assets`, zhouxueyang/LIBERO-Pro at `3a33dc4cfbea7ff5eac6764e9c557500972b6259`. The evaluator passes each BDDL and its matching saved state by absolute path. |
| `SIM_PYTHON` | `/mowenjie/fluffy-bassoon-2/setups/libero-meta-learning-20261117-r1/env/bin/python`. Python 3.11.14, MuJoCo 3.2.3, robosuite 1.4.0, NumPy 1.26.4. The simulator runs in a separate child process. |
| `POLICY_PYTHON` | `/mowenjie/fluffy-bassoon/envs/openpi-215abfb/bin/python`. Python 3.11.17 with the qualified headless OpenCV overlay, version 5.1.0. |
| `OPENPI_SOURCE`, `PI05_CHECKPOINT` | `/mowenjie/fluffy-bassoon/third_party/openpi-215abfb` at `215abfb217dbac7d5f1273282331b9b1866c0479`. Policy config `pi05_libero`, local weights `/mowenjie/fluffy-bassoon/inputs/pi05-libero-openpi-20261027-r1`. |
| `SYMMETA_SOURCE` | `/mowenjie/fluffy-bassoon-2/setups/xemb-icml-campaign-20261121-r1/source-ff2ddf0a` at `ff2ddf0ab03e6fbe243979556d8d8f2ebb9bc29c`. Supplies the Stage 2A streaming reader. |
| `SYMMETA_CHECKPOINT` | `/mowenjie/fluffy-bassoon-2/outputs/xemb-icml-campaign-20261121-r1/gpu03v2/stage1/checkpoints/equiv_v2_dis_dz-passive_attempt1-seed41.msgpack`. |

The original README links the Hugging Face release for parallel evaluation. Its December 2025 pin contains 160 BDDL/init pairs across 16 suite/perturbation combinations. The four original precomputed families are object (`object`), position (`swap`), semantic (`lan`) and task (`task`). The authors provide environment-replacement code, but the published snapshot has no `*_env` state pairs. The reported 53% includes that fifth family. The 32-case paired pilot uses the four released families. On Baige all 64 local asset SHA-256 digests matched the locked manifest, whose digest was `8c63b4fa0afcabae5e59669657c64a25d61da0ffda8050d19b4662b4a66d601c`. [Original leaderboard and setup](https://github.com/hongyuxiyohung/LIBERO-PRO/blob/e235a970758ecd78ecbb585fca6b915a8d44d18d/README.md) · [Pinned data](https://huggingface.co/datasets/zhouxueyang/LIBERO-Pro/tree/3a33dc4cfbea7ff5eac6764e9c557500972b6259) · [Case manifest](CASES.json)

The official pinned OpenPI server maps LIBERO to `pi05_libero` and `gs://openpi-assets/checkpoints/pi05_libero`. Its LIBERO client uses 256-pixel native cameras, 180° rotation and padded 224-pixel inputs, pose axis-angle and finger state, ten settling controls, and five controls per policy query. The horizon is 220 for Spatial, 280 for Object, 300 for Goal and 520 for LIBERO-10. Here the released BDDL supplies the case-preserving prompt. Both methods use the same motor and matched query noise. [Official inference instructions](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/examples/libero/README.md) · [Official client](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/examples/libero/main.py) · [Pilot protocol](PROTOCOL.md)

## Linux verification

`ssh baige-dev` reached `aibox-r855e239307b-6bfb688cdd-lz2dx`, Linux x86_64. The project PFS mount at `/mowenjie` reported 2.0 TiB total, 259.2 GiB available before staging and 1.16 million inodes available after staging. A fresh file passed write, sync and readback. The container exposes GPFS capacity via `df` and `statvfs`; it provides neither `quota` nor `mmlsquota` for an independent per-user quota readout.

Remote GitHub access timed out at 12 seconds. The Mac fetched the public Git histories from the named GitHub and Hugging Face origins. Complete-history Git bundles were copied with `scp` over the existing alias. Both passed Linux `git bundle verify` and supplied clean detached checkouts. All four Git pins in the interface passed a clean-worktree check. The simulator imports from the new checkout directly through `PYTHONPATH`. All generated caches stay in run-specific scratch, including a copy of the 4.3 MB PaliGemma tokenizer.

| CPU check | Result and durable Baige receipt |
| --- | --- |
| Frozen policy | All 16 checkpoint files (12,439,085,481 bytes) matched the recorded sizes and SHA-256 digests. `READY.json` SHA-256 `ec064cce32f5003e2e3979c5712d7d6385018f6bfcc59cb2161105881e70d170`. OpenPI resolved from the pinned checkout. |
| Streaming reader | Reader SHA-256 `c12bace5cb391499d6abd7cc5767f94b1519de7d77940efb0cfcb71ca0f686a4`. The pinned loader restored 33 parameter arrays for `factored_equiv_v2`, tag `equiv_v2_dis_dz`, on `TFRT_CPU_0`. Policy and reader receipt `$runtime/qualify-policy.h2gCXT/policy.json`. |
| Tokenizer | The isolated copy matched SHA-256 `8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6`. Official tokenization of the prompt with an eight-value fixture state produced 49 valid tokens in the configured 200-token array. Receipt `$runtime/qualify-tokenizer.lWXVaw/tokenizer.json`. |
| Native reset | Locked case `libero_goal_task/01/28`, seed 7. Released state 28 of 50 matched all 79 values exactly. Native tabletop model `MountedPanda`, `OSC_POSE`, seven actions, 20 Hz. The literal BDDL requests the plate although the file stem names the bowl. |
| Native cameras and step | Mesa llvmpipe (LLVM 15.0.7) rendered both 256×256×3 RGB observations. Two independent process resets reproduced agentview SHA-256 `3fd71f0b1b6cec76a19693f9b53a20ac637624c113febe0bb8a60cbc8e65636d` and wrist `88f6171c64134572859fed3a6865afbc604a99d4f5b26f4c0e3c16176cf84d2f`. One native settling control returned reward 0, done false and success false. Receipts `$runtime/qualify-sim.YARzNR/sim.json` and `$runtime/qualify-sim.wmVqSw/sim.json`. |

The evaluator owner corrected the Panda subclass check and qualified matched prefix execution using a CPU fixture motor. The real frozen-policy inference and first scored native pair require the parent's distributed GPU allocation. [Evaluator checks](CHECKS.md) · [Prior paired runtime and checkpoint provenance](../symmeta-fk-ur5e-20261123-r1/RUNTIME.md)

To verify or reproduce on the CPU workspace from the parent's deployed checkout, run the following. `install` accepts the retained `release.bundle` and `assets.bundle` in the runtime directory or can fetch the pinned public histories. It preserves existing checkouts and fails on a revision or cleanliness mismatch. `smoke` creates a fresh receipt directory, masks CUDA and selects Mesa software EGL.

```bash
here=outputs/symmeta-original-pro-20261123-r1
bash "$here/prepare_runtime.sh" verify
bash "$here/prepare_runtime.sh" smoke
```

For the evaluator's CPU wiring check, source `prepare_runtime.sh` and call `original_pro_paths`, `original_pro_cpu` and `original_pro_caches "$scratch"`, with a fresh writable scratch directory. The GPU wrapper exports the same interpreter, source and checkpoint paths.

## Parent launch on Baige Distributed Training

The existing Beijing console showed a working PyTorch allocation on **Pro6K → 算法Z组** (`aihcq-rh0wwk9muei4`) and the **复制** action. The earlier paired job `job-pc0ioesm8tle` on this route has a retained passing GPU probe, a merged summary and exit code 0. These checks establish the route. Queue availability and personal remaining GPU quota are time-dependent and need a fresh read in the copy form at admission. [Baige training console](https://console.bce.baidu.com/aihc/tasks) · [Baige copy-job instructions](https://cloud.baidu.com/doc/AIHC/s/nlibc3jme)

1. Main deploys the reviewed commit through Git into a clean detached checkout on the shared PFS. The run directory is ignored by default in this repository. Include the evaluator, manifest and scripts in that commit and verify the full SHA on the target.
2. In **大规模训练与推理 → 分布式训练**, copy the completed paired job. Select one PyTorch worker with **one NVIDIA RPBZZZ6 / RTX PRO 6000D**, 12 CPU cores, 96 GiB RAM and 8 GiB shared memory. Match the validated image `ccr-registry.baidubce.com/aihc/nvcr.io/nvidia/pytorch:2.7.0-cu12.8.0-py3.12-ubuntu24.04`, the project's read-write PFS at `/mowenjie`, and `NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics`. Set the scheduler maximum to 90 minutes, disable automatic retries and give the copy a fresh name.
3. Set the task's working directory to that detached checkout and use the command below, substituting the independently verified literal 40-character deployment SHA for `$PIN`. The last argument bounds the wrapper including startup. A single 90-minute allocation is at most 1.5 GPU-hours. The wrapper checks the Git pins, allocated GPU count/type, NVIDIA EGL renderer and hashes before calling `eval.py run --manifest ... --output ...`.

```bash
bash outputs/symmeta-original-pro-20261123-r1/run.sh "$PIN" pilot01 5400
```

The attempt directory `/mowenjie/fluffy-bassoon-2/outputs/symmeta-original-pro-20261123-r1/pilot01` must be fresh. `launch.txt`, `cases.input.json`, `gpu.json`, `run.log` and `exit.txt` belong to the wrapper. The evaluator writes under `eval/`. Confirm the first completed real pair and its matching prefix, then check every 20 minutes through the final `eval/summary.json` and exit code. Wrapper Bash syntax, the Linux pin check and the rejection of a non-scheduler host passed during preparation. [Launch wrapper](run.sh) · [CPU smoke](smoke_runtime.py)
