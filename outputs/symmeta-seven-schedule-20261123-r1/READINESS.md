# Qualification and parent handoff

**Ready on deployment commit `ba3c41e25575d07d7bd8bbfaae2797e7cf9dc0a9`.** Baige CPU attempt `cpu03` exited zero in 38 seconds from the clean detached production checkout. The parent subsequently launched `gpu01` under Baige Distributed Training job `job-v9luygbnvwq2`. At 16:07 Shanghai on November 23, the exclusive one-device probe had passed on the Pro6K (`NVIDIA RPBZZZ6`), and seed 41 had started its 3,500 updates. The parent owns the running job. [CPU log](qualification/cpu.log) · [Independent readback](qualification/readback.json)

## Real-input qualification

- The Stage 1 loader read the original 1,344 passive source traces in archived order, verified the ordered tape digest and fitted the original nominal gain (0.054141017027535904). Each arm contributes 672 traces and each of seven shifts contributes 192. The archived dev audit and official `gpu09test` comparison ran on their respective inputs.
- The original Stage 1 training function in the development base matches the original job revision `d5638457` before the optional schedule argument. Static comparison also confirmed the archive reader, DIS computation, Perceiver CNN and attention helpers, and factor loss. The schedule-only branch uses the trainer’s original sampler and JAX initialization. The readback records a fingerprint of all 3,500 sampled index batches for each seed.
- Each of seeds 41, 42 and 43 ran the first two updates of the full schedule on CPU, at batch 64. Count 0 applied zero LR, count 1 applied 0.0000015000114, count 200 is the peak 0.000300000014, and count 3,499 applies 0.0000150000687. All three smoke checkpoints reloaded with the original 32-slot architecture and finite parameter changes from initialization.
- The two arms cover 288 matched trace–seed pairs across 96 supports, 64 OOD, and eight tasks. Independent readback verified all 576 predictions’ circular yaw, normalized planar operator error, gain, gripper and factor-correctness values. It independently reconstructed the seed-averaged OOD yaw difference and both whole-task bootstrap bounds. The local compressed archive contains the complete CPU rows and summary. The tiny smoke updates serve only to qualify the pipeline. [CPU rows and summary](qualification/cpu-metrics.tar.gz)
- The CPU launcher guard rejected the development host before creating an attempt. Python parsing, Bash syntax and Git whitespace checks passed on macOS. The successful `cpu03` readback used exactly the deployed files. The earlier provisional `cpu01` smoke passed. A second patch-based staging copy, `cpu02`, omitted the trainer opt-in when the parent advanced the Git base during capture and stopped before any update. Its log remains under the run root.
- The three saved original checkpoints reproduced the historical GPU OOD yaw figures within 0.018° on CPU. The comparison produced these recorded values.

| Reader seed | Original OOD yaw MAE, CPU | Original OOD yaw MAE, PR7 GPU | Original operator error on all 96, CPU |
| ---: | ---: | ---: | ---: |
| 41 | 97.08268° | 97.07525° | 0.980559 |
| 42 | 92.32526° | 92.34247° | 0.934743 |
| 43 | 97.47418° | 97.47559° | 0.984341 |

## Exact production invocation

The parent used the following entry point inside the one-GPU scheduler container. The run root already contains `gpu01`, and the launcher's exclusive directory creation protects this attempt. Use a fresh attempt ID only for a separately authorized retry after diagnosis.

```bash
bash /mowenjie/fluffy-bassoon-2/setups/symmeta-perceiver-followups-20261123-r1/src-ba3c41e25575/outputs/symmeta-seven-schedule-20261123-r1/launch.sh ba3c41e25575d07d7bd8bbfaae2797e7cf9dc0a9 gpu01
```

The shared run root is `baige-dev:/mowenjie/fluffy-bassoon-2/outputs/symmeta-seven-schedule-20261123-r1/`.

| Relative path | Contents |
| --- | --- |
| `gpu01/launch.txt`, `gpu01/gpu.json`, `gpu01/run.log`, `gpu01/exit.txt` | Production revision, allocation, progress and terminal status |
| `gpu01/results/checkpoints/perceiver_icba_dis-passive_attempt1-seed{41,42,43}.msgpack` | Three production candidate checkpoints |
| `gpu01/results/paired-cases-{41,42,43}.json` | Matched raw predictions and errors for 96 official supports per seed |
| `gpu01/results/contract.json`, `gpu01/results/dev-audit-{41,42,43}.json` | Runtime, schedule, immutable inputs and Stage 1 archive-dev diagnostics |
| `gpu01/results/summary.json`, `gpu01/results/RESULT.md` | Production paired estimates and 95% task-cluster intervals |
| `cpu03/results/` | Qualification checkpoints, case rows, summary, contract and dev audits |

To reproduce CPU qualification, use a fresh `qa` location on the shared volume with `JAX_PLATFORMS=cpu`, empty `CUDA_VISIBLE_DEVICES`, `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `OPENCV_FOR_THREADS_NUM=1`, `XLA_FLAGS=--xla_cpu_multi_thread_eigen=false`, and the OpenPI Python and `PYTHONPATH` from the launcher. The qualified invocation was `taskset -c 0-3 "$python" -Bu "$here/run.py" --cpu-qualification --revision "$pin" --output "$qa/results"`.
