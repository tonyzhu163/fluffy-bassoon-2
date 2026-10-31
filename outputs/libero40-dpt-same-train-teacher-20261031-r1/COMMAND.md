# Optional immutable same-TRAIN-teacher DPT generation

Prepared only. This CPU task is independent of source/target collection and changes neither legacy100 data nor its teacher-transfer metrics. No GPU, simulator, SGD, VLA inference, environment installation or fallback is permitted. Retain partial outputs on backend failure or the original 20-minute deadline.

Use an actual Baige allocation with **0 GPUs, 4 CPUs, 16 GiB RAM, 20 minutes**. The launcher first attests actual zero-device visibility and cgroup resources using qualified Python3.12; focused writer, adapter and legacy tests run there. Qualified OpenPI Python3.11 + DSRL then evaluates the eight frozen TRAIN PixelSAC actors on CPU using recorded canonical SAC inputs. It verifies each pure actor digest against the full checkpoint and checks an actual TRAIN target replay within 1e-5. No model work is permitted on the development host or Mac.

```bash
bash "$PINNED/scripts/diagnostics/vla_source40_dpt_same_teacher_cpu.sh" \
  "$SOURCE" libero40-dpt-same-train-teacher-cpu-20261031-r1 \
  /mowenjie/fluffy-bassoon-2/outputs/libero40-source-pipeline-cpu-20261031-r2/snapshot/admission.json \
  26303151bc26bc6809348b7b3e3351205619efbfb21f34da4b70229a33198fe2
```

`$PINNED` must be a clean worktree at the integrated writer + label-admission adapter commit; `$SOURCE` is that full exact commit. Fresh output only. Admission cost remains **457851 source actions**, counted once; derived labels add zero actions and zero GPU seconds.

Outputs: per-goal derived VAL DPT NPZ + SHA-bound sidecar, `generation/binding.json`, focused test log, actual CPU/GPU attestation, cumulative CPU clock, and a fresh assembled generation. Metadata-only `load_binding(...)` binds the eight original behavior streams separately from each matching TRAIN label actor. Assembly admission replaces only VAL DPT labels and verifies byte-equivalent non-target fields; the wrapper also round-trips the fresh assembly and proves all AD and TRAIN arrays unchanged. Originals and default admissions remain unchanged.

The new metric is `heldout_state_same_train_teacher`; legacy `teacher_transfer` remains a different metric. Descriptive old/new target deltas do not establish robot performance, actor optimality or in-context adaptation. Fixed500 policy selection remains unchanged; source collection can proceed if this optional CPU path fails.
