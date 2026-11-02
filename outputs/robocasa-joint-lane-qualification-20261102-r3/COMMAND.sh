#!/usr/bin/env bash
set -Eeuo pipefail
: "${TASK_META:?}" "${TASK_CODE_PIN:?}"
export TASK_STARTED=$(date +%s)
export TASK_OUT=/mowenjie/fluffy-bassoon-2/outputs/robocasa-joint-lane-qualification-20261102-r3/gpu
mkdir "$TASK_OUT"
export PYTHONPATH=$TASK_META/src PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export FLUFFY_GPU_EXECUTION=baige NCCL_P2P_DISABLE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
export TMPDIR=/tmp/joint-lane-qualification
mkdir -p "$TMPDIR"
exec > >(tee "$TASK_OUT/console.log") 2>&1
trap 'rc=$?; printf "%s\n" "$rc" > "$TASK_OUT/exit-code.txt"; date +%s > "$TASK_OUT/ended-unix.txt"' EXIT
printf '%s\n' "$TASK_STARTED" > "$TASK_OUT/started-unix.txt"
timeout --signal=TERM --kill-after=30s 930 /mowenjie/fluffy-bassoon-2/setups/robocasa365-native-20261101-r1/env/bin/python -B "$TASK_META/outputs/robocasa-joint-lane-qualification-20261102-r3/WORKLOAD.py"
