#!/usr/bin/env bash
set -Eeuo pipefail
: "${TASK_META:?}" "${TASK_CODE_PIN:?}"
export TASK_STARTED=$(date +%s)
export TASK_OUT=/mowenjie/fluffy-bassoon-2/outputs/robocasa-rollout-capacity-1632-20261103-r1/gpu
mkdir "$TASK_OUT"
export PYTHONPATH=$TASK_META/src PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export FLUFFY_GPU_EXECUTION=baige NCCL_P2P_DISABLE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
export TMPDIR=/tmp/rollout-capacity-1632
mkdir -p "$TMPDIR"
exec > >(tee "$TASK_OUT/console.log") 2>&1
trap 'rc=$?; printf "%s\n" "$rc" > "$TASK_OUT/exit-code.txt"; date +%s > "$TASK_OUT/ended-unix.txt"' EXIT
timeout --signal=TERM --kill-after=30s 1770 /mowenjie/fluffy-bassoon-2/setups/robocasa365-native-20261101-r1/env/bin/python -B "$TASK_META/outputs/robocasa-rollout-capacity-1632-20261103-r1/WORKLOAD.py"
