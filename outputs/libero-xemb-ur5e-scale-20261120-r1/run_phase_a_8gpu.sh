#!/usr/bin/env bash
set -euo pipefail

COMMIT="${1:?commit required}"
RUN_TAG="${2:-gpu03phasea}"
SRC_TAG="${3:-gpu01}"
NUM_GPUS=8

BASE="/mowenjie/fluffy-bassoon-2"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="$BASE/outputs/libero-xemb-ur5e-scale-20261120-r1/$RUN_TAG"
SRC="$BASE/outputs/libero-xemb-ur5e-scale-20261120-r1/$SRC_TAG"
PY="$BASE/setups/robocasa365-native-20261101-r1/env/bin/python"
SCRIPT="$REPO/outputs/libero-xemb-ur5e-scale-20261120-r1/phase_a_hybrid.py"

mkdir -p "$OUT/logs" "$OUT/train" "$OUT/live"
ACTUAL_COMMIT="$(git -C "$REPO" rev-parse HEAD)"
test "$ACTUAL_COMMIT" = "$COMMIT"

export PYTHONPATH="$REPO/src:$REPO/outputs/libero-xemb-ur5e-sysid-20261120-r1:$REPO/outputs/libero-xemb-ur5e-scale-20261120-r1"
export NCCL_P2P_DISABLE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.55
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONUNBUFFERED=1

START_TS="$(date +%s)"
echo "$COMMIT $(hostname) $$ phase-a-hybrid" > "$OUT/launch.txt"
date -u +"%Y-%m-%dT%H:%M:%SZ" > "$OUT/gpu-start.txt"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
"$PY" -Bu "$REPO/scripts/[train]gpu_probe.py" \
  --expected-count "$NUM_GPUS" --purpose training --baige-container --baige-exclusive-training \
  --output "$OUT/gpu.json" > "$OUT/probe.log" 2>&1

# Step 1: Train 3-seed 10-channel hybrid reader on GPU 0 (~3 min).
CUDA_VISIBLE_DEVICES=0 "$PY" "$SCRIPT" train-hybrid \
  --archive "$SRC/archive/archive.json" \
  --out-dir "$OUT/train" \
  --steps 3000 --batch-size 64 --seeds 41 42 43 --cnn-width 96 --mlp-width 192 \
  > "$OUT/logs/train-hybrid.log" 2>&1

# Step 2: 8-GPU sharded live closed-loop eval (96 UR5e queries x 6 arms = 576 episodes, ~18 min).
PIDS=()
for i in $(seq 0 $((NUM_GPUS - 1))); do
  CUDA_VISIBLE_DEVICES="$i" MUJOCO_EGL_DEVICE_ID="$i" "$PY" "$SCRIPT" eval-shard \
    --archive "$SRC/archive/archive.json" \
    --base-train-dir "$SRC/train" \
    --hybrid-train-dir "$OUT/train" \
    --output-dir "$OUT/live" \
    --shard-idx "$i" \
    --num-shards "$NUM_GPUS" \
    --seed-ckpt 41 \
    > "$OUT/logs/eval-shard-$i.log" 2>&1 &
  PIDS+=($!)
done

FAIL=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    FAIL=1
  fi
done
if [ "$FAIL" -ne 0 ]; then
  echo "1 $(date +%s) $START_TS" > "$OUT/exit.txt"
  exit 1
fi

CUDA_VISIBLE_DEVICES=0 "$PY" "$SCRIPT" merge \
  --output-dir "$OUT/live" \
  --num-shards "$NUM_GPUS" \
  > "$OUT/logs/merge.log" 2>&1

echo "0 $(date +%s) $START_TS" > "$OUT/exit.txt"
