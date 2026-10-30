#!/usr/bin/env bash
# Existing approved allocation only. Exactly ONE full-memory native-horizon PPO batch.
set -Eeuo pipefail
TASK_STARTED=$(date +%s)
TASK_SOURCE=${1:?reviewed exact commit}
TASK_RUN=${2:?fresh immutable qualification identity}
TASK_BUDGET=${3:?root-created exact20k/2400s funding record}
TASK_RESUME=${4:-}
case "$TASK_RESUME" in ''|--resume) ;; *) exit 2;; esac
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_OLD=/mowenjie/fluffy-bassoon
TASK_WT=$TASK_ROOT/worktrees/$TASK_RUN
TASK_OUT=$TASK_ROOT/outputs/$TASK_RUN
TASK_MODEL=$TASK_OLD/envs/openpi-215abfb/bin/python
TASK_PROBE=$TASK_OLD/envs/lerobot-v044-8fff0fde/bin/python
TASK_OPENPI=$TASK_OLD/third_party/openpi-215abfb
TASK_CUDA=$TASK_ROOT/outputs/libero-pi05-student-cuda-20261029-r1/cu128-overlay
TASK_LIBERO=$TASK_ROOT/deps/libero-dsrl-87edbd1-20261029-r1/repo
TASK_SCHEDULE=$TASK_ROOT/outputs/libero40-source-1m-20261029-r1/schedule.json
TASK_RESUME_ARGS=()
if [ "$TASK_RESUME" = --resume ]; then
  test -f "$TASK_OUT/worker-time.json"
  TASK_RESUME_ARGS=(--resume)
else
  test ! -e "$TASK_OUT"
  mkdir -p "$TASK_OUT"
fi
cd "$TASK_WT"
test "$(git rev-parse HEAD)" = "$TASK_SOURCE"
test -z "$(git status --porcelain)"
exec 9> "$TASK_OUT/launcher.lock"
flock -n 9
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export PYTHONPATH=$TASK_WT/src
TASK_CLOCK=$("$TASK_MODEL" -m fluffy_bassoon.meta_rl.vla_source40_batch14_qualification begin \
  --root "$TASK_OUT" --budget "$TASK_BUDGET" --schedule "$TASK_SCHEDULE" \
  --source "$TASK_SOURCE" --run "$TASK_RUN" --started-unix "$TASK_STARTED" "${TASK_RESUME_ARGS[@]}")
IFS=$'\t' read -r TASK_ATTEMPT TASK_STOP TASK_SLICE <<< "$TASK_CLOCK"
TASK_TELEMETRY_PID=''
exec > >(tee "$TASK_ATTEMPT/console.log") 2>&1
task_cleanup() {
  TASK_RC=$?
  trap - EXIT
  if [ -n "$TASK_TELEMETRY_PID" ]; then
    kill "$TASK_TELEMETRY_PID" 2>/dev/null || true
    wait "$TASK_TELEMETRY_PID" 2>/dev/null || true
  fi
  PYTHONPATH=$TASK_WT/src "$TASK_MODEL" -m fluffy_bassoon.meta_rl.vla_source40_ppo_worker finish \
    --root "$TASK_OUT" --attempt "$TASK_ATTEMPT" --exit-code "$TASK_RC" || true
  PYTHONPATH=$TASK_WT/src "$TASK_MODEL" -m fluffy_bassoon.meta_rl.vla_source40_batch14_qualification report \
    --root "$TASK_OUT" --attempt "$TASK_ATTEMPT" --exit-code "$TASK_RC" || true
  printf '%s\n' "$TASK_RC" > "$TASK_ATTEMPT/exit-code.txt"
  printf '%s\n' "$TASK_RC" > "$TASK_OUT/exit-code.txt"
  exit "$TASK_RC"
}
trap task_cleanup EXIT
export FLUFFY_GPU_EXECUTION=baige CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=disabled
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
# Python3.10 probe/tests must not see the Python3.11 cu128 model overlay.
PYTHONPATH=$TASK_WT/src "$TASK_PROBE" 'scripts/[train]gpu_probe.py' --expected-count 1 --purpose training \
  --baige-container --baige-exclusive-training --output "$TASK_ATTEMPT/gpu-probe.json"
export LYCO_BAIGE_GPU_MAP=$TASK_ATTEMPT/gpu-probe.json
TASK_GPU_UUID=$("$TASK_MODEL" - "$TASK_ATTEMPT/gpu-probe.json" <<'PYGPU'
import json,sys
print(json.load(open(sys.argv[1]))['devices'][0]['uuid'])
PYGPU
)
nvidia-smi --id="$TASK_GPU_UUID" --query-gpu=timestamp,uuid,memory.used,utilization.gpu \
  --format=csv,noheader,nounits --loop-ms=1000 > "$TASK_ATTEMPT/device-telemetry.csv" 2> "$TASK_ATTEMPT/device-telemetry.stderr" &
TASK_TELEMETRY_PID=$!
CUDA_VISIBLE_DEVICES='' PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  PYTHONPATH=$TASK_ROOT/envs/pytest-overlay-20261029-r1:$TASK_WT/src \
  "$TASK_PROBE" -m pytest -q -p no:cacheprovider \
  tests/test_vla_rl2_batch.py tests/test_vla_source40_batch14.py \
  tests/test_vla_reset_identity.py tests/test_vla_source40_ppo_worker.py \
  tests/test_vla_source40.py::test_resume_charges_unresolved_execution_intent_conservatively \
  > "$TASK_ATTEMPT/focused-tests.log" 2>&1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0
export LIBERO_CONFIG_PATH=$TASK_ROOT/cache/libero-dsrl-87edbd1-20261029-r1
export OPENPI_DATA_HOME=$TASK_OLD/cache/openpi-215abfb
export MAGICK_HOME=/mowenjie/libs/libero-plus-imagemagick
export MAGICK_CODER_MODULE_PATH=$MAGICK_HOME/usr/lib/x86_64-linux-gnu/ImageMagick-6.9.11/modules-Q16/coders
export LD_LIBRARY_PATH=$MAGICK_HOME/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PYTHONPATH=$TASK_CUDA:$TASK_WT/src:$TASK_OPENPI/src:$TASK_OPENPI/packages/openpi-client/src
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$TASK_ROOT/envs/pytest-overlay-20261029-r1:$PYTHONPATH \
  "$TASK_MODEL" -m pytest -q -p no:cacheprovider \
  tests/test_vla_source40.py::test_motor_rng_round_trip_preserves_next_random_draw \
  > "$TASK_ATTEMPT/motor-rng-tests.log" 2>&1
TASK_DRIVER_RESUME=()
if [ -e "$TASK_OUT/experiment" ]; then TASK_DRIVER_RESUME=(--resume); fi
TASK_TIMEOUT=$((TASK_STOP - $(date +%s) - 60))
test "$TASK_TIMEOUT" -gt 420
# The trainer checks between complete K4 trials; keep300s between its soft
# stop and outer timeout for the longest native trial and checkpoint cleanup.
timeout --signal=INT --kill-after=30s "$TASK_TIMEOUT" "$TASK_MODEL" \
  -m fluffy_bassoon.meta_rl.vla_source40_batch14 \
  --schedule "$TASK_SCHEDULE" --output "$TASK_OUT/experiment" \
  --checkpoint "$TASK_OLD/inputs/pi05-libero-openpi-20261027-r1" --openpi-root "$TASK_OPENPI" \
  --simulator-python /mowenjie/envs/wam-libero-plus-py310/bin/python \
  --simulator-pythonpath "$TASK_WT/src:$TASK_LIBERO:$TASK_OPENPI/packages/openpi-client/src:$TASK_OLD/envs/pi05-plus-client-deps" \
  --source-commit "$TASK_SOURCE" --expected-hostname "$(hostname)" \
  --max-total-actions 20000 --stop-unix "$((TASK_STOP - 360))" --updates 1 --seed 2 --memory-mode full \
  --width 128 --layers 2 --heads 4 --context-length 300 "${TASK_DRIVER_RESUME[@]}"
