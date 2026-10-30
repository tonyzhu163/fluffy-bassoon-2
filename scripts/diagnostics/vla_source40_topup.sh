#!/usr/bin/env bash
# Requires an already approved allocation. No scheduler submission or GPU grant.
set -Eeuo pipefail
TASK_STARTED=$(date +%s)
TASK_SOURCE=${1:?reviewed40hex commit}
TASK_RUN=${2:?fresh immutable output identity}
TASK_PLAN=${3:?prepared immutable topup plan}
TASK_SLOT=${4:?source_task:NN/train_or_validation}
TASK_LEDGER=${5:?canonical cumulative topup ledger}
TASK_SLICE=${6:?invocation slice seconds excluding60startup allowance}
TASK_WORKER=${7:?cumulative worker seconds including setup and retries}
TASK_RESUME=${8:-}
case "$TASK_RESUME" in ''|--resume) ;; *) exit 2;; esac
case "$TASK_RUN" in *[!a-zA-Z0-9-]*|'') exit 2;; esac
TASK_CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_OLD=/mowenjie/fluffy-bassoon
TASK_OUTPUT=$TASK_ROOT/outputs/$TASK_RUN
TASK_MODEL=$TASK_OLD/envs/openpi-215abfb/bin/python
TASK_TORCH=$TASK_OLD/envs/lerobot-v044-8fff0fde/bin/python
TASK_TEST=$TASK_ROOT/envs/pytest-overlay-20261029-r1
TASK_DSRL=$TASK_ROOT/deps/dsrl-pi0-7f48937-20261029-r1/repo
TASK_LIBERO=$TASK_ROOT/deps/libero-dsrl-87edbd1-20261029-r1/repo
TASK_OPENPI=$TASK_OLD/third_party/openpi-215abfb
TASK_OVERLAY=$TASK_ROOT/envs/dsrl-pi0-overlay-20261029-r1
TASK_SIM=/mowenjie/envs/wam-libero-plus-py310/bin/python
cd "$TASK_CODE"
test "$(git rev-parse HEAD)" = "$TASK_SOURCE"
test -z "$(git status --porcelain)"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export PYTHONPATH=$TASK_CODE/src
TASK_RESUME_ARGS=()
if [ "$TASK_RESUME" = --resume ]; then TASK_RESUME_ARGS=(--resume); fi
TASK_ATTEMPT=$("$TASK_MODEL" -m fluffy_bassoon.meta_rl.vla_source40_topup clock-begin \
  --plan "$TASK_PLAN" --output "$TASK_OUTPUT" --ledger "$TASK_LEDGER" --slot "$TASK_SLOT" \
  --source-commit "$TASK_SOURCE" --slice-seconds "$TASK_SLICE" --worker-seconds "$TASK_WORKER" \
  --started-unix "$TASK_STARTED" "${TASK_RESUME_ARGS[@]}")
test -f "$TASK_ATTEMPT/attempt.json"
exec > >(tee "$TASK_ATTEMPT/console.log") 2>&1
TASK_SIM_PID=
task_cleanup() {
  TASK_RC=$?
  trap - EXIT
  if [ -n "$TASK_SIM_PID" ]; then
    kill "$TASK_SIM_PID" 2>/dev/null || true
    wait "$TASK_SIM_PID" 2>/dev/null || true
  fi
  PYTHONPATH=$TASK_CODE/src "$TASK_MODEL" -m fluffy_bassoon.meta_rl.vla_source40_topup clock-finish \
    --plan "$TASK_PLAN" --ledger "$TASK_LEDGER" --output "$TASK_OUTPUT" \
    --attempt "$TASK_ATTEMPT" --exit-code "$TASK_RC" || true
  printf '%s\n' "$TASK_RC" > "$TASK_OUTPUT/exit-code.txt"
  exit "$TASK_RC"
}
trap task_cleanup EXIT
export FLUFFY_GPU_EXECUTION=baige
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0
export OPENPI_DATA_HOME=$TASK_OLD/cache/openpi-215abfb
export LIBERO_CONFIG_PATH=$TASK_ROOT/cache/libero-dsrl-87edbd1-20261029-r1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 WANDB_MODE=disabled
export MAGICK_HOME=/mowenjie/libs/libero-plus-imagemagick
export MAGICK_CODER_MODULE_PATH=$MAGICK_HOME/usr/lib/x86_64-linux-gnu/ImageMagick-6.9.11/modules-Q16/coders
export LD_LIBRARY_PATH=$MAGICK_HOME/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
"$TASK_TORCH" 'scripts/[train]gpu_probe.py' --expected-count 1 --purpose training \
  --baige-container --baige-exclusive-training --output "$TASK_ATTEMPT/gpu-probe.json"
export LYCO_BAIGE_GPU_MAP=$TASK_ATTEMPT/gpu-probe.json
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH="$TASK_TEST:$TASK_CODE/src" \
  "$TASK_TORCH" -m pytest -q -p no:cacheprovider \
  tests/test_vla_source40_topup.py tests/test_vla_source40.py tests/test_vla_source40_ppo_worker.py \
  > "$TASK_ATTEMPT/focused-tests.log" 2>&1
TASK_SUITE=${TASK_SLOT%%:*}
TASK_NUMBER=${TASK_SLOT#*:}
TASK_NUMBER=${TASK_NUMBER%%/*}
PYTHONPATH=$TASK_CODE/src:$TASK_LIBERO:$TASK_OPENPI/packages/openpi-client/src:$TASK_OLD/envs/pi05-plus-client-deps \
  "$TASK_SIM" -m fluffy_bassoon.meta_rl.vla_libero --mode env-server \
  --output "$TASK_ATTEMPT/simulator" --port 8765 --suite "$TASK_SUITE" --task-id "$TASK_NUMBER" \
  --horizon 400 --fresh-environment-on-reset > "$TASK_ATTEMPT/simulator.log" 2>&1 &
TASK_SIM_PID=$!
for TASK_POLL in $(seq 1 180); do
  if [ -e "$TASK_ATTEMPT/simulator/simulator-ready.json" ]; then break; fi
  kill -0 "$TASK_SIM_PID"
  sleep 1
done
test -e "$TASK_ATTEMPT/simulator/simulator-ready.json"
TASK_STOP=$((TASK_STARTED + TASK_SLICE - 60))
TASK_OUTER=$((TASK_STARTED + TASK_SLICE - 20 - $(date +%s)))
test "$TASK_OUTER" -gt 180
export PYTHONPATH=$TASK_OVERLAY:$TASK_CODE/src:$TASK_DSRL:$TASK_OPENPI/src:$TASK_OPENPI/packages/openpi-client/src
timeout --signal=TERM --kill-after=10s "${TASK_OUTER}s" "$TASK_MODEL" \
  -m fluffy_bassoon.meta_rl.vla_source40_topup run --plan "$TASK_PLAN" --ledger "$TASK_LEDGER" \
  --output "$TASK_OUTPUT" --slot "$TASK_SLOT" --source-commit "$TASK_SOURCE" \
  --expected-hostname "$(hostname)" --stop-unix "$TASK_STOP" --port 8765 \
  --checkpoint "$TASK_OLD/inputs/pi05-libero-openpi-20261027-r1" \
  --openpi-root "$TASK_OPENPI" --dsrl-root "$TASK_DSRL" "${TASK_RESUME_ARGS[@]}"
