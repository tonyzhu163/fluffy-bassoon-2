#!/usr/bin/env bash
# Run only in the already-authorized exclusive single-GPU Baige container.
set -Eeuo pipefail
TASK_SOURCE=${1:?exact integrated source commit}
TASK_RUN=${2:?unique output identity}
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_OLD=/mowenjie/fluffy-bassoon
TASK_WORKTREE=$TASK_ROOT/worktrees/$TASK_RUN
TASK_OUTPUT=$TASK_ROOT/outputs/$TASK_RUN
TASK_DSRL=$TASK_ROOT/deps/dsrl-pi0-7f48937-20261029-r1/repo
TASK_OVERLAY=$TASK_ROOT/envs/dsrl-pi0-overlay-20261029-r1
TASK_OPENPI=$TASK_OLD/third_party/openpi-215abfb
TASK_PLUS=$TASK_OLD/third_party/libero-plus-4976dc3-r3
TASK_MODEL=$TASK_OLD/envs/openpi-215abfb/bin/python
TASK_SIM=/mowenjie/envs/wam-libero-plus-py310/bin/python
TASK_TORCH=$TASK_OLD/envs/lerobot-v044-8fff0fde/bin/python
TASK_TEST_OVERLAY=$TASK_ROOT/envs/pytest-overlay-20261029-r1
TASK_CHECKPOINT=$TASK_ROOT/inputs/pi0-libero-openpi-g1748456809412170-20261029-r1
test ! -e "$TASK_OUTPUT"
mkdir -p "$TASK_OUTPUT"
cd "$TASK_WORKTREE"
test "$(git rev-parse HEAD)" = "$TASK_SOURCE"
test -z "$(git status --porcelain)"
exec > >(tee "$TASK_OUTPUT/console.log") 2>&1
TASK_SIM_PID=
task_cleanup() {
  TASK_RC=$?
  trap - EXIT
  if [ -n "$TASK_SIM_PID" ]; then
    kill "$TASK_SIM_PID" 2>/dev/null || true
    wait "$TASK_SIM_PID" 2>/dev/null || true
  fi
  printf '%s\n' "$TASK_RC" > "$TASK_OUTPUT/exit-code.txt"
  exit "$TASK_RC"
}
trap task_cleanup EXIT
export FLUFFY_GPU_EXECUTION=baige CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0
export OPENPI_DATA_HOME=$TASK_OLD/cache/openpi-215abfb
export LIBERO_CONFIG_PATH=$TASK_OLD/cache/pi05-libero-plus-4976dc3-r3
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 WANDB_MODE=disabled
export MAGICK_HOME=/mowenjie/libs/libero-plus-imagemagick
export MAGICK_CODER_MODULE_PATH=$MAGICK_HOME/usr/lib/x86_64-linux-gnu/ImageMagick-6.9.11/modules-Q16/coders
export LD_LIBRARY_PATH=$MAGICK_HOME/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PYTHONPATH=$TASK_WORKTREE/src
"$TASK_TORCH" 'scripts/[train]gpu_probe.py' --expected-count 1 --purpose training \
  --baige-container --output "$TASK_OUTPUT/gpu-probe.json"
export LYCO_BAIGE_GPU_MAP=$TASK_OUTPUT/gpu-probe.json
CUDA_VISIBLE_DEVICES='' PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH="$TASK_TEST_OVERLAY:$TASK_WORKTREE/src" \
  "$TASK_TORCH" -m pytest -q -p no:cacheprovider tests/test_vla_history.py \
  tests/test_vla_retry_teacher.py tests/test_vla_performance_report.py tests/test_vla_libero.py \
  > "$TASK_OUTPUT/focused-tests.log" 2>&1
export PYTHONPATH=$TASK_WORKTREE/src:$TASK_PLUS:$TASK_OPENPI/packages/openpi-client/src:$TASK_OLD/envs/pi05-plus-client-deps
"$TASK_SIM" -m fluffy_bassoon.meta_rl.vla_libero --mode env-server \
  --output "$TASK_OUTPUT/simulator" --port 8765 --suite libero_90 --task-id 57 \
  --horizon 400 > "$TASK_OUTPUT/simulator.log" 2>&1 &
TASK_SIM_PID=$!
for TASK_POLL in $(seq 1 60); do
  if [ -e "$TASK_OUTPUT/simulator/simulator-ready.json" ]; then break; fi
  kill -0 "$TASK_SIM_PID"
  sleep 1
done
test -e "$TASK_OUTPUT/simulator/simulator-ready.json"
export PYTHONPATH=$TASK_OVERLAY:$TASK_WORKTREE/src:$TASK_DSRL:$TASK_OPENPI/src:$TASK_OPENPI/packages/openpi-client/src
"$TASK_MODEL" -m fluffy_bassoon.meta_rl.vla_libero --mode teacher \
  --output "$TASK_OUTPUT/experiment" --port 8765 --suite libero_90 --task-id 57 \
  --openpi-root "$TASK_OPENPI" --dsrl-root "$TASK_DSRL" --checkpoint "$TASK_CHECKPOINT" \
  --model-config pi0_libero --chunk 20 --horizon 400 --source-commit "$TASK_SOURCE" \
  --max-steps 4000 --warmup-chunks 32 --batch-size 32 --utd 20 \
  --eval-states 40 --attempts 2 --history-updates 10 --history-device cpu --history-objective ad
