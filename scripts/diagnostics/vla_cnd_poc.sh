#!/usr/bin/env bash
# Fixed teacher gate, continuous noise histories, and frozen student robot POC.
# Run only inside the authorized exclusive single-GPU Baige allocation.
set -Eeuo pipefail
TASK_SOURCE=${1:?exact integrated source commit}
TASK_RUN=${2:?fresh CND output identity}
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_OLD=/mowenjie/fluffy-bassoon
TASK_WORKTREE=$TASK_ROOT/worktrees/$TASK_RUN
TASK_OUTPUT=$TASK_ROOT/outputs/$TASK_RUN
TASK_DSRL=$TASK_ROOT/deps/dsrl-pi0-7f48937-20261029-r1/repo
TASK_OVERLAY=$TASK_ROOT/envs/dsrl-pi0-overlay-20261029-r1
TASK_OPENPI=$TASK_OLD/third_party/openpi-215abfb
TASK_LIBERO=$TASK_ROOT/deps/libero-dsrl-87edbd1-20261029-r1/repo
TASK_MODEL=$TASK_OLD/envs/openpi-215abfb/bin/python
TASK_SIM=/mowenjie/envs/wam-libero-plus-py310/bin/python
TASK_TORCH=$TASK_OLD/envs/lerobot-v044-8fff0fde/bin/python
TASK_TEST_OVERLAY=$TASK_ROOT/envs/pytest-overlay-20261029-r1
TASK_CHECKPOINT=$TASK_ROOT/inputs/pi0-libero-openpi-g1748456809412170-20261029-r1
TASK_PRIOR=${3:-$TASK_ROOT/outputs/libero-vla-baige-20261029-qual-r3/experiment}
test ! -e "$TASK_OUTPUT"
test -f "$TASK_PRIOR/source-recovery.pkl"
test -f "$TASK_PRIOR/contract.json"
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
export LIBERO_CONFIG_PATH=$TASK_ROOT/cache/libero-dsrl-87edbd1-20261029-r1
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
  tests/test_vla_features.py tests/test_vla_multitask.py tests/test_vla_rl2.py \
  tests/test_vla_noise_distillation.py tests/test_vla_cnd_run.py \
  > "$TASK_OUTPUT/focused-tests.log" 2>&1
export PYTHONPATH=$TASK_WORKTREE/src:$TASK_LIBERO:$TASK_OPENPI/packages/openpi-client/src:$TASK_OLD/envs/pi05-plus-client-deps
"$TASK_SIM" -m fluffy_bassoon.meta_rl.vla_libero --mode env-server \
  --output "$TASK_OUTPUT/simulator" --port 8765 --suite libero_90 --task-id 57 \
  --horizon 400 > "$TASK_OUTPUT/simulator.log" 2>&1 &
TASK_SIM_PID=$!
for TASK_POLL in $(seq 1 600); do
  if [ -e "$TASK_OUTPUT/simulator/simulator-ready.json" ]; then break; fi
  kill -0 "$TASK_SIM_PID"
  sleep 1
done
if [ ! -e "$TASK_OUTPUT/simulator/simulator-ready.json" ]; then
  tail -n 100 "$TASK_OUTPUT/simulator.log"
  printf 'Simulator failed readiness after 600 seconds\n' >&2
  exit 2
fi
export PYTHONPATH=$TASK_OVERLAY:$TASK_WORKTREE/src:$TASK_DSRL:$TASK_OPENPI/src:$TASK_OPENPI/packages/openpi-client/src
"$TASK_MODEL" -m fluffy_bassoon.meta_rl.vla_cnd_run \
  --output "$TASK_OUTPUT/experiment" --port 8765 --suite libero_90 --task-id 57 \
  --openpi-root "$TASK_OPENPI" --dsrl-root "$TASK_DSRL" --checkpoint "$TASK_CHECKPOINT" \
  --model-config pi0_libero --feature-interface sac --chunk 20 --horizon 400 \
  --source-commit "$TASK_SOURCE" --prior-recovery "$TASK_PRIOR/source-recovery.pkl" \
  --prior-contract "$TASK_PRIOR/contract.json" --train-states 0 1 2 3 \
  --validation-states 36 37 --calibration-states 24 25 26 27 28 29 30 31 \
  --min-teacher-success .7 --min-episodes 4 --max-episodes 6 \
  --anneal-decisions 35 55 --anneal-powers .5 1 2 --max-collection-steps 14400 \
  --history-updates 500 --history-device cuda --eval-states 40 41 42 43 --attempts 4 \
  --batch-size 32 --utd 20 --online-dsrl
