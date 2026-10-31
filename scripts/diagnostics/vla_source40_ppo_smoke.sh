#!/usr/bin/env bash
# Fresh on-policy PPO runtime qualification, one GPU <=40min; source cap9500.
set -Eeuo pipefail
TASK_SOURCE=${1:?exact reviewed integrated commit}
TASK_RUN=${2:?fresh output identity}
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_OLD=/mowenjie/fluffy-bassoon
TASK_WORKTREE=$TASK_ROOT/worktrees/$TASK_RUN
TASK_OUTPUT=$TASK_ROOT/outputs/$TASK_RUN
TASK_MODEL=$TASK_OLD/envs/openpi-215abfb/bin/python
TASK_PROBE=$TASK_OLD/envs/lerobot-v044-8fff0fde/bin/python
TASK_DSRL=$TASK_ROOT/deps/dsrl-pi0-7f48937-20261029-r1/repo
TASK_DSRL_OVERLAY=$TASK_ROOT/envs/dsrl-pi0-overlay-20261029-r1
TASK_OPENPI=$TASK_OLD/third_party/openpi-215abfb
TASK_CUDA=$TASK_ROOT/outputs/libero-pi05-student-cuda-20261029-r1/cu128-overlay
TASK_TEST=$TASK_ROOT/envs/pytest-overlay-20261029-r1
TASK_SIM=/mowenjie/envs/wam-libero-plus-py310/bin/python
TASK_LIBERO=$TASK_ROOT/deps/libero-dsrl-87edbd1-20261029-r1/repo
TASK_CHECKPOINT=$TASK_OLD/inputs/pi05-libero-openpi-20261027-r1
TASK_PLAN=$TASK_ROOT/outputs/libero40-source-1m-20261029-r1
test ! -e "$TASK_OUTPUT"
mkdir -p "$TASK_OUTPUT"
cd "$TASK_WORKTREE"
test "$(git rev-parse HEAD)" = "$TASK_SOURCE"
test -z "$(git status --porcelain)"
exec > >(tee "$TASK_OUTPUT/console.log") 2>&1
task_cleanup() {
  TASK_RC=$?
  trap - EXIT
  if [ -n "${TASK_SIM_PID:-}" ]; then kill "$TASK_SIM_PID" 2>/dev/null || true; wait "$TASK_SIM_PID" 2>/dev/null || true; fi
  printf '%s\n' "$TASK_RC" > "$TASK_OUTPUT/exit-code.txt"
  exit "$TASK_RC"
}
trap task_cleanup EXIT
TASK_SIM_PID=
export FLUFFY_GPU_EXECUTION=baige CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=disabled
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTHONPATH=$TASK_WORKTREE/src
"$TASK_PROBE" 'scripts/[train]gpu_probe.py' --expected-count 1 --purpose training \
  --baige-container --baige-exclusive-training --output "$TASK_OUTPUT/gpu-probe.json"
export LYCO_BAIGE_GPU_MAP=$TASK_OUTPUT/gpu-probe.json
CUDA_VISIBLE_DEVICES='' PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$TASK_TEST:$TASK_WORKTREE/src \
  "$TASK_PROBE" -m pytest -q -p no:cacheprovider tests/test_vla*.py tests/test_gpu_probe.py \
  > "$TASK_OUTPUT/focused-tests.log" 2>&1

export PYTHONPATH=$TASK_WORKTREE/src
"$TASK_MODEL" -m fluffy_bassoon.meta_rl.vla_source40_campaign reserve-ppo \
  --campaign "$TASK_PLAN/campaign.json" --ledger "$TASK_PLAN/ppo-ledger.json" \
  --run "$TASK_RUN" --commit "$TASK_SOURCE" --action-cap 9500
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0
export LIBERO_CONFIG_PATH=$TASK_ROOT/cache/libero-dsrl-87edbd1-20261029-r1
export OPENPI_DATA_HOME=$TASK_OLD/cache/openpi-215abfb
export MAGICK_HOME=/mowenjie/libs/libero-plus-imagemagick
export MAGICK_CODER_MODULE_PATH=$MAGICK_HOME/usr/lib/x86_64-linux-gnu/ImageMagick-6.9.11/modules-Q16/coders
export LD_LIBRARY_PATH=$MAGICK_HOME/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
PYTHONPATH=$TASK_WORKTREE/src:$TASK_LIBERO:$TASK_OPENPI/packages/openpi-client/src:$TASK_OLD/envs/pi05-plus-client-deps \
  "$TASK_SIM" -m fluffy_bassoon.meta_rl.vla_libero --mode env-server \
  --output "$TASK_OUTPUT/simulator" --port 8765 --suite libero_spatial --task-id 0 \
  --horizon 400 --fresh-environment-on-reset > "$TASK_OUTPUT/simulator.log" 2>&1 &
TASK_SIM_PID=$!
for TASK_POLL in $(seq 1 600); do
  if [ -e "$TASK_OUTPUT/simulator/simulator-ready.json" ]; then break; fi
  kill -0 "$TASK_SIM_PID"
  sleep 1
done
test -e "$TASK_OUTPUT/simulator/simulator-ready.json"
export PYTHONPATH=$TASK_CUDA:$TASK_DSRL_OVERLAY:$TASK_WORKTREE/src:$TASK_OPENPI/src:$TASK_OPENPI/packages/openpi-client/src
"$TASK_MODEL" -m fluffy_bassoon.meta_rl.vla_source40_ppo \
  --output "$TASK_OUTPUT/experiment" --checkpoint "$TASK_CHECKPOINT" --openpi-root "$TASK_OPENPI" \
  --suite libero_spatial --task-id 0 --partition train --source-commit "$TASK_SOURCE" \
  --expected-hostname "$(hostname)" --max-total-actions 9500 --max-elapsed-seconds 1500 \
  --updates 8 --attempts 4 --seed 1 --width 32 --layers 1 --heads 4 --context-length 128
