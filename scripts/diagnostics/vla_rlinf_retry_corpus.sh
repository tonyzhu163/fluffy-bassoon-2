#!/usr/bin/env bash
# One original-LIBERO source task; publish real completed AD/DPT trials.
set -Eeuo pipefail
TASK_SOURCE=${1:?exact reviewed source commit}
TASK_RUN=${2:?fresh retry corpus output identity}
TASK_SUITE=${3:-libero_90}
TASK_TASK=${4:-47}
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_OLD=/mowenjie/fluffy-bassoon
TASK_WORKTREE=$TASK_ROOT/worktrees/$TASK_RUN
TASK_OUTPUT=$TASK_ROOT/outputs/$TASK_RUN
TASK_MODEL=$TASK_OLD/envs/maniskill-pi05-rlinf-e1a82ebf-20261028-r1/bin/python
TASK_SIM=/mowenjie/envs/wam-libero-plus-py310/bin/python
TASK_PROBE=$TASK_OLD/envs/lerobot-v044-8fff0fde/bin/python
TASK_RLINF=$TASK_OLD/third_party/rlinf-e1a82ebf
TASK_OPENPI=$TASK_OLD/envs/maniskill-pi05-rlinf-e1a82ebf-20261028-r1/lib/python3.11/site-packages/openpi
TASK_LIBERO=$TASK_ROOT/deps/libero-dsrl-87edbd1-20261029-r1/repo
TASK_DSRL=$TASK_ROOT/deps/dsrl-pi0-7f48937-20261029-r1/repo
TASK_OVERLAY=$TASK_ROOT/envs/dsrl-pi0-overlay-20261029-r1
TASK_CHECKPOINT=$TASK_ROOT/inputs/rlinf-pi05-libero130-sft-6222623f-20261029-r1
TASK_SPLIT=$TASK_WORKTREE/outputs/libero-pi05-corpus-20261029-r1/split.json
cd "$TASK_WORKTREE"
test "$(git rev-parse HEAD)" = "$TASK_SOURCE"
test -z "$(git status --porcelain)"
test ! -e "$TASK_OUTPUT"
mkdir -p "$TASK_OUTPUT"
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
export OPENPI_DATA_HOME=$TASK_OLD/inputs/rlinf-openpi-tokenizer-befaa248-20261028-r1
export LIBERO_CONFIG_PATH=$TASK_ROOT/cache/libero-dsrl-87edbd1-20261029-r1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 WANDB_MODE=disabled
export MAGICK_HOME=/mowenjie/libs/libero-plus-imagemagick
export MAGICK_CODER_MODULE_PATH=$MAGICK_HOME/usr/lib/x86_64-linux-gnu/ImageMagick-6.9.11/modules-Q16/coders
export LD_LIBRARY_PATH=$MAGICK_HOME/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PYTHONPATH=$TASK_WORKTREE/src
"$TASK_PROBE" 'scripts/[train]gpu_probe.py' --expected-count 1 --purpose training \
  --baige-container --baige-exclusive-training --output "$TASK_OUTPUT/gpu-probe.json"
export LYCO_BAIGE_GPU_MAP=$TASK_OUTPUT/gpu-probe.json
CUDA_VISIBLE_DEVICES='' PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  PYTHONPATH="$TASK_ROOT/envs/pytest-overlay-20261029-r1:$TASK_WORKTREE/src" \
  "$TASK_PROBE" -m pytest -q -p no:cacheprovider tests/test_vla*.py tests/test_gpu_probe.py \
  > "$TASK_OUTPUT/focused-tests.log" 2>&1
CUDA_VISIBLE_DEVICES='' "$TASK_MODEL" - "$TASK_SPLIT" "$TASK_SUITE" "$TASK_TASK" <<'PY'
import sys
from fluffy_bassoon.meta_rl.vla_cnd_corpus import load_frozen_split
_, tasks = load_frozen_split(sys.argv[1])
rows = [row for row in tasks.values() if row['suite'] == sys.argv[2] and row['task_id'] == int(sys.argv[3])]
assert len(rows) == 1 and rows[0]['split'] == 'train' and rows[0]['teacher_collection_allowed']
PY
export PYTHONPATH=$TASK_WORKTREE/src:$TASK_LIBERO:$TASK_OLD/envs/pi05-plus-client-deps
"$TASK_SIM" -m fluffy_bassoon.meta_rl.vla_libero --mode env-server \
  --output "$TASK_OUTPUT/simulator" --port 8765 --suite "$TASK_SUITE" --task-id "$TASK_TASK" \
  --horizon 400 --fresh-environment-on-reset > "$TASK_OUTPUT/simulator.log" 2>&1 &
TASK_SIM_PID=$!
for TASK_POLL in $(seq 1 600); do
  if [ -e "$TASK_OUTPUT/simulator/simulator-ready.json" ]; then break; fi
  kill -0 "$TASK_SIM_PID"
  sleep 1
done
test -e "$TASK_OUTPUT/simulator/simulator-ready.json"
export PYTHONPATH=$TASK_WORKTREE/src:$TASK_RLINF:$TASK_DSRL:$TASK_OVERLAY
"$TASK_MODEL" -m fluffy_bassoon.meta_rl.vla_rlinf_retry_corpus \
  --output "$TASK_OUTPUT/experiment" --source-commit "$TASK_SOURCE" --frozen-split "$TASK_SPLIT" \
  --suite "$TASK_SUITE" --task-id "$TASK_TASK" --rlinf-root "$TASK_RLINF" \
  --openpi-root "$TASK_OPENPI" --dsrl-root "$TASK_DSRL" --checkpoint "$TASK_CHECKPOINT" \
  --port 8765 --seed 0
