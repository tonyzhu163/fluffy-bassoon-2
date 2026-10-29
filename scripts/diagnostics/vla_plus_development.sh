#!/usr/bin/env bash
# Two declared Plus development cases in one exclusive scheduler GPU allocation.
set -Eeuo pipefail
TASK_SOURCE=${1:?exact reviewed integrated source commit}
TASK_RUN=${2:?fresh Plus development run identity}
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_OLD=/mowenjie/fluffy-bassoon
TASK_WORKTREE=$TASK_ROOT/worktrees/$TASK_RUN
TASK_OUTPUT=$TASK_ROOT/outputs/$TASK_RUN
TASK_PLUS=$TASK_OLD/third_party/libero-plus-4976dc3-r3
TASK_SIM=/mowenjie/envs/wam-libero-plus-py310/bin/python
TASK_MODEL=$TASK_OLD/envs/maniskill-pi05-rlinf-e1a82ebf-20261028-r1/bin/python
TASK_PROBE=$TASK_OLD/envs/lerobot-v044-8fff0fde/bin/python
TASK_RLINF=$TASK_OLD/third_party/rlinf-e1a82ebf
TASK_OPENPI=$TASK_OLD/envs/maniskill-pi05-rlinf-e1a82ebf-20261028-r1/lib/python3.11/site-packages/openpi
TASK_DSRL=$TASK_ROOT/deps/dsrl-pi0-7f48937-20261029-r1/repo
TASK_OVERLAY=$TASK_ROOT/envs/dsrl-pi0-overlay-20261029-r1
TASK_CHECKPOINT=$TASK_ROOT/inputs/rlinf-pi05-libero130-sft-6222623f-20261029-r1
cd "$TASK_WORKTREE"
test "$(git rev-parse HEAD)" = "$TASK_SOURCE"
test -z "$(git status --porcelain)"
test ! -e "$TASK_OUTPUT"
test -f "$TASK_CHECKPOINT/READY.json"
test -f src/fluffy_bassoon/meta_rl/vla_shifted_poc.py
test -f src/fluffy_bassoon/meta_rl/vla_rlinf_libero.py
mkdir -p "$TASK_OUTPUT"
exec > >(tee "$TASK_OUTPUT/console.log") 2>&1
TASK_SIM_PIDS=()
task_cleanup() {
  TASK_RC=$?
  trap - EXIT
  for TASK_PID in "${TASK_SIM_PIDS[@]}"; do
    kill "$TASK_PID" 2>/dev/null || true
    wait "$TASK_PID" 2>/dev/null || true
  done
  printf '%s\n' "$TASK_RC" > "$TASK_OUTPUT/exit-code.txt"
  exit "$TASK_RC"
}
trap task_cleanup EXIT
export FLUFFY_GPU_EXECUTION=baige CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0
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
# Pin both actual official cases and lineage before any motor outcomes.
CUDA_VISIBLE_DEVICES='' "$TASK_PROBE" - "$TASK_PLUS" "$TASK_OUTPUT/development-cases.json" <<'PY'
import json, pathlib, sys
from fluffy_bassoon.meta_rl.vla_plus_worker import case_contract
cases=[dict(port=port, **case_contract(sys.argv[1], index)) for port,index in ((8765,667),(8766,290))]
pathlib.Path(sys.argv[2]).write_text(json.dumps(dict(benchmark='LIBERO-Plus', development=True,
    cases=cases, horizon=280, settle_steps=10, attempts=4,
    model='RLinf pi0.5 LIBERO130 SFT revision6222623f; receipt verified by motor loader',
    claim='native and ordinary DSRL development POC; no trained meta-controller evaluation'),indent=2)+'\n')
PY
export PYTHONPATH=$TASK_WORKTREE/src:$TASK_PLUS:$TASK_OLD/envs/pi05-plus-client-deps
for TASK_PAIR in 8765:667 8766:290; do
  TASK_PORT=${TASK_PAIR%:*}
  TASK_CASE=${TASK_PAIR#*:}
  "$TASK_SIM" -m fluffy_bassoon.meta_rl.vla_plus_worker \
    --output "$TASK_OUTPUT/simulator-$TASK_PORT" --source-commit "$TASK_SOURCE" \
    --libero-root "$TASK_PLUS" --case-index "$TASK_CASE" --port "$TASK_PORT" \
    --horizon 280 --settle-steps 10 > "$TASK_OUTPUT/simulator-$TASK_PORT.log" 2>&1 &
  TASK_SIM_PIDS+=("$!")
done
for TASK_POLL in $(seq 1 600); do
  if [ -e "$TASK_OUTPUT/simulator-8765/simulator-ready.json" ] && \
     [ -e "$TASK_OUTPUT/simulator-8766/simulator-ready.json" ]; then break; fi
  for TASK_PID in "${TASK_SIM_PIDS[@]}"; do kill -0 "$TASK_PID"; done
  sleep 1
done
test -e "$TASK_OUTPUT/simulator-8765/simulator-ready.json"
test -e "$TASK_OUTPUT/simulator-8766/simulator-ready.json"
# Use the installed RLinf OpenPI runtime, not the older OpenPI215 JAX tree.
export PYTHONPATH=$TASK_OVERLAY:$TASK_WORKTREE/src:$TASK_RLINF:$TASK_DSRL
"$TASK_MODEL" -m fluffy_bassoon.meta_rl.vla_shifted_poc \
  --output "$TASK_OUTPUT/experiment" --source-commit "$TASK_SOURCE" --benchmark plus \
  --ports 8765 8766 --checkpoint "$TASK_CHECKPOINT" --rlinf-root "$TASK_RLINF" \
  --openpi-root "$TASK_OPENPI" --dsrl-root "$TASK_DSRL" \
  --chunk 10 --horizon 280 --attempts 4 --utd 20 --batch-size 32 --warmup-chunks 32
