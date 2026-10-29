#!/usr/bin/env bash
# Fresh pi0.5 technical pilot. Schedule on one exclusive GPU for <=45 minutes.
# No pi0 teacher, corpus generation, student training or semantic OOD claim.
set -Eeuo pipefail
TASK_SOURCE=${1:?exact integrated source commit}
TASK_RUN=${2:?fresh pi0.5 output identity}
TASK_SUITE=${3:-libero_90}
TASK_TASK=${4:-57}
TASK_PHASE=${5:-teacher}
if [ "$TASK_PHASE" != teacher ] && [ "$TASK_PHASE" != teacher-gate ]; then
  printf 'Unknown phase: %s\n' "$TASK_PHASE" >&2
  exit 2
fi
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
TASK_CHECKPOINT=$TASK_OLD/inputs/pi05-libero-openpi-20261027-r1
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
export LIBERO_CONFIG_PATH=$TASK_ROOT/cache/libero-dsrl-87edbd1-20261029-r1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 WANDB_MODE=disabled
export MAGICK_HOME=/mowenjie/libs/libero-plus-imagemagick
export MAGICK_CODER_MODULE_PATH=$MAGICK_HOME/usr/lib/x86_64-linux-gnu/ImageMagick-6.9.11/modules-Q16/coders
export LD_LIBRARY_PATH=$MAGICK_HOME/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PYTHONPATH=$TASK_WORKTREE/src
"$TASK_TORCH" 'scripts/[train]gpu_probe.py' --expected-count 1 --purpose training \
  --baige-container --baige-exclusive-training --output "$TASK_OUTPUT/gpu-probe.json"
export LYCO_BAIGE_GPU_MAP=$TASK_OUTPUT/gpu-probe.json
CUDA_VISIBLE_DEVICES='' PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH="$TASK_TEST_OVERLAY:$TASK_WORKTREE/src" \
  "$TASK_TORCH" -m pytest -q -p no:cacheprovider tests/test_vla*.py tests/test_gpu_probe.py \
  > "$TASK_OUTPUT/focused-tests.log" 2>&1
export PYTHONPATH=$TASK_OVERLAY:$TASK_WORKTREE/src:$TASK_DSRL:$TASK_OPENPI/src:$TASK_OPENPI/packages/openpi-client/src
"$TASK_MODEL" - "$TASK_CHECKPOINT" "$TASK_OUTPUT/configuration-qualification.json" <<'PY'
import hashlib, json, pathlib, sys
from openpi.training import config
from fluffy_bassoon.meta_rl.vla_features import FEATURE_KIND
checkpoint, destination = map(pathlib.Path, sys.argv[1:])
receipt = json.loads((checkpoint / 'READY.json').read_text())
receipt_hash = hashlib.sha256((checkpoint / 'READY.json').read_bytes()).hexdigest()
conf = config.get_config('pi05_libero')
assert conf.model.pi05 and conf.model.action_horizon == 10 and conf.model.action_dim == 32
assert receipt['status'] == 'verified' and receipt['checkpoint'] == 'gs://openpi-assets/checkpoints/pi05_libero'
assert receipt_hash == 'ec064cce32f5003e2e3979c5712d7d6385018f6bfcc59cb2161105881e70d170'
assert FEATURE_KIND == 'openpi_final_prefix_visual_mean_language_mean_public_state_v2'
normalization = checkpoint / 'assets/physical-intelligence/libero/norm_stats.json'
assert hashlib.sha256(normalization.read_bytes()).hexdigest() == 'b3a44bb2810436fb62917decaea58bd4d9110255df527dea21e8fd40c960bd84'
for item in receipt['files']:
    assert (checkpoint / item['name']).stat().st_size == item['bytes'], item['name']
destination.write_text(json.dumps(dict(model_config='pi05_libero', native_horizon=10,
    action_dim=32, executed_chunk=10, flow_steps=10, checkpoint_receipt_sha256=receipt_hash,
    feature_kind=FEATURE_KIND, fresh_teacher=True,
    scope='same-task technical pilot; no semantic OOD claim'), indent=2) + '\n')
PY
task_start_simulator() {
  local TASK_SIM_NAME=$1
  PYTHONPATH=$TASK_WORKTREE/src:$TASK_LIBERO:$TASK_OPENPI/packages/openpi-client/src:$TASK_OLD/envs/pi05-plus-client-deps \
    "$TASK_SIM" -m fluffy_bassoon.meta_rl.vla_libero --mode env-server \
    --output "$TASK_OUTPUT/$TASK_SIM_NAME" --port 8765 --suite "$TASK_SUITE" --task-id "$TASK_TASK" \
    --horizon 400 > "$TASK_OUTPUT/$TASK_SIM_NAME.log" 2>&1 &
  TASK_SIM_PID=$!
  for TASK_POLL in $(seq 1 600); do
    if [ -e "$TASK_OUTPUT/$TASK_SIM_NAME/simulator-ready.json" ]; then break; fi
    kill -0 "$TASK_SIM_PID"
    sleep 1
  done
  if [ ! -e "$TASK_OUTPUT/$TASK_SIM_NAME/simulator-ready.json" ]; then
    tail -n 100 "$TASK_OUTPUT/$TASK_SIM_NAME.log"
    printf 'Simulator failed readiness after 600 seconds\n' >&2
    exit 2
  fi
}
task_start_simulator simulator
export PYTHONPATH=$TASK_OVERLAY:$TASK_WORKTREE/src:$TASK_DSRL:$TASK_OPENPI/src:$TASK_OPENPI/packages/openpi-client/src
"$TASK_MODEL" -m fluffy_bassoon.meta_rl.vla_libero --mode teacher \
  --output "$TASK_OUTPUT/experiment" --port 8765 --suite "$TASK_SUITE" --task-id "$TASK_TASK" \
  --openpi-root "$TASK_OPENPI" --dsrl-root "$TASK_DSRL" --checkpoint "$TASK_CHECKPOINT" \
  --model-config pi05_libero --feature-interface vla-prefix --chunk 10 --horizon 400 \
  --source-commit "$TASK_SOURCE" --max-steps 12000 --warmup-chunks 32 --batch-size 32 --utd 20 \
  --eval-states 40 41 42 43 --attempts 2 --history-updates 0
"$TASK_MODEL" -m fluffy_bassoon.meta_rl.vla_performance_report \
  --episodes "$TASK_OUTPUT/experiment/episodes.jsonl" --output "$TASK_OUTPUT/performance.json" \
  --plot "$TASK_OUTPUT/success-by-attempt.png"
if [ "$TASK_PHASE" = teacher-gate ]; then
  # Only evaluate the completed fresh pi0.5 teacher. A passing gate does not
  # authorize this launcher to generate the separate corpus or train students.
  # The first driver closes its single-client simulator on exit. Reap that
  # owned worker, then start a fresh worker/readiness receipt for the gate.
  wait "$TASK_SIM_PID"
  TASK_SIM_PID=
  task_start_simulator simulator-gate
  "$TASK_MODEL" -m fluffy_bassoon.meta_rl.vla_cnd_run --qualify-only \
    --output "$TASK_OUTPUT/teacher-gate" --port 8765 --suite "$TASK_SUITE" --task-id "$TASK_TASK" \
    --openpi-root "$TASK_OPENPI" --dsrl-root "$TASK_DSRL" --checkpoint "$TASK_CHECKPOINT" \
    --model-config pi05_libero --feature-interface sac --chunk 10 --horizon 400 \
    --source-commit "$TASK_SOURCE" --prior-recovery "$TASK_OUTPUT/experiment/source-recovery.pkl" \
    --prior-contract "$TASK_OUTPUT/experiment/contract.json" \
    --calibration-states 24 25 26 27 28 29 30 31 --min-teacher-success .7
fi
