#!/usr/bin/env bash
# Common-motor frozen validation after both source PPO arms are terminal/ready.
# One approved GPU,4h cumulative across setup, shutdown and exact-source retries.
set -Eeuo pipefail
TASK_SOURCE=${1:?exact reviewed validation commit}
TASK_RUN=${2:?fresh validation identity}
TASK_FULL_RUN=${3:?terminal full source-PPO identity}
TASK_RESET_RUN=${4:?terminal separately trained reset source-PPO identity}
TASK_SLICE=${5:?allocated slice seconds including setup/shutdown}
TASK_RESUME=${6:-}
case "$TASK_RESUME" in ''|--resume) ;; *) exit 2;; esac
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_OLD=/mowenjie/fluffy-bassoon
TASK_WT=$TASK_ROOT/worktrees/$TASK_RUN
TASK_OUT=$TASK_ROOT/outputs/$TASK_RUN
TASK_MODEL=$TASK_OLD/envs/openpi-215abfb/bin/python
TASK_PROBE=$TASK_OLD/envs/lerobot-v044-8fff0fde/bin/python
TASK_PLAN=$TASK_ROOT/outputs/libero40-source-1m-20261029-r1
TASK_OPENPI=$TASK_OLD/third_party/openpi-215abfb
TASK_CUDA=$TASK_ROOT/outputs/libero-pi05-student-cuda-20261029-r1/cu128-overlay
TASK_LIBERO=$TASK_ROOT/deps/libero-dsrl-87edbd1-20261029-r1/repo
TASK_STARTED=$(date +%s)
TASK_RESUME_ARGS=()
if [ "$TASK_RESUME" = --resume ]; then
  test -f "$TASK_OUT/experiment/validation-recovery.pt"
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
"$TASK_MODEL" - "$TASK_OUT/models.json" "$TASK_ROOT/outputs/$TASK_FULL_RUN" \
  "$TASK_ROOT/outputs/$TASK_RESET_RUN" "$TASK_PLAN/ppo-ledger.json" "$TASK_RESUME" <<'PYMODELS'
import fcntl, hashlib, json, pathlib, sys
output, full, reset = map(pathlib.Path, sys.argv[1:4])
from fluffy_bassoon.meta_rl.vla_source40_ppo_lineage import terminal_source
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
models = {}
for mode, root in (('full', full), ('episode', reset)):
    # The source launcher owns this lock for its entire attempt. Never admit a live writer.
    with (root / 'launcher.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        models[mode] = terminal_source(root, mode, pathlib.Path(sys.argv[4]))
value = dict(format='source40_frozen_ppo_validation_models_v1', models=models)
if sys.argv[5] == '--resume':
    if json.loads(output.read_text()) != value:
        raise ValueError('Frozen validation model/time inputs changed on resume')
else:
    output.write_text(json.dumps(value, indent=2) + '\n')
PYMODELS
TASK_MODELS_SHA=$(sha256sum "$TASK_OUT/models.json" | cut -d' ' -f1)
TASK_ATTEMPT=$("$TASK_MODEL" - "$TASK_OUT" "$TASK_SOURCE" "$TASK_RUN" "$TASK_MODELS_SHA" \
  "$TASK_PLAN/campaign.json" "$TASK_PLAN/schedule.json" "$TASK_SLICE" "$TASK_RESUME" "$TASK_STARTED" <<'PYTIME'
import hashlib, pathlib, sys
from fluffy_bassoon.meta_rl.vla_source40_ppo_worker import begin
root, source, run, models_sha, campaign, schedule, seconds, resume, started = sys.argv[1:]
contract = dict(operation='frozen_source_ppo_validation', source_commit=source, run_id=run,
    models_sha256=models_sha, maximum_total_actions=56000, maximum_driver_seconds=14400,
    campaign_sha256=hashlib.sha256(pathlib.Path(campaign).read_bytes()).hexdigest(),
    schedule_sha256=hashlib.sha256(pathlib.Path(schedule).read_bytes()).hexdigest(), optimizer_updates=0)
print(begin(root, contract, slice_seconds=int(seconds), worker_seconds=14400, resume=resume == '--resume', now=float(started)))
PYTIME
)
exec > >(tee "$TASK_ATTEMPT/console.log") 2>&1
task_cleanup() {
  TASK_RC=$?
  trap - EXIT
  PYTHONPATH=$TASK_WT/src "$TASK_MODEL" -m fluffy_bassoon.meta_rl.vla_source40_ppo_worker finish \
    --root "$TASK_OUT" --attempt "$TASK_ATTEMPT" --exit-code "$TASK_RC" || true
  printf '%s\n' "$TASK_RC" > "$TASK_ATTEMPT/exit-code.txt"
  printf '%s\n' "$TASK_RC" > "$TASK_OUT/exit-code.txt"
  exit "$TASK_RC"
}
trap task_cleanup EXIT
"$TASK_MODEL" -m fluffy_bassoon.meta_rl.vla_source40_campaign reserve-ppo \
  --campaign "$TASK_PLAN/campaign.json" --ledger "$TASK_PLAN/ppo-ledger.json" \
  --run "$TASK_RUN" --commit "$TASK_SOURCE" --action-cap 56000 "${TASK_RESUME_ARGS[@]}"
export FLUFFY_GPU_EXECUTION=baige CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=disabled
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
"$TASK_PROBE" 'scripts/[train]gpu_probe.py' --expected-count 1 --purpose training \
  --baige-container --baige-exclusive-training --output "$TASK_ATTEMPT/gpu-probe.json"
export LYCO_BAIGE_GPU_MAP=$TASK_ATTEMPT/gpu-probe.json
CUDA_VISIBLE_DEVICES='' PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  PYTHONPATH=$TASK_ROOT/envs/pytest-overlay-20261029-r1:$TASK_WT/src \
  "$TASK_PROBE" -m pytest -q -p no:cacheprovider tests/test_vla_source40_frozen_validation.py tests/test_vla_source40_ppo_worker.py tests/test_vla_source40_ppo_lineage.py \
  > "$TASK_ATTEMPT/focused-tests.log" 2>&1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0
export LIBERO_CONFIG_PATH=$TASK_ROOT/cache/libero-dsrl-87edbd1-20261029-r1
export OPENPI_DATA_HOME=$TASK_OLD/cache/openpi-215abfb
export MAGICK_HOME=/mowenjie/libs/libero-plus-imagemagick
export MAGICK_CODER_MODULE_PATH=$MAGICK_HOME/usr/lib/x86_64-linux-gnu/ImageMagick-6.9.11/modules-Q16/coders
export LD_LIBRARY_PATH=$MAGICK_HOME/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PYTHONPATH=$TASK_CUDA:$TASK_WT/src:$TASK_OPENPI/src:$TASK_OPENPI/packages/openpi-client/src
TASK_TIMEOUT=$(( TASK_SLICE - ( $(date +%s) - TASK_STARTED ) - 60 ))
test "$TASK_TIMEOUT" -gt 60
timeout --signal=INT --kill-after=30s "$TASK_TIMEOUT" "$TASK_MODEL" \
  -m fluffy_bassoon.meta_rl.vla_source40_frozen_validation \
  --output "$TASK_OUT/experiment" --models "$TASK_OUT/models.json" --models-sha256 "$TASK_MODELS_SHA" \
  --campaign "$TASK_PLAN/campaign.json" --schedule "$TASK_PLAN/schedule.json" \
  --checkpoint "$TASK_OLD/inputs/pi05-libero-openpi-20261027-r1" --openpi-root "$TASK_OPENPI" \
  --simulator-python /mowenjie/envs/wam-libero-plus-py310/bin/python \
  --simulator-pythonpath "$TASK_WT/src:$TASK_LIBERO:$TASK_OPENPI/packages/openpi-client/src:$TASK_OLD/envs/pi05-plus-client-deps" \
  --source-commit "$TASK_SOURCE" --expected-hostname "$(hostname)" \
  --max-total-actions 56000 --max-elapsed-seconds 14400 "${TASK_RESUME_ARGS[@]}"
