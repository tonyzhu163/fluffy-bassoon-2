#!/usr/bin/env bash
# Future source-PPO target evaluation in an existing root-authorized allocation.
# Reads an existing target escrow; never reserves actions or submits a job.
set -Eeuo pipefail
TASK_SOURCE=${1:?exact reviewed target-evaluation commit}
TASK_RUN=${2:?fresh target evaluation identity}
TASK_MODELS=${3:?hash-bound terminal source models manifest}
TASK_CASE=${4:?predeclared case/simulator plan JSON}
TASK_ACTIONS=${5:?existing target action escrow}
TASK_WORKER_SECONDS=${6:?approved cumulative target GPU seconds}
TASK_SLICE=${7:?this allocated slice seconds including setup/shutdown}
TASK_LEDGER=${8:?canonical ledger already containing this target reservation}
shift 8
TASK_RESUME=
TASK_EXTRA=()
while [ "$#" -gt 0 ]; do
  if [ "$1" = --resume ]; then TASK_RESUME=--resume; else TASK_EXTRA+=("$1"); fi
  shift
done
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_OLD=/mowenjie/fluffy-bassoon
TASK_WT=$TASK_ROOT/worktrees/$TASK_RUN
TASK_OUT=$TASK_ROOT/outputs/$TASK_RUN
TASK_MODEL=$TASK_OLD/envs/openpi-215abfb/bin/python
TASK_PROBE=$TASK_OLD/envs/lerobot-v044-8fff0fde/bin/python
TASK_OPENPI=$TASK_OLD/third_party/openpi-215abfb
TASK_CUDA=$TASK_ROOT/outputs/libero-pi05-student-cuda-20261029-r1/cu128-overlay
TASK_PLAN=$TASK_ROOT/outputs/libero40-source-1m-20261029-r1
TASK_STARTED=$(date +%s)
TASK_RESUME_ARGS=()
if [ "$TASK_RESUME" = --resume ]; then
  test -f "$TASK_OUT/experiment/evaluation-recovery.pkl"
  cmp "$TASK_MODELS" "$TASK_OUT/models-input.json"
  cmp "$TASK_CASE" "$TASK_OUT/case-plan.json"
  TASK_RESUME_ARGS=(--resume)
else
  test ! -e "$TASK_OUT"
  mkdir -p "$TASK_OUT"
  cp "$TASK_MODELS" "$TASK_OUT/models-input.json"
  cp "$TASK_CASE" "$TASK_OUT/case-plan.json"
fi
cd "$TASK_WT"
test "$(git rev-parse HEAD)" = "$TASK_SOURCE"
test -z "$(git status --porcelain)"
exec 9> "$TASK_OUT/launcher.lock"
flock -n 9
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export PYTHONPATH=$TASK_WT/src
TASK_MODELS_SHA=$(sha256sum "$TASK_OUT/models-input.json" | cut -d' ' -f1)
TASK_CASE_SHA=$(sha256sum "$TASK_OUT/case-plan.json" | cut -d' ' -f1)
TASK_ATTEMPT=$("$TASK_MODEL" - "$TASK_OUT" "$TASK_SOURCE" "$TASK_RUN" "$TASK_MODELS_SHA" "$TASK_CASE_SHA" \
  "$TASK_ACTIONS" "$TASK_WORKER_SECONDS" "$TASK_SLICE" "$TASK_LEDGER" "$TASK_RESUME" "${TASK_EXTRA[@]}" <<'PYTIME'
import hashlib, json, pathlib, sys
from fluffy_bassoon.meta_rl.vla_source40_ppo_worker import begin
root, source, run, models, case, actions, worker, seconds, ledger_path, resume, *extra = sys.argv[1:]
ledger = json.loads(pathlib.Path(ledger_path).read_text())
row = ledger.get('reservations', {}).get(run)
if (row is None or row.get('source_commit') != source or row.get('maximum_total_actions') != int(actions)
        or row.get('purpose') != 'target_evaluation' or row.get('source_pretraining_data') is not False
        or row.get('fresh_ppo_training_data') is not False):
    raise ValueError('Root must record an exact target-only action escrow before this launcher runs')
record_sha = hashlib.sha256(json.dumps(row, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
contract = dict(operation='frozen_source_ppo_target', source_commit=source, run_id=run,
    models_sha256=models, case_plan_sha256=case, maximum_total_actions=int(actions),
    maximum_driver_seconds=int(worker), escrow_record_sha256=record_sha, extra_arguments=extra,
    deployment_optimizer_updates=0, target_data_in_pretraining=False)
print(begin(root, contract, slice_seconds=int(seconds), worker_seconds=int(worker), resume=resume == '--resume'))
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
export FLUFFY_GPU_EXECUTION=baige CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=disabled
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
"$TASK_PROBE" 'scripts/[train]gpu_probe.py' --expected-count 1 --purpose training \
  --baige-container --baige-exclusive-training --output "$TASK_ATTEMPT/gpu-probe.json"
export LYCO_BAIGE_GPU_MAP=$TASK_ATTEMPT/gpu-probe.json
CUDA_VISIBLE_DEVICES='' PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  PYTHONPATH=$TASK_ROOT/envs/pytest-overlay-20261029-r1:$TASK_WT/src \
  "$TASK_PROBE" -m pytest -q -p no:cacheprovider tests/test_vla_source40_ppo_target.py tests/test_vla_source40_evaluate.py \
  > "$TASK_ATTEMPT/focused-tests.log" 2>&1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0
export OPENPI_DATA_HOME=$TASK_OLD/cache/openpi-215abfb
export MAGICK_HOME=/mowenjie/libs/libero-plus-imagemagick
export MAGICK_CODER_MODULE_PATH=$MAGICK_HOME/usr/lib/x86_64-linux-gnu/ImageMagick-6.9.11/modules-Q16/coders
export LD_LIBRARY_PATH=$MAGICK_HOME/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export LIBERO_CONFIG_PATH=$TASK_ROOT/cache/libero-dsrl-87edbd1-20261029-r1
export PYTHONPATH=$TASK_CUDA:$TASK_WT/src:$TASK_OPENPI/src:$TASK_OPENPI/packages/openpi-client/src
TASK_TIMEOUT=$(( TASK_SLICE - ( $(date +%s) - TASK_STARTED ) - 60 ))
test "$TASK_TIMEOUT" -gt 60
timeout --signal=INT --kill-after=30s "$TASK_TIMEOUT" "$TASK_MODEL" \
  -m fluffy_bassoon.meta_rl.vla_source40_ppo_target \
  --output "$TASK_OUT/experiment" --run-id "$TASK_RUN" --source-commit "$TASK_SOURCE" \
  --expected-hostname "$(hostname)" --models "$TASK_OUT/models-input.json" --models-sha256 "$TASK_MODELS_SHA" \
  --case-plan "$TASK_OUT/case-plan.json" --case-plan-sha256 "$TASK_CASE_SHA" --schedule "$TASK_PLAN/schedule.json" \
  --checkpoint "$TASK_OLD/inputs/pi05-libero-openpi-20261027-r1" --openpi-root "$TASK_OPENPI" \
  --ledger "$TASK_LEDGER" --max-total-actions "$TASK_ACTIONS" --max-elapsed-seconds "$TASK_WORKER_SECONDS" \
  "${TASK_RESUME_ARGS[@]}" "${TASK_EXTRA[@]}"
