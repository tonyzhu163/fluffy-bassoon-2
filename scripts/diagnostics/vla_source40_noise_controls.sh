#!/usr/bin/env bash
# One of two explicitly authorized60-minute noise-control workers.
# No reservation/job submission; all setup, failed tries and teardown are charged.
set -Eeuo pipefail
TASK_STARTED=$(date +%s)
TASK_SOURCE=${1:?exact reviewed source}
TASK_RUN=${2:?immutable worker run identity}
TASK_WORKER=${3:?predeclared worker index0..1}
TASK_PLAN=${4:?immutable all90 noise-control plan JSON}
TASK_LEDGER=${5:?separate authorized152k/2h target ledger}
TASK_SLICE=${6:?this allocation seconds including setup; initially3540}
TASK_RESUME=${7:-}
case "$TASK_RESUME" in ''|--resume) ;; *) exit 2;; esac
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_OLD=/mowenjie/fluffy-bassoon
TASK_WT=$TASK_ROOT/worktrees/$TASK_RUN
TASK_OUT=$TASK_ROOT/outputs/$TASK_RUN
TASK_MODEL=$TASK_OLD/envs/openpi-215abfb/bin/python
TASK_PROBE=$TASK_OLD/envs/lerobot-v044-8fff0fde/bin/python
TASK_OPENPI=$TASK_OLD/third_party/openpi-215abfb
TASK_CUDA=$TASK_ROOT/outputs/libero-pi05-student-cuda-20261029-r1/cu128-overlay
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
TASK_PLAN_SHA=$(sha256sum "$TASK_PLAN" | cut -d' ' -f1)
TASK_ATTEMPT=$("$TASK_MODEL" - "$TASK_OUT" "$TASK_SOURCE" "$TASK_RUN" "$TASK_WORKER" "$TASK_PLAN" \
  "$TASK_PLAN_SHA" "$TASK_LEDGER" "$TASK_SLICE" "$TASK_RESUME" "$TASK_STARTED" <<'PYTIME'
import hashlib, json, pathlib, sys
from fluffy_bassoon.meta_rl.vla_source40_ppo_worker import begin
root, source, run, worker, plan_path, plan_sha, ledger_path, seconds, resume, started = sys.argv[1:]
plan = json.loads(pathlib.Path(plan_path).read_text())
ledger = json.loads(pathlib.Path(ledger_path).read_text())
row = ledger.get('reservations', {}).get(run)
w = plan['workers'][int(worker)]
if (ledger.get('format') != 'libero90_noise_controls_budget_v1' or ledger.get('maximum_total_actions') != 152000
        or ledger.get('maximum_gpu_seconds') != 7200 or ledger.get('source_campaign_immutable') is not True
        or ledger.get('noise_plan_sha256') != plan['noise_plan_sha256'] or row is None
        or row.get('run_id') != run or row.get('source_commit') != source or row.get('worker') != int(worker)
        or row.get('maximum_total_actions') != w['maximum_total_actions'] or row.get('worker_seconds') != 3600
        or row.get('purpose') != 'target_evaluation' or row.get('source_pretraining_data') is not False
        or row.get('fresh_ppo_training_data') is not False):
    raise ValueError('Exact pre-existing target-only worker reservation is required before any model/simulator work')
record_sha = hashlib.sha256(json.dumps(row, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
contract = dict(operation='source40_libero90_noise_controls', source_commit=source, run_id=run, worker=int(worker),
    noise_plan_sha256=plan['noise_plan_sha256'], plan_file_sha256=plan_sha,
    maximum_total_actions=w['maximum_total_actions'], maximum_driver_seconds=3600,
    escrow_record_sha256=record_sha, deployment_optimizer_updates=0, source_data_from_targets=False)
print(begin(root, contract, slice_seconds=int(seconds), worker_seconds=3600,
    resume=resume == '--resume', now=float(started)))
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
"$TASK_PROBE" 'scripts/[train]gpu_probe.py' --expected-count 1 --purpose evaluation \
  --baige-container --output "$TASK_ATTEMPT/gpu-probe.json"
export LYCO_BAIGE_GPU_MAP=$TASK_ATTEMPT/gpu-probe.json
CUDA_VISIBLE_DEVICES='' PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  PYTHONPATH=$TASK_ROOT/envs/pytest-overlay-20261029-r1:$TASK_WT/src \
  "$TASK_PROBE" -m pytest -q -p no:cacheprovider \
  tests/test_vla_source40_noise_controls.py tests/test_vla_common_runtime_evaluate.py \
  tests/test_vla_source40_evaluate.py tests/test_vla_source40_ppo_worker.py tests/test_gpu_probe.py \
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
  -m fluffy_bassoon.meta_rl.vla_source40_noise_controls \
  --output "$TASK_OUT/experiment" --run-id "$TASK_RUN" --source-commit "$TASK_SOURCE" \
  --expected-hostname "$(hostname)" --worker "$TASK_WORKER" --plan "$TASK_PLAN" --plan-sha256 "$TASK_PLAN_SHA" \
  --stop-before-unix "$(( TASK_STARTED + TASK_SLICE - 125 ))" \
  --ledger "$TASK_LEDGER" \
  --checkpoint "$TASK_OLD/inputs/pi05-libero-openpi-20261027-r1" --openpi-root "$TASK_OPENPI" \
  "${TASK_RESUME_ARGS[@]}"
