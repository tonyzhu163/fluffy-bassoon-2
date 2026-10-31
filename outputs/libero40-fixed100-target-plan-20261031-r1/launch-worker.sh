#!/usr/bin/env bash
# One of six root-authorized target workers. No reservation or job submission.
# The original one-hour cumulative clock includes setup, tests and failed tries.
set -Eeuo pipefail
TASK_STARTED=$(date +%s)
TASK_SOURCE=${1:?exact reviewed source}
TASK_RUN=${2:?immutable worker run identity}
TASK_WORKER=${3:?predeclared worker index0..5}
TASK_PLAN=${4:?immutable first-wave plan JSON}
TASK_MATRIX=${5:?completed profile-bound source40 matrix JSON}
TASK_ASSEMBLY=${6:?source40 training assembly JSON}
TASK_LEDGER=${7:?separate authorized profile-specific target ledger}
TASK_SLICE=${8:?this allocation seconds including setup; initially3540}
TASK_RESUME=${9:-}
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
TASK_MATRIX_SHA=$(sha256sum "$TASK_MATRIX" | cut -d' ' -f1)
TASK_ASSEMBLY_SHA=$(sha256sum "$TASK_ASSEMBLY" | cut -d' ' -f1)
TASK_ATTEMPT=$("$TASK_MODEL" - "$TASK_OUT" "$TASK_SOURCE" "$TASK_RUN" "$TASK_WORKER" "$TASK_PLAN" \
  "$TASK_PLAN_SHA" "$TASK_MATRIX_SHA" "$TASK_ASSEMBLY_SHA" "$TASK_LEDGER" "$TASK_SLICE" "$TASK_RESUME" "$TASK_STARTED" <<'PYTIME'
import hashlib, json, pathlib, sys
from fluffy_bassoon.meta_rl.vla_source40_ppo_worker import begin
from fluffy_bassoon.meta_rl.vla_source40_target_firstwave_plan import load_plan, profile_settings, LEGACY_PROFILE
root, source, run, worker, plan_path, plan_sha, matrix_sha, assembly_sha, ledger_path, seconds, resume, started = sys.argv[1:]
plan = load_plan(plan_path, plan_sha, source)
settings = profile_settings(plan['contract'].get('profile', LEGACY_PROFILE))
ledger = json.loads(pathlib.Path(ledger_path).read_text())
row = ledger.get('reservations', {}).get(run)
w = plan['workers'][int(worker)]
if (ledger.get('format') != 'libero_target_wave_budget_v1' or ledger.get('maximum_total_actions') != settings['maximum_total_actions']
        or ledger.get('maximum_gpu_seconds') != settings['maximum_gpu_seconds'] or ledger.get('source_campaign_immutable') is not True
        or ledger.get('wave_sha256') != plan['firstwave_sha256'] or row is None
        or row.get('run_id') != run or row.get('source_commit') != source or row.get('worker') != int(worker)
        or row.get('maximum_total_actions') != w['action_cap'] or row.get('worker_seconds') != settings['worker_seconds']
        or row.get('purpose') != 'target_evaluation' or row.get('source_pretraining_data') is not False
        or row.get('fresh_ppo_training_data') is not False):
    raise ValueError('Exact pre-existing target-only worker reservation is required before any model/simulator work')
record_sha = hashlib.sha256(json.dumps(row, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
contract = dict(operation='source40_ad_target_firstwave', source_commit=source, run_id=run, worker=int(worker),
    wave_sha256=plan['firstwave_sha256'], plan_file_sha256=plan_sha, matrix_file_sha256=matrix_sha,
    assembly_file_sha256=assembly_sha, maximum_total_actions=w['action_cap'], maximum_driver_seconds=settings['worker_seconds'],
    escrow_record_sha256=record_sha, deployment_optimizer_updates=0, source_data_from_targets=False)
print(begin(root, contract, slice_seconds=int(seconds), worker_seconds=settings['worker_seconds'],
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
# Existing e764 code passed its allocated CPU qualification; no redundant tests here.
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0
export LIBERO_CONFIG_PATH=$TASK_ROOT/cache/libero-dsrl-87edbd1-20261029-r1
export OPENPI_DATA_HOME=$TASK_OLD/cache/openpi-215abfb
export MAGICK_HOME=/mowenjie/libs/libero-plus-imagemagick
export MAGICK_CODER_MODULE_PATH=$MAGICK_HOME/usr/lib/x86_64-linux-gnu/ImageMagick-6.9.11/modules-Q16/coders
export LD_LIBRARY_PATH=$MAGICK_HOME/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PYTHONPATH=$TASK_CUDA:$TASK_WT/src:$TASK_OPENPI/src:$TASK_OPENPI/packages/openpi-client/src
TASK_TIMEOUT=$(( TASK_SLICE - ( $(date +%s) - TASK_STARTED ) - 60 ))
test "$TASK_TIMEOUT" -gt 60
TASK_DEADLINE_ARGS=()
TASK_PROFILE=$(PYTHONPATH=$TASK_WT/src "$TASK_MODEL" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["contract"].get("profile", "legacy500"))' "$TASK_PLAN")
if [ "$TASK_PROFILE" = interim100-libero90 ]; then
  TASK_DEADLINE_ARGS=(--stop-before-unix "$(( TASK_STARTED + TASK_SLICE - 125 ))")
fi
timeout --signal=INT --kill-after=30s "$TASK_TIMEOUT" "$TASK_MODEL" \
  -m fluffy_bassoon.meta_rl.vla_source40_target_firstwave \
  --output "$TASK_OUT/experiment" --run-id "$TASK_RUN" --source-commit "$TASK_SOURCE" \
  --expected-hostname "$(hostname)" --worker "$TASK_WORKER" --plan "$TASK_PLAN" --plan-sha256 "$TASK_PLAN_SHA" \
  --matrix "$TASK_MATRIX" --matrix-sha256 "$TASK_MATRIX_SHA" \
  --assembly "$TASK_ASSEMBLY" --assembly-sha256 "$TASK_ASSEMBLY_SHA" --ledger "$TASK_LEDGER" \
  --checkpoint "$TASK_OLD/inputs/pi05-libero-openpi-20261027-r1" --openpi-root "$TASK_OPENPI" \
  "${TASK_RESUME_ARGS[@]}" "${TASK_DEADLINE_ARGS[@]}"
