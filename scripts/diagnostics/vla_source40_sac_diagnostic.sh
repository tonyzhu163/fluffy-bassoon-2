#!/usr/bin/env bash
# Existing allocated GPU only. Legacy default uses the original PPO remainder;
# optional topup mode uses the NEW20k escrow inside the cumulative source1M.
# No scheduler submission, GPU grant or actor training.
set -Eeuo pipefail
TASK_SOURCE=${1:?reviewed exact source commit}
TASK_RUN=${2:?fresh immutable competence diagnostic identity}
TASK_ALIAS=${3:?explicit terminal-validation remainder alias JSON}
TASK_ALIAS_SHA=${4:?reviewed alias SHA256}
TASK_SLICE=${5:?allocated seconds including setup/shutdown within alias time}
TASK_RESUME=${6:-}
TASK_BUDGET_KIND=${7:-legacy}
case "$TASK_RESUME" in ''|--resume) ;; *) exit 2;; esac
case "$TASK_BUDGET_KIND" in legacy|topup) ;; *) exit 2;; esac
TASK_STARTED=$(date +%s)
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_OLD=/mowenjie/fluffy-bassoon
TASK_WT=$TASK_ROOT/worktrees/$TASK_RUN
TASK_OUT=$TASK_ROOT/outputs/$TASK_RUN
TASK_MODEL=$TASK_OLD/envs/openpi-215abfb/bin/python
TASK_PROBE=$TASK_OLD/envs/lerobot-v044-8fff0fde/bin/python
TASK_PLAN=$TASK_ROOT/outputs/libero40-source-1m-20261029-r1
TASK_OPENPI=$TASK_OLD/third_party/openpi-215abfb
TASK_CUDA=$TASK_ROOT/outputs/libero-pi05-student-cuda-20261029-r1/cu128-overlay
TASK_DSRL=$TASK_ROOT/deps/dsrl-pi0-7f48937-20261029-r1/repo
TASK_DSRL_OVERLAY=$TASK_ROOT/envs/dsrl-pi0-overlay-20261029-r1
TASK_LIBERO=$TASK_ROOT/deps/libero-dsrl-87edbd1-20261029-r1/repo
cd "$TASK_WT"
test "$(git rev-parse HEAD)" = "$TASK_SOURCE"
test -z "$(git status --porcelain)"
TASK_RESUME_ARGS=()
if [ "$TASK_RESUME" = --resume ]; then
  test -f "$TASK_OUT/experiment/diagnostic-recovery.pkl"
  TASK_RESUME_ARGS=(--resume)
else
  test ! -e "$TASK_OUT"
  mkdir -p "$TASK_OUT"
fi
exec 9> "$TASK_OUT/launcher.lock"
flock -n 9
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export PYTHONPATH=$TASK_CUDA:$TASK_DSRL_OVERLAY:$TASK_WT/src:$TASK_OPENPI/src:$TASK_OPENPI/packages/openpi-client/src
TASK_ATTEMPT=$("$TASK_MODEL" - "$TASK_OUT" "$TASK_SOURCE" "$TASK_RUN" "$TASK_ALIAS" "$TASK_ALIAS_SHA" \
  "$TASK_SLICE" "$TASK_RESUME" "$TASK_STARTED" "$TASK_BUDGET_KIND" <<'PYTIME'
import pathlib, sys
from fluffy_bassoon.meta_rl.vla_source40_ppo_worker import begin
root, source, run, alias_path, alias_sha, seconds, resume, started, kind = sys.argv[1:]
if kind == 'topup':
    from fluffy_bassoon.meta_rl.vla_source40_topup_competence import load_alias
else:
    from fluffy_bassoon.meta_rl.vla_source40_sac_diagnostic import load_alias
alias = load_alias(alias_path, alias_sha, run_id=run, source_commit=source)
contract = dict(operation='source40_per_task_sac_competence', source_commit=source, run_id=run,
    alias_path=alias_path, alias_sha256=alias_sha, maximum_total_actions=20000,
    maximum_worker_seconds=alias['maximum_worker_seconds'], optimizer_updates=0)
if kind == 'topup':
    from fluffy_bassoon.meta_rl.vla_source40_topup import load_plan, time_hold
    contract['plan_sha256'] = alias['plan_sha256']
    time_hold(load_plan(alias['plan_path']), alias['canonical_topup_ledger'], root, alias['maximum_worker_seconds'])
print(begin(pathlib.Path(root), contract, slice_seconds=int(seconds),
    worker_seconds=alias['maximum_worker_seconds'], resume=resume == '--resume', now=float(started)))
PYTIME
)
exec > >(tee "$TASK_ATTEMPT/console.log") 2>&1
task_cleanup() {
  TASK_RC=$?
  trap - EXIT
  if [ "$TASK_BUDGET_KIND" = topup ]; then
    PYTHONPATH=$TASK_WT/src "$TASK_MODEL" -m fluffy_bassoon.meta_rl.vla_source40_topup clock-finish \
      --output "$TASK_OUT" --attempt "$TASK_ATTEMPT" --exit-code "$TASK_RC" \
      --plan "$TASK_TOPUP_PLAN" --ledger "$TASK_TOPUP_LEDGER" || true
  else
    PYTHONPATH=$TASK_WT/src "$TASK_MODEL" -m fluffy_bassoon.meta_rl.vla_source40_ppo_worker finish \
      --root "$TASK_OUT" --attempt "$TASK_ATTEMPT" --exit-code "$TASK_RC" || true
  fi
  printf '%s\n' "$TASK_RC" > "$TASK_ATTEMPT/exit-code.txt"
  printf '%s\n' "$TASK_RC" > "$TASK_OUT/exit-code.txt"
  exit "$TASK_RC"
}
TASK_DIAGNOSTIC_MODULE=(-m fluffy_bassoon.meta_rl.vla_source40_sac_diagnostic)
TASK_SOURCE_SCHEDULE=$TASK_PLAN/schedule.json
TASK_SOURCE_LEDGER=$TASK_PLAN/ledger.json
TASK_OUTPUTS_ROOT=$TASK_ROOT/outputs
if [ "$TASK_BUDGET_KIND" = topup ]; then
  TASK_DIAGNOSTIC_MODULE=(-m fluffy_bassoon.meta_rl.vla_source40_topup_competence run)
  mapfile -t TASK_TOPUP_FIELDS < <("$TASK_MODEL" - "$TASK_ALIAS" <<'PYALIAS'
import json, sys
value = json.load(open(sys.argv[1]))
for key in ('plan_path', 'canonical_topup_ledger', 'schedule', 'source_ledger', 'outputs_root'):
    print(value[key])
PYALIAS
  )
  TASK_TOPUP_PLAN=${TASK_TOPUP_FIELDS[0]}
  TASK_TOPUP_LEDGER=${TASK_TOPUP_FIELDS[1]}
  TASK_SOURCE_SCHEDULE=${TASK_TOPUP_FIELDS[2]}
  TASK_SOURCE_LEDGER=${TASK_TOPUP_FIELDS[3]}
  TASK_OUTPUTS_ROOT=${TASK_TOPUP_FIELDS[4]}
fi
trap task_cleanup EXIT
export FLUFFY_GPU_EXECUTION=baige CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=disabled
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
PYTHONPATH=$TASK_WT/src "$TASK_PROBE" 'scripts/[train]gpu_probe.py' --expected-count 1 --purpose training \
  --baige-container --baige-exclusive-training --output "$TASK_ATTEMPT/gpu-probe.json"
export LYCO_BAIGE_GPU_MAP=$TASK_ATTEMPT/gpu-probe.json
CUDA_VISIBLE_DEVICES='' PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  PYTHONPATH=$TASK_ROOT/envs/pytest-overlay-20261029-r1:$TASK_WT/src \
  "$TASK_PROBE" -m pytest -q -p no:cacheprovider tests/test_vla_source40_sac_diagnostic.py \
  > "$TASK_ATTEMPT/focused-tests.log" 2>&1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0
export LIBERO_CONFIG_PATH=$TASK_ROOT/cache/libero-dsrl-87edbd1-20261029-r1
export OPENPI_DATA_HOME=$TASK_OLD/cache/openpi-215abfb
export MAGICK_HOME=/mowenjie/libs/libero-plus-imagemagick
export MAGICK_CODER_MODULE_PATH=$MAGICK_HOME/usr/lib/x86_64-linux-gnu/ImageMagick-6.9.11/modules-Q16/coders
export LD_LIBRARY_PATH=$MAGICK_HOME/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
TASK_TIMEOUT=$((TASK_SLICE - $(date +%s) + TASK_STARTED - 60))
test "$TASK_TIMEOUT" -gt 240
TASK_STOP_UNIX=$(( $(date +%s) + TASK_TIMEOUT - 60 ))
timeout --signal=INT --kill-after=30s "$TASK_TIMEOUT" "$TASK_MODEL" \
  "${TASK_DIAGNOSTIC_MODULE[@]}" \
  --output "$TASK_OUT/experiment" --alias "$TASK_ALIAS" --alias-sha256 "$TASK_ALIAS_SHA" \
  --schedule "$TASK_SOURCE_SCHEDULE" --source-ledger "$TASK_SOURCE_LEDGER" --outputs-root "$TASK_OUTPUTS_ROOT" \
  --checkpoint "$TASK_OLD/inputs/pi05-libero-openpi-20261027-r1" --openpi-root "$TASK_OPENPI" --dsrl-root "$TASK_DSRL" \
  --simulator-python /mowenjie/envs/wam-libero-plus-py310/bin/python \
  --simulator-pythonpath "$TASK_WT/src:$TASK_LIBERO:$TASK_OPENPI/packages/openpi-client/src:$TASK_OLD/envs/pi05-plus-client-deps" \
  --source-commit "$TASK_SOURCE" --expected-hostname "$(hostname)" \
  --invocation-stop-unix "$TASK_STOP_UNIX" "${TASK_RESUME_ARGS[@]}"
