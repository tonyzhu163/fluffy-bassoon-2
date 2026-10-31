#!/usr/bin/env bash
# Existing approved allocation only. This script never requests resources.
set -Eeuo pipefail
TASK_STARTED=$(date +%s)
TASK_SOURCE=${1:?exact reviewed integrated source commit}
TASK_RUN=${2:?fresh immutable task57 identity}
TASK_GRANT=${3:?approved task57 grant JSON}
TASK_GRANT_SHA=${4:?grant SHA256}
shift 4
TASK_SLICE=7140
if [ "$#" -gt 0 ] && [[ "$1" != --* ]]; then TASK_SLICE=$1; shift; fi
TASK_RESUME=()
TASK_PAUSE=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --resume) TASK_RESUME=(--resume); shift;;
    --pause-after-updates) TASK_PAUSE=(--pause-after-updates "${2:?update pause}"); shift 2;;
    *) printf 'Unknown task57 option: %s\n' "$1" >&2; exit 2;;
  esac
done
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_OLD=/mowenjie/fluffy-bassoon
TASK_WT=$TASK_ROOT/worktrees/$TASK_RUN
TASK_OUT=$TASK_ROOT/outputs/$TASK_RUN
TASK_MODEL=$TASK_OLD/envs/openpi-215abfb/bin/python
TASK_PROBE=$TASK_OLD/envs/lerobot-v044-8fff0fde/bin/python
TASK_OPENPI=$TASK_OLD/third_party/openpi-215abfb
TASK_DSRL=$TASK_ROOT/deps/dsrl-pi0-7f48937-20261029-r1/repo
TASK_DSRL_OVERLAY=$TASK_ROOT/envs/dsrl-pi0-overlay-20261029-r1
TASK_LIBERO=$TASK_ROOT/deps/libero-dsrl-87edbd1-20261029-r1/repo
TASK_SIM=/mowenjie/envs/wam-libero-plus-py310/bin/python
if [ "${#TASK_RESUME[@]}" -eq 0 ]; then test ! -e "$TASK_OUT"; mkdir -p "$TASK_OUT";
else test -f "$TASK_OUT/worker-time.json"; fi
exec 9> "$TASK_OUT/launcher.lock"
flock -n 9
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export PYTHONPATH=$TASK_WT/src
TASK_CLOCK=$("$TASK_MODEL" -m fluffy_bassoon.meta_rl.vla_target_online_dsrl_task57 begin \
  --root "$TASK_OUT" --source "$TASK_SOURCE" --run "$TASK_RUN" \
  --grant "$TASK_GRANT" --grant-sha256 "$TASK_GRANT_SHA" \
  --slice-seconds "$TASK_SLICE" --started-unix "$TASK_STARTED" "${TASK_RESUME[@]}")
IFS=$'\t' read -r TASK_ATTEMPT TASK_STOP TASK_ACTUAL_SLICE <<< "$TASK_CLOCK"
exec > >(tee "$TASK_ATTEMPT/console.log") 2>&1
task_cleanup() {
  TASK_RC=$?
  trap - EXIT
  PYTHONPATH=$TASK_WT/src "$TASK_MODEL" -m fluffy_bassoon.meta_rl.vla_target_online_dsrl_task57 finish \
    --root "$TASK_OUT" --attempt "$TASK_ATTEMPT" --exit-code "$TASK_RC" || true
  printf '%s\n' "$TASK_RC" > "$TASK_ATTEMPT/exit-code.txt"
  printf '%s\n' "$TASK_RC" > "$TASK_OUT/exit-code.txt"
  exit "$TASK_RC"
}
trap task_cleanup EXIT
cd "$TASK_WT"
test "$(git rev-parse HEAD)" = "$TASK_SOURCE"
test -z "$(git status --porcelain)"
TASK_DRIVER_RESUME=()
if [ -e "$TASK_OUT/experiment" ]; then
  test "${#TASK_RESUME[@]}" -gt 0
  TASK_DRIVER_RESUME=(--resume)
fi
export FLUFFY_GPU_EXECUTION=baige CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=disabled
"$TASK_PROBE" 'scripts/[train]gpu_probe.py' --expected-count 1 --purpose training \
  --baige-container --baige-exclusive-training --output "$TASK_ATTEMPT/gpu-probe.json"
export LYCO_BAIGE_GPU_MAP=$TASK_ATTEMPT/gpu-probe.json
# New glue/recovery tests are qualified once on allocated0GPU Linux before
# this exact source is admitted. Do not repeat CPU tests on paid GPU time.
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0
export OPENPI_DATA_HOME=$TASK_OLD/cache/openpi-215abfb
export LIBERO_CONFIG_PATH=$TASK_ROOT/cache/libero-dsrl-87edbd1-20261029-r1
export MAGICK_HOME=/mowenjie/libs/libero-plus-imagemagick
export MAGICK_CODER_MODULE_PATH=$MAGICK_HOME/usr/lib/x86_64-linux-gnu/ImageMagick-6.9.11/modules-Q16/coders
export LD_LIBRARY_PATH=$MAGICK_HOME/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PYTHONPATH=$TASK_DSRL_OVERLAY:$TASK_WT/src:$TASK_OPENPI/src:$TASK_OPENPI/packages/openpi-client/src
TASK_TIMEOUT=$(( TASK_STOP - $(date +%s) - 60 ))
test "$TASK_TIMEOUT" -gt 300
timeout --signal=INT --kill-after=30s "$TASK_TIMEOUT" "$TASK_MODEL" \
  -m fluffy_bassoon.meta_rl.vla_target_online_dsrl_task57 run \
  --output "$TASK_OUT/experiment" --source-commit "$TASK_SOURCE" --run-id "$TASK_RUN" \
  --expected-hostname "$(hostname)" --grant "$TASK_GRANT" --grant-sha256 "$TASK_GRANT_SHA" \
  --checkpoint "$TASK_OLD/inputs/pi05-libero-openpi-20261027-r1" \
  --openpi-root "$TASK_OPENPI" --dsrl-root "$TASK_DSRL" --libero-root "$TASK_LIBERO" \
  --simulator-python "$TASK_SIM" \
  --simulator-pythonpath "$TASK_WT/src:$TASK_LIBERO:$TASK_OPENPI/packages/openpi-client/src:$TASK_OLD/envs/pi05-plus-client-deps" \
  --stop-unix "$(( TASK_STOP - 180 ))" "${TASK_DRIVER_RESUME[@]}" "${TASK_PAUSE[@]}"
