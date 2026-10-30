#!/usr/bin/env bash
# Run only inside an already approved one-GPU allocation; reserves no budget.
set -Eeuo pipefail
TASK_SOURCE=${1:?exact reviewed commit}
TASK_RUN=${2:?fresh immutable target baseline identity}
TASK_SECONDS=7200
shift 2
if [ "$#" -gt 0 ] && [ "$1" != --resume ]; then TASK_SECONDS=$1; shift; fi
TASK_RESUME=()
if [ "$#" -gt 0 ]; then test "$1" = --resume; TASK_RESUME=(--resume); shift; fi
test "$#" -eq 0
test "$TASK_SECONDS" -eq 7200
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_OLD=/mowenjie/fluffy-bassoon
TASK_WT=$TASK_ROOT/worktrees/$TASK_RUN
TASK_OUT=$TASK_ROOT/outputs/$TASK_RUN
TASK_MODEL=$TASK_OLD/envs/openpi-215abfb/bin/python
TASK_PROBE=$TASK_OLD/envs/lerobot-v044-8fff0fde/bin/python
TASK_OPENPI=$TASK_OLD/third_party/openpi-215abfb
TASK_DSRL=$TASK_ROOT/deps/dsrl-pi0-7f48937-20261029-r1/repo
TASK_DSRL_OVERLAY=$TASK_ROOT/envs/dsrl-pi0-overlay-20261029-r1
TASK_TEST=$TASK_ROOT/envs/pytest-overlay-20261029-r1
TASK_LIBERO=$TASK_ROOT/deps/libero-dsrl-87edbd1-20261029-r1/repo
TASK_SIM=/mowenjie/envs/wam-libero-plus-py310/bin/python
TASK_STARTED=$(date +%s)
if [ "${#TASK_RESUME[@]}" -eq 0 ]; then test ! -e "$TASK_OUT"; mkdir -p "$TASK_OUT";
else test -f "$TASK_OUT/worker-time.json"; fi
exec 9> "$TASK_OUT/launcher.lock"
flock -n 9
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export PYTHONPATH=$TASK_WT/src
TASK_CLOCK=$("$TASK_MODEL" -m fluffy_bassoon.meta_rl.vla_target_online_dsrl_clock begin \
  --root "$TASK_OUT" --source "$TASK_SOURCE" --run "$TASK_RUN" \
  --started-unix "$TASK_STARTED" "${TASK_RESUME[@]}")
IFS=$'\t' read -r TASK_ATTEMPT TASK_STOP_UNIX TASK_SLICE_SECONDS <<< "$TASK_CLOCK"
test -d "$TASK_ATTEMPT"
exec > >(tee "$TASK_ATTEMPT/console.log") 2>&1
task_cleanup() {
  TASK_RC=$?
  trap - EXIT
  PYTHONPATH=$TASK_WT/src "$TASK_MODEL" -m fluffy_bassoon.meta_rl.vla_target_online_dsrl_clock finish \
    --root "$TASK_OUT" --attempt "$TASK_ATTEMPT" --exit-code "$TASK_RC" || true
  printf '%s\n' "$TASK_RC" > "$TASK_ATTEMPT/exit-code.txt"
  printf '%s\n' "$TASK_RC" > "$TASK_OUT/exit-code.txt"
  exit "$TASK_RC"
}
trap task_cleanup EXIT
cd "$TASK_WT"
test "$(git rev-parse HEAD)" = "$TASK_SOURCE"
test -z "$(git status --porcelain)"
# A failed setup attempt can consume time before creating any science state.
# Resume its clock, but start a fresh driver only when no experiment exists.
TASK_DRIVER_RESUME=()
if [ -f "$TASK_OUT/experiment/contract.json" ]; then
  test "${#TASK_RESUME[@]}" -gt 0
  TASK_DRIVER_RESUME=(--resume)
else
  test ! -e "$TASK_OUT/experiment"
fi
export FLUFFY_GPU_EXECUTION=baige CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=disabled
export PYTHONPATH=$TASK_WT/src
"$TASK_PROBE" 'scripts/[train]gpu_probe.py' --expected-count 1 --purpose training \
  --baige-container --baige-exclusive-training --output "$TASK_ATTEMPT/gpu-probe.json"
export LYCO_BAIGE_GPU_MAP=$TASK_ATTEMPT/gpu-probe.json
CUDA_VISIBLE_DEVICES='' PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$TASK_TEST:$TASK_WT/src \
  "$TASK_PROBE" -m pytest -q -p no:cacheprovider \
  tests/test_vla_target_online_dsrl.py tests/test_vla_target_online_dsrl_six.py \
  tests/test_vla_target_online_dsrl_clock.py tests/test_vla_source40_ppo_worker.py \
  tests/test_vla_source40.py::test_sac_persists_across_resets_and_updates_only_after_real_warmup \
  tests/test_vla_source40.py::test_crossing_warmup_keeps_gaussian_and_weights_fixed_until_episode_end \
  tests/test_vla_source40.py::test_resume_charges_unresolved_execution_intent_conservatively \
  tests/test_vla_source40.py::test_physical_ledger_includes_failed_settling_and_unresolved_step_intents \
  > "$TASK_ATTEMPT/focused-tests.log" 2>&1
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$TASK_TEST:$TASK_DSRL_OVERLAY:$TASK_WT/src:$TASK_OPENPI/src:$TASK_OPENPI/packages/openpi-client/src \
  "$TASK_MODEL" -m pytest -q -p no:cacheprovider \
  tests/test_vla_source40.py::test_motor_rng_round_trip_preserves_next_random_draw \
  > "$TASK_ATTEMPT/motor-rng-tests.log" 2>&1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0
export OPENPI_DATA_HOME=$TASK_OLD/cache/openpi-215abfb
export LIBERO_CONFIG_PATH=$TASK_ROOT/cache/libero-dsrl-87edbd1-20261029-r1
export MAGICK_HOME=/mowenjie/libs/libero-plus-imagemagick
export MAGICK_CODER_MODULE_PATH=$MAGICK_HOME/usr/lib/x86_64-linux-gnu/ImageMagick-6.9.11/modules-Q16/coders
export LD_LIBRARY_PATH=$MAGICK_HOME/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PYTHONPATH=$TASK_DSRL_OVERLAY:$TASK_WT/src:$TASK_OPENPI/src:$TASK_OPENPI/packages/openpi-client/src
TASK_REMAINING=$(( TASK_STOP_UNIX - $(date +%s) - 60 ))
test "$TASK_REMAINING" -gt 180
timeout --signal=INT --kill-after=30s "$TASK_REMAINING" "$TASK_MODEL" \
  -m fluffy_bassoon.meta_rl.vla_target_online_dsrl_six \
  --output "$TASK_OUT/experiment" --source-commit "$TASK_SOURCE" --expected-hostname "$(hostname)" \
  --checkpoint "$TASK_OLD/inputs/pi05-libero-openpi-20261027-r1" \
  --openpi-root "$TASK_OPENPI" --dsrl-root "$TASK_DSRL" --libero-root "$TASK_LIBERO" \
  --simulator-python "$TASK_SIM" \
  --simulator-pythonpath "$TASK_WT/src:$TASK_LIBERO:$TASK_OPENPI/packages/openpi-client/src:$TASK_OLD/envs/pi05-plus-client-deps" \
  --max-total-actions 64000 --max-elapsed-seconds 7200 --stop-unix "$TASK_STOP_UNIX" \
  --learner-seed 0 "${TASK_DRIVER_RESUME[@]}"
