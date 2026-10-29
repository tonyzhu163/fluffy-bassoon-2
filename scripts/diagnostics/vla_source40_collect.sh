#!/usr/bin/env bash
# Run one source40 task/partition from an explicit complete-case plan.
# Requires an existing approved allocation; this script does not submit jobs.
# Full-case wall time/storage must be measured and budgeted separately.
set -Eeuo pipefail
TASK_SOURCE=${1:?exact reviewed integrated commit}
TASK_RUN=${2:?fresh output identity}
TASK_SCHEDULE=${3:?reviewed complete-case schedule path}
TASK_SUITE=${4:?source suite}
TASK_NUMBER=${5:?source task index}
TASK_PARTITION=${6:?source partition}
TASK_RESUME=${7:-}
case "$TASK_RESUME" in ''|--resume) ;; *) exit 2;; esac
case "$TASK_SUITE" in libero_spatial|libero_object|libero_goal|libero_10) ;; *) exit 2;; esac
case "$TASK_PARTITION" in train|validation) ;; *) exit 2;; esac
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_OLD=/mowenjie/fluffy-bassoon
TASK_WORKTREE=$TASK_ROOT/worktrees/$TASK_RUN
TASK_OUTPUT=$TASK_ROOT/outputs/$TASK_RUN
TASK_DSRL=$TASK_ROOT/deps/dsrl-pi0-7f48937-20261029-r1/repo
TASK_LIBERO=$TASK_ROOT/deps/libero-dsrl-87edbd1-20261029-r1/repo
TASK_OVERLAY=$TASK_ROOT/envs/dsrl-pi0-overlay-20261029-r1
TASK_OPENPI=$TASK_OLD/third_party/openpi-215abfb
TASK_MODEL=$TASK_OLD/envs/openpi-215abfb/bin/python
TASK_SIM=/mowenjie/envs/wam-libero-plus-py310/bin/python
TASK_TORCH=$TASK_OLD/envs/lerobot-v044-8fff0fde/bin/python
TASK_TEST_OVERLAY=$TASK_ROOT/envs/pytest-overlay-20261029-r1
TASK_CHECKPOINT=$TASK_OLD/inputs/pi05-libero-openpi-20261027-r1
if [ "$TASK_RESUME" = --resume ]; then
  test -f "$TASK_OUTPUT/experiment/source-recovery.pkl"
  test -f "$TASK_OUTPUT/schedule.json"
  cmp "$TASK_SCHEDULE" "$TASK_OUTPUT/schedule.json"
else
  test ! -e "$TASK_OUTPUT"
  mkdir -p "$TASK_OUTPUT"
fi
TASK_ATTEMPT=$TASK_OUTPUT/attempts/$(date -u +%Y%m%dT%H%M%S)-$$
mkdir -p "$TASK_ATTEMPT"
cd "$TASK_WORKTREE"
test "$(git rev-parse HEAD)" = "$TASK_SOURCE"
test -z "$(git status --porcelain)"
exec > >(tee "$TASK_ATTEMPT/console.log") 2>&1
TASK_SIM_PID=
task_cleanup() {
  TASK_RC=$?
  trap - EXIT
  if [ -n "$TASK_SIM_PID" ]; then
    kill "$TASK_SIM_PID" 2>/dev/null || true
    wait "$TASK_SIM_PID" 2>/dev/null || true
  fi
  printf '%s\n' "$TASK_RC" > "$TASK_ATTEMPT/exit-code.txt"
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
  --baige-container --baige-exclusive-training --output "$TASK_ATTEMPT/gpu-probe.json"
export LYCO_BAIGE_GPU_MAP=$TASK_ATTEMPT/gpu-probe.json
CUDA_VISIBLE_DEVICES='' PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH="$TASK_TEST_OVERLAY:$TASK_WORKTREE/src" \
  "$TASK_TORCH" -m pytest -q -p no:cacheprovider tests/test_vla*.py tests/test_gpu_probe.py \
  > "$TASK_ATTEMPT/focused-tests.log" 2>&1
printf -v TASK_KEY '%s:%02d' "$TASK_SUITE" "$TASK_NUMBER"
"$TASK_MODEL" - "$TASK_SCHEDULE" "$TASK_OUTPUT/schedule.json" "$TASK_KEY" "$TASK_PARTITION" "$TASK_OUTPUT/budget.json" <<'PYPLAN'
import json, pathlib, shutil, sys
from fluffy_bassoon.meta_rl.vla_source40_schedule import load_schedule, selected_partition_budget
original, destination, key, partition, budget_path = sys.argv[1:]
plan = load_schedule(original)
assert plan.get('collection_mode') == 'complete_cases', 'Require explicit complete-case mode'
assert any(task['task_key'] == key for task in plan['tasks']), key
if pathlib.Path(original).resolve() != pathlib.Path(destination).resolve():
    if pathlib.Path(destination).exists():
        assert pathlib.Path(original).read_bytes() == pathlib.Path(destination).read_bytes()
    else:
        shutil.copyfile(original, destination)
budget = selected_partition_budget(plan, partition)
pathlib.Path(budget_path).write_text(json.dumps(budget, indent=2) + '\n')
print(json.dumps(dict(task=key, partition=partition, schedule_sha256=plan['schedule_sha256'], budget=budget), indent=2), flush=True)
PYPLAN
TASK_MAX_STEPS=$("$TASK_MODEL" - "$TASK_OUTPUT/budget.json" <<'PYBUDGET'
import json, pathlib, sys
print(json.loads(pathlib.Path(sys.argv[1]).read_text())['maximum_primitive_steps'])
PYBUDGET
)
export PYTHONPATH=$TASK_OVERLAY:$TASK_WORKTREE/src:$TASK_DSRL:$TASK_OPENPI/src:$TASK_OPENPI/packages/openpi-client/src
"$TASK_MODEL" - "$TASK_CHECKPOINT" <<'PY'
import hashlib, json, pathlib, sys
import pickle
import jax
import numpy as np
from openpi.training import config
from fluffy_bassoon.meta_rl.vla_source40_corpus import save_motor_rng, restore_motor_rng
path = pathlib.Path(sys.argv[1])
receipt = json.loads((path/'READY.json').read_text())
assert hashlib.sha256((path/'READY.json').read_bytes()).hexdigest() == 'ec064cce32f5003e2e3979c5712d7d6385018f6bfcc59cb2161105881e70d170'
assert receipt['status'] == 'verified' and receipt['checkpoint'] == 'gs://openpi-assets/checkpoints/pi05_libero'
model = config.get_config('pi05_libero').model
assert model.pi05 and model.action_horizon == 10 and model.action_dim == 32
assert hashlib.sha256((path/'assets/physical-intelligence/libero/norm_stats.json').read_bytes()).hexdigest() == 'b3a44bb2810436fb62917decaea58bd4d9110255df527dea21e8fd40c960bd84'
for row in receipt['files']:
    assert (path/row['name']).stat().st_size == row['bytes']
for key in (jax.random.key(17), jax.random.PRNGKey(17)):
    restored = restore_motor_rng(pickle.loads(pickle.dumps(save_motor_rng(key))))
    np.testing.assert_array_equal(jax.random.key_data(jax.random.split(restored)),
                                  jax.random.key_data(jax.random.split(key)))
print('Motor PRNG serialization round-trip passed for typed and legacy keys', flush=True)
PY
PYTHONPATH=$TASK_WORKTREE/src:$TASK_LIBERO:$TASK_OPENPI/packages/openpi-client/src:$TASK_OLD/envs/pi05-plus-client-deps \
  "$TASK_SIM" -m fluffy_bassoon.meta_rl.vla_libero --mode env-server \
  --output "$TASK_ATTEMPT/simulator" --port 8765 --suite "$TASK_SUITE" --task-id "$TASK_NUMBER" \
  --horizon 400 --fresh-environment-on-reset > "$TASK_ATTEMPT/simulator.log" 2>&1 &
TASK_SIM_PID=$!
for TASK_POLL in $(seq 1 600); do
  if [ -e "$TASK_ATTEMPT/simulator/simulator-ready.json" ]; then break; fi
  kill -0 "$TASK_SIM_PID"
  sleep 1
done
test -e "$TASK_ATTEMPT/simulator/simulator-ready.json"
TASK_RESUME_ARGS=()
if [ "$TASK_RESUME" = --resume ]; then TASK_RESUME_ARGS=(--resume); fi
"$TASK_MODEL" -m fluffy_bassoon.meta_rl.vla_source40_corpus \
  --schedule "$TASK_OUTPUT/schedule.json" --task-key "$TASK_KEY" --partition "$TASK_PARTITION" \
  --output "$TASK_OUTPUT/experiment" --port 8765 --openpi-root "$TASK_OPENPI" \
  --dsrl-root "$TASK_DSRL" --checkpoint "$TASK_CHECKPOINT" --source-commit "$TASK_SOURCE" \
  --max-primitive-steps "$TASK_MAX_STEPS" "${TASK_RESUME_ARGS[@]}"
