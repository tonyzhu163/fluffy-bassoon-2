#!/usr/bin/env bash
# Zero-simulator-action motor diagnostic inside an existing <=10min GPU allocation.
set -Eeuo pipefail
TASK_SOURCE=${1:?exact reviewed motor diagnostic commit}
TASK_RUN=${2:?fresh diagnostic run identity}
TASK_REFERENCE=${3:-libero40-subset-target-20261030-r2}
TASK_DSRL_RUN=${4:-libero40-target-online-dsrl-20261030-r2}
TASK_ORDER=${5:-typed-first}
case "$TASK_ORDER" in typed-first|legacy-first) ;; *) exit 2;; esac
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_OLD=/mowenjie/fluffy-bassoon
TASK_WT=$TASK_ROOT/worktrees/$TASK_RUN
TASK_OUT=$TASK_ROOT/outputs/$TASK_RUN
TASK_MODEL=$TASK_OLD/envs/openpi-215abfb/bin/python
TASK_PROBE=$TASK_OLD/envs/lerobot-v044-8fff0fde/bin/python
TASK_OPENPI=$TASK_OLD/third_party/openpi-215abfb
TASK_DSRL=$TASK_ROOT/deps/dsrl-pi0-7f48937-20261029-r1/repo
TASK_OVERLAY=$TASK_ROOT/envs/dsrl-pi0-overlay-20261029-r1
TASK_CUDA=$TASK_ROOT/outputs/libero-pi05-student-cuda-20261029-r1/cu128-overlay
test ! -e "$TASK_OUT"
mkdir -p "$TASK_OUT"
cd "$TASK_WT"
test "$(git rev-parse HEAD)" = "$TASK_SOURCE"
test -z "$(git status --porcelain)"
exec > >(tee "$TASK_OUT/console.log") 2>&1
task_cleanup() {
  TASK_RC=$?
  trap - EXIT
  printf '%s\n' "$TASK_RC" > "$TASK_OUT/exit-code.txt"
  exit "$TASK_RC"
}
trap task_cleanup EXIT
export FLUFFY_GPU_EXECUTION=baige CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=disabled
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTHONPATH=$TASK_WT/src
"$TASK_PROBE" 'scripts/[train]gpu_probe.py' --expected-count 1 --purpose training \
  --baige-container --baige-exclusive-training --output "$TASK_OUT/gpu-probe.json"
export LYCO_BAIGE_GPU_MAP=$TASK_OUT/gpu-probe.json
CUDA_VISIBLE_DEVICES='' PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  PYTHONPATH=$TASK_ROOT/envs/pytest-overlay-20261029-r1:$TASK_WT/src \
  "$TASK_PROBE" -m pytest -q -p no:cacheprovider tests/test_vla_motor_replay.py \
  > "$TASK_OUT/focused-tests.log" 2>&1
"$TASK_MODEL" - "$TASK_ROOT/outputs/$TASK_REFERENCE/experiment" \
  "$TASK_ROOT/outputs/$TASK_DSRL_RUN/experiment" "$TASK_OUT/inputs.json" <<'PY'
import hashlib, json, pathlib, sys
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
old, new, output = map(pathlib.Path, sys.argv[1:])
entries = []
for task, group in ((0, 'group-000'), (1, 'group-014')):
    reference, dsrl = old / group / 'index.json', new / f'case-{task:02d}' / 'index.json'
    entries.append(dict(task_id=task, reference_index=str(reference), reference_index_sha256=sha(reference),
        reference_contract_sha256=sha(reference.parent / 'contract.json'), dsrl_index=str(dsrl),
        dsrl_index_sha256=sha(dsrl), dsrl_contract_sha256=sha(dsrl.parent / 'contract.json')))
output.write_text(json.dumps(dict(format='openpi_motor_replay_inputs_v1', cases=entries), indent=2) + '\n')
PY
TASK_INPUTS_SHA=$(sha256sum "$TASK_OUT/inputs.json" | cut -d' ' -f1)
export OPENPI_DATA_HOME=$TASK_OLD/cache/openpi-215abfb
export PYTHONPATH=$TASK_CUDA:$TASK_OVERLAY:$TASK_WT/src:$TASK_DSRL:$TASK_OPENPI/src:$TASK_OPENPI/packages/openpi-client/src
timeout --signal=INT --kill-after=15s 540 "$TASK_MODEL" \
  -m fluffy_bassoon.meta_rl.vla_motor_replay \
  --inputs "$TASK_OUT/inputs.json" --inputs-sha256 "$TASK_INPUTS_SHA" \
  --output "$TASK_OUT/experiment" --checkpoint "$TASK_OLD/inputs/pi05-libero-openpi-20261027-r1" \
  --openpi-root "$TASK_OPENPI" --dsrl-root "$TASK_DSRL" \
  --source-commit "$TASK_SOURCE" --expected-hostname "$(hostname)" \
  --key-order "$TASK_ORDER" --max-elapsed-seconds 480
