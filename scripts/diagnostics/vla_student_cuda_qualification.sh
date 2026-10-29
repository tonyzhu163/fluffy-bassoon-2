#!/usr/bin/env bash
# Schedule one exclusive GPU for <=5 minutes; no shared environment is changed.
set -Eeuo pipefail
TASK_SOURCE=${1:?exact integrated source commit}
TASK_RUN=${2:?fresh student CUDA qualification output identity}
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_OLD=/mowenjie/fluffy-bassoon
TASK_WORKTREE=$TASK_ROOT/worktrees/$TASK_RUN
TASK_OUTPUT=$TASK_ROOT/outputs/$TASK_RUN
TASK_OPENPI=$TASK_OLD/third_party/openpi-215abfb
TASK_MODEL=$TASK_OLD/envs/openpi-215abfb/bin/python
TASK_PROBE=$TASK_OLD/envs/lerobot-v044-8fff0fde/bin/python
TASK_CUDA_PACKAGES=$TASK_OLD/envs/simulator/lib/python3.11/site-packages
TASK_DSRL_OVERLAY=$TASK_ROOT/envs/dsrl-pi0-overlay-20261029-r1
test ! -e "$TASK_OUTPUT"
mkdir -p "$TASK_OUTPUT"
cd "$TASK_WORKTREE"
test "$(git rev-parse HEAD)" = "$TASK_SOURCE"
test -z "$(git status --porcelain)"
exec > >(tee "$TASK_OUTPUT/console.log") 2>&1
trap 'TASK_RC=$?; printf "%s\n" "$TASK_RC" > "$TASK_OUTPUT/exit-code.txt"' EXIT
export FLUFFY_GPU_EXECUTION=baige CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 WANDB_MODE=disabled
export OPENPI_DATA_HOME=$TASK_OLD/cache/openpi-215abfb
export PYTHONPATH=$TASK_WORKTREE/src
"$TASK_PROBE" 'scripts/[train]gpu_probe.py' --expected-count 1 --purpose training \
  --baige-container --output "$TASK_OUTPUT/gpu-probe.json"
export LYCO_BAIGE_GPU_MAP=$TASK_OUTPUT/gpu-probe.json
TASK_CUDA_OVERLAY=$TASK_OUTPUT/cu128-overlay
"$TASK_PROBE" - "$TASK_CUDA_PACKAGES" "$TASK_CUDA_OVERLAY" "$TASK_OUTPUT/overlay.json" <<'PY'
import json, pathlib, sys
source, overlay, receipt = map(pathlib.Path, sys.argv[1:])
overlay.mkdir()
names = ['torch', 'torchvision', 'functorch', 'torchgen', 'triton', 'nvidia']
for pattern in ('torch-*.dist-info', 'torchvision-*.dist-info', 'triton-*.dist-info', 'nvidia_*.dist-info'):
    names.extend(p.name for p in source.glob(pattern))
links = []
for name in names:
    target = source / name
    assert target.exists(), target
    (overlay / name).symlink_to(target, target_is_directory=target.is_dir())
    links.append(dict(name=name, target=str(target)))
receipt.write_text(json.dumps(dict(source_python_abi='cp311', torch_version='2.7.1+cu128',
    source=str(source), overlay=str(overlay), links=links,
    method='run-local read-only symlinks; original environments unchanged'), indent=2) + '\n')
PY
TASK_BASE_PATH=$TASK_DSRL_OVERLAY:$TASK_WORKTREE/src:$TASK_OPENPI/src:$TASK_OPENPI/packages/openpi-client/src
if timeout --signal=TERM --kill-after=10s 110s env PYTHONPATH="$TASK_BASE_PATH" \
    "$TASK_MODEL" -m fluffy_bassoon.meta_rl.vla_student_cuda_diagnostic \
    --output "$TASK_OUTPUT/original.json" --source-commit "$TASK_SOURCE" --label original-openpi; then
  TASK_ORIGINAL_RC=0
else
  TASK_ORIGINAL_RC=$?
fi
# Always try the isolated cu128 overlay, including after an original-runtime failure.
if timeout --signal=TERM --kill-after=10s 110s env PYTHONPATH="$TASK_CUDA_OVERLAY:$TASK_BASE_PATH" \
    "$TASK_MODEL" -m fluffy_bassoon.meta_rl.vla_student_cuda_diagnostic \
    --output "$TASK_OUTPUT/cu128.json" --source-commit "$TASK_SOURCE" --label isolated-cu128 \
    --expected-torch-cuda 12.8 --expected-torch-version 2.7.1+cu128; then
  TASK_CU128_RC=0
else
  TASK_CU128_RC=$?
fi
"$TASK_PROBE" - "$TASK_OUTPUT/summary.json" "$TASK_ORIGINAL_RC" "$TASK_CU128_RC" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
path.write_text(json.dumps(dict(original_exit_code=int(sys.argv[2]), cu128_overlay_exit_code=int(sys.argv[3]),
    cu128_overlay_qualified=int(sys.argv[3]) == 0, gpu_count=1,
    scope='two real history-student CUDA updates bracketed by same-process JAX GPU convolutions'), indent=2) + '\n')
PY
exit "$TASK_CU128_RC"
