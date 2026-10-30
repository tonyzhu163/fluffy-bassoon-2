#!/usr/bin/env bash
# Existing allocated GPU only. This does not submit or authorize an allocation.
set -Eeuo pipefail
TASK_SOURCE=${1:?exact reviewed worker commit}
TASK_WORKER=${2:?worker index0..7}
TASK_RECEIPT=${3:?fresh absolute worker receipt directory}
TASK_WALL=${4:-7140}
TASK_PREFIX=${5:-}
TASK_STARTED=$(date +%s)
TASK_HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_PROBE=/mowenjie/fluffy-bassoon/envs/lerobot-v044-8fff0fde/bin/python
TASK_MODEL=/mowenjie/fluffy-bassoon/envs/openpi-215abfb/bin/python
TASK_PREFLIGHT=${TASK_RECEIPT}.preflight
test "$TASK_WALL" -le 7140
test "$TASK_WALL" -ge 60
test ! -e "$TASK_RECEIPT"
test ! -e "$TASK_PREFLIGHT"
mkdir -p "$TASK_PREFLIGHT"
task_remaining() {
  TASK_REMAINING=$((TASK_WALL - $(date +%s) + TASK_STARTED - 30))
  test "$TASK_REMAINING" -ge 60
}
test "$(git -C "$TASK_HERE" rev-parse HEAD)" = "$TASK_SOURCE"
test -z "$(git -C "$TASK_HERE" status --porcelain)"
cd "$TASK_HERE"
export FLUFFY_GPU_EXECUTION=baige CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTHONPATH=$TASK_HERE/src
task_remaining
timeout --signal=TERM --kill-after=10s "$TASK_REMAINING" "$TASK_PROBE" 'scripts/[train]gpu_probe.py' \
  --expected-count 1 --purpose training --baige-container --baige-exclusive-training \
  --output "$TASK_PREFLIGHT/gpu-probe.json"
export LYCO_BAIGE_GPU_MAP=$TASK_PREFLIGHT/gpu-probe.json
task_remaining
CUDA_VISIBLE_DEVICES='' PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  PYTHONPATH=$TASK_ROOT/envs/pytest-overlay-20261029-r1:$TASK_HERE/src \
  timeout --signal=TERM --kill-after=10s "$TASK_REMAINING" "$TASK_PROBE" -m pytest -q -p no:cacheprovider \
  tests/test_vla*.py tests/test_gpu_probe.py > "$TASK_PREFLIGHT/focused-tests.log" 2>&1
"$TASK_MODEL" - "$TASK_PREFLIGHT" "$TASK_SOURCE" "$TASK_WORKER" "$TASK_STARTED" "$TASK_WALL" <<'PY'
import hashlib, json, pathlib, socket, sys, time
root, source, worker, started, wall = sys.argv[1:]
root = pathlib.Path(root)
probe, tests = root/'gpu-probe.json', root/'focused-tests.log'
value = dict(source_commit=source, worker=int(worker), hostname=socket.gethostname(),
    allocation_started_unix=int(started), maximum_outer_seconds=int(wall), completed_unix=time.time(),
    probe_path=str(probe), probe_sha256=hashlib.sha256(probe.read_bytes()).hexdigest(),
    tests_path=str(tests), tests_sha256=hashlib.sha256(tests.read_bytes()).hexdigest(), tests_exit_code=0)
(root/'receipt.json').write_text(json.dumps(value, indent=2)+'\n')
PY
task_remaining
TASK_PREFIX_ARGS=()
if [ -n "$TASK_PREFIX" ]; then TASK_PREFIX_ARGS=(--fresh-run-prefix "$TASK_PREFIX"); fi
exec "$TASK_MODEL" \
  -m fluffy_bassoon.meta_rl.vla_source40_campaign_worker run \
  --source-commit "$TASK_SOURCE" --worker "$TASK_WORKER" --workers 8 \
  --expected-hostname "$(hostname)" --output "$TASK_RECEIPT" \
  --max-elapsed-seconds "$TASK_REMAINING" --child-timeout-seconds 2400 \
  --preflight-receipt "$TASK_PREFLIGHT/receipt.json" "${TASK_PREFIX_ARGS[@]}"
