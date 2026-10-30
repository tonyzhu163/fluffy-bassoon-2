#!/usr/bin/env bash
# Existing funded1800-second allocation only; consumes a CPU-built manifest.
set -Eeuo pipefail
TASK_STARTED=$(date +%s)
TASK_SOURCE=${1:?exact reviewed commit}
TASK_RUN=${2:?fresh interim fitting identity}
TASK_MANIFEST=${3:?existing immutable CPU-produced topup-inputs.json}
TASK_MANIFEST_SHA=${4:?manifest file SHA256}
TASK_RESUME=${5:-}
case "$TASK_RESUME" in ''|--resume) ;; *) exit 2;; esac
TASK_CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_OLD=/mowenjie/fluffy-bassoon
TASK_OUT=$TASK_ROOT/outputs/$TASK_RUN
TASK_MODEL=$TASK_OLD/envs/openpi-215abfb/bin/python
TASK_PROBE=$TASK_OLD/envs/lerobot-v044-8fff0fde/bin/python
TASK_TEST=$TASK_ROOT/envs/pytest-overlay-20261029-r1
TASK_CUDA=$TASK_ROOT/outputs/libero-pi05-student-cuda-20261029-r1/cu128-overlay
cd "$TASK_CODE"
test "$(git rev-parse HEAD)" = "$TASK_SOURCE"
test -z "$(git status --porcelain)"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export PYTHONPATH=$TASK_CODE/src
TASK_ARGS=()
if [ "$TASK_RESUME" = --resume ]; then TASK_ARGS=(--resume); fi
TASK_CLOCK=$("$TASK_MODEL" -m fluffy_bassoon.meta_rl.vla_source40_topup_interim_fit begin \
  --root "$TASK_OUT" --manifest "$TASK_MANIFEST" --manifest-sha256 "$TASK_MANIFEST_SHA" \
  --source-commit "$TASK_SOURCE" --run "$TASK_RUN" --started "$TASK_STARTED" "${TASK_ARGS[@]}")
TASK_ATTEMPT=$("$TASK_MODEL" -c 'import json,sys; print(json.loads(sys.argv[1])["attempt"])' "$TASK_CLOCK")
TASK_SLICE=$("$TASK_MODEL" -c 'import json,sys; print(json.loads(sys.argv[1])["slice_seconds"])' "$TASK_CLOCK")
exec > >(tee "$TASK_ATTEMPT/console.log") 2>&1
task_cleanup() {
  TASK_RC=$?
  trap - EXIT
  PYTHONPATH=$TASK_CODE/src "$TASK_MODEL" -m fluffy_bassoon.meta_rl.vla_source40_ppo_worker finish \
    --root "$TASK_OUT" --attempt "$TASK_ATTEMPT" --exit-code "$TASK_RC" || true
  printf '%s\n' "$TASK_RC" > "$TASK_OUT/exit-code.txt"
  exit "$TASK_RC"
}
trap task_cleanup EXIT
exec 9> "$TASK_OUT/launcher.lock"
flock -n 9
export FLUFFY_GPU_EXECUTION=baige OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
"$TASK_PROBE" 'scripts/[train]gpu_probe.py' --expected-count 1 --purpose training \
  --baige-container --baige-exclusive-training --output "$TASK_ATTEMPT/gpu-probe.json"
export LYCO_BAIGE_GPU_MAP=$TASK_ATTEMPT/gpu-probe.json
# Only the new command/profile/clock contract tests; the CPU assembly gate stays separate.
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$TASK_TEST:$TASK_CODE/src \
  "$TASK_PROBE" -m pytest -q -p no:cacheprovider tests/test_vla_source40_topup_interim_fit.py \
  > "$TASK_ATTEMPT/focused-tests.log" 2>&1
export PYTHONPATH=$TASK_CUDA:$TASK_CODE/src
"$TASK_MODEL" - "$TASK_ATTEMPT/student-runtime.json" <<'PYRUNTIME'
import json, pathlib, sys
assert sys.version_info[:2] == (3,11), sys.version
import torch
assert torch.__version__ == '2.7.1+cu128' and torch.version.cuda == '12.8'
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
value = (torch.arange(16,device='cuda',dtype=torch.float32).square().sum()).item()
assert value == 1240
pathlib.Path(sys.argv[1]).write_text(json.dumps(dict(python=sys.version,executable=sys.executable,
    torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(0),
    cuda_test_sum=value),indent=2)+'\n')
PYRUNTIME
TASK_TRAIN_RESUME=()
if [ "$TASK_RESUME" = --resume ] && [ -f "$TASK_OUT/experiment/contract.json" ]; then TASK_TRAIN_RESUME=(--resume); fi
TASK_REMAINING=$((TASK_STARTED + TASK_SLICE - $(date +%s) - 30))
test "$TASK_REMAINING" -gt 120
# External deadline leaves30seconds for shutdown/clock receipt; setup/retries count.
timeout --signal=INT --kill-after=20s "${TASK_REMAINING}s" "$TASK_MODEL" \
  -m fluffy_bassoon.meta_rl.vla_source40_topup_interim_fit train \
  --root "$TASK_OUT" --manifest "$TASK_MANIFEST" --manifest-sha256 "$TASK_MANIFEST_SHA" \
  --source-commit "$TASK_SOURCE" --run "$TASK_RUN" --expected-hostname "$(hostname)" \
  --receipt "$TASK_ATTEMPT/fitting.json" "${TASK_TRAIN_RESUME[@]}"
