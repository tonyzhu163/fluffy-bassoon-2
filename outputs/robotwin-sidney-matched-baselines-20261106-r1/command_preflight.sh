#!/usr/bin/env bash
set -Eeuo pipefail
TASK_SOURCE=${1:?commit}
TASK_OUT=${2:?fresh output}
TASK_CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
TASK_RUN=$TASK_CODE/outputs/robotwin-sidney-matched-baselines-20261106-r1
TASK_PYTHON=/mowenjie/fluffy-bassoon-2/envs/robotwin-vlrla-20261029-r1/bin/python
[[ $(uname -s) == Linux && $(hostname) == job-* ]]
[[ $(git -C "$TASK_CODE" rev-parse HEAD) == "$TASK_SOURCE" && ! -e $TASK_OUT ]]
mkdir "$TASK_OUT"
exec > "$TASK_OUT/worker.log" 2>&1
TASK_STARTED=$(date +%s)
finish() {
  TASK_RC=$?
  trap - EXIT
  printf '{"exit_code":%s,"started_unix":%s,"ended_unix":%s,"worker_gpu_seconds":%s}\n' \
    "$TASK_RC" "$TASK_STARTED" "$(date +%s)" "$(($(date +%s)-TASK_STARTED))" > "$TASK_OUT/EXIT.json"
}
trap finish EXIT
export REPO_PATH=/mowenjie/fluffy-bassoon-2/sources/robotwin-baseline-smoke-20261028-r1/RLinf
export ROBOTWIN_PATH=/mowenjie/fluffy-bassoon-2/sources/robotwin-baseline-smoke-20261028-r1/RoboTwin
export PYTHONPATH=$TASK_CODE/src:$REPO_PATH:$ROBOTWIN_PATH:$TASK_RUN:$TASK_CODE/outputs/robotwin-sidney-rlinf-ppo-20261105-r1
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4
export FLUFFY_GPU_EXECUTION=baige NCCL_P2P_DISABLE=1
"$TASK_PYTHON" - "$TASK_RUN/COMMAND-PREFLIGHT-ADMISSION.json" <<'PY'
import json,sys,shutil
p=json.load(open(sys.argv[1]));assert p['status']=='admitted'
assert p['completed_additional_conservative_gpu_seconds']+p['legacy_gpu_seconds']+p['preflight_maximum_gpu_seconds']+p['prospective_pair_gpu_seconds']<=p['human_cumulative_gpu_hours']*3600
assert shutil.disk_usage('/mowenjie/fluffy-bassoon-2').free>=p['required_forward_free_gib']*2**30
PY
if [[ ${CUDA_VISIBLE_DEVICES+x} != x ]]; then export CUDA_VISIBLE_DEVICES=0; fi
# Mandatory outer allocation health check; the four command checks below are CPU-only.
"$TASK_PYTHON" "$TASK_CODE/scripts/[train]gpu_probe.py" --baige-container \
  --baige-exclusive-training --purpose training --expected-count 1 --output "$TASK_OUT/gpu-probe.json"
export CUDA_VISIBLE_DEVICES=
export NATIVE_PPO_WORLD=4 NATIVE_PPO_PLACEMENT=0-3 NATIVE_PPO_MODE=train
export NATIVE_PPO_OUTPUT=$TASK_OUT NATIVE_PPO_ACCOUNTING_ROOT=$TASK_OUT/action-accounting
export NATIVE_PPO_DEADLINE=$((TASK_STARTED+180))
export NATIVE_PPO_MODEL=/mowenjie/fluffy-bassoon-2/cache/robotwin-sidney-rlinf-20261105-r1
export NATIVE_PPO_INITIALIZER_REVISION=e49e2ab6c11f07511573b67261bd129e88d0a416
timeout --signal=INT --kill-after=10s 150 "$TASK_PYTHON" "$TASK_RUN/configuration.py" validate_commands > "$TASK_OUT/RESULT.json"
