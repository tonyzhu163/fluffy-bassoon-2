#!/usr/bin/env bash
set -Eeuo pipefail
TASK_SOURCE=${1:?reviewed project commit}
TASK_OUT=${2:?fresh output directory}
TASK_CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
TASK_RUN=$TASK_CODE/outputs/robotwin-sidney-initial-performance-20261106-r1
TASK_PYTHON=/mowenjie/fluffy-bassoon-2/envs/robotwin-vlrla-20261029-r1/bin/python
[[ $(uname -s) == Linux && $(hostname) == job-* ]]
[[ $(git -C "$TASK_CODE" rev-parse HEAD) == "$TASK_SOURCE" ]]
[[ -z $(git -C "$TASK_CODE" status --porcelain --untracked-files=no) ]]
[[ ! -e $TASK_OUT ]]
mkdir -p "$TASK_OUT"
exec > "$TASK_OUT/worker.log" 2>&1
export TASK_CODE TASK_SOURCE
export REPO_PATH=/mowenjie/fluffy-bassoon-2/sources/robotwin-baseline-smoke-20261028-r1/RLinf
export ROBOTWIN_PATH=/mowenjie/fluffy-bassoon-2/sources/robotwin-baseline-smoke-20261028-r1/RoboTwin
export ASSETS_PATH=$ROBOTWIN_PATH ROBOTWIN_ASSETS_PATH=$ROBOTWIN_PATH
export EMBODIED_PATH=$REPO_PATH/examples/embodiment
export PYTHONPATH=$TASK_CODE/src:$REPO_PATH:$ROBOTWIN_PATH
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export ROBOT_PLATFORM=ALOHA MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export FLUFFY_GPU_EXECUTION=baige NCCL_P2P_DISABLE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
if [[ ${CUDA_VISIBLE_DEVICES+x} != x ]]; then export CUDA_VISIBLE_DEVICES=0; fi
TASK_STARTED=$(date +%s)
export TASK_DEADLINE=$((TASK_STARTED+14220))
TASK_MEMORY_PID=
finish() {
  TASK_RC=$?
  trap - EXIT
  if [[ -n $TASK_MEMORY_PID ]]; then kill "$TASK_MEMORY_PID" 2>/dev/null || true; fi
  TASK_ENDED=$(date +%s)
  printf '{"exit_code":%s,"started_unix":%s,"ended_unix":%s,"worker_gpu_seconds":%s}\n' \
    "$TASK_RC" "$TASK_STARTED" "$TASK_ENDED" "$((TASK_ENDED-TASK_STARTED))" > "$TASK_OUT/EXIT.json"
}
trap finish EXIT
"$TASK_PYTHON" "$TASK_CODE/scripts/[train]gpu_probe.py" --baige-container \
  --purpose evaluation --expected-count 1 --output "$TASK_OUT/gpu-probe.json"
export LYCO_BAIGE_GPU_MAP=$TASK_OUT/gpu-probe.json
"$TASK_PYTHON" "$TASK_CODE/outputs/robotwin-sidney-rlinf-ppo-20261105-r1/host_memory.py" \
  "$TASK_OUT" --minimum-gib 128 --once
"$TASK_PYTHON" "$TASK_CODE/outputs/robotwin-sidney-rlinf-ppo-20261105-r1/host_memory.py" \
  "$TASK_OUT" --minimum-gib 128 > "$TASK_OUT/host-memory.log" 2>&1 &
TASK_MEMORY_PID=$!
"$TASK_PYTHON" - "$TASK_RUN" "$TASK_OUT" <<'PY'
import hashlib,json,os,shutil,sys
from pathlib import Path
run,out=map(Path,sys.argv[1:]);plan=json.loads((run/'CONTRACT.json').read_text())
assert plan['prior_conservative_gpu_seconds']+plan['reserved_gpu_seconds']<=50*3600
assert plan['reserved_gpu_seconds']==14460 and plan['optimizer_updates']==0
assert hashlib.sha256((run/'CASES.json').read_bytes()).hexdigest()==plan['case_manifest_sha256']
free=shutil.disk_usage(out).free
assert free>=(286+12)*2**30
probe=out/'write-probe'
with probe.open('xb') as handle:
    handle.write(b'robotwin-frozen-initial-policy\n');handle.flush();os.fsync(handle.fileno())
probe.unlink()
(out/'admission.json').write_text(json.dumps(dict(status='complete',free_bytes=free,
    required_free_bytes=298*2**30,cumulative_worst_gpu_seconds=130567+14460,
    gpu_count=1,optimizer_updates=0,write_fsync=True))+'\n')
PY
mkdir -p /root/.sapien/physx
if [[ ! -e /root/.sapien/physx/105.1-physx-5.3.1.patch0 ]]; then
  ln -s /mowenjie/fluffy-bassoon/inputs/maniskill-scene-assets-20261028-r1/physx/105.1-physx-5.3.1.patch0 \
    /root/.sapien/physx/105.1-physx-5.3.1.patch0
fi
timeout --signal=INT --kill-after=30s 14040 "$TASK_PYTHON" "$TASK_RUN/evaluate_initial.py" \
  --output "$TASK_OUT" --manifest "$TASK_RUN/CASES.json"
