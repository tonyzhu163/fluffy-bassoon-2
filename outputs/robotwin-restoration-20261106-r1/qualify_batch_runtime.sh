#!/usr/bin/env bash
set -Eeuo pipefail
TASK_SOURCE=${1:?reviewed commit}
TASK_OUT=${2:?fresh output identity}
TASK_CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
TASK_PYTHON=/mowenjie/fluffy-bassoon-2/envs/robotwin-vlrla-20261029-r1/bin/python
[[ $(uname -s) == Linux && $(hostname) == job-* ]]
[[ $(git -C "$TASK_CODE" rev-parse HEAD) == "$TASK_SOURCE" ]]
[[ -z $(git -C "$TASK_CODE" status --porcelain --untracked-files=no) && ! -e $TASK_OUT ]]
mkdir -p "$TASK_OUT"
exec > "$TASK_OUT/worker.log" 2>&1
TASK_RUN=$TASK_CODE/outputs/robotwin-sidney-matched-baselines-20261106-r1
TASK_SHARED=$TASK_CODE/outputs/robotwin-sidney-rlinf-ppo-20261105-r1
TASK_LIMIT=1740
if [[ ${CUDA_VISIBLE_DEVICES+x} != x ]]; then export CUDA_VISIBLE_DEVICES=0; fi
source "$TASK_RUN/runtime_environment.sh"
TASK_STARTED=$(date +%s)
export TASK_DEADLINE=$((TASK_STARTED+1740))
finish() {
  TASK_RC=$?
  trap - EXIT
  TASK_ENDED=$(date +%s)
  printf '{"exit_code":%s,"started_unix":%s,"ended_unix":%s,"worker_gpu_seconds":%s}\n' \
    "$TASK_RC" "$TASK_STARTED" "$TASK_ENDED" "$((TASK_ENDED-TASK_STARTED))" > "$TASK_OUT/EXIT.json"
}
trap finish EXIT
"$TASK_PYTHON" "$TASK_CODE/scripts/[train]gpu_probe.py" --baige-container \
  --purpose evaluation --expected-count 1 --output "$TASK_OUT/gpu-probe.json"
export LYCO_BAIGE_GPU_MAP=$TASK_OUT/gpu-probe.json
"$TASK_PYTHON" "$TASK_SHARED/host_memory.py" "$TASK_OUT" --minimum-gib 128 --once
"$TASK_PYTHON" - "$TASK_CODE" "$TASK_OUT" <<'PY'
import json,os,shutil,sys
from pathlib import Path
code,out=map(Path,sys.argv[1:]);p=json.loads((code/'outputs/robotwin-restoration-20261106-r1/BATCH01-CONTRACT.json').read_text())
assert p['closed_conservative_gpu_seconds']+p['reserved_gpu_seconds']==p['all_worst_gpu_seconds']<=180000
assert p['minimum_free_gib']==p['other_and_floor_gib']+p['maximum_output_gib']==286.5
free=shutil.disk_usage(out).free/2**30
assert free >= p['minimum_free_gib']
probe=out/'write-probe'
with probe.open('xb') as stream:stream.write(b'robotwin-process-runtime\n');stream.flush();os.fsync(stream.fileno())
probe.unlink()
(out/'admission.json').write_text(json.dumps(dict(free_gib=free,minimum_free_gib=p['minimum_free_gib'],
    write_fsync=True,worst_gpu_seconds=p['all_worst_gpu_seconds']))+'\n')
PY
mkdir -p /root/.sapien/physx
if [[ ! -e /root/.sapien/physx/105.1-physx-5.3.1.patch0 ]]; then
  ln -s /mowenjie/fluffy-bassoon/inputs/maniskill-scene-assets-20261028-r1/physx/105.1-physx-5.3.1.patch0 \
    /root/.sapien/physx/105.1-physx-5.3.1.patch0
fi
timeout --signal=INT --kill-after=30s 1680 "$TASK_PYTHON" \
  "$TASK_CODE/outputs/robotwin-restoration-20261106-r1/qualify_batch_runtime.py" "$TASK_OUT"
