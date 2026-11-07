#!/usr/bin/env bash
set -Eeuo pipefail
TASK_SOURCE=${1:?reviewed commit}
TASK_CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
TASK_DRIVER=$TASK_CODE/outputs/robotwin-restoration-20261106-r1
TASK_OUT=/mnt/bos/adgfyl/strong-plant-archives/robotwin-restoration-20261106-r1/studentprecisiongpu03-signals
TASK_PYTHON=/mowenjie/fluffy-bassoon-2/envs/robotwin-vlrla-20261029-r1/bin/python
export TASK_CODE TASK_SOURCE TASK_DRIVER TASK_OUT GIT_OPTIONAL_LOCKS=0
export TASK_PRECISION_CONTRACT=STUDENTPRECISIONGPU03-CONTRACT.json
[[ $(hostname) == job-* && $(git -C "$TASK_CODE" rev-parse HEAD) == "$TASK_SOURCE" ]]
[[ -z $(git -C "$TASK_CODE" status --porcelain --untracked-files=no) && ! -e $TASK_OUT ]]
mkdir "$TASK_OUT"
exec > "$TASK_OUT/worker.log" 2>&1
TASK_STARTED=$(date +%s)
export TASK_DEADLINE=$((TASK_STARTED+120))
finish() {
 TASK_RC=$?
 trap - EXIT
 TASK_ENDED=$(date +%s)
 printf '{"exit_code":%s,"worker_gpu_seconds":%s}\n' "$TASK_RC" "$((TASK_ENDED-TASK_STARTED))" > "$TASK_OUT/EXIT.json"
}
trap finish EXIT
TASK_RUN=$TASK_CODE/outputs/robotwin-sidney-matched-baselines-20261106-r1
TASK_SHARED=$TASK_CODE/outputs/robotwin-sidney-rlinf-ppo-20261105-r1
if [[ ${CUDA_VISIBLE_DEVICES+x} != x ]]; then export CUDA_VISIBLE_DEVICES=0; fi
source "$TASK_RUN/runtime_environment.sh"
export PYTHONPATH=$TASK_DRIVER:$PYTHONPATH
"$TASK_PYTHON" "$TASK_CODE/scripts/[train]gpu_probe.py" --baige-container --purpose evaluation \
 --expected-count 1 --output "$TASK_OUT/gpu-probe.json"
"$TASK_PYTHON" "$TASK_SHARED/host_memory.py" "$TASK_OUT" --minimum-gib 128 --once
"$TASK_PYTHON" - "$TASK_DRIVER" <<'PY'
import json,sys
from pathlib import Path
from fluffy_bassoon.meta_rl.rlinf_native_ppo import sha
r=Path(sys.argv[1]);p=json.loads((r/'STUDENTPRECISIONGPU03-CONTRACT.json').read_bytes())
assert p['status']=='admitted' and p['all_worst_gpu_seconds']<=p['hard_cap_gpu_seconds']==219600
assert p['closed_gpu_seconds']+p['ppo_reserved_gpu_seconds']+p['reserved_gpu_seconds']==p['all_worst_gpu_seconds']
for name,h in p['source_files'].items():assert sha(r.parents[1]/name)==h
assert sha(r/'qualify_student_precision_gpu.py')==p['entry_sha256']
assert sha(r/'qualify_student_precision_gpu03.sh')==p['launcher_sha256']
probe=json.loads((Path(__import__('os').environ['TASK_OUT'])/'gpu-probe.json').read_bytes())
assert len(probe['devices'])==1 and probe['devices'][0]['cuda_name']=='NVIDIA RPBZZZ6'
assert probe['devices'][0]['cuda_total_bytes']>=90*2**30
PY
TASK_REMAINING=$((TASK_DEADLINE-$(date +%s)-10))
[[ $TASK_REMAINING -gt 60 ]]
timeout --signal=INT --kill-after=10s "$TASK_REMAINING" "$TASK_PYTHON" \
 "$TASK_DRIVER/qualify_student_precision_gpu.py" > "$TASK_OUT/calibration.log" 2>&1
