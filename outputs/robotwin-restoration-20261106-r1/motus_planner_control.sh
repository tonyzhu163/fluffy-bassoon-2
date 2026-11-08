#!/usr/bin/env bash
set -Eeuo pipefail
export TASK_SOURCE=${1:?reviewed commit}
export TASK_CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
TASK_DRIVER=$TASK_CODE/outputs/robotwin-restoration-20261106-r1
TASK_PLAN_NAME=${2:?fresh admitted contract basename}
[[ $TASK_PLAN_NAME == MOTUS-PLANNER-CONTROL01-CONTRACT.json ]]
TASK_PLAN=$TASK_DRIVER/$TASK_PLAN_NAME
TASK_PYTHON=/mowenjie/fluffy-bassoon-2/envs/robotwin-vlrla-20261029-r1/bin/python
TASK_BASE=/mnt/bos/adgfyl/strong-plant-archives/robotwin-restoration-20261106-r1
[[ $(uname -s) == Linux && $(hostname) == job-* ]]
[[ $(git -C "$TASK_CODE" rev-parse HEAD) == "$TASK_SOURCE" ]]
[[ -z $(git -C "$TASK_CODE" status --porcelain --untracked-files=no) ]]
"$TASK_PYTHON" - "$TASK_PLAN" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]))
assert p['status']=='admitted' and p['cpu_preflight_passed']
assert p['reserved_gpu_seconds']==600 and p['worker_timeout_seconds']==470
assert p['all_worst_gpu_seconds']==p['closed_conservative_gpu_seconds']+p['other_active_reserved_gpu_seconds']+p['reserved_gpu_seconds']<=p['approved_cap_gpu_seconds']==360000
assert not p['budget_extension_approved']
PY
export TASK_OUT=$TASK_BASE/motus-planner-control01
TASK_SIGNALS=$TASK_BASE/motus-planner-control01-signals
export TASK_LOCAL=/tmp/rt-planner-control01
[[ ! -e $TASK_OUT && ! -e $TASK_SIGNALS && ! -e $TASK_LOCAL ]]
mkdir "$TASK_SIGNALS"
exec > "$TASK_SIGNALS/worker.log" 2>&1
TASK_STARTED=$(date +%s)
finish() {
 TASK_RC=$?
 trap - EXIT
 TASK_ENDED=$(date +%s)
 printf '{"exit_code":%s,"started_unix":%s,"ended_unix":%s,"worker_gpu_seconds":%s}\n' \
  "$TASK_RC" "$TASK_STARTED" "$TASK_ENDED" "$((TASK_ENDED-TASK_STARTED))" > "$TASK_SIGNALS/EXIT.json"
}
trap finish EXIT
export TASK_LIMIT=470 TASK_DEADLINE=$((TASK_STARTED+470))
TASK_RUN=$TASK_CODE/outputs/robotwin-sidney-matched-baselines-20261106-r1
TASK_SHARED=$TASK_CODE/outputs/robotwin-sidney-rlinf-ppo-20261105-r1
# The shared source defaults to four IDs when the scheduler leaves CVD unset.
# Its single allocated container device is logical0; preserve any explicit CVD.
if [[ ${CUDA_VISIBLE_DEVICES+x} != x ]]; then export CUDA_VISIBLE_DEVICES=0; fi
source "$TASK_RUN/runtime_environment.sh"
export PYTHONPATH=$TASK_DRIVER:$PYTHONPATH
"$TASK_PYTHON" "$TASK_CODE/scripts/[train]gpu_probe.py" --baige-container --purpose evaluation --expected-count 1 --output "$TASK_SIGNALS/gpu-probe.json"
export LYCO_BAIGE_GPU_MAP=$TASK_SIGNALS/gpu-probe.json
"$TASK_PYTHON" "$TASK_SHARED/host_memory.py" "$TASK_SIGNALS" --minimum-gib 128 --once
"$TASK_PYTHON" - "$TASK_PLAN" <<'PY'
import hashlib,json,os,shutil,sys
from pathlib import Path
p=json.load(open(sys.argv[1]));code=Path(os.environ['TASK_CODE'])
digest=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
for name,value in p['source_files'].items():assert digest(code/name)==value
assert digest(p['CPU_result'])==p['CPU_sha256']
assert digest(p['original_configuration'])==p['original_configuration_sha256']
local=Path(os.environ['TASK_LOCAL']);out=Path(os.environ['TASK_OUT'])
assert shutil.disk_usage(local.parent).free>=8*2**30
assert any(os.path.ismount(x) for x in (out.parent,Path('/mnt/bos/adgfyl'),Path('/mnt/bos')))
probe=out.parent/'motus-planner-control01-write-probe'
with probe.open('xb') as f:f.write(b'robotwin-fresh-scenes');f.flush();os.fsync(f.fileno())
assert probe.read_bytes()==b'robotwin-fresh-scenes';probe.unlink()
local.mkdir();(local/'assets').symlink_to(Path(os.environ['ROBOTWIN_PATH'])/'assets',target_is_directory=True)
font=Path(os.environ['ROBOTWIN_PATH'])/'Times New Roman.ttf'
if font.is_file():(local/font.name).symlink_to(font)
PY
export XDG_CACHE_HOME=$TASK_LOCAL/cache TORCH_HOME=$TASK_LOCAL/torch TMPDIR=$TASK_LOCAL/tmp
mkdir -p "$XDG_CACHE_HOME" "$TORCH_HOME" "$TMPDIR" /root/.sapien/physx
if [[ ! -e /root/.sapien/physx/105.1-physx-5.3.1.patch0 ]]; then
 ln -s /mowenjie/fluffy-bassoon/inputs/maniskill-scene-assets-20261028-r1/physx/105.1-physx-5.3.1.patch0 /root/.sapien/physx/105.1-physx-5.3.1.patch0
fi
cd "$TASK_LOCAL"
TASK_REMAINING=$((TASK_DEADLINE-$(date +%s)-90))
[[ $TASK_REMAINING -ge 180 ]]
timeout --signal=TERM --kill-after=20s "$TASK_REMAINING" "$TASK_PYTHON" "$TASK_DRIVER/motus_planner_control.py" > "$TASK_SIGNALS/qualify.log" 2>&1
[[ $(du -sb "$TASK_OUT" "$TASK_SIGNALS" | awk '{s+=$1} END {print s}') -le 67108864 ]]

