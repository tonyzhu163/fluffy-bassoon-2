#!/usr/bin/env bash
set -Eeuo pipefail
TASK_SOURCE=${1:?reviewed commit}
TASK_CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
TASK_DRIVER=$TASK_CODE/outputs/robotwin-restoration-20261106-r1
TASK_ARCHIVE=/mnt/bos/adgfyl/strong-plant-archives/robotwin-restoration-20261106-r1
TASK_OUT=$TASK_ARCHIVE/motuscamera01-signals
TASK_LOCAL=/tmp/rtcamera1
TASK_PYTHON=/mowenjie/fluffy-bassoon-2/envs/robotwin-vlrla-20261029-r1/bin/python
export TASK_SOURCE TASK_CODE TASK_DRIVER TASK_OUT TASK_LOCAL GIT_OPTIONAL_LOCKS=0
[[ $(uname -s) == Linux && $(hostname) == job-* ]]
[[ $(git -C "$TASK_CODE" rev-parse HEAD) == "$TASK_SOURCE" ]]
[[ -z $(git -C "$TASK_CODE" status --porcelain --untracked-files=no) ]]
[[ ! -e $TASK_OUT && ! -e $TASK_LOCAL ]]
"$TASK_PYTHON" - "$TASK_DRIVER/MOTUSCAMERA01-CONTRACT.json" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]))
assert p['status']=='admitted' and p['cpu_preflight_passed']
assert p['closed_conservative_gpu_seconds']+p['reserved_gpu_seconds']==p['all_worst_gpu_seconds']<=p['hard_cap_gpu_seconds']==360000
assert p['reserved_gpu_seconds']==1800 and p['worker_seconds']==1500
PY
mkdir "$TASK_OUT" "$TASK_LOCAL"
exec > "$TASK_OUT/worker.log" 2>&1
TASK_STARTED=$(date +%s)
export TASK_DEADLINE=$((TASK_STARTED+1500))
export TASK_RESOURCE_LIVE=$TASK_LOCAL/resources-live.json
TASK_RUN=$TASK_CODE/outputs/robotwin-sidney-matched-baselines-20261106-r1
TASK_SHARED=$TASK_CODE/outputs/robotwin-sidney-rlinf-ppo-20261105-r1
TASK_LIMIT=1500
if [[ ${CUDA_VISIBLE_DEVICES+x} != x ]]; then export CUDA_VISIBLE_DEVICES=0; fi
source "$TASK_RUN/runtime_environment.sh"
export PYTHONPATH=$TASK_DRIVER:$PYTHONPATH
TASK_MONITOR_PID=
finish() {
 TASK_RC=$?
 trap - EXIT
 if [[ -n $TASK_MONITOR_PID ]]; then
  kill "$TASK_MONITOR_PID" 2>/dev/null || true
  wait "$TASK_MONITOR_PID" 2>/dev/null || true
 fi
 TASK_ENDED=$(date +%s)
 printf '{"exit_code":%s,"started_unix":%s,"ended_unix":%s,"worker_gpu_seconds":%s}\n' \
  "$TASK_RC" "$TASK_STARTED" "$TASK_ENDED" "$((TASK_ENDED-TASK_STARTED))" > "$TASK_OUT/EXIT.json"
}
trap finish EXIT
"$TASK_PYTHON" "$TASK_CODE/scripts/[train]gpu_probe.py" --baige-container --purpose evaluation \
 --expected-count 1 --output "$TASK_OUT/gpu-probe.json"
export LYCO_BAIGE_GPU_MAP=$TASK_OUT/gpu-probe.json
"$TASK_PYTHON" "$TASK_SHARED/host_memory.py" "$TASK_OUT" --minimum-gib 128 --once
"$TASK_PYTHON" - "$TASK_DRIVER" <<'PY'
import json,os,shutil,sys
from pathlib import Path
from fluffy_bassoon.meta_rl.rlinf_native_ppo import sha,write
root=Path(sys.argv[1]);p=json.loads((root/'MOTUSCAMERA01-CONTRACT.json').read_bytes())
for name,digest in p['source_files'].items():assert sha(root.parents[1]/name)==digest
assert sha(root/'probe_motus_camera.py')==p['entry_sha256']
assert sha(root/'probe_motus_camera.sh')==p['launcher_sha256']
assert sha(Path(p['cpu_preflight_result']))==p['cpu_preflight_sha256']
probe=json.loads((Path(os.environ['TASK_OUT'])/'gpu-probe.json').read_bytes())
assert probe['status']=='passed' and len(probe['devices'])==1
assert probe['devices'][0]['cuda_name']==p['qualified_cuda_name']
assert probe['devices'][0]['cuda_total_bytes']>=90*2**30
assert shutil.disk_usage(os.environ['TASK_LOCAL']).free>=16*2**30
path=Path(os.environ['TASK_OUT'])/'write-probe'
with path.open('xb') as f:f.write(b'paired-camera-horizon-probe');f.flush();os.fsync(f.fileno())
assert path.read_bytes()==b'paired-camera-horizon-probe';path.unlink()
write(Path(os.environ['TASK_OUT'])/'admission.json',dict(source_commit=os.environ['TASK_SOURCE'],
 write_fsync=True,new_worker_pfs_output_bytes=0,reserved_gpu_seconds=1800,optimizer_updates=0))
PY
export XDG_CACHE_HOME=$TASK_LOCAL/cache OPENPI_DATA_HOME=$TASK_LOCAL/openpi
export HF_HOME=$TASK_LOCAL/hf TORCH_HOME=$TASK_LOCAL/torch TMPDIR=$TASK_LOCAL/tmp
mkdir -p "$XDG_CACHE_HOME" "$OPENPI_DATA_HOME" "$HF_HOME" "$TORCH_HOME" "$TMPDIR" /root/.sapien/physx
if [[ ! -e /root/.sapien/physx/105.1-physx-5.3.1.patch0 ]]; then
 ln -s /mowenjie/fluffy-bassoon/inputs/maniskill-scene-assets-20261028-r1/physx/105.1-physx-5.3.1.patch0 \
 /root/.sapien/physx/105.1-physx-5.3.1.patch0
fi
"$TASK_PYTHON" "$TASK_DRIVER/observe_motus_resources.py" "$TASK_OUT" 1 > "$TASK_OUT/resources.log" 2>&1 &
TASK_MONITOR_PID=$!
TASK_REMAINING=$((TASK_DEADLINE-$(date +%s)-60))
[[ $TASK_REMAINING -gt 180 ]]
timeout --signal=INT --kill-after=20s "$TASK_REMAINING" "$TASK_PYTHON" \
 "$TASK_DRIVER/probe_motus_camera.py" > "$TASK_OUT/evaluation.log" 2>&1
