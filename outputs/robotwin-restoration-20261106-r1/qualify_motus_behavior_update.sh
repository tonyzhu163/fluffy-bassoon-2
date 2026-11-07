#!/usr/bin/env bash
set -Eeuo pipefail
TASK_SOURCE=${1:?reviewed source commit}
TASK_OUT=${2:?fresh project BOS signals identity}
export TASK_SOURCE TASK_OUT GIT_OPTIONAL_LOCKS=0
TASK_CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
TASK_ARCHIVE=/mnt/bos/adgfyl/strong-plant-archives/robotwin-restoration-20261106-r1
TASK_PYTHON=/mowenjie/fluffy-bassoon-2/envs/robotwin-vlrla-20261029-r1/bin/python
TASK_DRIVER=$TASK_CODE/outputs/robotwin-restoration-20261106-r1
[[ $(uname -s) == Linux && $(hostname) == job-* ]]
[[ $(git -C "$TASK_CODE" rev-parse HEAD) == "$TASK_SOURCE" ]]
[[ -z $(git -C "$TASK_CODE" status --porcelain --untracked-files=no) ]]
[[ $TASK_OUT == "$TASK_ARCHIVE/motusbehaviorupdate01-signals" && ! -e $TASK_OUT ]]
# Approval changes the contract; a prepared source is deliberately unlaunchable.
"$TASK_PYTHON" - "$TASK_DRIVER/MOTUSBEHAVIORUPDATE01-CONTRACT.json" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]))
assert p['status']=='admitted' and p['cpu_preflight_passed']
assert p['closed_conservative_gpu_seconds']+p['other_active_reserved_gpu_seconds']+p['reserved_gpu_seconds']==p['all_worst_gpu_seconds']<=p['approved_cap_gpu_seconds']
PY
mkdir "$TASK_OUT"
exec > "$TASK_OUT/worker.log" 2>&1
TASK_STARTED=$(date +%s)
TASK_RUN=$TASK_CODE/outputs/robotwin-sidney-matched-baselines-20261106-r1
TASK_SHARED=$TASK_CODE/outputs/robotwin-sidney-rlinf-ppo-20261105-r1
TASK_LIMIT=720
if [[ ${CUDA_VISIBLE_DEVICES+x} != x ]]; then export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7; fi
source "$TASK_RUN/runtime_environment.sh"
export NATIVE_PPO_WORLD=8 NATIVE_PPO_PLACEMENT=0-7
export PYTHONPATH=$TASK_DRIVER:$PYTHONPATH
export TASK_LOCAL=/tmp/rtbu1 TASK_COLD=$TASK_ARCHIVE/motusbehaviorupdate01
export RAY_TMPDIR=$TASK_LOCAL/r
export TASK_RESOURCE_LIVE=$TASK_LOCAL/resources-live.json
export NATIVE_PPO_MODEL=/mowenjie/fluffy-bassoon-2/cache/robotwin-ppo-pilot-20261105-r1-restore02/pi05-motus-effc6e1a-r1
export NATIVE_PPO_INITIALIZER_REVISION=effc6e1a99bd4d0edf9447e8ba313def67649cfe
export NATIVE_PPO_DEADLINE=$((TASK_STARTED+TASK_LIMIT))
export NATIVE_PPO_OUTPUT=$TASK_COLD/state NATIVE_PPO_ACCOUNTING_ROOT=$TASK_COLD/action-accounting
export NATIVE_PPO_ATTEMPT=ppo-resume01
TASK_MONITOR_PID=
stop_monitor() {
 if [[ -n $TASK_MONITOR_PID ]]; then
  kill "$TASK_MONITOR_PID" 2>/dev/null || true
  wait "$TASK_MONITOR_PID" 2>/dev/null || true
  TASK_MONITOR_PID=
  if [[ -f $TASK_RESOURCE_LIVE ]]; then
   timeout 35s "$TASK_PYTHON" -c 'import json,sys;from fluffy_bassoon.meta_rl.rlinf_native_ppo import write;write(sys.argv[2],json.load(open(sys.argv[1])))' "$TASK_RESOURCE_LIVE" "$TASK_OUT/resources.json" || return $?
  fi
 fi
}
finish() {
 TASK_RC=$?
 trap - EXIT
 TASK_MONITOR_RC=0
 stop_monitor || TASK_MONITOR_RC=$?
 if [[ $TASK_RC == 0 && $TASK_MONITOR_RC != 0 ]]; then TASK_RC=$TASK_MONITOR_RC; fi
 TASK_ENDED=$(date +%s)
 printf '{"exit_code":%s,"started_unix":%s,"ended_unix":%s,"worker_gpu_seconds":%s}\n' \
  "$TASK_RC" "$TASK_STARTED" "$TASK_ENDED" "$(((TASK_ENDED-TASK_STARTED)*8))" > "$TASK_OUT/EXIT.json"
}
trap finish EXIT
"$TASK_PYTHON" "$TASK_CODE/scripts/[train]gpu_probe.py" --baige-container \
 --baige-exclusive-training --purpose training --expected-count 8 --output "$TASK_OUT/gpu-probe.json"
export LYCO_BAIGE_GPU_MAP=$TASK_OUT/gpu-probe.json
"$TASK_PYTHON" "$TASK_SHARED/host_memory.py" "$TASK_OUT" --minimum-gib 512 --once
[[ $(git -C "$REPO_PATH" rev-parse HEAD) == c605b9d55ed93a180119b099e6f836abff293b08 ]]
[[ $(git -C "$ROBOTWIN_PATH" rev-parse HEAD) == dca9ec682688821c42944eacd7f5dd9bfb15396f ]]
[[ -z $(git -C "$REPO_PATH" status --porcelain --untracked-files=no) ]]
[[ -z $(git -C "$ROBOTWIN_PATH" status --porcelain --untracked-files=no) ]]
"$TASK_PYTHON" - "$TASK_DRIVER" <<'PY'
import json,os,shutil,sys
from pathlib import Path
from fluffy_bassoon.meta_rl.rlinf_native_ppo import sha,write
root=Path(sys.argv[1]);p=json.loads((root/'MOTUSBEHAVIORUPDATE01-CONTRACT.json').read_bytes())
for key,name in [('entry_sha256','qualify_motus_behavior_update.py'),('launcher_sha256','qualify_motus_behavior_update.sh')]:
 assert sha(root/name)==p[key]
for name,expected in p['source_files'].items():assert sha(root.parents[1]/name)==expected
local,cold,out=map(lambda key:Path(os.environ[key]),('TASK_LOCAL','TASK_COLD','TASK_OUT'))
assert not local.exists() and not cold.exists()
assert shutil.disk_usage(local.parent).free>=p['local_staging_cap_gib']*2**30
assert any(os.path.ismount(x) for x in (cold.parent,Path('/mnt/bos/adgfyl'),Path('/mnt/bos')))
probe=out/'write-probe'
with probe.open('xb') as f:f.write(b'robotwin-optimized-ppo');f.flush();os.fsync(f.fileno())
assert probe.read_bytes()==b'robotwin-optimized-ppo';probe.unlink()
local.mkdir();cold.mkdir();(cold/'state').mkdir()
assert sha(Path(p['cpu_preflight_result']))==p['cpu_preflight_sha256']
checkpoint=Path(p['checkpoint1'])
assert sha(checkpoint/'MANIFEST.json')==p['checkpoint1_manifest_sha256']
manifest=json.loads((checkpoint/'MANIFEST.json').read_bytes())
staged=local/'global_step_1'
shutil.copytree(checkpoint,staged)
for name,row in manifest['files'].items():
 assert (staged/name).stat().st_size==row['bytes'] and sha(staged/name)==row['sha256']
assert sha(staged/'MANIFEST.json')==p['checkpoint1_manifest_sha256']
write(out/'input-staging.json',dict(checkpoint1_staged=True,manifest_sha256=p['checkpoint1_manifest_sha256'],
 source_commit=os.environ['TASK_SOURCE'],optimizer_updates=0,simulator_actions=0))
model=Path(os.environ['NATIVE_PPO_MODEL'])
assert sha(model/'model.safetensors')==p['model_sha256']
assert sha(model/'assets/pi0.5_clean_randomize_joint_training/norm_stats.json')==p['norm_stats_sha256']
(local/'assets').symlink_to(Path(os.environ['ROBOTWIN_PATH'])/'assets',target_is_directory=True)
font=Path(os.environ['ROBOTWIN_PATH'])/'Times New Roman.ttf'
if font.is_file():(local/font.name).symlink_to(font)
write(out/'admission.json',dict(source_commit=os.environ['TASK_SOURCE'],model_hashes_verified=True,
 new_worker_pfs_output_bytes=0,bos_write_read_fsync=True,
 local_staging_cap_gib=p['local_staging_cap_gib'],bos_run_cap_gib=p['bos_run_cap_gib'],
 optimizer_updates=0,simulator_actions=0,checkpoint1_read_only=True))
PY
export XDG_CACHE_HOME=$TASK_LOCAL/cache OPENPI_DATA_HOME=$TASK_LOCAL/openpi
export HF_HOME=$TASK_LOCAL/hf TORCH_HOME=$TASK_LOCAL/torch TMPDIR=$TASK_LOCAL/tmp
mkdir -p "$XDG_CACHE_HOME" "$OPENPI_DATA_HOME" "$HF_HOME" "$TORCH_HOME" "$TMPDIR" /root/.sapien/physx
if [[ ! -e /root/.sapien/physx/105.1-physx-5.3.1.patch0 ]]; then
 ln -s /mowenjie/fluffy-bassoon/inputs/maniskill-scene-assets-20261028-r1/physx/105.1-physx-5.3.1.patch0 \
 /root/.sapien/physx/105.1-physx-5.3.1.patch0
fi
"$TASK_PYTHON" "$TASK_DRIVER/observe_motus_resources.py" "$TASK_OUT" 8 > "$TASK_OUT/resources.log" 2>&1 &
TASK_MONITOR_PID=$!
cd "$TASK_LOCAL"
TASK_REMAINING=$((NATIVE_PPO_DEADLINE-$(date +%s)-60))
[[ $TASK_REMAINING -gt 180 ]]
export TASK_DEADLINE=$NATIVE_PPO_DEADLINE
timeout --signal=INT --kill-after=20s "$TASK_REMAINING" "$TASK_PYTHON" "$TASK_DRIVER/qualify_motus_behavior_update.py" > "$TASK_OUT/audit.log" 2>&1
stop_monitor
"$TASK_PYTHON" - "$TASK_OUT" <<'PY'
import json,sys
from pathlib import Path
from fluffy_bassoon.meta_rl.rlinf_native_ppo import write
out=Path(sys.argv[1]);r=json.loads((out/'RESULT.json').read_bytes())
assert r['status']=='passed_actual_B32_update_checkpoint_roundtrip' and len(r['actors'])==8
assert r['optimizer_updates']==15 and r['simulator_actions']==0
resources=json.loads((out/'resources.json').read_bytes());assert resources['sampled_reserve_passed']
write(out/'COMPLETE.json',dict(status='passed_actual_B32_update_checkpoint_roundtrip',
 optimizer_updates=15,simulator_actions=0,independent_cpu_readback_pending=True))
PY
