#!/usr/bin/env bash
# Fresh C0->C2 qualification before reserving the longer repaired learning curve.
set -Eeuo pipefail
TASK_SOURCE=${1:?reviewed source commit}
TASK_METHOD=${2:?ppo or grpo}
[[ $TASK_METHOD == ppo || $TASK_METHOD == grpo ]]
TASK_CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
TASK_DRIVER=$TASK_CODE/outputs/robotwin-restoration-20261106-r1
TASK_ARCHIVE=/mnt/bos/adgfyl/strong-plant-archives/robotwin-restoration-20261106-r1/repaired-motus-20261109-r1
TASK_COLD=$TASK_ARCHIVE/$TASK_METHOD-prefix01
TASK_OUT=$TASK_COLD/signals
TASK_LOCAL=/tmp/rt-repaired-$TASK_METHOD-prefix01
TASK_PYTHON=/mowenjie/fluffy-bassoon-2/envs/robotwin-vlrla-20261029-r1/bin/python
export TASK_SOURCE TASK_METHOD TASK_CODE TASK_DRIVER TASK_OUT TASK_LOCAL TASK_COLD GIT_OPTIONAL_LOCKS=0
[[ $(uname -s) == Linux && $(hostname) == job-* ]]
[[ $(git -C "$TASK_CODE" rev-parse HEAD) == "$TASK_SOURCE" ]]
[[ -z $(git -C "$TASK_CODE" status --porcelain --untracked-files=no) ]]
[[ ! -e $TASK_COLD && ! -e $TASK_LOCAL ]]
"$TASK_PYTHON" - "$TASK_ARCHIVE/controller-pilot01-readback.json" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]))
assert p['status']=='passed_independent_linux_cpu_physical_trace_reduction'
assert p['gate_passed'] and p['complete_episodes']==48 and p['optimizer_updates']==0
assert p['source_commit']=='3f2228ee8299104c3813d53f5cf0b9159b34c39e'
PY
mkdir "$TASK_COLD" "$TASK_LOCAL"
mkdir "$TASK_OUT" "$TASK_COLD/state"
exec > "$TASK_OUT/worker.log" 2>&1
TASK_STARTED=$(date +%s)
TASK_LIMIT=9000
export TASK_DEADLINE=$((TASK_STARTED+TASK_LIMIT)) NATIVE_PPO_DEADLINE=$((TASK_STARTED+TASK_LIMIT))
export TASK_RESOURCE_LIVE=$TASK_LOCAL/resources-live.json
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
  "$TASK_RC" "$TASK_STARTED" "$TASK_ENDED" "$(((TASK_ENDED-TASK_STARTED)*8))" > "$TASK_OUT/EXIT.json"
}
trap finish EXIT
TASK_RUN=$TASK_CODE/outputs/robotwin-sidney-matched-baselines-20261106-r1
TASK_SHARED=$TASK_CODE/outputs/robotwin-sidney-rlinf-ppo-20261105-r1
if [[ ${CUDA_VISIBLE_DEVICES+x} != x ]]; then export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7; fi
source "$TASK_RUN/runtime_environment.sh"
export NATIVE_PPO_WORLD=8 NATIVE_PPO_PLACEMENT=0-7
export PYTHONPATH=$TASK_DRIVER:$PYTHONPATH
export RAY_TMPDIR=$TASK_LOCAL/r
export NATIVE_PPO_MODEL=/mowenjie/fluffy-bassoon-2/cache/robotwin-ppo-pilot-20261105-r1-restore02/pi05-motus-effc6e1a-r1
export NATIVE_PPO_INITIALIZER_REVISION=effc6e1a99bd4d0edf9447e8ba313def67649cfe
export NATIVE_PPO_OUTPUT=$TASK_COLD/state NATIVE_PPO_ACCOUNTING_ROOT=$TASK_COLD/action-accounting
export NATIVE_PPO_ATTEMPT=repaired-$TASK_METHOD-prefix01
"$TASK_PYTHON" "$TASK_CODE/scripts/[train]gpu_probe.py" --baige-container --baige-exclusive-training \
 --purpose training --expected-count 8 --output "$TASK_OUT/gpu-probe.json"
export LYCO_BAIGE_GPU_MAP=$TASK_OUT/gpu-probe.json
"$TASK_PYTHON" "$TASK_SHARED/host_memory.py" "$TASK_OUT" --minimum-gib 1024 --once
"$TASK_PYTHON" - <<'PY'
import json,os,shutil,subprocess
from pathlib import Path
from fluffy_bassoon.meta_rl.rlinf_native_ppo import sha
assert shutil.disk_usage(os.environ['TASK_LOCAL']).free>=384*2**30
assert os.path.ismount('/mnt/bos/adgfyl/strong-plant-archives/robotwin-restoration-20261106-r1') or os.path.ismount('/mnt/bos/adgfyl')
for folder,expected in ((os.environ['REPO_PATH'],'c605b9d55ed93a180119b099e6f836abff293b08'),
                        (os.environ['ROBOTWIN_PATH'],'dca9ec682688821c42944eacd7f5dd9bfb15396f')):
 assert subprocess.check_output(['git','-C',folder,'rev-parse','HEAD'],text=True).strip()==expected
 assert not subprocess.check_output(['git','-C',folder,'status','--porcelain','--untracked-files=no'],text=True).strip()
reference=json.loads((Path(os.environ['TASK_DRIVER'])/'MOTOR01-CONTRACT.json').read_bytes())
model=Path(os.environ['NATIVE_PPO_MODEL'])
assert sha(model/'model.safetensors')==reference['model_sha256']
assert sha(model/'assets/pi0.5_clean_randomize_joint_training/norm_stats.json')==reference['norm_stats_sha256']
p=Path(os.environ['TASK_OUT'])/'write-probe'
with p.open('xb') as f:f.write(b'repaired-motus-prefix');f.flush();os.fsync(f.fileno())
assert p.read_bytes()==b'repaired-motus-prefix';p.unlink()
local=Path(os.environ['TASK_LOCAL']);native=Path(os.environ['ROBOTWIN_PATH'])
(local/'assets').symlink_to(native/'assets',target_is_directory=True)
if (native/'Times New Roman.ttf').is_file():(local/'Times New Roman.ttf').symlink_to(native/'Times New Roman.ttf')
PY
export XDG_CACHE_HOME=$TASK_LOCAL/cache OPENPI_DATA_HOME=$TASK_LOCAL/openpi
export HF_HOME=$TASK_LOCAL/hf TORCH_HOME=$TASK_LOCAL/torch TMPDIR=$TASK_LOCAL/tmp
mkdir -p "$XDG_CACHE_HOME" "$OPENPI_DATA_HOME" "$HF_HOME" "$TORCH_HOME" "$TMPDIR" /root/.sapien/physx
if [[ ! -e /root/.sapien/physx/105.1-physx-5.3.1.patch0 ]]; then
 ln -s /mowenjie/fluffy-bassoon/inputs/maniskill-scene-assets-20261028-r1/physx/105.1-physx-5.3.1.patch0 \
 /root/.sapien/physx/105.1-physx-5.3.1.patch0
fi
"$TASK_PYTHON" - "$TASK_LOCAL/controller" > "$TASK_OUT/controller.json" <<'PY'
import json,os,sys
from fluffy_bassoon.rl.robotwin_batch import accounted_controller_source
print(json.dumps(accounted_controller_source(os.environ['ROBOTWIN_PATH'],sys.argv[1],waypoint_controller=True)))
PY
export TASK_CONTROLLER=$TASK_LOCAL/controller/envs/_base_task.py
export TASK_CONTROLLER_SHA=$(sha256sum "$TASK_CONTROLLER" | cut -d' ' -f1)
"$TASK_PYTHON" "$TASK_DRIVER/motus_configuration.py" validate --optimized-setup --repaired-recipe > "$TASK_OUT/configs.json"
mapfile -t TASK_OVERRIDES < <("$TASK_PYTHON" "$TASK_DRIVER/motus_configuration.py" "$TASK_METHOD" --optimized-setup --repaired-recipe)
[[ ${#TASK_OVERRIDES[@]} -ge 20 ]]
"$TASK_PYTHON" "$TASK_DRIVER/observe_motus_resources.py" "$TASK_OUT" 8 > "$TASK_OUT/resources.log" 2>&1 &
TASK_MONITOR_PID=$!
TASK_ENTRY=motus_optimized_entry.py
if [[ $TASK_METHOD == grpo ]]; then TASK_ENTRY=motus_grpo_entry.py; fi
cd "$TASK_LOCAL"
TASK_REMAINING=$((TASK_DEADLINE-$(date +%s)-90))
[[ $TASK_REMAINING -gt 600 ]]
timeout --signal=INT --kill-after=30s "$TASK_REMAINING" "$TASK_PYTHON" "$TASK_DRIVER/$TASK_ENTRY" \
 "${TASK_OVERRIDES[@]}" native_pilot.stop_after_iteration=2 > "$TASK_OUT/train.log" 2>&1
"$TASK_PYTHON" - <<'PY'
import json,os
from pathlib import Path
from fluffy_bassoon.meta_rl.rlinf_native_ppo import write
state=Path(os.environ['NATIVE_PPO_OUTPUT']);result=json.loads((state/'RESULT.json').read_bytes())
assert result['status']=='completed_admitted_prefix' and result['completed_iterations']==2
assert result['native_adam_updates_per_rank']==60
for step in range(3):
 assert json.loads((state/f'checkpoint-receipts/step-{step:04d}.json').read_bytes())['completed']
 assert json.loads((state/f'dev/iteration-{step:04d}.json').read_bytes())['completed']
write(Path(os.environ['TASK_OUT'])/'COMPLETE.json',dict(status='completed_repaired_two_collection_prefix',
 source_commit=os.environ['TASK_SOURCE'],algorithm=os.environ['TASK_METHOD'],
 adam_updates_per_rank=60,full_curve_pending=True,independent_readback_pending=True))
PY
