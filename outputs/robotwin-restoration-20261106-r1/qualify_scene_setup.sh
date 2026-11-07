#!/usr/bin/env bash
set -Eeuo pipefail
TASK_SOURCE=${1:?verified commit}
TASK_OUT=${2:?fresh output}
export TASK_SOURCE TASK_OUT GIT_OPTIONAL_LOCKS=0
TASK_CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
TASK_DRIVER=$TASK_CODE/outputs/robotwin-restoration-20261106-r1
TASK_ARCHIVE=/mnt/bos/adgfyl/strong-plant-archives/robotwin-restoration-20261106-r1
TASK_PYTHON=/mowenjie/fluffy-bassoon-2/envs/robotwin-vlrla-20261029-r1/bin/python
[[ $(uname -s) == Linux && $(hostname) == job-* ]]
[[ $(git -C "$TASK_CODE" rev-parse HEAD) == "$TASK_SOURCE" ]]
[[ -z $(git -C "$TASK_CODE" status --porcelain --untracked-files=no) ]]
[[ $TASK_OUT == "$TASK_ARCHIVE/scenesetup01-signals" && ! -e $TASK_OUT ]]
mkdir "$TASK_OUT"
exec > "$TASK_OUT/worker.log" 2>&1
TASK_STARTED=$(date +%s)
TASK_LIMIT=2200
TASK_RUN=$TASK_CODE/outputs/robotwin-sidney-matched-baselines-20261106-r1
TASK_SHARED=$TASK_CODE/outputs/robotwin-sidney-rlinf-ppo-20261105-r1
source "$TASK_RUN/runtime_environment.sh"
export TASK_LOCAL=/tmp/rtscene1 TASK_DEADLINE=$((TASK_STARTED+TASK_LIMIT))
TASK_MONITOR_PID=
stop_monitor() {
 if [[ -n $TASK_MONITOR_PID ]]; then
  kill "$TASK_MONITOR_PID" 2>/dev/null || true
  wait "$TASK_MONITOR_PID" 2>/dev/null || true
  TASK_MONITOR_PID=
 fi
}
finish() {
 TASK_RC=$?
 trap - EXIT
 stop_monitor
 TASK_ENDED=$(date +%s)
 printf '{"exit_code":%s,"started_unix":%s,"ended_unix":%s,"worker_gpu_seconds":%s}\n' \
  "$TASK_RC" "$TASK_STARTED" "$TASK_ENDED" "$((TASK_ENDED-TASK_STARTED))" > "$TASK_OUT/EXIT.json"
}
trap finish EXIT
"$TASK_PYTHON" "$TASK_CODE/scripts/[train]gpu_probe.py" --baige-container \
 --purpose evaluation --expected-count 1 --output "$TASK_OUT/gpu-probe.json"
export LYCO_BAIGE_GPU_MAP=$TASK_OUT/gpu-probe.json
"$TASK_PYTHON" "$TASK_SHARED/host_memory.py" "$TASK_OUT" --minimum-gib 128 --once
[[ $(git -C "$REPO_PATH" rev-parse HEAD) == c605b9d55ed93a180119b099e6f836abff293b08 ]]
[[ $(git -C "$ROBOTWIN_PATH" rev-parse HEAD) == dca9ec682688821c42944eacd7f5dd9bfb15396f ]]
[[ -z $(git -C "$REPO_PATH" status --porcelain --untracked-files=no) ]]
[[ -z $(git -C "$ROBOTWIN_PATH" status --porcelain --untracked-files=no) ]]
"$TASK_PYTHON" - "$TASK_DRIVER" <<'PY'
import json,os,shutil,sys
from pathlib import Path
from fluffy_bassoon.meta_rl.rlinf_native_ppo import sha,write
root=Path(sys.argv[1]);p=json.loads((root/'SCENESETUP01-CONTRACT.json').read_bytes())
assert p['status']=='admitted'
assert p['closed_conservative_gpu_seconds']+p['reserved_gpu_seconds']==p['all_worst_gpu_seconds']<=180000
for key,name in [('entry_sha256','qualify_scene_setup.py'),('launcher_sha256','qualify_scene_setup.sh'),('observer_sha256','observe_motus_resources.py')]:
 assert sha(root/name)==p[key]
local,out=Path(os.environ['TASK_LOCAL']),Path(os.environ['TASK_OUT'])
assert not local.exists() and shutil.disk_usage(local.parent).free>=4*2**30
archive=out.parent
assert any(os.path.ismount(x) for x in [archive,Path('/mnt/bos/adgfyl'),Path('/mnt/bos')])
probe=out/'write-probe'
with probe.open('xb') as f:f.write(b'robotwin-deferred-setup');f.flush();os.fsync(f.fileno())
assert probe.read_bytes()==b'robotwin-deferred-setup';probe.unlink();local.mkdir()
(local/'assets').symlink_to(Path(os.environ['ROBOTWIN_PATH'])/'assets',target_is_directory=True)
font=Path(os.environ['ROBOTWIN_PATH'])/'Times New Roman.ttf'
if font.is_file():(local/font.name).symlink_to(font)
write(out/'admission.json',dict(source_commit=os.environ['TASK_SOURCE'],gpus=1,
 optimizer_updates=0,model_calls=0,new_worker_pfs_output_bytes=0,bos_write_read_fsync=True,
 maximum_all_in_actions=5120,reserved_gpu_seconds=p['reserved_gpu_seconds']))
PY
export XDG_CACHE_HOME=$TASK_LOCAL/cache HF_HOME=$TASK_LOCAL/hf TORCH_HOME=$TASK_LOCAL/torch TMPDIR=$TASK_LOCAL/tmp
mkdir -p "$XDG_CACHE_HOME" "$HF_HOME" "$TORCH_HOME" "$TMPDIR" /root/.sapien/physx
if [[ ! -e /root/.sapien/physx/105.1-physx-5.3.1.patch0 ]]; then
 ln -s /mowenjie/fluffy-bassoon/inputs/maniskill-scene-assets-20261028-r1/physx/105.1-physx-5.3.1.patch0 \
 /root/.sapien/physx/105.1-physx-5.3.1.patch0
fi
"$TASK_PYTHON" - "$TASK_LOCAL/controller" > "$TASK_OUT/controller.json" <<'PY'
import json,os,sys
from fluffy_bassoon.rl.robotwin_batch import accounted_controller_source
print(json.dumps(accounted_controller_source(os.environ['ROBOTWIN_PATH'],sys.argv[1])))
PY
export TASK_CONTROLLER=$TASK_LOCAL/controller/envs/_base_task.py
export TASK_CONTROLLER_SHA=$(sha256sum "$TASK_CONTROLLER" | cut -d ' ' -f1)
"$TASK_PYTHON" "$TASK_DRIVER/observe_motus_resources.py" "$TASK_OUT" 1 > "$TASK_OUT/resources.log" 2>&1 &
TASK_MONITOR_PID=$!
cd "$TASK_LOCAL"
"$TASK_PYTHON" "$TASK_DRIVER/qualify_scene_setup.py" --stage > "$TASK_OUT/stage.log" 2>&1
TASK_REMAINING=$((TASK_DEADLINE-$(date +%s)-45))
[[ $TASK_REMAINING -ge 600 ]]
timeout --signal=INT --kill-after=30s "$TASK_REMAINING" "$TASK_PYTHON" "$TASK_DRIVER/qualify_scene_setup.py" \
 > "$TASK_OUT/replay.log" 2>&1
stop_monitor
"$TASK_PYTHON" - "$TASK_OUT" <<'PY'
import json,sys
from pathlib import Path
from fluffy_bassoon.meta_rl.rlinf_native_ppo import sha,write
root=Path(sys.argv[1]);r=json.loads((root/'RESULT.json').read_bytes())
assert r['status']=='passed' and r['deferred_episodes']==16 and r['default_control_episodes']==4
assert r['optimizer_updates']==r['model_calls']==0
assert json.loads((root/'resources.json').read_bytes())['sampled_reserve_passed']
files={str(p.relative_to(root)):dict(bytes=p.stat().st_size,sha256=sha(p))
 for p in sorted(root.rglob('*')) if p.is_file() and p.suffix=='.json'}
assert sum(x['bytes'] for x in files.values()) <= 64*2**20
write(root/'FILES.json',dict(files=files))
write(root/'COMPLETE.json',dict(status='complete_deferred_setup_replay',deferred_episodes=16,
 default_control_episodes=4,original_native_outputs_exact=True,model_calls=0,optimizer_updates=0,
 combined_training_memory_qualified=False,larger_scene_count_qualified=False,
 result_sha256=sha(root/'RESULT.json'),independent_readback_pending=True))
PY
