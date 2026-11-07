#!/usr/bin/env bash
set -Eeuo pipefail
TASK_SOURCE=${1:?reviewed source commit}
TASK_OUT=${2:?fresh project BOS output}
export TASK_SOURCE TASK_OUT GIT_OPTIONAL_LOCKS=0
TASK_CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
TASK_ARCHIVE=/mnt/bos/adgfyl/strong-plant-archives/robotwin-restoration-20261106-r1
TASK_PYTHON=/mowenjie/fluffy-bassoon-2/envs/robotwin-vlrla-20261029-r1/bin/python
[[ $(uname -s) == Linux && $(hostname) == job-* ]]
[[ $(git -C "$TASK_CODE" rev-parse HEAD) == "$TASK_SOURCE" ]]
[[ -z $(git -C "$TASK_CODE" status --porcelain --untracked-files=no) ]]
[[ $TASK_OUT == "$TASK_ARCHIVE/motusdev01-signals" && ! -e $TASK_OUT ]]
mkdir "$TASK_OUT"
exec > "$TASK_OUT/worker.log" 2>&1
TASK_STARTED=$(date +%s)
TASK_LIMIT=570
TASK_RUN=$TASK_CODE/outputs/robotwin-sidney-matched-baselines-20261106-r1
TASK_SHARED=$TASK_CODE/outputs/robotwin-sidney-rlinf-ppo-20261105-r1
source "$TASK_RUN/runtime_environment.sh"
TASK_DRIVER=$TASK_CODE/outputs/robotwin-restoration-20261106-r1
export PYTHONPATH=$TASK_DRIVER:$PYTHONPATH
export TASK_LOCAL=/tmp/rtdev1 TASK_DEADLINE=$((TASK_STARTED+TASK_LIMIT))
export RAY_TMPDIR=$TASK_LOCAL/r
export NATIVE_PPO_MODEL=/mowenjie/fluffy-bassoon-2/cache/robotwin-ppo-pilot-20261105-r1-restore02/pi05-motus-effc6e1a-r1
export NATIVE_PPO_INITIALIZER_REVISION=effc6e1a99bd4d0edf9447e8ba313def67649cfe
export NATIVE_PPO_DEADLINE=$TASK_DEADLINE
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
  "$TASK_RC" "$TASK_STARTED" "$TASK_ENDED" "$(((TASK_ENDED-TASK_STARTED)*4))" > "$TASK_OUT/EXIT.json"
}
trap finish EXIT
"$TASK_PYTHON" "$TASK_CODE/scripts/[train]gpu_probe.py" --baige-container \
 --purpose evaluation --expected-count 4 --output "$TASK_OUT/gpu-probe.json"
export LYCO_BAIGE_GPU_MAP=$TASK_OUT/gpu-probe.json
"$TASK_PYTHON" "$TASK_SHARED/host_memory.py" "$TASK_OUT" --minimum-gib 256 --once
[[ $(git -C "$REPO_PATH" rev-parse HEAD) == c605b9d55ed93a180119b099e6f836abff293b08 ]]
[[ $(git -C "$ROBOTWIN_PATH" rev-parse HEAD) == dca9ec682688821c42944eacd7f5dd9bfb15396f ]]
[[ -z $(git -C "$REPO_PATH" status --porcelain --untracked-files=no) ]]
[[ -z $(git -C "$ROBOTWIN_PATH" status --porcelain --untracked-files=no) ]]
"$TASK_PYTHON" - "$TASK_DRIVER" <<'PY'
import json,os,shutil,sys
from pathlib import Path
from fluffy_bassoon.meta_rl.rlinf_native_ppo import sha,write
driver=Path(sys.argv[1]);p=json.loads((driver/'MOTUSDEV01-CONTRACT.json').read_bytes())
assert p['status']=='admitted'
assert p['closed_conservative_gpu_seconds']+p['reserved_gpu_seconds']==p['all_worst_gpu_seconds']<=180000
for field,name in [('entry_sha256','evaluate_motus_dev.py'),('launcher_sha256','evaluate_motus_dev.sh')]:
 assert sha(driver/name)==p[field]
local,out=Path(os.environ['TASK_LOCAL']),Path(os.environ['TASK_OUT'])
assert not local.exists() and shutil.disk_usage(local.parent).free>=32*2**30
archive=out.parent
assert any(os.path.ismount(x) for x in (archive,Path('/mnt/bos/adgfyl'),Path('/mnt/bos')))
probe=out/'write-probe'
with probe.open('xb') as stream:
 stream.write(b'motus-fixed-native-dev');stream.flush();os.fsync(stream.fileno())
assert probe.read_bytes()==b'motus-fixed-native-dev';probe.unlink();local.mkdir()
assert sha(archive/'cpudev-inputs02/RESULT.json')==p['baseline_inputs_sha256']
cold=archive/'motusreplay01/checkpoints/global_step_1'
assert sha(cold/'MANIFEST.json')==p['checkpoint_manifest_sha256']
manifest=json.loads((cold/'MANIFEST.json').read_bytes())
paths=[k for k in manifest['files'] if k.endswith('/full_weights.pt')]
assert len(paths)==1
weights=cold/paths[0];assert sha(weights)==p['inference_weights_sha256']==manifest['files'][paths[0]]['sha256']
shutil.copyfile(weights,local/'full_weights.pt');assert sha(local/'full_weights.pt')==p['inference_weights_sha256']
model=Path(os.environ['NATIVE_PPO_MODEL'])
assert sha(model/'model.safetensors')==p['model_sha256']
assert sha(model/'assets/pi0.5_clean_randomize_joint_training/norm_stats.json')==p['norm_stats_sha256']
(local/'assets').symlink_to(Path(os.environ['ROBOTWIN_PATH'])/'assets',target_is_directory=True)
font=Path(os.environ['ROBOTWIN_PATH'])/'Times New Roman.ttf'
if font.is_file():(local/font.name).symlink_to(font)
write(out/'admission.json',dict(source_commit=os.environ['TASK_SOURCE'],gpus=4,optimizer_updates=0,
 stageback_weights_sha256=sha(local/'full_weights.pt'),baseline_inputs_sha256=p['baseline_inputs_sha256'],
 new_worker_pfs_output_bytes=0,bos_write_read_fsync=True,maximum_all_in_actions=4096))
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
print(json.dumps(accounted_controller_source(os.environ['ROBOTWIN_PATH'],sys.argv[1])))
PY
export TASK_CONTROLLER=$TASK_LOCAL/controller/envs/_base_task.py
export TASK_CONTROLLER_SHA=$(sha256sum "$TASK_CONTROLLER" | cut -d ' ' -f1)
"$TASK_PYTHON" "$TASK_DRIVER/observe_motus_resources.py" "$TASK_OUT" > "$TASK_OUT/resources.log" 2>&1 &
TASK_MONITOR_PID=$!
cd "$TASK_LOCAL"
TASK_REMAINING=$((TASK_DEADLINE-$(date +%s)-45))
[[ $TASK_REMAINING -ge 360 ]]
TASK_EVAL_RC=0
timeout --signal=INT --kill-after=30s "$TASK_REMAINING" "$TASK_PYTHON" "$TASK_DRIVER/evaluate_motus_dev.py" \
 > "$TASK_OUT/evaluate.log" 2>&1 || TASK_EVAL_RC=$?
"$TASK_PYTHON" -m ray.scripts.scripts stop --force > "$TASK_OUT/ray-stop.log" 2>&1 || true
[[ $TASK_EVAL_RC == 0 ]] || exit "$TASK_EVAL_RC"
stop_monitor
"$TASK_PYTHON" - "$TASK_OUT" <<'PY'
import json,sys
from pathlib import Path
from fluffy_bassoon.meta_rl.rlinf_native_ppo import sha,write
root=Path(sys.argv[1]);r=json.loads((root/'state/RESULT.json').read_bytes())
assert r['status']=='complete' and r['episodes']==16 and r['optimizer_updates_during_eval']==0
assert len(r['environments'])==len(r['frozen_rollouts'])==4
assert all(v['passed'] for v in r['baseline_calibration'])
assert all(v['frozen_parameters_unchanged'] for v in r['frozen_rollouts'])
assert json.loads((root/'resources.json').read_bytes())['sampled_reserve_passed']
files={str(p.relative_to(root)):dict(bytes=p.stat().st_size,sha256=sha(p))
 for p in sorted((root/'state').rglob('*')) if p.is_file()}
assert sum(r['bytes'] for r in files.values())<=4*2**30
write(root/'FILES.json',dict(files=files))
write(root/'COMPLETE.json',dict(status='complete_native_source_dev',episodes=16,optimizer_updates=0,
 frozen=True,baseline_inputs_calibrated=True,paired_initial_observations=True,
 shared_train_eval_engineering=True,independent_trace_readback_still_required=True,
 result_sha256=sha(root/'state/RESULT.json')))
PY
