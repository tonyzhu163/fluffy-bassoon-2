#!/usr/bin/env bash
set -Eeuo pipefail
TASK_SOURCE=${1:?reviewed source commit}
TASK_OUT=${2:?fresh project BOS signals identity}
export TASK_OUT
TASK_CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
TASK_ARCHIVE=/mnt/bos/adgfyl/strong-plant-archives/robotwin-restoration-20261106-r1
TASK_PYTHON=/mowenjie/fluffy-bassoon-2/envs/robotwin-vlrla-20261029-r1/bin/python
[[ $(uname -s) == Linux && $(hostname) == job-* ]]
export GIT_OPTIONAL_LOCKS=0
[[ $(git -C "$TASK_CODE" rev-parse HEAD) == "$TASK_SOURCE" ]]
[[ -z $(git -C "$TASK_CODE" status --porcelain --untracked-files=no) ]]
[[ $TASK_OUT == "$TASK_ARCHIVE/motusppo04-signals" && ! -e $TASK_OUT ]]
mkdir -p "$TASK_OUT"
exec > "$TASK_OUT/worker.log" 2>&1
TASK_STARTED=$(date +%s)
TASK_RUN=$TASK_CODE/outputs/robotwin-sidney-matched-baselines-20261106-r1
TASK_SHARED=$TASK_CODE/outputs/robotwin-sidney-rlinf-ppo-20261105-r1
TASK_LIMIT=3300
source "$TASK_RUN/runtime_environment.sh"
TASK_DRIVER=$TASK_CODE/outputs/robotwin-restoration-20261106-r1
export PYTHONPATH=$TASK_DRIVER:$PYTHONPATH
export TASK_LOCAL=/tmp/rtmp4 TASK_COLD=$TASK_ARCHIVE/motusppo04
export RAY_TMPDIR=$TASK_LOCAL/r
export NATIVE_PPO_MODEL=/mowenjie/fluffy-bassoon-2/cache/robotwin-ppo-pilot-20261105-r1-restore02/pi05-motus-effc6e1a-r1
export NATIVE_PPO_INITIALIZER_REVISION=effc6e1a99bd4d0edf9447e8ba313def67649cfe
export NATIVE_PPO_DEADLINE=$((TASK_STARTED+TASK_LIMIT))
export NATIVE_PPO_OUTPUT=$TASK_COLD/state NATIVE_PPO_ACCOUNTING_ROOT=$TASK_COLD/action-accounting
export NATIVE_PPO_ATTEMPT=ppo-prefix04
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
    "$TASK_RC" "$TASK_STARTED" "$TASK_ENDED" "$(((TASK_ENDED-TASK_STARTED)*4))" > "$TASK_OUT/EXIT.json"
}
trap finish EXIT
"$TASK_PYTHON" "$TASK_CODE/scripts/[train]gpu_probe.py" --baige-container \
  --baige-exclusive-training --purpose training --expected-count 4 --output "$TASK_OUT/gpu-probe.json"
export LYCO_BAIGE_GPU_MAP=$TASK_OUT/gpu-probe.json
"$TASK_PYTHON" "$TASK_SHARED/host_memory.py" "$TASK_OUT" --minimum-gib 1024 --once
[[ $(git -C "$REPO_PATH" rev-parse HEAD) == c605b9d55ed93a180119b099e6f836abff293b08 ]]
[[ $(git -C "$ROBOTWIN_PATH" rev-parse HEAD) == dca9ec682688821c42944eacd7f5dd9bfb15396f ]]
[[ -z $(git -C "$REPO_PATH" status --porcelain --untracked-files=no) ]]
[[ -z $(git -C "$ROBOTWIN_PATH" status --porcelain --untracked-files=no) ]]
"$TASK_PYTHON" - "$TASK_DRIVER" <<'PY'
import json,os,shutil,sys
from pathlib import Path
from fluffy_bassoon.meta_rl.rlinf_native_ppo import sha,write
driver=Path(sys.argv[1]);contract=json.loads((driver/'MOTUSPPO04-CONTRACT.json').read_bytes())
assert contract['status']=='admitted'
assert contract['closed_conservative_gpu_seconds']+contract['reserved_gpu_seconds']==contract['all_worst_gpu_seconds']<=180000
local,cold,out=map(lambda key:Path(os.environ[key]),('TASK_LOCAL','TASK_COLD','TASK_OUT'))
assert not local.exists() and not cold.exists()
assert shutil.disk_usage(local.parent).free>=160*2**30
archive=cold.parent
assert archive.is_dir() and any(os.path.ismount(p) for p in (archive,Path('/mnt/bos/adgfyl'),Path('/mnt/bos')))
probe=archive/'motusppo04-write-probe'
with probe.open('xb') as stream:
 stream.write(b'robotwin-native-motus-teacher');stream.flush();os.fsync(stream.fileno())
assert probe.read_bytes()==b'robotwin-native-motus-teacher'
probe.unlink();local.mkdir();cold.mkdir();(cold/'state').mkdir()
model=Path(os.environ['NATIVE_PPO_MODEL'])
assert sha(model/'model.safetensors')==contract['model_sha256']
assert sha(model/'assets/pi0.5_clean_randomize_joint_training/norm_stats.json')==contract['norm_stats_sha256']
# Preserve every native relative asset lookup while keeping new relative data
# on this job's local disk. External source/model/cache files remain read-only.
(local/'assets').symlink_to(Path(os.environ['ROBOTWIN_PATH'])/'assets',target_is_directory=True)
font=Path(os.environ['ROBOTWIN_PATH'])/'Times New Roman.ttf'
if font.is_file():(local/font.name).symlink_to(font)
write(out/'admission.json',dict(source_commit=os.environ['TASK_SOURCE'],model_hashes_verified=True,
 new_worker_pfs_output_bytes=0,local_free_bytes=shutil.disk_usage(local).free,
 shared_readonly_pfs_free_bytes=shutil.disk_usage('/mowenjie/fluffy-bassoon-2').free,
 bos_write_read_fsync=True,local_staging_cap_gib=160,bos_run_cap_gib=96,
 conservative_action_envelope_precommitted=500000,
 unrecoverable_journal_rule='Charge the complete precommitted envelope; admit no incomplete corpus boundary'))
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
"$TASK_PYTHON" "$TASK_DRIVER/motus_configuration.py" validate > "$TASK_OUT/configs.json"
"$TASK_PYTHON" "$TASK_DRIVER/observe_motus_resources.py" "$TASK_OUT" > "$TASK_OUT/resources.log" 2>&1 &
TASK_MONITOR_PID=$!
mapfile -t TASK_OVERRIDES < <("$TASK_PYTHON" "$TASK_DRIVER/motus_configuration.py" ppo)
[[ ${#TASK_OVERRIDES[@]} -ge 20 ]]
cd "$TASK_LOCAL"
TASK_REMAINING=$((NATIVE_PPO_DEADLINE-$(date +%s)-60))
[[ $TASK_REMAINING -gt 600 ]]
timeout --signal=INT --kill-after=30s "$TASK_REMAINING" "$TASK_PYTHON" "$TASK_DRIVER/motus_teacher_entry.py" \
  "${TASK_OVERRIDES[@]}" > "$TASK_OUT/train.log" 2>&1
"$TASK_PYTHON" - "$TASK_OUT" "$NATIVE_PPO_OUTPUT" <<'PY'
import json,sys
from pathlib import Path
out,state=map(Path,sys.argv[1:]);result=json.loads((state/'RESULT.json').read_bytes())
assert result['status']=='completed_admitted_prefix' and result['completed_iterations']==1
assert result['native_adam_updates_per_rank']==15
assert json.loads((state/'checkpoint-receipts/step-0001.json').read_bytes())['completed']
for folder in ('first-replay','native-values','learner-reference'):
 files=list((state/folder).glob('*.json'));assert len(files)==4
 assert all(json.loads(p.read_bytes())['passed'] for p in files)
assert len(list((state/'group-resets').glob('*.json')))==16
assert all(json.loads(p.read_bytes())['passed'] for p in (state/'group-resets').glob('*.json'))
assert (state/'dev/iteration-0000.json').is_file() and (state/'dev/iteration-0001.json').is_file()
print(json.dumps(dict(status='completed_verified_prefix',scientific_campaign_complete=False)))
PY
