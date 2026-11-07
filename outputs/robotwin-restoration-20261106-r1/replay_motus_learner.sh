#!/usr/bin/env bash
set -Eeuo pipefail
TASK_SOURCE=${1:?reviewed source commit}
TASK_OUT=${2:?fresh project BOS signals identity}
export TASK_SOURCE TASK_OUT GIT_OPTIONAL_LOCKS=0
TASK_CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
TASK_ARCHIVE=/mnt/bos/adgfyl/strong-plant-archives/robotwin-restoration-20261106-r1
TASK_PYTHON=/mowenjie/fluffy-bassoon-2/envs/robotwin-vlrla-20261029-r1/bin/python
[[ $(uname -s) == Linux && $(hostname) == job-* ]]
[[ $(git -C "$TASK_CODE" rev-parse HEAD) == "$TASK_SOURCE" ]]
[[ -z $(git -C "$TASK_CODE" status --porcelain --untracked-files=no) ]]
[[ $TASK_OUT == "$TASK_ARCHIVE/motusreplay01-signals" && ! -e $TASK_OUT ]]
mkdir "$TASK_OUT"
exec > "$TASK_OUT/worker.log" 2>&1
TASK_STARTED=$(date +%s)
TASK_LIMIT=2040
TASK_RUN=$TASK_CODE/outputs/robotwin-sidney-matched-baselines-20261106-r1
TASK_SHARED=$TASK_CODE/outputs/robotwin-sidney-rlinf-ppo-20261105-r1
source "$TASK_RUN/runtime_environment.sh"
TASK_DRIVER=$TASK_CODE/outputs/robotwin-restoration-20261106-r1
export PYTHONPATH=$TASK_DRIVER:$PYTHONPATH
export TASK_LOCAL=/tmp/rtlr1 TASK_DEADLINE=$((TASK_STARTED+TASK_LIMIT))
export TASK_COLD=$TASK_ARCHIVE/motusreplay01
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
 --baige-exclusive-training --purpose training --expected-count 4 --output "$TASK_OUT/gpu-probe.json"
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
driver=Path(sys.argv[1]);p=json.loads((driver/'MOTUSREPLAY01-CONTRACT.json').read_bytes())
assert p['status']=='admitted'
assert p['closed_conservative_gpu_seconds']+p['reserved_gpu_seconds']==p['all_worst_gpu_seconds']<=180000
assert sha(driver/'replay_motus_learner.py')==p['entry_sha256']
assert sha(driver/'replay_motus_learner.sh')==p['launcher_sha256']
local,cold,out=map(lambda n:Path(os.environ[n]),('TASK_LOCAL','TASK_COLD','TASK_OUT'))
assert not local.exists() and not cold.exists()
assert shutil.disk_usage(local.parent).free>=96*2**30
assert cold.parent.is_dir() and any(os.path.ismount(x) for x in (cold.parent,Path('/mnt/bos/adgfyl'),Path('/mnt/bos')))
probe=cold.parent/'motusreplay01-write-probe'
with probe.open('xb') as stream:
 stream.write(b'native-motus-learner-recovery');stream.flush();os.fsync(stream.fileno())
assert probe.read_bytes()==b'native-motus-learner-recovery'
probe.unlink();local.mkdir();cold.mkdir()
assert len((os.environ['RAY_TMPDIR']+'/ray/session_2026-11-07_09-00-00_666653_9999999/sockets/plasma_store').encode())<107
audit=cold.parent/'collection-audit02/RESULT.json'
assert sha(audit)==p['collection_audit_sha256']
assert json.loads(audit.read_bytes())['status']=='learner_replay_inputs_complete'
initial=cold.parent/'motusppo04/checkpoints/global_step_0'
assert sha(initial/'MANIFEST.json')==p['checkpoint0_manifest_sha256']
model=Path(os.environ['NATIVE_PPO_MODEL'])
assert sha(model/'model.safetensors')==p['model_sha256']
assert sha(model/'assets/pi0.5_clean_randomize_joint_training/norm_stats.json')==p['norm_stats_sha256']
write(out/'admission.json',dict(source_commit=os.environ['TASK_SOURCE'],new_worker_pfs_output_bytes=0,
 bos_write_read_fsync=True,model_hashes_verified=True,collection_audit_sha256=sha(audit),
 local_staging_cap_gib=96,bos_checkpoint_cap_gib=48,new_simulator_controls=0,
 original_checkpoint_and_collection_unchanged=True))
PY
export XDG_CACHE_HOME=$TASK_LOCAL/cache OPENPI_DATA_HOME=$TASK_LOCAL/openpi
export HF_HOME=$TASK_LOCAL/hf TORCH_HOME=$TASK_LOCAL/torch TMPDIR=$TASK_LOCAL/tmp
mkdir -p "$XDG_CACHE_HOME" "$OPENPI_DATA_HOME" "$HF_HOME" "$TORCH_HOME" "$TMPDIR"
"$TASK_PYTHON" "$TASK_DRIVER/observe_motus_resources.py" "$TASK_OUT" > "$TASK_OUT/resources.log" 2>&1 &
TASK_MONITOR_PID=$!
cd "$TASK_LOCAL"
TASK_TRAIN_RC=0
TASK_REMAINING=$((TASK_DEADLINE-$(date +%s)-60))
[[ $TASK_REMAINING -ge 1320 ]]
timeout --signal=INT --kill-after=30s 1320 "$TASK_PYTHON" "$TASK_DRIVER/replay_motus_learner.py" \
 --phase train > "$TASK_OUT/train.log" 2>&1 || TASK_TRAIN_RC=$?
# This allocated container is exclusive. Stop only its local Ray processes.
"$TASK_PYTHON" -m ray.scripts.scripts stop --force > "$TASK_OUT/ray-stop-train.log" 2>&1 || true
[[ $TASK_TRAIN_RC == 0 ]] || exit "$TASK_TRAIN_RC"
TASK_REMAINING=$((TASK_DEADLINE-$(date +%s)-45))
[[ $TASK_REMAINING -ge 300 ]]
TASK_RESTORE_LIMIT=$TASK_REMAINING
if [[ $TASK_RESTORE_LIMIT -gt 630 ]]; then TASK_RESTORE_LIMIT=630; fi
TASK_RESTORE_RC=0
timeout --signal=INT --kill-after=30s "$TASK_RESTORE_LIMIT" "$TASK_PYTHON" "$TASK_DRIVER/replay_motus_learner.py" \
 --phase restore > "$TASK_OUT/restore.log" 2>&1 || TASK_RESTORE_RC=$?
"$TASK_PYTHON" -m ray.scripts.scripts stop --force > "$TASK_OUT/ray-stop-restore.log" 2>&1 || true
[[ $TASK_RESTORE_RC == 0 ]] || exit "$TASK_RESTORE_RC"
stop_monitor
"$TASK_PYTHON" - "$TASK_OUT" <<'PY'
import json,sys
from pathlib import Path
from fluffy_bassoon.meta_rl.rlinf_native_ppo import write
root=Path(sys.argv[1]);train=json.loads((root/'train/RESULT.json').read_bytes());restore=json.loads((root/'restore/RESULT.json').read_bytes())
assert train['status']==restore['status']=='complete'
assert train['retained_optimizer_steps']==restore['retained_optimizer_steps']==15
assert restore['full_owned_restore_exact'] and restore['next_update_exact']
assert next(r for r in restore['exports'] if r['rank']==0)['qualified']
resources=json.loads((root/'resources.json').read_bytes())
assert resources['sampled_reserve_passed'],'Reject this execution profile below the8GiB reserve'
write(root/'COMPLETE.json',dict(status='complete_learner_recovery',retained_native_optimizer_steps=15,
 full_owned_restore_exact=True,next_update_exact=True,inference_export_qualified=True,
 new_simulator_controls=0,exact_all_worker_simulator_resume=False,
 scientific_source_admission=False,complete_six_task_campaign=False))
PY
