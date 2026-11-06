#!/usr/bin/env bash
set -Eeuo pipefail
TASK_SOURCE=${1:?reviewed source commit}
TASK_OUT=${2:?fresh BOS signals output}
TASK_CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
TASK_PYTHON=/mowenjie/fluffy-bassoon-2/envs/robotwin-vlrla-20261029-r1/bin/python
TASK_ARCHIVE=/mnt/bos/adgfyl/strong-plant-archives/robotwin-restoration-20261106-r1
export GIT_OPTIONAL_LOCKS=0
[[ $(uname -s) == Linux && $(hostname) == job-* ]]
[[ $(git -C "$TASK_CODE" rev-parse HEAD) == "$TASK_SOURCE" ]]
[[ -z $(git -C "$TASK_CODE" status --porcelain --untracked-files=no) ]]
[[ $TASK_OUT == "$TASK_ARCHIVE/actorrestore01-signals" && ! -e $TASK_OUT ]]
mountpoint -q "$TASK_ARCHIVE"
mkdir "$TASK_OUT"
exec > "$TASK_OUT/worker.log" 2>&1
TASK_STARTED=$(date +%s)
TASK_LIMIT=1080
TASK_RUN=$TASK_CODE/outputs/robotwin-sidney-matched-baselines-20261106-r1
TASK_SHARED=$TASK_CODE/outputs/robotwin-sidney-rlinf-ppo-20261105-r1
source "$TASK_RUN/runtime_environment.sh"
[[ $(git -C "$REPO_PATH" rev-parse HEAD) == c605b9d55ed93a180119b099e6f836abff293b08 ]]
[[ -z $(git -C "$REPO_PATH" status --porcelain) ]]
export TASK_OUT TASK_DEADLINE=$((TASK_STARTED+1080)) TASK_LOCAL=/tmp/rtar1
export TASK_COLD=$TASK_ARCHIVE/actorprofile01
# RLinf and Ray append /ray/session_.../sockets/plasma_store to this path.
export RAY_TMPDIR=$TASK_LOCAL/r
export NATIVE_PPO_MODEL=/mowenjie/fluffy-bassoon-2/cache/robotwin-ppo-pilot-20261105-r1-restore02/pi05-motus-effc6e1a-r1
export NATIVE_PPO_INITIALIZER_REVISION=effc6e1a99bd4d0edf9447e8ba313def67649cfe
export NATIVE_PPO_DEADLINE=$TASK_DEADLINE
finish() {
 TASK_RC=$?
 trap - EXIT
 TASK_ENDED=$(date +%s)
 printf '{"exit_code":%s,"started_unix":%s,"ended_unix":%s,"worker_gpu_seconds":%s}\n' \
  "$TASK_RC" "$TASK_STARTED" "$TASK_ENDED" "$(((TASK_ENDED-TASK_STARTED)*4))" > "$TASK_OUT/EXIT.json"
}
trap finish EXIT
"$TASK_PYTHON" "$TASK_CODE/scripts/[train]gpu_probe.py" --baige-container \
 --baige-exclusive-training --purpose training --expected-count 4 --output "$TASK_OUT/gpu-probe.json"
export LYCO_BAIGE_GPU_MAP=$TASK_OUT/gpu-probe.json
"$TASK_PYTHON" "$TASK_SHARED/host_memory.py" "$TASK_OUT" --minimum-gib 256 --once
"$TASK_PYTHON" - "$TASK_CODE" <<'PY'
import hashlib,json,os,shutil,sys
from pathlib import Path
code=Path(sys.argv[1]);out=Path(os.environ['TASK_OUT']);local=Path(os.environ['TASK_LOCAL'])
p=json.loads((code/'outputs/robotwin-restoration-20261106-r1/ACTORRESTORE01-CONTRACT.json').read_text())
assert p['closed_conservative_gpu_seconds']+p['reserved_gpu_seconds']==p['all_worst_gpu_seconds']<=180000
assert hashlib.sha256((code/'outputs/robotwin-restoration-20261106-r1/profile_motus_actor.py').read_bytes()).hexdigest()==p['profile_python_sha256']
assert len((os.environ['RAY_TMPDIR']+'/ray/session_2026-11-07_03-50-08_666653_9999999/sockets/plasma_store').encode())<107
assert not local.exists() and shutil.disk_usage(local.parent).free>=64*2**30
prior=out.parent/'actorprofile01-signals'
assert json.loads((prior/'profile/RESULT.json').read_text())['status']=='complete'
assert json.loads((prior/'EXIT.json').read_text())['exit_code']==1
receipt=json.loads((prior/'CHECKPOINT-PRESERVED.json').read_text())
assert receipt['status']=='complete' and receipt['cold_full_readback'] and receipt['local_full_stageback']
assert receipt['manifest_sha256']==p['checkpoint_manifest_sha256']
manifest=Path(os.environ['TASK_COLD'])/'MANIFEST.json'
assert hashlib.sha256(manifest.read_bytes()).hexdigest()==p['checkpoint_manifest_sha256']
assert json.loads(manifest.read_text())['source_commit']==p['checkpoint_source_commit']
assert hashlib.sha256((prior/'profile/RECOVERY-EXPECTED.json').read_bytes()).hexdigest()==p['expected_recovery_sha256']
local.mkdir();(out/'profile').mkdir()
shutil.copyfile(prior/'CHECKPOINT-PRESERVED.json',out/'CHECKPOINT-PRESERVED.json')
shutil.copyfile(prior/'profile/RECOVERY-EXPECTED.json',out/'profile/RECOVERY-EXPECTED.json')
model=Path(os.environ['NATIVE_PPO_MODEL'])
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for block in iter(lambda:stream.read(8*2**20),b''):h.update(block)
 return h.hexdigest()
assert sha(model/'model.safetensors')==p['model_sha256']
assert sha(model/'assets/pi0.5_clean_randomize_joint_training/norm_stats.json')==p['norm_stats_sha256']
(out/'admission.json').write_text(json.dumps(dict(short_ray_socket_path=True,original_checkpoint_unchanged=True,
 new_worker_pfs_output_bytes=0,source_commit=os.environ['TASK_SOURCE']))+chr(10))
PY
export XDG_CACHE_HOME=$TASK_LOCAL/cache OPENPI_DATA_HOME=$TASK_LOCAL/openpi
export HF_HOME=$TASK_LOCAL/hf TORCH_HOME=$TASK_LOCAL/torch
mkdir -p "$XDG_CACHE_HOME" "$OPENPI_DATA_HOME" "$HF_HOME" "$TORCH_HOME"
cd "$TASK_LOCAL"
export TASK_PROFILE_PHASE=restore NATIVE_PPO_OUTPUT=$TASK_OUT/restore
mkdir "$NATIVE_PPO_OUTPUT"
TASK_RESTORE_RC=0
timeout --signal=INT --kill-after=30s 900 "$TASK_PYTHON" \
 "$TASK_CODE/outputs/robotwin-restoration-20261106-r1/profile_motus_actor.py" \
 actor.micro_batch_size=64 > "$TASK_OUT/restore/console.log" 2>&1 || TASK_RESTORE_RC=$?
"$TASK_PYTHON" -m ray.scripts.scripts stop --force > "$TASK_OUT/ray-stop.log" 2>&1 || true
exit "$TASK_RESTORE_RC"
