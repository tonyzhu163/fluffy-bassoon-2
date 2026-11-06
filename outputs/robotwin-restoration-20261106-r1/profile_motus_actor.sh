#!/usr/bin/env bash
set -Eeuo pipefail
TASK_SOURCE=${1:?reviewed source commit}
TASK_OUT=${2:?fresh hot output}
TASK_CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
TASK_PYTHON=/mowenjie/fluffy-bassoon-2/envs/robotwin-vlrla-20261029-r1/bin/python
[[ $(uname -s) == Linux && $(hostname) == job-* ]]
[[ $(git -C "$TASK_CODE" rev-parse HEAD) == "$TASK_SOURCE" ]]
[[ -z $(git -C "$TASK_CODE" status --porcelain --untracked-files=no) && ! -e $TASK_OUT ]]
mkdir -p "$TASK_OUT"
exec > "$TASK_OUT/worker.log" 2>&1
TASK_STARTED=$(date +%s)
TASK_RUN=$TASK_CODE/outputs/robotwin-sidney-matched-baselines-20261106-r1
TASK_SHARED=$TASK_CODE/outputs/robotwin-sidney-rlinf-ppo-20261105-r1
TASK_LIMIT=2640
source "$TASK_RUN/runtime_environment.sh"
export TASK_OUT TASK_DEADLINE=$((TASK_STARTED+2640))
export TASK_LOCAL=/tmp/mwj-robotwin-motus-actorprofile01
export TASK_COLD=/mnt/bos/adgfyl/strong-plant-archives/robotwin-restoration-20261106-r1/actorprofile01
export RAY_TMPDIR=$TASK_LOCAL/ray
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
"$TASK_PYTHON" - "$TASK_CODE" "$TASK_OUT" <<'PY'
import json,os,shutil,sys
from pathlib import Path
from hashlib import sha256
code,out=map(Path,sys.argv[1:]);p=json.loads((code/'outputs/robotwin-restoration-20261106-r1/ACTORPROFILE01-CONTRACT.json').read_text())
assert p['closed_conservative_gpu_seconds']+p['reserved_gpu_seconds']==p['all_worst_gpu_seconds']<=180000
assert shutil.disk_usage(out).free>=286.5*2**30
local,cold=Path(os.environ['TASK_LOCAL']),Path(os.environ['TASK_COLD'])
assert not local.exists() and not cold.exists()
assert shutil.disk_usage(local.parent).free>=64*2**30
bucket=Path('/mnt/bos/adgfyl')
assert bucket.is_dir() and (os.path.ismount(bucket) or os.path.ismount(bucket.parent)), 'Persistent BOS mount required before model work'
probe=bucket/'strong-plant-archives/robotwin-restoration-20261106-r1/actorprofile01-write-probe'
with probe.open('xb') as stream:stream.write(b'robotwin-native-actor-checkpoint\n');stream.flush();os.fsync(stream.fileno())
assert probe.read_bytes()==b'robotwin-native-actor-checkpoint\n'
probe.unlink();local.mkdir()
model=Path(os.environ['NATIVE_PPO_MODEL'])
def sha(path):
 h=sha256()
 with path.open('rb') as stream:
  for block in iter(lambda:stream.read(8*2**20),b''):h.update(block)
 return h.hexdigest()
assert sha(model/'model.safetensors')==p['model_sha256']
assert sha(model/'assets/pi0.5_clean_randomize_joint_training/norm_stats.json')==p['norm_stats_sha256']
(out/'admission.json').write_text(json.dumps(dict(hot_free_gib=shutil.disk_usage(out).free/2**30,
 local_free_gib=shutil.disk_usage(local).free/2**30,bos_mounted=True,bos_write_read_fsync=True,
 model_hashes_verified=True,other_and_floor_gib=286,hot_output_cap_gib=.5,cold_checkpoint_cap_gib=32))+'\n')
PY
export TASK_PROFILE_PHASE=profile NATIVE_PPO_OUTPUT=$TASK_OUT/profile
mkdir "$NATIVE_PPO_OUTPUT"
TASK_PROFILE_RC=0
timeout --signal=INT --kill-after=30s 1920 "$TASK_PYTHON" \
 "$TASK_CODE/outputs/robotwin-restoration-20261106-r1/profile_motus_actor.py" \
 actor.micro_batch_size=64 > "$TASK_OUT/profile/console.log" 2>&1 || TASK_PROFILE_RC=$?
printf '{"exit_code":%s}\n' "$TASK_PROFILE_RC" > "$TASK_OUT/profile/EXIT.json"
# This scheduler container is exclusive to this job; stop only its local Ray.
"$TASK_PYTHON" -m ray.scripts.scripts stop --force > "$TASK_OUT/ray-stop-profile.log" 2>&1 || true
if [[ -e $TASK_OUT/CHECKPOINT-PRESERVED.json && -e $TASK_OUT/profile/RECOVERY-EXPECTED.json ]]; then
 export TASK_PROFILE_PHASE=restore NATIVE_PPO_OUTPUT=$TASK_OUT/restore
 mkdir "$NATIVE_PPO_OUTPUT"
 timeout --signal=INT --kill-after=30s 570 "$TASK_PYTHON" \
  "$TASK_CODE/outputs/robotwin-restoration-20261106-r1/profile_motus_actor.py" \
  actor.micro_batch_size=64 > "$TASK_OUT/restore/console.log" 2>&1
 "$TASK_PYTHON" -m ray.scripts.scripts stop --force > "$TASK_OUT/ray-stop-restore.log" 2>&1 || true
fi
exit "$TASK_PROFILE_RC"
