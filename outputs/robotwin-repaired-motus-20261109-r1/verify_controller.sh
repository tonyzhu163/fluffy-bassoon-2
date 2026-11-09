#!/usr/bin/env bash
set -Eeuo pipefail
TASK_SOURCE=${1:?reviewed source commit}
TASK_CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
TASK_DRIVER=$TASK_CODE/outputs/robotwin-restoration-20261106-r1
TASK_OUT=/mnt/bos/adgfyl/strong-plant-archives/robotwin-repaired-motus-20261109-r1/controller-pilot01
TASK_LOCAL=/tmp/rt-repaired-controller-pilot01
TASK_PYTHON=/mowenjie/fluffy-bassoon-2/envs/robotwin-vlrla-20261029-r1/bin/python
export TASK_SOURCE TASK_CODE TASK_DRIVER TASK_OUT TASK_LOCAL GIT_OPTIONAL_LOCKS=0
[[ $(uname -s) == Linux && $(hostname) == job-* ]]
[[ $(git -C "$TASK_CODE" rev-parse HEAD) == "$TASK_SOURCE" ]]
[[ -z $(git -C "$TASK_CODE" status --porcelain --untracked-files=no) ]]
[[ ! -e $TASK_OUT && ! -e $TASK_LOCAL ]]
mkdir -p "$(dirname "$TASK_OUT")"
mkdir "$TASK_OUT" "$TASK_LOCAL"
exec > "$TASK_OUT/worker.log" 2>&1
TASK_STARTED=$(date +%s)
export TASK_DEADLINE=$((TASK_STARTED+4800))
finish() {
 TASK_RC=$?
 trap - EXIT
 TASK_ENDED=$(date +%s)
 printf '{"exit_code":%s,"started_unix":%s,"ended_unix":%s,"worker_gpu_seconds":%s}\n' \
  "$TASK_RC" "$TASK_STARTED" "$TASK_ENDED" "$((TASK_ENDED-TASK_STARTED))" > "$TASK_OUT/EXIT.json"
}
trap finish EXIT
TASK_RUN=$TASK_CODE/outputs/robotwin-sidney-matched-baselines-20261106-r1
TASK_SHARED=$TASK_CODE/outputs/robotwin-sidney-rlinf-ppo-20261105-r1
TASK_LIMIT=4800
if [[ ${CUDA_VISIBLE_DEVICES+x} != x ]]; then export CUDA_VISIBLE_DEVICES=0; fi
source "$TASK_RUN/runtime_environment.sh"
export PYTHONPATH=$TASK_DRIVER:$PYTHONPATH
export NATIVE_PPO_MODEL=/mowenjie/fluffy-bassoon-2/cache/robotwin-ppo-pilot-20261105-r1-restore02/pi05-motus-effc6e1a-r1
export NATIVE_PPO_INITIALIZER_REVISION=effc6e1a99bd4d0edf9447e8ba313def67649cfe
"$TASK_PYTHON" "$TASK_CODE/scripts/[train]gpu_probe.py" --baige-container \
 --purpose evaluation --expected-count 1 --output "$TASK_OUT/gpu-probe.json"
export LYCO_BAIGE_GPU_MAP=$TASK_OUT/gpu-probe.json
"$TASK_PYTHON" "$TASK_SHARED/host_memory.py" "$TASK_OUT" --minimum-gib 128 --once
"$TASK_PYTHON" - <<'PY'
import json,os,shutil,subprocess
from pathlib import Path
from fluffy_bassoon.meta_rl.rlinf_native_ppo import write
for folder,expected in ((os.environ['REPO_PATH'],'c605b9d55ed93a180119b099e6f836abff293b08'),
                        (os.environ['ROBOTWIN_PATH'],'dca9ec682688821c42944eacd7f5dd9bfb15396f')):
 assert subprocess.check_output(['git','-C',folder,'rev-parse','HEAD'],text=True).strip()==expected
 assert not subprocess.check_output(['git','-C',folder,'status','--porcelain','--untracked-files=no'],text=True).strip()
probe=json.loads((Path(os.environ['TASK_OUT'])/'gpu-probe.json').read_bytes())
assert probe['status']=='passed' and len(probe['devices'])==1
assert probe['devices'][0]['cuda_total_bytes']>=90*2**30
assert shutil.disk_usage(os.environ['TASK_LOCAL']).free>=32*2**30
path=Path(os.environ['TASK_OUT'])/'write-probe'
with path.open('xb') as f:f.write(b'repaired-controller-pilot');f.flush();os.fsync(f.fileno())
assert path.read_bytes()==b'repaired-controller-pilot';path.unlink()
write(Path(os.environ['TASK_OUT'])/'admission.json',dict(source_commit=os.environ['TASK_SOURCE'],
 write_read_fsync=True,new_worker_pfs_output_bytes=0,reserved_gpu_seconds=5400,
 model_and_simulator_construction_pending=True))
PY
export XDG_CACHE_HOME=$TASK_LOCAL/cache OPENPI_DATA_HOME=$TASK_LOCAL/openpi
export HF_HOME=$TASK_LOCAL/hf TORCH_HOME=$TASK_LOCAL/torch TMPDIR=$TASK_LOCAL/tmp
mkdir -p "$XDG_CACHE_HOME" "$OPENPI_DATA_HOME" "$HF_HOME" "$TORCH_HOME" "$TMPDIR" /root/.sapien/physx
if [[ ! -e /root/.sapien/physx/105.1-physx-5.3.1.patch0 ]]; then
 ln -s /mowenjie/fluffy-bassoon/inputs/maniskill-scene-assets-20261028-r1/physx/105.1-physx-5.3.1.patch0 \
 /root/.sapien/physx/105.1-physx-5.3.1.patch0
fi
cd "$TASK_LOCAL"
ln -s "$ROBOTWIN_PATH/assets" assets
if [[ -f "$ROBOTWIN_PATH/Times New Roman.ttf" ]]; then ln -s "$ROBOTWIN_PATH/Times New Roman.ttf" .; fi
TASK_REMAINING=$((TASK_DEADLINE-$(date +%s)-60))
[[ $TASK_REMAINING -gt 300 ]]
timeout --signal=INT --kill-after=20s "$TASK_REMAINING" "$TASK_PYTHON" \
 "$TASK_CODE/outputs/robotwin-repaired-motus-20261109-r1/verify_controller.py" \
 --output-dir "$TASK_OUT" --local-dir "$TASK_LOCAL" --max-cases 16 > "$TASK_OUT/evaluation.log" 2>&1
