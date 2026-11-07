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
[[ $TASK_OUT == "$TASK_ARCHIVE/reconstruction01-signals" && ! -e $TASK_OUT ]]
mkdir "$TASK_OUT"
exec > "$TASK_OUT/worker.log" 2>&1
TASK_STARTED=$(date +%s)
TASK_LIMIT=970
TASK_RUN=$TASK_CODE/outputs/robotwin-sidney-matched-baselines-20261106-r1
TASK_SHARED=$TASK_CODE/outputs/robotwin-sidney-rlinf-ppo-20261105-r1
if [[ ${CUDA_VISIBLE_DEVICES+x} != x ]]; then export CUDA_VISIBLE_DEVICES=0; fi
source "$TASK_RUN/runtime_environment.sh"
export TASK_LOCAL=/tmp/rtrec1 TASK_DEADLINE=$((TASK_STARTED+TASK_LIMIT))
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
[[ -z $(git -C "$REPO_PATH" status --porcelain --untracked-files=no) ]]
"$TASK_PYTHON" - "$TASK_DRIVER" "${BASH_SOURCE[0]}" <<'PY'
import json,os,shutil,sys
from pathlib import Path
from fluffy_bassoon.meta_rl.rlinf_native_ppo import sha,write
root=Path(sys.argv[1]);launcher=Path(sys.argv[2]).resolve();assert launcher.parent==root.resolve()
p=json.loads((root/'RECONSTRUCTION01-CONTRACT.json').read_bytes())
assert p['status']=='admitted'
assert p['closed_conservative_gpu_seconds']+p['reserved_gpu_seconds']==p['all_worst_gpu_seconds']<=180000
for key,name in [('entry_sha256','reconstruct_frozen_base.py'),('launcher_sha256',launcher.name),('observer_sha256','observe_motus_resources.py')]:assert sha(root/name)==p[key]
for name,expected in p['source_files'].items():assert sha(root.parents[1]/name)==expected
local,out=Path(os.environ['TASK_LOCAL']),Path(os.environ['TASK_OUT'])
assert not local.exists() and shutil.disk_usage(local.parent).free>=24*2**30
assert any(os.path.ismount(x) for x in [out.parent,Path('/mnt/bos/adgfyl'),Path('/mnt/bos')])
probe=out/'write-probe'
with probe.open('xb') as f:f.write(b'frozen-base-reconstruction');f.flush();os.fsync(f.fileno())
assert probe.read_bytes()==b'frozen-base-reconstruction';probe.unlink();local.mkdir()
model=Path('/mowenjie/fluffy-bassoon-2/cache/robotwin-ppo-pilot-20261105-r1-restore02/pi05-motus-effc6e1a-r1')
assert sha(model/'model.safetensors')==p['model_sha256']
assert sha(model/'assets/pi0.5_clean_randomize_joint_training/norm_stats.json')==p['norm_stats_sha256']
write(out/'admission.json',dict(source_commit=os.environ['TASK_SOURCE'],gpus=1,
 optimizer_updates=0,simulator_actions=0,new_worker_pfs_output_bytes=0,bos_write_read_fsync=True,
 reserved_gpu_seconds=p['reserved_gpu_seconds']))
PY
export XDG_CACHE_HOME=$TASK_LOCAL/cache OPENPI_DATA_HOME=$TASK_LOCAL/openpi HF_HOME=$TASK_LOCAL/hf TORCH_HOME=$TASK_LOCAL/torch TMPDIR=$TASK_LOCAL/tmp
mkdir -p "$XDG_CACHE_HOME" "$OPENPI_DATA_HOME" "$HF_HOME" "$TORCH_HOME" "$TMPDIR"
"$TASK_PYTHON" "$TASK_DRIVER/observe_motus_resources.py" "$TASK_OUT" 1 > "$TASK_OUT/resources.log" 2>&1 &
TASK_MONITOR_PID=$!
"$TASK_PYTHON" "$TASK_DRIVER/reconstruct_frozen_base.py" --stage > "$TASK_OUT/stage.log" 2>&1
TASK_REMAINING=$((TASK_DEADLINE-$(date +%s)-45))
[[ $TASK_REMAINING -ge 600 ]]
timeout --signal=INT --kill-after=20s "$TASK_REMAINING" "$TASK_PYTHON" "$TASK_DRIVER/reconstruct_frozen_base.py" \
 > "$TASK_OUT/reconstruction.log" 2>&1
stop_monitor
"$TASK_PYTHON" - "$TASK_OUT" <<'PY'
import json,sys
from pathlib import Path
from fluffy_bassoon.meta_rl.rlinf_native_ppo import sha,write
root=Path(sys.argv[1]);r=json.loads((root/'RESULT.json').read_bytes())
assert r['status']=='passed' and len(r['chunks'])==96 and r['issued_command_rows']==6144
assert r['model_sampler_rows']==6208 and r['all_teacher_decodes_exact']
assert r['optimizer_updates']==r['simulator_actions']==0
assert json.loads((root/'resources.json').read_bytes())['sampled_reserve_passed']
files={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(root.iterdir())
 if p.is_file() and (p.suffix=='.json' or p.name.endswith('.pt.gz'))}
assert sum(row['bytes'] for row in files.values())<=128*2**20
write(root/'FILES.json',dict(files=files))
write(root/'COMPLETE.json',dict(status='complete_frozen_reconstruction',issued_command_rows=6144,
 native_batch=64,original_gaussian=True,all_teacher_decodes_exact=True,
 optimizer_updates=0,simulator_actions=0,source_corpus_admitted=False,
 result_sha256=sha(root/'RESULT.json'),independent_readback_pending=True))
PY
