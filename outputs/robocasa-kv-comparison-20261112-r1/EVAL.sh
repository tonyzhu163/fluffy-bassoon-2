#!/usr/bin/env bash
# Complete the saved three-task KV comparison; no refitting or altered cases.
set -Eeuo pipefail
RUNNER_PIN=${1:?reviewed main commit}
OBJECTIVE=${2:?dpt, dit, or ic_iql}
ATTEMPT=${3:?fresh attempt identity}
case "$OBJECTIVE" in dpt|dit|ic_iql) ;; *) exit 2 ;; esac
[[ $ATTEMPT =~ ^[a-z0-9-]+$ ]]
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
ROOT=/mowenjie/fluffy-bassoon-2
RUN=robocasa-kv-comparison-20261112-r1
OLD=robocasa-three-task-kv-velocity-meta-20261105-r1
SCI=$ROOT/sources/steering-packed-87984718
SCI_PIN=87984718c9199797d31a158c47225cc695623afe
PY=$ROOT/setups/robocasa365-native-20261101-r1/env/bin/python
[[ $(uname -s) == Linux && ${HOSTNAME:-} == job-* ]]
[[ $(git -C "$HERE" rev-parse HEAD) == "$RUNNER_PIN" ]]
[[ -z $(git -C "$HERE" status --porcelain --untracked-files=no) ]]
[[ $(git -C "$SCI" rev-parse HEAD) == "$SCI_PIN" ]]
export RLINF_PATH=$ROOT/sources/critic-selected-a1229239-r1
[[ $(git -C "$RLINF_PATH" rev-parse HEAD) == 8f9bfa7661928ef8c17494b04f4d7bf3125634e1 ]]
export PYTHONPATH=$SCI/src:$RLINF_PATH
export OPENPI_DATA_HOME=$ROOT/setups/robocasa365-native-20261101-r1/cache/.cache/openpi
export ROBOCASA_ASSETS_PATH=$ROOT/setups/robocasa365-native-20261101-r1/assets
export STEERING_MIN_FREE_GIB=220 STEERING_WAVE_ALLOWANCE_SECONDS=600
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NCCL_P2P_DISABLE=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl FLUFFY_GPU_EXECUTION=baige
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export STEERING_CODE=$SCI STEERING_COMMIT=$SCI_PIN
TASK_LOG=$ROOT/outputs/$RUN/jobs/$OBJECTIVE-$ATTEMPT
[[ ! -e $TASK_LOG ]]
mkdir -p "$TASK_LOG"
exec > >(tee "$TASK_LOG/console.log") 2>&1
BEGAN=$(date +%s)
trap 'RC=$?; printf "{\"exit_code\":%s,\"started_unix\":%s,\"ended_unix\":%s,\"GPUs\":1}\n" "$RC" "$BEGAN" "$(date +%s)" >"$TASK_LOG/EXIT.json"' EXIT
if [[ ${CUDA_VISIBLE_DEVICES+x} != x ]]; then
    visible_uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader)
    [[ "$visible_uuid" == "${NVIDIA_VISIBLE_DEVICES:?}" && "$visible_uuid" == GPU-* && "$visible_uuid" != *$'\n'* && "$visible_uuid" != *,* ]]
    export CUDA_VISIBLE_DEVICES=0
fi
"$PY" -B - "$ROOT" "$RUN" "$OLD" "$OBJECTIVE" "$RUNNER_PIN" "$HERE" <<'PY'
import hashlib,json,shutil,sys,time
from pathlib import Path
root,run,old,objective,pin,code=sys.argv[1:]
root=Path(root);run=root/'outputs'/run
contract=json.loads((Path(code)/'outputs'/run.name/'CONTRACT.json').read_text())
assert contract['approved_historical_cap_GPU_hours']==170
assert contract['reserved_evaluation_GPU_hours']==22.5
assert contract['historical_conservative_upper_bound_GPU_hours']+22.5<=170
assert contract['all_new_work_reserved_GPU_hours']<=50
assert shutil.disk_usage(run).free >= 240*2**30, '220 GiB floor plus20 GiB shared readout buffer'
heartbeat=json.loads((run/'monitor/HOT-STORAGE.json').read_text())
assert heartbeat['packed_archive_enabled'] and time.time()-heartbeat['checked_unix']<120
idx=json.loads((root/'outputs'/old/'finalized-fits-20261106-r1/INDEX.json').read_text())
for name in ('candidate.pt','training-status.json','contract.json'):
    a=idx['models']['kv_bridge/'+objective]['artifacts'][name]
    p=Path(a['path'])
    with p.open('rb') as f: assert hashlib.file_digest(f,'sha256').hexdigest()==a['sha256']
print(json.dumps(dict(admitted=True,runner_commit=pin,objective=objective,
    candidate_sha256=idx['models']['kv_bridge/'+objective]['artifacts']['candidate.pt']['sha256'],
    historical_cap_GPU_hours=170,reserved_new_evaluation_GPU_hours=22.5,
    maximum_this_job_GPU_hours=7.5,free_PFS_GiB=shutil.disk_usage(run).free/2**30)))
PY
PROBE=$TASK_LOG/GPU-PROBE.json
"$PY" -B "$HERE/scripts/[train]gpu_probe.py" --expected-count 1 --purpose evaluation --baige-container --output "$PROBE"
DEADLINE=$((BEGAN+450*60-180))
for SPLIT in ID OOD; do
    OUT=$ROOT/outputs/$RUN/readouts/kv_bridge/$OBJECTIVE/$SPLIT
    REMAINING=$((DEADLINE-$(date +%s)))
    [[ $REMAINING -gt 600 ]]
    timeout --signal=TERM --kill-after=15s "${REMAINING}s" "$PY" -B "$HERE/outputs/$RUN/READOUT.py" \
        --code "$SCI" --commit "$SCI_PIN" \
        --fit "$ROOT/outputs/$OLD/finalized-fits-20261106-r1/kv_bridge/$OBJECTIVE" \
        --output "$OUT" --probe "$PROBE" --interface kv_bridge --objective "$OBJECTIVE" \
        --split "$SPLIT" --deadline "$DEADLINE" --attempt "$ATTEMPT"
    [[ -f "$OUT/RESULT.json" ]] || { cat "$OUT/OPERATIONAL-CHECKPOINT.json"; exit 3; }
done
