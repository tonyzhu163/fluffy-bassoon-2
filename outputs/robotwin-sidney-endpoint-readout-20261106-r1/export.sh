#!/usr/bin/env bash
set -Eeuo pipefail
TASK_SOURCE=${1:?reviewed source commit}
TASK_OUT=${2:?fresh output identity}
TASK_FUNDING=${3:?this attempt funding contract}
TASK_CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
TASK_ENDPOINT=$TASK_CODE/outputs/robotwin-sidney-endpoint-readout-20261106-r1
TASK_RUN=$TASK_CODE/outputs/robotwin-sidney-matched-baselines-20261106-r1
TASK_SHARED=$TASK_CODE/outputs/robotwin-sidney-rlinf-ppo-20261105-r1
TASK_PYTHON=/mowenjie/fluffy-bassoon-2/envs/robotwin-vlrla-20261029-r1/bin/python
TASK_LIMIT=1740
[[ $(uname -s) == Linux && $(hostname) == job-* ]]
[[ $(git -C "$TASK_CODE" rev-parse HEAD) == "$TASK_SOURCE" ]]
[[ -z $(git -C "$TASK_CODE" status --porcelain --untracked-files=no) && ! -e $TASK_OUT ]]
mkdir -p "$TASK_OUT"
exec > "$TASK_OUT/worker.log" 2>&1
source "$TASK_RUN/runtime_environment.sh"
TASK_STARTED=$(date +%s)
TASK_DEADLINE=$((TASK_STARTED+1740))
export TASK_DEADLINE
TASK_MEMORY_PID=
finish() {
  TASK_RC=$?
  trap - EXIT
  if [[ -n $TASK_MEMORY_PID ]]; then kill "$TASK_MEMORY_PID" 2>/dev/null || true; fi
  TASK_ENDED=$(date +%s)
  printf '{"exit_code":%s,"started_unix":%s,"ended_unix":%s,"worker_gpu_seconds":%s}\n' \
    "$TASK_RC" "$TASK_STARTED" "$TASK_ENDED" "$((4*(TASK_ENDED-TASK_STARTED)))" > "$TASK_OUT/EXIT.json"
}
trap finish EXIT
"$TASK_PYTHON" "$TASK_CODE/scripts/[train]gpu_probe.py" --baige-container \
  --purpose evaluation --expected-count 4 --output "$TASK_OUT/gpu-probe.json"
export LYCO_BAIGE_GPU_MAP=$TASK_OUT/gpu-probe.json
"$TASK_PYTHON" "$TASK_SHARED/host_memory.py" "$TASK_OUT" --minimum-gib 256 --once
"$TASK_PYTHON" "$TASK_SHARED/host_memory.py" "$TASK_OUT" --minimum-gib 256 > "$TASK_OUT/host-memory.log" 2>&1 &
TASK_MEMORY_PID=$!
[[ $(git -C "$REPO_PATH" rev-parse HEAD) == c605b9d55ed93a180119b099e6f836abff293b08 ]]
[[ -z $(git -C "$REPO_PATH" status --porcelain --untracked-files=no) ]]
"$TASK_PYTHON" - "$TASK_FUNDING" "$TASK_OUT" <<'PY'
import json,os,shutil,sys
from pathlib import Path
plan_path,out=map(Path,sys.argv[1:]);p=json.loads(plan_path.read_text())
assert p['carry_forward_worst_gpu_seconds']+p['export_reserved_gpu_seconds']+p['readout_reserved_gpu_seconds']<=180000
assert p['export_reserved_gpu_seconds']==7440 and p['optimizer_updates']==0
assert shutil.disk_usage(out).free>=334*2**30
previous=Path('/mowenjie/fluffy-bassoon-2/outputs/robotwin-sidney-initial-performance-20261106-r1/eval01')
assert json.loads((previous/'RESULT.json').read_text())['status']=='complete'
assert json.loads((previous/'EXIT.json').read_text())['exit_code']==0
probe=out/'write-probe'
with probe.open('xb') as stream:stream.write(b'frozen-native-endpoint-export\n');stream.flush();os.fsync(stream.fileno())
probe.unlink()
(out/'admission.json').write_text(json.dumps(dict(status='complete',gpus=4,
    free_bytes=shutil.disk_usage(out).free,required_free_bytes=334*2**30,
    worst_all_reserved_gpu_seconds=p['carry_forward_worst_gpu_seconds']+7440+p['readout_reserved_gpu_seconds'],
    optimizer_updates=0,new_simulator_controls=0))+'\n')
PY
for TASK_METHOD in ppo grpo; do
  TASK_REMAINING=$((TASK_DEADLINE-$(date +%s)-30))
  [[ $TASK_REMAINING -gt 120 ]]
  timeout --signal=INT --kill-after=20s "$TASK_REMAINING" "$TASK_PYTHON" "$TASK_ENDPOINT/export_policy.py" \
    --algorithm "$TASK_METHOD" --output "$TASK_OUT/$TASK_METHOD" > "$TASK_OUT/$TASK_METHOD.log" 2>&1
  "$TASK_PYTHON" -m ray.scripts.scripts stop --force > "$TASK_OUT/ray-cleanup-$TASK_METHOD.log" 2>&1
done
"$TASK_PYTHON" - "$TASK_OUT" <<'PY'
import json,sys
from pathlib import Path
r=Path(sys.argv[1]);arms={m:json.loads((r/m/'RESULT.json').read_text()) for m in ('ppo','grpo')}
assert all(v['status']=='complete' and v['optimizer_steps_per_rank']==[40]*4 for v in arms.values())
used=sum(p.stat().st_blocks*512 for p in r.rglob('*') if p.is_file());assert used<=24*2**30
(r/'RESULT.json').write_text(json.dumps(dict(status='complete',arms=arms,optimizer_updates=0,
    new_simulator_controls=0,allocated_output_bytes=used,plain_inference_weights_qualified=True),indent=2)+'\n')
PY
