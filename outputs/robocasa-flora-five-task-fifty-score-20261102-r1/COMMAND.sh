#!/usr/bin/env bash
set -Eeuo pipefail
: "${TASK_META:?}" "${TASK_CODE_PIN:?}" "${RANK:?}" "${WORLD_SIZE:?}"
TASK_STARTED=${TASK_STARTED:-$(date +%s)}
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_RUN=robocasa-flora-five-task-fifty-score-20261102-r1
TASK_NODE_RANK=$RANK
test "$WORLD_SIZE" = 4
test "$TASK_NODE_RANK" -ge 0
test "$TASK_NODE_RANK" -lt 4
TASK_NODE=$(printf '%02d' "$TASK_NODE_RANK")
TASK_OUT=$TASK_ROOT/outputs/$TASK_RUN/node-$TASK_NODE
TASK_INPUT=$TASK_META/outputs/$TASK_RUN
TASK_PY=$TASK_ROOT/setups/robocasa365-native-20261101-r1/env/bin/python
export TASK_STARTED TASK_ROOT TASK_RUN TASK_OUT TASK_INPUT TASK_PY TASK_META TASK_CODE_PIN TASK_NODE_RANK
test ! -e "$TASK_OUT"
mkdir -p "$TASK_OUT/tmp" "$TASK_OUT/cache"
export PYTHONPATH=$TASK_META/src PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
TASK_ATTEMPT=$("$TASK_PY" -B - <<'CLOCK'
import json,os
from pathlib import Path
from fluffy_bassoon.meta_rl.rlinf_flora_ppo import sha
from fluffy_bassoon.meta_rl.vla_source40_ppo_worker import begin
path=Path(os.environ['TASK_INPUT'])/'ADMISSION.json'
ticket=json.loads(path.read_text())
assert ticket['admitted'] is True and ticket['maximum_gpu_seconds']==28800
assert ticket['inclusive_wall_seconds_per_node']==3600 and ticket['robot_actions']==0
contract=dict(run_id=os.environ['TASK_RUN'],source_commit=os.environ['TASK_CODE_PIN'],
    admission_sha256=sha(path),node_rank=int(os.environ['TASK_NODE_RANK']),node_count=4,
    gpu_count=2,global_gpu_count=8,wall_ceiling_seconds=3600,maximum_gpu_seconds=7200,
    global_maximum_gpu_seconds=28800,robot_actions=0,scope='five fixed native50 baseline scoring only')
print(begin(Path(os.environ['TASK_OUT'])/'clock',contract,slice_seconds=3540,
    worker_seconds=3600,now=int(os.environ['TASK_STARTED'])))
CLOCK
)
export TASK_ATTEMPT
exec > >(tee "$TASK_OUT/console.log") 2>&1
cleanup() {
 trap - EXIT INT TERM
 "$TASK_PY" -B - <<'CLOSE' || true
import json,os
from pathlib import Path
from fluffy_bassoon.meta_rl.vla_source40_ppo_worker import finish
out=Path(os.environ['TASK_OUT'])
row=finish(out/'clock',os.environ['TASK_ATTEMPT'],int(os.environ['TASK_RC']))
row['charged_gpu_seconds']=row['charged_seconds']*2
(out/'terminal-charge.json').write_text(json.dumps(row,indent=2)+'\n')
print(json.dumps(row))
CLOSE
 printf '%s\n' "$TASK_RC" > "$TASK_OUT/exit-code.txt"
 exit "$TASK_RC"
}
export TASK_RC=0
trap 'TASK_RC=$?;export TASK_RC;cleanup' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled FLUFFY_GPU_EXECUTION=baige NCCL_P2P_DISABLE=1
mkdir -p /tmp/fb-50-parent-$TASK_NODE
export TMPDIR=/tmp/fb-50-parent-$TASK_NODE XDG_CACHE_HOME=$TASK_OUT/cache HF_HOME=$TASK_OUT/cache/huggingface TORCHINDUCTOR_CACHE_DIR=$TASK_OUT/cache/torch
TASK_REMAINING=$((TASK_STARTED+3540-$(date +%s)))
test "$TASK_REMAINING" -gt 0
timeout --signal=TERM --kill-after=60s "$TASK_REMAINING" "$TASK_PY" -B "$TASK_INPUT/WORKLOAD.py"
