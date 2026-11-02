#!/usr/bin/env bash
set -Eeuo pipefail
: "${TASK_META:?}" "${TASK_CODE_PIN:?}" "${RANK:?}" "${WORLD_SIZE:?}"
TASK_STARTED=${TASK_STARTED:-$(date +%s)}
export TASK_ROOT=/mowenjie/fluffy-bassoon-2 TASK_RUN=robocasa-flora-rubric-replay-20261102-r1
export TASK_NODE_RANK=$RANK TASK_STARTED
test "$WORLD_SIZE" = 2
TASK_NODE=$(printf '%02d' "$TASK_NODE_RANK")
export TASK_OUT=$TASK_ROOT/outputs/$TASK_RUN/node-$TASK_NODE
export TASK_INPUT=$TASK_META/outputs/$TASK_RUN
export TASK_PY=$TASK_ROOT/setups/robocasa365-native-20261101-r1/env/bin/python
test ! -e "$TASK_OUT"
mkdir -p "$TASK_OUT/cache"
export PYTHONPATH=$TASK_META/src PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
TASK_ATTEMPT=$("$TASK_PY" -B - <<'PY'
import os
from pathlib import Path
from fluffy_bassoon.meta_rl.vla_source40_ppo_worker import begin
contract=dict(run_id=os.environ['TASK_RUN'],source_commit=os.environ['TASK_CODE_PIN'],node_rank=int(os.environ['TASK_NODE_RANK']),
    node_count=2,gpu_count=2,global_gpu_count=4,wall_ceiling_seconds=3300,maximum_gpu_seconds=6600,
    global_maximum_gpu_seconds=13200,robot_actions=0,scope='634 retained videos; original and one frozen rubric; no PPO')
print(begin(Path(os.environ['TASK_OUT'])/'clock',contract,slice_seconds=3240,worker_seconds=3300,now=int(os.environ['TASK_STARTED'])))
PY
)
export TASK_ATTEMPT
exec > >(tee "$TASK_OUT/console.log") 2>&1
cleanup() {
 trap - EXIT INT TERM
 "$TASK_PY" -B - <<'PY' || true
import os,json
from pathlib import Path
from fluffy_bassoon.meta_rl.vla_source40_ppo_worker import finish
p=Path(os.environ['TASK_OUT']);r=finish(p/'clock',os.environ['TASK_ATTEMPT'],int(os.environ['TASK_RC']))
r['charged_gpu_seconds']=2*r['charged_seconds'];(p/'terminal-charge.json').write_text(json.dumps(r,indent=2)+'\n')
PY
 printf '%s\n' "$TASK_RC" > "$TASK_OUT/exit-code.txt"
 exit "$TASK_RC"
}
export TASK_RC=0
trap 'TASK_RC=$?;export TASK_RC;cleanup' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled NCCL_P2P_DISABLE=1
export TMPDIR=/tmp/rubric-parent-$TASK_NODE XDG_CACHE_HOME=$TASK_OUT/cache TORCHINDUCTOR_CACHE_DIR=$TASK_OUT/cache/torch
mkdir -p "$TMPDIR"
TASK_REMAINING=$((TASK_STARTED+3240-$(date +%s)))
test "$TASK_REMAINING" -gt 0
timeout --signal=TERM --kill-after=30s "$TASK_REMAINING" "$TASK_PY" -B "$TASK_INPUT/WORKLOAD.py"
