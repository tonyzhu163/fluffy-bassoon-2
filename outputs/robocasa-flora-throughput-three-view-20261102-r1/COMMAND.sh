#!/usr/bin/env bash
set -Eeuo pipefail
: "${TASK_META:?}" "${TASK_CODE_PIN:?}" "${TASK_CORE_QUALIFICATION:?}" "${RANK:?}" "${WORLD_SIZE:?}"
TASK_STARTED=${TASK_STARTED:-$(date +%s)}
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_RUN=robocasa-flora-throughput-three-view-20261102-r1
TASK_NODE_RANK=$RANK
TASK_NODE_COUNT=$WORLD_SIZE
test "$TASK_NODE_COUNT" = 4
test "$TASK_NODE_RANK" -ge 0
test "$TASK_NODE_RANK" -lt 4
TASK_NODE=$(printf '%02d' "$TASK_NODE_RANK")
TASK_OUT=$TASK_ROOT/outputs/$TASK_RUN/node-$TASK_NODE
TASK_INPUT=$TASK_META/outputs/$TASK_RUN
TASK_PY=$TASK_ROOT/setups/robocasa365-native-20261101-r1/env/bin/python
export TASK_STARTED TASK_ROOT TASK_RUN TASK_OUT TASK_INPUT TASK_PY TASK_META TASK_CODE_PIN TASK_ROLE TASK_NODE_RANK TASK_NODE_COUNT TASK_CORE_QUALIFICATION
test ! -e "$TASK_OUT"
mkdir -p "$TASK_OUT/tmp" "$TASK_OUT/cache"
export PYTHONPATH=$TASK_META/src PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
TASK_ATTEMPT=$("$TASK_PY" -B - <<'CLOCK'
import os,json,hashlib
from pathlib import Path
from fluffy_bassoon.meta_rl.vla_source40_ppo_worker import begin
p=Path(os.environ['TASK_INPUT'])/'ADMISSION.json'
assert os.environ.get('TASK_ROLE', 'Flora-batch')=='Flora-batch'
c=dict(source_commit=os.environ['TASK_CODE_PIN'],run_id=os.environ['TASK_RUN']+'-Flora-batch-node'+os.environ['TASK_NODE_RANK'],ticket_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),gpu_count=2,global_gpu_count=8,node_rank=int(os.environ['TASK_NODE_RANK']),node_count=4,wall_ceiling_seconds=3000,maximum_gpu_seconds=6000,global_maximum_gpu_seconds=24000,actions=0)
print(begin(Path(os.environ['TASK_OUT'])/'clock',c,slice_seconds=2940,worker_seconds=3000,now=int(os.environ['TASK_STARTED'])))
CLOCK
)
export TASK_ATTEMPT
exec > >(tee "$TASK_OUT/console.log") 2>&1
cleanup() {
 trap - EXIT INT TERM
 "$TASK_PY" -B - <<'CLOSE' || true
import os,json
from pathlib import Path
from fluffy_bassoon.meta_rl.vla_source40_ppo_worker import finish
clock=Path(os.environ['TASK_OUT'])/'clock'
row=finish(clock,os.environ['TASK_ATTEMPT'],int(os.environ['TASK_RC']))
row['charged_gpu_seconds']=row['charged_seconds']*2
(Path(os.environ['TASK_OUT'])/'terminal-charge.json').write_text(json.dumps(row,indent=2)+'\n');print(json.dumps(row))
CLOSE
 printf '%s\n' "$TASK_RC" > "$TASK_OUT/exit-code.txt"
 exit "$TASK_RC"
}
export TASK_RC=0
trap 'TASK_RC=$?;export TASK_RC;cleanup' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
TASK_RLINF=$TASK_ROOT/sources/robocasa365-public-pi05-nnx-leaves-20261101-r1/RLinf
export TASK_RLINF RLINF_PATH=$TASK_RLINF
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled FLUFFY_GPU_EXECUTION=baige
export NCCL_P2P_DISABLE=1
mkdir -p /tmp/fb-3v1-parent-$TASK_NODE
export TMPDIR=/tmp/fb-3v1-parent-$TASK_NODE XDG_CACHE_HOME=$TASK_OUT/cache HF_HOME=$TASK_OUT/cache/huggingface TORCHINDUCTOR_CACHE_DIR=$TASK_OUT/cache/torch
TASK_SLICE=2940
TASK_REMAINING=$((TASK_STARTED+TASK_SLICE-$(date +%s)))
timeout --signal=TERM --kill-after=60s "$TASK_REMAINING" "$TASK_PY" -B "$TASK_INPUT/WORKLOAD.py"
