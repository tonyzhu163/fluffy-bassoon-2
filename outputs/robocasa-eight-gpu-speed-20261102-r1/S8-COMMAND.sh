#!/usr/bin/env bash
set -Eeuo pipefail
: "${TASK_META:?}" "${TASK_CODE_PIN:?}"
TASK_STARTED=${TASK_STARTED:-$(date +%s)}
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_RUN=robocasa-eight-gpu-speed-20261102-r1
TASK_OUT=$TASK_ROOT/outputs/$TASK_RUN/S8-job
TASK_INPUT=$TASK_META/outputs/$TASK_RUN
TASK_PY=$TASK_ROOT/setups/robocasa365-native-20261101-r1/env/bin/python
export TASK_STARTED TASK_ROOT TASK_RUN TASK_OUT TASK_INPUT TASK_PY TASK_META TASK_CODE_PIN TASK_ROLE
test ! -e "$TASK_OUT"
mkdir -p "$TASK_OUT/tmp" "$TASK_OUT/cache"
export PYTHONPATH=$TASK_META/src PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
TASK_ATTEMPT=$("$TASK_PY" -B - <<'CLOCK'
import os,json,hashlib
from pathlib import Path
from fluffy_bassoon.meta_rl.vla_source40_ppo_worker import begin
p=Path(os.environ['TASK_INPUT'])/'S8-ADMISSION.json'
assert os.environ['TASK_ROLE']=='S8'
c=dict(source_commit=os.environ['TASK_CODE_PIN'],run_id=os.environ['TASK_RUN']+'-S8',ticket_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),gpu_count=8,wall_ceiling_seconds=14400,maximum_gpu_seconds=115200,actions=958250)
print(begin(Path(os.environ['TASK_OUT'])/'clock',c,slice_seconds=14340,worker_seconds=14400,now=int(os.environ['TASK_STARTED'])))
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
row['charged_gpu_seconds']=row['charged_seconds']*8
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
mkdir -p /tmp/fb-parent
export TMPDIR=/tmp/fb-parent XDG_CACHE_HOME=$TASK_OUT/cache HF_HOME=$TASK_OUT/cache/huggingface TORCHINDUCTOR_CACHE_DIR=$TASK_OUT/cache/torch
TASK_SLICE=14340
TASK_REMAINING=$((TASK_STARTED+TASK_SLICE-$(date +%s)))
timeout --signal=TERM --kill-after=60s "$TASK_REMAINING" "$TASK_PY" -B "$TASK_INPUT/S8-WORKLOAD.py"
