#!/usr/bin/env bash
set -Eeuo pipefail
: "${TASK_META:?}" "${TASK_CODE_PIN:?}"
TASK_STARTED=${TASK_STARTED:-$(date +%s)}
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_RUN=robocasa-eight-gpu-speed-20261102-r1
TASK_NODE_RANK=${RANK:?}
TASK_OUT=$TASK_ROOT/outputs/$TASK_RUN/S8-distributed-r2-job/node-$(printf "%02d" "$TASK_NODE_RANK")
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
original=Path(os.environ['TASK_ROOT'])/'outputs/robocasa-eight-gpu-speed-20261102-r1/S8-distributed-job'/('node-'+format(int(os.environ['RANK']),'02d'))/'clock'
saved=json.loads((original/'worker-time.json').read_text())
assert saved['worker_seconds']==14400 and all(row['status']=='finished' for row in saved['attempts'])
assert sum(row['charged_seconds'] for row in saved['attempts'])+13920<=14400
print(begin(original,saved['contract'],slice_seconds=13860,worker_seconds=14400,resume=True,now=int(os.environ['TASK_STARTED'])))
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
clock=Path(os.environ['TASK_ROOT'])/'outputs/robocasa-eight-gpu-speed-20261102-r1/S8-distributed-job'/('node-'+format(int(os.environ['RANK']),'02d'))/'clock'
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
mkdir -p /tmp/fb-parent
export TMPDIR=/tmp/fb-parent XDG_CACHE_HOME=$TASK_OUT/cache HF_HOME=$TASK_OUT/cache/huggingface TORCHINDUCTOR_CACHE_DIR=$TASK_OUT/cache/torch
TASK_SLICE=13860
TASK_REMAINING=$((TASK_STARTED+TASK_SLICE-$(date +%s)))
timeout --signal=TERM --kill-after=60s "$TASK_REMAINING" "$TASK_PY" -B "$TASK_INPUT/S8-DISTRIBUTED-R2-WORKLOAD.py"
