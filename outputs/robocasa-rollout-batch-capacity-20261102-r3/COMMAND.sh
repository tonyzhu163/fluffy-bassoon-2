#!/usr/bin/env bash
set -Eeuo pipefail
: "${TASK_META:?}" "${TASK_CODE_PIN:?}"
TASK_STARTED=${TASK_STARTED:-$(date +%s)}
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_RUN=robocasa-rollout-batch-capacity-20261102-r3
TASK_OUT=$TASK_ROOT/outputs/$TASK_RUN/job/node-$(printf '%02d' "${RANK:?}")
TASK_PY=$TASK_ROOT/setups/robocasa365-native-20261101-r1/env/bin/python
export TASK_STARTED TASK_ROOT TASK_RUN TASK_OUT TASK_PY TASK_META TASK_CODE_PIN
test ! -e "$TASK_OUT"; mkdir -p "$TASK_OUT/cache"
export PYTHONPATH=$TASK_META/src PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
TASK_ATTEMPT=$("$TASK_PY" -B - <<'CLOCK'
import os
from pathlib import Path
from fluffy_bassoon.meta_rl.vla_source40_ppo_worker import begin
out=Path(os.environ['TASK_OUT'])
contract=dict(source_commit=os.environ['TASK_CODE_PIN'],purpose='same eight full750 native cases with1/2/4/8lanes per sharedGPUmodel, two replicas per setting; noSGD',
    requested_gpus=2,cpus=64,ram_gib=256,shm_gib=16,worker_seconds=1620,
    robot_actions_global_ceiling=56192,global_gpus=8,maximum_gpu_seconds_global=12960)
print(begin(out/'clock',contract,slice_seconds=1560,worker_seconds=1620,now=int(os.environ['TASK_STARTED'])))
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
out=Path(os.environ['TASK_OUT']);row=finish(out/'clock',os.environ['TASK_ATTEMPT'],int(os.environ['TASK_RC']))
row['charged_gpu_seconds']=row['charged_seconds']*2
(out/'terminal-charge.json').write_text(json.dumps(row,indent=2)+'\n');print(json.dumps(row))
CLOSE
 printf '%s\n' "$TASK_RC" > "$TASK_OUT/exit-code.txt"
 exit "$TASK_RC"
}
export TASK_RC=0
trap 'TASK_RC=$?;export TASK_RC;cleanup' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
export TASK_RLINF=$TASK_ROOT/sources/robocasa365-public-pi05-nnx-leaves-20261101-r1/RLinf
export RLINF_PATH=$TASK_RLINF FLUFFY_GPU_EXECUTION=baige NCCL_P2P_DISABLE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
mkdir -p /tmp/fb-capacity
export TMPDIR=/tmp/fb-capacity XDG_CACHE_HOME=$TASK_OUT/cache
TASK_REMAINING=$((TASK_STARTED+1560-$(date +%s)))
timeout --signal=TERM --kill-after=30s "$TASK_REMAINING" "$TASK_PY" -B "$TASK_META/outputs/$TASK_RUN/WORKLOAD.py"
