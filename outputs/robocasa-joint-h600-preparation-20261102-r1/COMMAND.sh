#!/usr/bin/env bash
set -Eeuo pipefail
: "${TASK_META:?}" "${TASK_CODE_PIN:?}" "${RANK:?}"
export TASK_RUN=robocasa-joint-h600-preparation-20261102-r1
export TASK_ROOT=/mowenjie/fluffy-bassoon-2 TASK_STARTED=${TASK_STARTED:-$(date +%s)}
export TASK_OUT=$TASK_ROOT/outputs/$TASK_RUN/job/node-$(printf '%02d' "$RANK")
export TASK_PY=$TASK_ROOT/setups/robocasa365-native-20261101-r1/env/bin/python
export TASK_RLINF=$TASK_ROOT/sources/robocasa365-public-pi05-nnx-leaves-20261101-r1/RLinf
test ! -e "$TASK_OUT"
mkdir -p "$TASK_OUT/cache"
export PYTHONPATH=$TASK_META/src PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export RLINF_PATH=$TASK_RLINF FLUFFY_GPU_EXECUTION=baige NCCL_P2P_DISABLE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
export XDG_CACHE_HOME=$TASK_OUT/cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
TASK_ATTEMPT=$("$TASK_PY" -B - <<'PY'
import os
from pathlib import Path
from fluffy_bassoon.meta_rl.vla_source40_ppo_worker import begin
contract=dict(source_commit=os.environ['TASK_CODE_PIN'],run_id=os.environ['TASK_RUN'],
 purpose='Approved600-control calibration and conditional common-before; zeroSGD',
 requested_gpus=2,global_gpus=8,cpus=64,ram_gib=256,shm_gib=32,
 maximum_gpu_seconds_global=26400,robot_actions_global_ceiling=None,new_output_GiB=128)
print(begin(Path(os.environ['TASK_OUT'])/'clock',contract,slice_seconds=3240,
 worker_seconds=3300,now=int(os.environ['TASK_STARTED'])))
PY
)
export TASK_ATTEMPT
exec > >(tee "$TASK_OUT/console.log") 2>&1
cleanup() {
 trap - EXIT INT TERM
 "$TASK_PY" -B - <<'PY' || true
import json,os
from pathlib import Path
from fluffy_bassoon.meta_rl.vla_source40_ppo_worker import finish
out=Path(os.environ['TASK_OUT'])
row=finish(out/'clock',os.environ['TASK_ATTEMPT'],int(os.environ['TASK_RC']))
row['charged_gpu_seconds']=row['charged_seconds']*2
(out/'terminal-charge.json').write_text(json.dumps(row,indent=2)+'\n')
PY
 printf '%s\n' "$TASK_RC" > "$TASK_OUT/exit-code.txt"
 exit "$TASK_RC"
}
export TASK_RC=0
trap 'TASK_RC=$?;export TASK_RC;cleanup' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
TASK_REMAINING=$((TASK_STARTED+3240-$(date +%s)))
test "$TASK_REMAINING" -gt 0
timeout --signal=TERM --kill-after=20s "$TASK_REMAINING" "$TASK_PY" -B "$TASK_META/outputs/$TASK_RUN/WORKLOAD.py"
