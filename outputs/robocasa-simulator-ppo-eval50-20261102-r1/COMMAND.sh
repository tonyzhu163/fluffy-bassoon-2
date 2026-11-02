#!/usr/bin/env bash
# Source-only entry: real task/action/storage/admission bindings remain required.
set -Eeuo pipefail
: "${TASK_META:?}" "${TASK_CODE_PIN:?}" "${TASK_RUN:?}" "${TASK_BINDINGS:?}" "${TASK_BINDINGS_SHA256:?}"
TASK_STARTED=${TASK_STARTED:-$(date +%s)}
TASK_LAUNCH_KIND=evaluation
TASK_WORKER_SECONDS=7200
TASK_SLICE_SECONDS=7140
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_OUT=$TASK_ROOT/outputs/$TASK_RUN/job/node-$(printf '%02d' "${RANK:?}")
TASK_PY=$TASK_ROOT/setups/robocasa365-native-20261101-r1/env/bin/python
export TASK_STARTED TASK_ROOT TASK_RUN TASK_OUT TASK_PY TASK_META TASK_CODE_PIN TASK_LAUNCH_KIND TASK_BINDINGS TASK_BINDINGS_SHA256
export TASK_WORKER_SECONDS TASK_SLICE_SECONDS
test ! -e "$TASK_OUT"
mkdir -p "$TASK_OUT/cache"
export PYTHONPATH=$TASK_META/src PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
TASK_ATTEMPT=$("$TASK_PY" -B - <<'CLOCK'
import os
from pathlib import Path
from fluffy_bassoon.meta_rl.vla_source40_ppo_worker import begin
contract=dict(source_commit=os.environ['TASK_CODE_PIN'],purpose='Evaluate preserved simulator PPO on50 ID and50 OOD cases',
 stage='evaluation',run_id=os.environ['TASK_RUN'],binding_sha256=os.environ['TASK_BINDINGS_SHA256'],
 requested_gpus=2,global_gpus=8,cpus=32,ram_gib=128,shm_gib=16,
 worker_seconds=7200,maximum_gpu_seconds_global=57600,
 robot_actions_global_ceiling=100000,new_output_GiB=24)
print(begin(Path(os.environ['TASK_OUT'])/'clock',contract,slice_seconds=7140,
 worker_seconds=7200,now=int(os.environ['TASK_STARTED'])))
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
export TASK_RLINF=$TASK_ROOT/sources/robocasa365-public-pi05-nnx-leaves-20261101-r1/RLinf
export RLINF_PATH=$TASK_RLINF FLUFFY_GPU_EXECUTION=baige NCCL_P2P_DISABLE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
mkdir -p /tmp/fb-eval50-${RANK}
export TMPDIR=/tmp/fb-eval50-${RANK} XDG_CACHE_HOME=$TASK_OUT/cache HF_HOME=$TASK_OUT/cache/huggingface
TASK_REMAINING=$((TASK_STARTED+TASK_SLICE_SECONDS-60-$(date +%s)))
test "$TASK_REMAINING" -gt 0
timeout --signal=TERM --kill-after=30s "$TASK_REMAINING" "$TASK_PY" -B "$TASK_META/outputs/robocasa-simulator-ppo-eval50-20261102-r1/WORKLOAD.py"
