#!/usr/bin/env bash
# Source-only template. Admission/budget/task/storage bindings do not exist yet.
set -Eeuo pipefail
: "${TASK_META:?}" "${TASK_CODE_PIN:?}" "${TASK_RUN:?}" "${TASK_LAUNCH_KIND:?}" "${TASK_BINDINGS:?}" "${TASK_BINDINGS_SHA256:?}"
case "$TASK_LAUNCH_KIND" in
 common) TASK_WORKER_SECONDS=7200; TASK_RAM_GIB=256; TASK_ACTION_CAP=450420; TASK_OUTPUT_GIB=48 ;;
 S) TASK_WORKER_SECONDS=21600; TASK_RAM_GIB=256; TASK_ACTION_CAP=925300; TASK_OUTPUT_GIB=80 ;;
 *) exit 64 ;;
esac
TASK_SLICE_SECONDS=$((TASK_WORKER_SECONDS-60))
TASK_STARTED=${TASK_STARTED:-$(date +%s)}
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_OUT=$TASK_ROOT/outputs/$TASK_RUN/job/node-$(printf '%02d' "${RANK:?}")
TASK_PY=$TASK_ROOT/setups/robocasa365-native-20261101-r1/env/bin/python
export TASK_STARTED TASK_ROOT TASK_RUN TASK_OUT TASK_PY TASK_META TASK_CODE_PIN TASK_LAUNCH_KIND TASK_BINDINGS TASK_BINDINGS_SHA256 TASK_RESUME_FROM TASK_RESUME_FROM_SHA256
export TASK_WORKER_SECONDS TASK_SLICE_SECONDS TASK_RAM_GIB TASK_ACTION_CAP TASK_OUTPUT_GIB
test ! -e "$TASK_OUT"
mkdir -p "$TASK_OUT/cache"
export PYTHONPATH=$TASK_META/src PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
TASK_ATTEMPT=$("$TASK_PY" -B - <<'CLOCK'
import os
from pathlib import Path
from fluffy_bassoon.meta_rl.vla_source40_ppo_worker import begin
out=Path(os.environ['TASK_OUT'])
seconds=int(os.environ['TASK_WORKER_SECONDS'])
contract=dict(source_commit=os.environ['TASK_CODE_PIN'],purpose='Five-task joint common-before or simulator-reward residual PPO',
 stage=os.environ['TASK_LAUNCH_KIND'],binding_sha256=os.environ['TASK_BINDINGS_SHA256'],
 requested_gpus=2,global_gpus=8,cpus=64,ram_gib=int(os.environ['TASK_RAM_GIB']),shm_gib=16,
 worker_seconds=seconds,maximum_gpu_seconds_global=seconds*8,
 robot_actions_global_ceiling=int(os.environ['TASK_ACTION_CAP']),new_output_GiB=int(os.environ['TASK_OUTPUT_GIB']))
print(begin(out/'clock',contract,slice_seconds=int(os.environ['TASK_SLICE_SECONDS']),worker_seconds=seconds,
 now=int(os.environ['TASK_STARTED'])))
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
mkdir -p /tmp/fb-j6
export TMPDIR=/tmp/fb-j6 XDG_CACHE_HOME=$TASK_OUT/cache HF_HOME=$TASK_OUT/cache/huggingface
TASK_REMAINING=$((TASK_STARTED+TASK_SLICE_SECONDS-60-$(date +%s)))
test "$TASK_REMAINING" -gt 0
timeout --signal=TERM --kill-after=30s "$TASK_REMAINING" "$TASK_PY" -B "$TASK_META/outputs/robocasa-joint-six-update-proposal-20261102-r1/NATIVE-WORKLOAD.py"
