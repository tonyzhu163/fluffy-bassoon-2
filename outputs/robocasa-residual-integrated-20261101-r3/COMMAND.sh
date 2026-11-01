#!/usr/bin/env bash
set -Eeuo pipefail
: "${TASK_META:?}" "${TASK_CODE_PIN:?}" "${TASK_ROLE:?}"
TASK_STARTED=${TASK_STARTED:-$(date +%s)}
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_RUN=robocasa-residual-integrated-20261101-r3
TASK_OUT=$TASK_ROOT/outputs/$TASK_RUN/$TASK_ROLE-job
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
p=Path(os.environ['TASK_INPUT'])/'ADMISSION.json';role=os.environ['TASK_ROLE'];gpus=2 if role=='F' else 1
if role=='F':
 original=Path(os.environ['TASK_ROOT'])/'outputs/robocasa-residual-integrated-20261101-r1/F-job/clock'
 saved=json.loads((original/'worker-time.json').read_text());assert saved==json.loads((Path(os.environ['TASK_INPUT'])/'EXPECTED-F-CLOSED-R2.json').read_text())
 print(begin(original,saved['contract'],slice_seconds=27660,worker_seconds=27960,resume=True,now=int(os.environ['TASK_STARTED'])))
else:
 c=dict(source_commit=os.environ['TASK_CODE_PIN'],run_id=os.environ['TASK_RUN']+'-S',ticket_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),gpu_count=1,wall_ceiling_seconds=27960,actions=971600)
 print(begin(Path(os.environ['TASK_OUT'])/'clock',c,slice_seconds=27900,worker_seconds=27960,now=int(os.environ['TASK_STARTED'])))
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
clock=(Path(os.environ['TASK_ROOT'])/'outputs/robocasa-residual-integrated-20261101-r1/F-job/clock' if os.environ['TASK_ROLE']=='F' else Path(os.environ['TASK_OUT'])/'clock')
row=finish(clock,os.environ['TASK_ATTEMPT'],int(os.environ['TASK_RC']))
row['charged_gpu_seconds']=row['charged_seconds']*(2 if os.environ['TASK_ROLE']=='F' else 1)
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
mkdir -p /tmp/fr-$TASK_ROLE
export TMPDIR=/tmp/fr-$TASK_ROLE XDG_CACHE_HOME=$TASK_OUT/cache HF_HOME=$TASK_OUT/cache/huggingface TORCHINDUCTOR_CACHE_DIR=$TASK_OUT/cache/torch
TASK_SLICE=27900
if [[ "$TASK_ROLE" == F ]]; then TASK_SLICE=27660; fi
TASK_REMAINING=$((TASK_STARTED+TASK_SLICE-$(date +%s)))
timeout --signal=TERM --kill-after=60s "$TASK_REMAINING" "$TASK_PY" -B "$TASK_INPUT/WORKLOAD.py"
