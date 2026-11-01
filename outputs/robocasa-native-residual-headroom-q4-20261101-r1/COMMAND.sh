#!/usr/bin/env bash
set -Eeuo pipefail
: "${TASK_META:?}" "${TASK_CODE_PIN:?}"
TASK_STARTED=${TASK_STARTED:-$(date +%s)}
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_RUN=robocasa-native-residual-headroom-q4-20261101-r1
TASK_OUT=$TASK_ROOT/outputs/$TASK_RUN
TASK_INPUT=$TASK_ROOT/inputs/$TASK_RUN
TASK_PY=$TASK_ROOT/setups/robocasa365-native-20261101-r1/env/bin/python
export TASK_STARTED TASK_ROOT TASK_RUN TASK_OUT TASK_INPUT TASK_PY TASK_META TASK_CODE_PIN
test ! -e "$TASK_OUT"
mkdir -p "$TASK_OUT/tmp" "$TASK_OUT/cache"
export PYTHONPATH=$TASK_META/src PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
TASK_ATTEMPT=$("$TASK_PY" -B - <<'CLOCK'
import os,json,hashlib
from pathlib import Path
from fluffy_bassoon.meta_rl.vla_source40_ppo_worker import begin
ticket=Path(os.environ['TASK_INPUT'])/'ADMISSION.json'
contract=dict(source_commit=os.environ['TASK_CODE_PIN'],run_id=os.environ['TASK_RUN'],ticket_sha256=hashlib.sha256(ticket.read_bytes()).hexdigest(),gpu_count=1,wall_ceiling_seconds=7200,actions=17648)
original=Path(os.environ['TASK_ROOT'])/'outputs/robocasa-native-residual-headroom-20261101-r1/clock'
saved=json.loads((original/'worker-time.json').read_text());assert saved==json.loads((Path(os.environ['TASK_INPUT'])/'EXPECTED-CLOSED-Q3.json').read_text())
print(begin(original,saved['contract'],slice_seconds=6720,worker_seconds=7200,resume=True,now=int(os.environ['TASK_STARTED'])))
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
print(json.dumps(finish(Path(os.environ['TASK_ROOT'])/'outputs/robocasa-native-residual-headroom-20261101-r1/clock',os.environ['TASK_ATTEMPT'],int(os.environ['TASK_RC']))))
CLOSE
 printf '%s\n' "$TASK_RC" > "$TASK_OUT/exit-code.txt"
 exit "$TASK_RC"
}
export TASK_RC=0
trap 'TASK_RC=$?;export TASK_RC;cleanup' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
TASK_RLINF=$TASK_ROOT/sources/robocasa365-public-pi05-nnx-leaves-20261101-r1/RLinf
export TASK_RLINF RLINF_PATH=$TASK_RLINF PYTHONPATH=$TASK_META/src:$TASK_RLINF
export ROBOCASA_ASSETS_PATH=$TASK_ROOT/setups/robocasa365-native-20261101-r1/assets
export OPENPI_DATA_HOME=$TASK_ROOT/setups/robocasa365-native-20261101-r1/cache/.cache/openpi
export ROBOCASA_NATIVE_OUTPUT=$TASK_OUT ROBOCASA_PI05_TORCH_MODEL=$TASK_ROOT/inputs/robocasa365-public-pi05-c484-torch-20261101-r2
export ROBOCASA_PI05_NATIVE_NORM=$TASK_ROOT/inputs/robocasa365-public-pi05-c484-75000-20261101-r1/assets/norm_stats.json
export FLUFFY_GPU_EXECUTION=baige
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
export TMPDIR=$TASK_OUT/tmp XDG_CACHE_HOME=$TASK_OUT/cache HF_HOME=$TASK_OUT/cache/huggingface TORCHINDUCTOR_CACHE_DIR=$TASK_OUT/cache/torch
TASK_REMAINING=$((TASK_STARTED+6420-$(date +%s)))
timeout --signal=TERM --kill-after=60s "$TASK_REMAINING" "$TASK_PY" -B "$TASK_INPUT/WORKLOAD.py"
