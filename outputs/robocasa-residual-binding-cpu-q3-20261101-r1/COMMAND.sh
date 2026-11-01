#!/usr/bin/env bash
set -Eeuo pipefail
TASK_STARTED=${TASK_STARTED:-$(date +%s)}
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_RUN=robocasa-residual-binding-cpu-q3-20261101-r1
TASK_OUT=$TASK_ROOT/outputs/$TASK_RUN
TASK_INPUT=$TASK_ROOT/inputs/robocasa-residual-binding-cpu-q3-20261101-r1
TASK_STAGE=$TASK_OUT/stage
TASK_RLINF=$TASK_ROOT/sources/robocasa365-public-pi05-policy-20261101-r1/RLinf
TASK_WT=$TASK_ROOT/worktrees/flora-ppo-fsdp-precision-dd773048-20261101-r1
TASK_PY=$TASK_ROOT/envs/robotwin-vlrla-20261029-r1/bin/python
TASK_DEADLINE=$((TASK_STARTED+2940))
: "${TASK_META:?}" "${TASK_CODE_PIN:?}"
export TASK_META TASK_CODE_PIN
TASK_NATIVE_PY=$TASK_ROOT/setups/robocasa365-native-20261101-r1/env/bin/python
TASK_SETUP_OUT=$TASK_ROOT/outputs/robocasa365-native-setup-cpu-q1-20261101-r1
TASK_PAYLOAD=$TASK_ROOT/inputs/robocasa365-public-pi05-c484-75000-20261101-r1
TASK_CONVERTED=$TASK_ROOT/inputs/robocasa365-public-pi05-c484-torch-20261101-r2
export TASK_STARTED TASK_ROOT TASK_RUN TASK_OUT TASK_INPUT TASK_STAGE TASK_RLINF TASK_WT TASK_PY TASK_DEADLINE TASK_META TASK_NATIVE_PY TASK_SETUP_OUT TASK_PAYLOAD TASK_CONVERTED
[[ ! -e "$TASK_OUT" && ! -e "$TASK_STAGE" ]]
mkdir -p "$TASK_OUT" "$TASK_STAGE/tmp"
export TMPDIR=$TASK_STAGE/tmp XDG_CACHE_HOME=$TASK_STAGE/cache/xdg
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONPATH=$TASK_WT/src
TASK_ATTEMPT=$("$TASK_PY" -B - <<'CLOCK'
import hashlib,json,os
from pathlib import Path
from fluffy_bassoon.meta_rl.vla_source40_ppo_worker import begin
original=Path(os.environ['TASK_ROOT'])/'outputs/robocasa365-public-policy-cpu-q1-20261101-r1/clock'
saved=json.loads((original/'worker-time.json').read_text());expected=json.loads((Path(os.environ['TASK_INPUT'])/'EXPECTED-CLOSED-CPU-Q2.json').read_text());assert saved==expected
print(begin(original,saved['contract'],slice_seconds=3120,worker_seconds=3600,resume=True,now=int(os.environ['TASK_STARTED'])))
CLOCK
)
export TASK_ATTEMPT
exec > >(tee "$TASK_OUT/console.log") 2>&1
cleanup() {
 trap - EXIT INT TERM
 PYTHONPATH=$TASK_WT/src "$TASK_PY" -B - <<'CLOSE' || true
import json,os
from pathlib import Path
from fluffy_bassoon.meta_rl.vla_source40_ppo_worker import finish
print(json.dumps(finish(Path(os.environ['TASK_ROOT'])/'outputs/robocasa365-public-policy-cpu-q1-20261101-r1/clock',os.environ['TASK_ATTEMPT'],int(os.environ['TASK_RC']))))
CLOSE
 printf '%s\n' "$TASK_RC" > "$TASK_OUT/exit-code.txt"
 exit "$TASK_RC"
}
export TASK_RC=0
trap 'TASK_RC=$?;export TASK_RC;cleanup' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$TASK_META/src:$TASK_RLINF ROBOCASA_ASSETS_PATH=$TASK_ROOT/setups/robocasa365-native-20261101-r1/assets
export TOKENIZER_DIR=$TASK_ROOT/setups/robocasa365-native-20261101-r1/cache/.cache/openpi
export RLINF_PATH=$TASK_RLINF ROBOCASA_NATIVE_OUTPUT=$TASK_OUT
export ROBOCASA_PI05_TORCH_MODEL=$TASK_CONVERTED ROBOCASA_PI05_NATIVE_NORM=$TASK_PAYLOAD/assets/norm_stats.json
export HF_HOME=$TASK_OUT/cache/huggingface
TASK_REMAINING=$((TASK_DEADLINE-$(date +%s)))
timeout --signal=TERM --kill-after=30s "$TASK_REMAINING" "$TASK_NATIVE_PY" -B "$TASK_INPUT/WORKLOAD.py"
