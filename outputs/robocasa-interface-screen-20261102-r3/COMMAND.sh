#!/usr/bin/env bash
set -Eeuo pipefail
: "${TASK_META:?}" "${TASK_CODE_PIN:?}" "${TASK_CARRY_SHA:?}" "${TASK_PRIOR_GPU_SECONDS:?}" "${RANK:?}" "${WORLD_SIZE:?}"
export TASK_STARTED=$(date +%s)
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_RUN=$TASK_ROOT/outputs/robocasa-interface-screen-20261102-r3
TASK_PY=$TASK_ROOT/setups/robocasa365-native-20261101-r1/env/bin/python
export PYTHONPATH=$TASK_META/src:$TASK_ROOT/sources/robocasa365-public-pi05-nnx-leaves-20261101-r1/RLinf
export RLINF_PATH=$TASK_ROOT/sources/robocasa365-public-pi05-nnx-leaves-20261101-r1/RLinf
export ROBOCASA_ASSETS_PATH=$TASK_ROOT/setups/robocasa365-native-20261101-r1/assets
export OPENPI_DATA_HOME=$TASK_ROOT/setups/robocasa365-native-20261101-r1/cache/.cache/openpi
export ROBOCASA_PI05_TORCH_MODEL=$TASK_ROOT/inputs/robocasa365-public-pi05-c484-torch-20261101-r2
export ROBOCASA_PI05_NATIVE_NORM=$TASK_ROOT/inputs/robocasa365-public-pi05-c484-75000-20261101-r1/assets/norm_stats.json
export ROBOCASA_NATIVE_OUTPUT=$TASK_RUN/job/node-$RANK/native
export FLUFFY_GPU_EXECUTION=baige NCCL_P2P_DISABLE=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
test ! -e "$TASK_RUN/job/node-$RANK"
mkdir -p "$TASK_RUN/job/node-$RANK"
exec > >(tee "$TASK_RUN/job/node-$RANK/console.log") 2>&1
finish() {
 TASK_RC=$?
 trap - EXIT
 set +e
 printf '{"exit_code":%s}\n' "$TASK_RC" > "$TASK_RUN/job/node-$RANK/worker-terminal.json"
 if [ "$RANK" = 0 ]; then
  "$TASK_PY" -B - "$TASK_RUN" "$((TASK_STARTED+9180))" <<'PY'
import json,sys,time
from pathlib import Path
root=Path(sys.argv[1]);deadline=int(sys.argv[2])
while True:
 paths=[root/f'job/node-{rank}/worker-terminal.json' for rank in range(4)]
 rows=[json.loads(path.read_text()) for path in paths if path.exists()]
 if any(row['exit_code'] for row in rows):raise SystemExit('A task worker failed; preserve and diagnose')
 if len(rows)==4:break
 if time.time()>deadline:raise SystemExit('Node completion deadline; preserve partial records')
 time.sleep(2)
PY
  TASK_BARRIER_RC=$?
  if [ "$TASK_BARRIER_RC" != 0 ]; then TASK_RC=$TASK_BARRIER_RC; fi
 fi
 TASK_FINISHED=$(date +%s)
 printf '{"started_unix":%s,"finished_unix":%s,"allocated_gpus":1,"observed_gpu_seconds":%s,"exit_code":%s}\n' \
  "$TASK_STARTED" "$TASK_FINISHED" "$((TASK_FINISHED-TASK_STARTED))" "$TASK_RC" > "$TASK_RUN/job/node-$RANK/clock.json"
 exit "$TASK_RC"
}
trap finish EXIT
timeout --signal=TERM --kill-after=30s 8820 "$TASK_PY" -B \
 "$TASK_META/outputs/robocasa-interface-screen-20261102-r3/WORKLOAD.py"
