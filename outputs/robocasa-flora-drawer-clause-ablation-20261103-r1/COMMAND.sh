#!/usr/bin/env bash
set -Eeuo pipefail
: "${TASK_META:?}" "${TASK_CODE_PIN:?}" "${RANK:?}" "${WORLD_SIZE:?}" "${NPROC_PER_NODE:?}"
export TASK_RUN=robocasa-flora-drawer-clause-ablation-20261103-r1
export TASK_STARTED=${TASK_STARTED:-$(date +%s)}
export PYTHONPATH=$TASK_META/src PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled NCCL_P2P_DISABLE=1
TASK_PY=/mowenjie/fluffy-bassoon-2/setups/robocasa365-native-20261101-r1/env/bin/python
finish_sample() {
  TASK_RC=$?
  export TASK_RC
  trap - EXIT INT TERM
  "$TASK_PY" -B - <<'PY'
import json,os,time
from pathlib import Path
p=Path('/mowenjie/fluffy-bassoon-2/outputs')/os.environ['TASK_RUN']
r=dict(source_commit=os.environ['TASK_CODE_PIN'],start_unix=int(os.environ['TASK_STARTED']),end_unix=int(time.time()),exit_code=int(os.environ['TASK_RC']),gpus=1,maximum_gpu_seconds=900,robot_actions=0)
r['charged_gpu_seconds']=r['end_unix']-r['start_unix']
(p/'terminal-charge.json').write_text(json.dumps(r,indent=2)+'\n')
PY
  exit "$TASK_RC"
}
trap finish_sample EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
timeout --signal=TERM --kill-after=20s 860 "$TASK_PY" -B "$TASK_META/outputs/$TASK_RUN/WORKLOAD.py"
