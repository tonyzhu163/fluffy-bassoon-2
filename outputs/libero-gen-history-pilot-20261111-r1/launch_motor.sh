#!/usr/bin/env bash
# First qualification allocation: one GPU, <=60 minutes (including startup).
set -euo pipefail
TASK_PROJECT=${1:?shared project root}
TASK_CODE=${2:?immutable project checkout}
TASK_COMMIT=${3:?expected full project SHA}
TASK_ID=${4:?fresh qualification identity such as qual01}
TASK_N_TEST=${5:-1}
[[ $TASK_ID =~ ^[a-z0-9-]+$ && $HOSTNAME == job-*-master-0 ]]
TASK_FILES="$TASK_CODE/outputs/libero-gen-history-pilot-20261111-r1"
TASK_RUN="$TASK_PROJECT/outputs/libero-gen-history-pilot-20261111-r1/$TASK_ID"
TASK_SETUP="$TASK_PROJECT/setups/libero-gen-history-pilot-20261111-r1"
TASK_PY="$TASK_SETUP/env/bin/python"
mkdir "$TASK_RUN"
TASK_STARTED=$(date +%s)
trap 'TASK_RC=$?; TASK_ENDED=$(date +%s); printf "{\"exit_code\":%s,\"started\":%s,\"ended\":%s,\"gpus\":1,\"hostname\":\"%s\"}\n" "$TASK_RC" "$TASK_STARTED" "$TASK_ENDED" "$HOSTNAME" > "$TASK_RUN/clock.json"' EXIT
test "$(git -C "$TASK_CODE" rev-parse HEAD)" = "$TASK_COMMIT"
git -C "$TASK_CODE" diff --quiet HEAD
export FLUFFY_GPU_EXECUTION=baige
export PYTHONPATH="$TASK_CODE/src"
export PYTHONDONTWRITEBYTECODE=1
export HF_HOME="$TASK_SETUP/huggingface-cache"
export HF_HUB_DISABLE_XET=1
export WANDB_MODE=disabled
export MPLCONFIGDIR="$TASK_RUN/matplotlib"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
# Preserve the scheduler's CUDA_VISIBLE_DEVICES and container isolation.
if [[ -z ${CUDA_VISIBLE_DEVICES:-} ]]; then
  TASK_UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader)
  [[ $TASK_UUID == GPU-* && $TASK_UUID != *$'\n'* && $TASK_UUID == "${NVIDIA_VISIBLE_DEVICES:?}" ]]
  export CUDA_VISIBLE_DEVICES=0
fi
"$TASK_PY" "$TASK_CODE/scripts/[train]gpu_probe.py" --expected-count 1 --purpose evaluation --baige-container --output "$TASK_RUN/gpu-probe.json"
"$TASK_PY" - "$TASK_FILES" "$TASK_SETUP" "$TASK_PROJECT" "$TASK_RUN" "$TASK_N_TEST" <<'PY'
from pathlib import Path
import datetime,json,subprocess,sys,time
files,setup,project,out=map(Path,sys.argv[1:5]); n=sys.argv[5]
started=time.time()
command=[sys.executable,'-Bu',str(files/'qualify_motor.py'),'--source',str(project/'cache/bpp-58125e6-20261111-r1'),'--assets',str(setup/'assets'),'--output',str(out/'motor'),'--n-test',n]
code=124
try:
 with (out/'motor.log').open('w') as log:
  code=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=3300).returncode
finally:
 (out/'process-clock.json').write_text(json.dumps({'started':started,'ended':time.time(),'exit_code':code,'gpu_count':1,'process_gpu_hours':(time.time()-started)/3600},indent=2)+'\n')
raise SystemExit(code)
PY
