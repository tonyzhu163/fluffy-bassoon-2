#!/usr/bin/env bash
# Custom motor qualification: one GPU, <=60 minutes, frozen released weights.
set -euo pipefail
TASK_PROJECT=${1:?shared project root}
TASK_CODE=${2:?immutable project checkout}
TASK_COMMIT=${3:?expected full project SHA}
TASK_ID=${4:?fresh qualification identity such as qual01}
TASK_MODE=${5:?smoke, panel or control}
[[ $TASK_MODE == smoke || $TASK_MODE == panel || $TASK_MODE == control ]]
[[ $TASK_ID =~ ^[a-z0-9-]+$ && $HOSTNAME == job-*-master-0 ]]
TASK_FILES="$TASK_CODE/outputs/libero-gen-custom-layouts-20261111-r1"
TASK_PARENT="$TASK_PROJECT/outputs/libero-gen-custom-layouts-20261111-r1"
mkdir -p "$TASK_PARENT"
TASK_RUN="$TASK_PARENT/$TASK_ID"
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
# The private GLVND dispatch packages repair the CPU image's missing libGL/EGL.
# NVIDIA vendor libraries remain supplied by the scheduler container runtime.
TASK_GL_LIB="$TASK_SETUP/os-libs01/prefix/usr/lib/x86_64-linux-gnu"
TASK_TORCH_LIB="$TASK_SETUP/env/lib/python3.10/site-packages/torch/lib"
test -f "$TASK_GL_LIB/libEGL.so.1"
test -f "$TASK_GL_LIB/libGL.so.1"
export LD_LIBRARY_PATH="$TASK_GL_LIB:$TASK_TORCH_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
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
"$TASK_PY" - "$TASK_FILES" "$TASK_SETUP" "$TASK_PROJECT" "$TASK_RUN" "$TASK_MODE" <<'PY'
from pathlib import Path
import datetime,json,subprocess,sys,time
files,setup,project,out=map(Path,sys.argv[1:5]); mode=sys.argv[5]
started=time.time()
import hashlib
command=[sys.executable,'-Bu',str(files/'motor.py'),'--source',str(project/'cache/bpp-58125e6-20261111-r1'),'--assets',str(setup/'assets'),'--output',str(out/'motor')]
if mode == 'control':
 reference=files.parent/'libero-gen-history-pilot-20261111-r1/panel03/motor/result.json'
 command += ['--native-control-reference',str(reference)]
else:
 admission=files/'admission.json'
 assert hashlib.sha256(admission.read_bytes()).hexdigest()=='35aeb8c3fe365da2bdffdaa37c3d713bcff2e3def3b38cd582936c161c29db74'
 indices='0,5,8,13' if mode=='smoke' else ','.join(map(str,range(16)))
 command += ['--geometry',str(project/'outputs/libero-gen-custom-layouts-20261111-r1/geometry02/geometry'),'--admission',str(admission),'--admission-sha256',hashlib.sha256(admission.read_bytes()).hexdigest(),'--indices',indices]
code=124
try:
 with (out/'motor.log').open('w') as log:
  code=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=1200 if mode == 'control' else 3300).returncode
 if code == 0 and mode != 'control':
  audit=[sys.executable,'-Bu',str(files/'audit_records.py'),'--motor',str(out/'motor'),'--geometry',str(project/'outputs/libero-gen-custom-layouts-20261111-r1/geometry02/geometry'),'--output',str(out/'audit.json')]
  with (out/'audit.log').open('w') as log:
   code=subprocess.run(audit,stdout=log,stderr=subprocess.STDOUT,timeout=180).returncode
finally:
 (out/'process-clock.json').write_text(json.dumps({'started':started,'ended':time.time(),'exit_code':code,'gpu_count':1,'process_gpu_hours':(time.time()-started)/3600},indent=2)+'\n')
raise SystemExit(code)
PY
