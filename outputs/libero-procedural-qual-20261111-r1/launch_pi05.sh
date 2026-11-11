#!/usr/bin/env bash
# One allocated GPU. Runtime preparation is a separate CPU operation.
set -euo pipefail
TASK_PROJECT=${1:?project root}
TASK_CODE=${2:?immutable checkout}
TASK_COMMIT=${3:?verified SHA}
TASK_ID=${4:?fresh identity}
TASK_MODE=${5:?native or generated}
TASK_GEOMETRY=${6:-}
[[ $TASK_MODE == native || $TASK_MODE == generated ]]
[[ $TASK_ID =~ ^[a-z0-9-]+$ && $HOSTNAME == job-*-master-0 ]]
test "$(git -C "$TASK_CODE" rev-parse HEAD)" = "$TASK_COMMIT"
git -C "$TASK_CODE" diff --quiet HEAD
TASK_FILES="$TASK_CODE/outputs/libero-procedural-qual-20261111-r1"
TASK_RUN="$TASK_PROJECT/outputs/libero-procedural-qual-20261111-r1/$TASK_ID"
TASK_SIM="$TASK_PROJECT/setups/libero-gen-history-pilot-20261111-r1"
TASK_SETUP="$TASK_PROJECT/setups/libero-procedural-pi05-20261111-r1"
test -f "$TASK_SETUP/checkpoint-verified.json"
mkdir -p "$(dirname "$TASK_RUN")"
mkdir "$TASK_RUN"
TASK_STARTED=$(date +%s)
trap 'TASK_RC=$?; TASK_ENDED=$(date +%s); printf "{\"exit_code\":%s,\"started\":%s,\"ended\":%s,\"gpus\":1,\"hostname\":\"%s\"}\n" "$TASK_RC" "$TASK_STARTED" "$TASK_ENDED" "$HOSTNAME" > "$TASK_RUN/clock.json"' EXIT
export FLUFFY_GPU_EXECUTION=baige
export PYTHONPATH="$TASK_CODE/src"
export PYTHONDONTWRITEBYTECODE=1
export HF_HOME="$TASK_SETUP/huggingface-cache"
export HF_HUB_DISABLE_XET=1
export OPENPI_DATA_HOME="$TASK_SETUP/openpi-cache"
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=0
export MPLCONFIGDIR="$TASK_RUN/matplotlib"
TASK_GL="$TASK_SIM/os-libs01/prefix/usr/lib/x86_64-linux-gnu"
export LD_LIBRARY_PATH="$TASK_GL:$TASK_SIM/env/lib/python3.10/site-packages/torch/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if [[ -z ${CUDA_VISIBLE_DEVICES:-} ]]; then
  TASK_UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader)
  [[ $TASK_UUID == GPU-* && $TASK_UUID != *$'\n'* && $TASK_UUID == "${NVIDIA_VISIBLE_DEVICES:?}" ]]
  export CUDA_VISIBLE_DEVICES=0
fi
"$TASK_SIM/env/bin/python" "$TASK_CODE/scripts/[train]gpu_probe.py" --expected-count 1 --purpose evaluation --baige-container --output "$TASK_RUN/gpu-probe.json"
"$TASK_SIM/env/bin/python" - "$TASK_PROJECT" "$TASK_FILES" "$TASK_SETUP" "$TASK_SIM" "$TASK_RUN" "$TASK_MODE" "$TASK_GEOMETRY" "$TASK_GL" <<'PY'
import json,os,pathlib,subprocess,sys,time,urllib.request
project,files,setup,sim,out=map(pathlib.Path,sys.argv[1:6]); mode,geometry,gl=sys.argv[6:9]
started=time.monotonic(); server=None; monitor=None
try:
 server_env=dict(os.environ,LD_LIBRARY_PATH=gl)
 with (out/'server.log').open('w') as slog, (out/'gpu-memory.csv').open('w') as mlog:
  server=subprocess.Popen([str(setup/'env/bin/python'),'-Bu',str(files/'pi05_server.py'),
      '--checkpoint',str(setup/'checkpoint'),'--output',str(out)],env=server_env,stdout=slog,stderr=subprocess.STDOUT)
  monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,name,memory.used,utilization.gpu',
      '--format=csv','--loop=5'],stdout=mlog,stderr=subprocess.STDOUT)
  deadline=time.monotonic()+900
  while True:
   if server.poll() is not None: raise RuntimeError(f'policy_server_exit:{server.returncode}')
   try:
    with urllib.request.urlopen('http://127.0.0.1:8027/healthz',timeout=2) as response:
     if response.status==200: break
   except OSError: pass
   if time.monotonic()>deadline: raise TimeoutError('policy_server_startup')
   time.sleep(2)
  client_env=dict(os.environ,PYTHONPATH=':'.join((str(setup/'client-deps'),
      str(setup/'openpi/packages/openpi-client/src'),os.environ['PYTHONPATH'])))
  command=[str(sim/'env/bin/python'),'-Bu',str(files/'pi05_eval.py'),'--source',str(project/'cache/bpp-58125e6-20261111-r1'),
      '--output',str(out/'episodes'),'--mode',mode]
  if mode=='generated': command += ['--geometry',geometry]
  with (out/'episodes.log').open('w') as log:
   subprocess.run(command,env=client_env,stdout=log,stderr=subprocess.STDOUT,check=True,
                  timeout=max(1,3600-(time.monotonic()-started)))
finally:
 for process in (server,monitor):
  if process is not None and process.poll() is None:
   process.terminate()
   try: process.wait(timeout=20)
   except subprocess.TimeoutExpired: process.kill(); process.wait()
 (out/'process-clock.json').write_text(json.dumps({'elapsed_seconds':time.monotonic()-started,'gpus':1})+'\n')
PY
