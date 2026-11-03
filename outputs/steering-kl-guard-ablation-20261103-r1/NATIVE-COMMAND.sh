#!/usr/bin/env bash
set -Eeuo pipefail
: "${TASK_CODE:?}" "${TASK_CODE_PIN:?}" "${TASK_TEMPLATE:?}"
test "${RANK:?}" = 0
test "${WORLD_SIZE:?}" = 1
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_RUN=$TASK_ROOT/outputs/steering-kl-guard-ablation-20261103-r1/gpu-r2
TASK_PY=$TASK_ROOT/setups/robocasa365-native-20261101-r1/env/bin/python
TASK_PROCESS_STARTED=$(date +%s)
export TASK_PROCESS_STARTED TASK_RUN TASK_CODE TASK_CODE_PIN TASK_TEMPLATE
test ! -e "$TASK_RUN"
mkdir "$TASK_RUN"
exec > >(tee "$TASK_RUN/console.log") 2>&1
finish() {
  TASK_EXIT_CODE=$?
  trap - EXIT
  TASK_FINISHED=$(date +%s)
  printf '{"started_unix":%s,"finished_unix":%s,"allocated_gpus":4,"process_gpu_seconds":%s,"exit_code":%s,"allocation_release_not_included":true}\n' \
    "$TASK_PROCESS_STARTED" "$TASK_FINISHED" "$((4*(TASK_FINISHED-TASK_PROCESS_STARTED)))" "$TASK_EXIT_CODE" > "$TASK_RUN/clock.json"
  exit "$TASK_EXIT_CODE"
}
trap finish EXIT
export RLINF_PATH=$TASK_ROOT/sources/robocasa365-public-pi05-nnx-leaves-20261101-r1/RLinf
export PYTHONPATH=$TASK_CODE/src:$RLINF_PATH
export ROBOCASA_ASSETS_PATH=$TASK_ROOT/setups/robocasa365-native-20261101-r1/assets
export OPENPI_DATA_HOME=$TASK_ROOT/setups/robocasa365-native-20261101-r1/cache/.cache/openpi
export FLUFFY_GPU_EXECUTION=baige MUJOCO_GL=egl PYOPENGL_PLATFORM=egl NCCL_P2P_DISABLE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
if test -z "${CUDA_VISIBLE_DEVICES:-}"; then
  # This exclusive scheduler container exposes only its four assigned GPUs.
  # Bind those existing namespace indices; never infer physical host indices.
  TASK_VISIBLE_GPU_COUNT=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | wc -l)
  test "$TASK_VISIBLE_GPU_COUNT" -eq 4
  export CUDA_VISIBLE_DEVICES=0,1,2,3
fi
export WANDB_MODE=disabled
unset WANDB_API_KEY WANDB_ACCESS_TOKEN WANDB_IDENTITY_TOKEN_FILE
export LYCO_BAIGE_GPU_MAP=$TASK_RUN/GPU-PROBE.json
timeout --signal=TERM --kill-after=15s 120s "$TASK_PY" -B "$TASK_CODE/scripts/[train]gpu_probe.py" \
  --expected-count 4 --output "$LYCO_BAIGE_GPU_MAP" --baige-container --purpose training --baige-exclusive-training
"$TASK_PY" -B - <<'PY'
import hashlib,json,os,pathlib,socket,subprocess,time
root=pathlib.Path(os.environ['TASK_RUN']);code=pathlib.Path(os.environ['TASK_CODE'])
pin=os.environ['TASK_CODE_PIN']
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=code,text=True).strip()==pin
assert not subprocess.check_output(['git','status','--porcelain'],cwd=code,text=True).strip()
p=json.loads(pathlib.Path(os.environ['TASK_TEMPLATE']).read_text())
assert p['source_commit']==pin and p['code']==str(code)
assert hashlib.sha256(pathlib.Path(p['CPU_result']).read_bytes()).hexdigest()==p['CPU_result_sha256']
quota,period=pathlib.Path('/sys/fs/cgroup/cpu.max').read_text().split()
available=len(os.sched_getaffinity(0))
if quota!='max':available=min(available,int(quota)/int(period))
assert available>=84
assert int(pathlib.Path('/sys/fs/cgroup/memory.max').read_text())>=832*2**30
shm=os.statvfs('/dev/shm');assert shm.f_blocks*shm.f_frsize>=192*2**30
storage=os.statvfs(root);assert storage.f_bavail*storage.f_frsize>=100*2**30
probe=json.loads(pathlib.Path(os.environ['LYCO_BAIGE_GPU_MAP']).read_text())
assert probe['status']=='passed' and probe['hostname']==socket.gethostname() and len(probe['devices'])==4
assert p['prior_GPU_seconds']==200 and p['budget_GPU_seconds']==21600
p.update(process_started_unix=int(os.environ['TASK_PROCESS_STARTED']),hostname=socket.gethostname(),
 resource_qualification=dict(effective_CPU=available,RAM_bytes=int(pathlib.Path('/sys/fs/cgroup/memory.max').read_text()),
 SHM_bytes=shm.f_blocks*shm.f_frsize,storage_free_bytes=storage.f_bavail*storage.f_frsize),
 native_env={k:os.environ[k] for k in ('RLINF_PATH','ROBOCASA_ASSETS_PATH','OPENPI_DATA_HOME','MUJOCO_GL','PYOPENGL_PLATFORM','NCCL_P2P_DISABLE')})
(root/'PROFILE.json').write_text(json.dumps(p,indent=2)+'\n')
deadline=p['process_started_unix']+(p['budget_GPU_seconds']-p['prior_GPU_seconds'])//4-180
(root/'deadline-unix.txt').write_text(str(deadline)+'\n')
print(json.dumps(dict(qualified=True,hostname=socket.gethostname(),source_commit=pin,GPUs=4,deadline=deadline,prior_GPU_seconds=p['prior_GPU_seconds'])),flush=True)
PY
TASK_DEADLINE=$(cat "$TASK_RUN/deadline-unix.txt")
TASK_REMAINING=$((TASK_DEADLINE+30-$(date +%s)))
test "$TASK_REMAINING" -gt 120
timeout --signal=TERM --kill-after=45s "${TASK_REMAINING}s" "$TASK_PY" -B \
  "$TASK_CODE/outputs/steering-kl-guard-ablation-20261103-r1/NATIVE-EVAL.py" \
  --profile "$TASK_RUN/PROFILE.json" --output "$TASK_RUN" --deadline-unix "$TASK_DEADLINE"
