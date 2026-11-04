#!/usr/bin/env bash
set -Eeuo pipefail
: "${TASK_CODE:?}" "${TASK_CODE_PIN:?}" "${TASK_QUALIFICATION:?}"
test "${RANK:?}" = 0
test "${WORLD_SIZE:?}" = 1
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_RUN=$TASK_ROOT/outputs/robocasa-buttons-replacement-qualification-20261104-r1/gpu-r1
TASK_PY=$TASK_ROOT/setups/robocasa365-native-20261101-r1/env/bin/python
TASK_PROCESS_STARTED=$(date +%s)
export TASK_PROCESS_STARTED TASK_RUN TASK_CODE TASK_CODE_PIN TASK_QUALIFICATION
test ! -e "$TASK_RUN"
mkdir "$TASK_RUN"
exec > >(tee "$TASK_RUN/console.log") 2>&1
finish() {
  TASK_EXIT_CODE=$?
  trap - EXIT
  TASK_FINISHED=$(date +%s)
  printf '{"started_unix":%s,"finished_unix":%s,"allocated_gpus":2,"process_gpu_seconds":%s,"exit_code":%s,"allocation_release_not_included":true}\n' \
    "$TASK_PROCESS_STARTED" "$TASK_FINISHED" "$((2*(TASK_FINISHED-TASK_PROCESS_STARTED)))" "$TASK_EXIT_CODE" > "$TASK_RUN/clock.json"
  exit "$TASK_EXIT_CODE"
}
trap finish EXIT
export RLINF_PATH=$TASK_ROOT/sources/robocasa365-public-pi05-nnx-leaves-20261101-r1/RLinf
export PYTHONPATH=$TASK_CODE/src:$RLINF_PATH
export ROBOCASA_ASSETS_PATH=$TASK_ROOT/setups/robocasa365-native-20261101-r1/assets
export OPENPI_DATA_HOME=$TASK_ROOT/setups/robocasa365-native-20261101-r1/cache/.cache/openpi
export FLUFFY_GPU_EXECUTION=baige MUJOCO_GL=egl PYOPENGL_PLATFORM=egl NCCL_P2P_DISABLE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 WANDB_MODE=disabled
if test -z "${CUDA_VISIBLE_DEVICES+x}"; then
  TASK_VISIBLE_GPU_COUNT=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | wc -l)
  test "$TASK_VISIBLE_GPU_COUNT" -eq 2
  export CUDA_VISIBLE_DEVICES=0,1
fi
export LYCO_BAIGE_GPU_MAP=$TASK_RUN/GPU-PROBE.json
timeout --signal=TERM --kill-after=15s 120s "$TASK_PY" -B "$TASK_CODE/scripts/[train]gpu_probe.py" \
  --expected-count 2 --output "$LYCO_BAIGE_GPU_MAP" --baige-container --purpose evaluation
"$TASK_PY" -B - <<'PY'
import hashlib,json,os,pathlib,socket,subprocess
root=pathlib.Path(os.environ['TASK_RUN']);code=pathlib.Path(os.environ['TASK_CODE'])
pin=os.environ['TASK_CODE_PIN']
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=code,text=True).strip()==pin
assert not subprocess.check_output(['git','status','--porcelain'],cwd=code,text=True).strip()
q=json.loads(pathlib.Path(os.environ['TASK_QUALIFICATION']).read_text())
assert q['completed'] and q['source_commit']==pin and q['new_GPU_seconds']==0
for name,checksum in q['core_files'].items():assert hashlib.sha256((code/name).read_bytes()).hexdigest()==checksum
cpu=pathlib.Path('/sys/fs/cgroup/cpu.max').read_text().split()
available=len(os.sched_getaffinity(0))
if cpu[0]!='max':available=min(available,int(cpu[0])/int(cpu[1]))
assert available>=32
assert int(pathlib.Path('/sys/fs/cgroup/memory.max').read_text())>=256*2**30
shm=os.statvfs('/dev/shm');assert shm.f_blocks*shm.f_frsize>=32*2**30
storage=os.statvfs(root);assert storage.f_bavail*storage.f_frsize>=100*2**30
probe=json.loads(pathlib.Path(os.environ['LYCO_BAIGE_GPU_MAP']).read_text())
assert probe['status']=='passed' and probe['hostname']==socket.gethostname() and len(probe["devices"])==2
profile=dict(source_commit=pin,code=str(code),output=str(root),initializer=q['initializer'],
 rlinf_execution_commit=q['rlinf_execution_commit'],gpu_probe=os.environ['LYCO_BAIGE_GPU_MAP'],
 deadline_unix=int(os.environ['TASK_PROCESS_STARTED'])+2640,maximum_process_seconds=2700,
 maximum_scheduler_seconds=3000,maximum_GPU_seconds=6000,lanes_per_GPU=4,
 effective_CPU=available,RAM_bytes=int(pathlib.Path('/sys/fs/cgroup/memory.max').read_text()),
 SHM_bytes=shm.f_blocks*shm.f_frsize,storage_free_bytes=storage.f_bavail*storage.f_frsize,
 CPU_qualification=os.environ['TASK_QUALIFICATION'])
(root/'PROFILE.json').write_text(json.dumps(profile,indent=2)+'\n')
print(json.dumps(dict(qualified=True,hostname=socket.gethostname(),source_commit=pin,GPUs=2)),flush=True)
PY
TASK_REMAINING=$((TASK_PROCESS_STARTED+2700-$(date +%s)))
test "$TASK_REMAINING" -gt 120
timeout --signal=TERM --kill-after=30s "${TASK_REMAINING}s" "$TASK_PY" -B \
  "$TASK_CODE/outputs/robocasa-buttons-replacement-qualification-20261104-r1/EVAL.py" --profile "$TASK_RUN/PROFILE.json"
