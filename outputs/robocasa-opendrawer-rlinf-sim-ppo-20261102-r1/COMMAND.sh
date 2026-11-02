#!/usr/bin/env bash
set -Eeuo pipefail
: "${TASK_META:?}" "${TASK_CODE_PIN:?}" "${TASK_ATTEMPT:?}" "${TASK_OUTPUT:?}"
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_PY=$TASK_ROOT/setups/robocasa365-native-20261101-r1/env/bin/python
export TASK_STARTED=$(date +%s)
export TASK_DEADLINE=$((TASK_STARTED+28080))
export PYTHONPATH=$TASK_META/src:$TASK_ROOT/sources/robocasa365-public-pi05-nnx-leaves-20261101-r1/RLinf
export RLINF_PATH=$TASK_ROOT/sources/robocasa365-public-pi05-nnx-leaves-20261101-r1/RLinf
export ROBOCASA_ASSETS_PATH=$TASK_ROOT/setups/robocasa365-native-20261101-r1/assets
export OPENPI_DATA_HOME=$TASK_ROOT/setups/robocasa365-native-20261101-r1/cache/.cache/openpi
export ROBOCASA_PI05_TORCH_MODEL=$TASK_ROOT/inputs/robocasa365-public-pi05-c484-torch-20261101-r2
export ROBOCASA_PI05_NATIVE_NORM=$TASK_ROOT/inputs/robocasa365-public-pi05-c484-75000-20261101-r1/assets/norm_stats.json
export ROBOCASA_SIM_PPO_OUTPUT=$TASK_OUTPUT
export ROBOCASA_SIM_PPO_ACCOUNTING=$TASK_ROOT/outputs/robocasa-opendrawer-rlinf-sim-ppo-20261102-r1/accounting
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl NCCL_P2P_DISABLE=1 FLUFFY_GPU_EXECUTION=baige
# Baige exposes the two allocated devices to the container, but does not set
# CUDA_VISIBLE_DEVICES. Preserve an existing selection; otherwise expose both
# local allocation ordinals for the probe and native RLinf placement.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 WANDB_MODE=disabled
test ! -e "$TASK_OUTPUT"
mkdir -p "$TASK_OUTPUT"
exec > >(tee "$TASK_OUTPUT/console.log") 2>&1
finish() {
 TASK_RC=$?
 trap - EXIT
 TASK_FINISHED=$(date +%s)
 printf '{"started_unix":%s,"finished_unix":%s,"allocated_gpus":2,"observed_gpu_seconds":%s,"exit_code":%s}\n' \
  "$TASK_STARTED" "$TASK_FINISHED" "$((2*(TASK_FINISHED-TASK_STARTED)))" "$TASK_RC" > "$TASK_OUTPUT/clock.json"
 exit "$TASK_RC"
}
trap finish EXIT
test "$(git -C "$TASK_META" rev-parse HEAD)" = "$TASK_CODE_PIN"
test -z "$(git -C "$TASK_META" status --porcelain)"
test "$(git -C "$RLINF_PATH" rev-parse HEAD)" = 8a1233c3fae6f5905adfda7684c455842b0ecfb8
"$TASK_PY" -B - <<'PY'
import hashlib,json,os,pathlib,socket,time
root=pathlib.Path(os.environ['TASK_OUTPUT'])
assert socket.gethostname().startswith('job-')
cpu=pathlib.Path('/sys/fs/cgroup/cpu.max').read_text().split()
assert int(cpu[0])/int(cpu[1])>=64
assert int(pathlib.Path('/sys/fs/cgroup/memory.max').read_text())>=256*(1<<30)
fs=os.statvfs(root)
assert fs.f_bavail*fs.f_frsize>=256*(1<<30)
shm=os.statvfs('/dev/shm');assert shm.f_blocks*shm.f_frsize>=32*(1<<30)
(root/'write-probe').write_text('Flow SDE allocated project write probe\n')
def digest(path):
 h=hashlib.sha256()
 with pathlib.Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
model=digest(pathlib.Path(os.environ['ROBOCASA_PI05_TORCH_MODEL'])/'model.safetensors')
norm=digest(os.environ['ROBOCASA_PI05_NATIVE_NORM'])
assert model=='661bf172a2ae9dfca75216697eeaa543408bcbdfa53c98132759b93e0e8993a5'
assert norm=='4aed1af411bd0e0f49d2b0e6d6832b11b8682917231a07173fbc30fd493bbdee'
(root/'manifest.json').write_text(json.dumps(dict(hostname=socket.gethostname(),source=os.environ['TASK_CODE_PIN'],
 runtime='8a1233c3fae6f5905adfda7684c455842b0ecfb8',model_sha256=model,norm_sha256=norm,
 allocated_gpus=2,CPU=64,RAM_GiB=256,SHM_GiB=32,free_bytes=fs.f_bavail*fs.f_frsize,
 quota='Shared PFS2TiB; no per-project quota interface exposed',started_unix=time.time(),
 nominal_policy_controls=972800,maximum_all_in_actions=1000000),indent=2)+'\n')
PY
"$TASK_PY" -B "$TASK_META/scripts/[train]gpu_probe.py" --expected-count 2 --purpose training \
 --baige-container --baige-exclusive-training --output "$TASK_OUTPUT/gpu-probe.json"
timeout --signal=TERM --kill-after=60s 28200 "$TASK_PY" -B "$TASK_META/scripts/[train]robocasa_flow_ppo.py"
