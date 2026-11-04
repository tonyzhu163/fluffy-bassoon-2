#!/usr/bin/env bash
set -Eeuo pipefail
: "${TASK_META:?}" "${TASK_CODE_PIN:?}" "${TASK_ATTEMPT:?}" "${TASK_OUTPUT:?}"
: "${ROBOCASA_FLOW_TASK:?}" "${ROBOCASA_SIM_PPO_ACCOUNTING:?}" "${TASK_SEGMENT:?}"
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_PY=$TASK_ROOT/setups/robocasa365-native-20261101-r1/env/bin/python
TASK_CONFIG=${TASK_CONFIG:-robocasa365_pi05_flow_sde_ppo_vlm_critic}
RLINF_EXPECTED_COMMIT=8f9bfa7661928ef8c17494b04f4d7bf3125634e1
export TASK_STARTED=$(date +%s)
export TASK_GPUS=${TASK_GPUS:-8} TASK_CPUS=${TASK_CPUS:-168} TASK_RAM_GIB=${TASK_RAM_GIB:-1280} TASK_SHM_GIB=${TASK_SHM_GIB:-32}
export TASK_WALL_SECONDS=${TASK_WALL_SECONDS:-14400}
export TASK_DEADLINE=$((TASK_STARTED+TASK_WALL_SECONDS-120))
export RLINF_PATH=$TASK_ROOT/sources/critic-selected-a1229239-r1
export PYTHONPATH=$TASK_META/src:$RLINF_PATH
export ROBOCASA_ASSETS_PATH=$TASK_ROOT/setups/robocasa365-native-20261101-r1/assets
export OPENPI_DATA_HOME=$TASK_ROOT/setups/robocasa365-native-20261101-r1/cache/.cache/openpi
export ROBOCASA_PI05_TORCH_MODEL=$TASK_ROOT/inputs/robocasa365-public-pi05-c484-torch-20261101-r2
export ROBOCASA_PI05_NATIVE_NORM=$TASK_ROOT/inputs/robocasa365-public-pi05-c484-75000-20261101-r1/assets/norm_stats.json
export ROBOCASA_SIM_PPO_OUTPUT=$TASK_OUTPUT
export ROBOCASA_SIM_PPO_ACCOUNTING
export ROBOCASA_SIM_PPO_ROLLOUT_PLACEMENT=${ROBOCASA_SIM_PPO_ROLLOUT_PLACEMENT:-0-7}
export ROBOCASA_SIM_PPO_ACTOR_PLACEMENT=${ROBOCASA_SIM_PPO_ACTOR_PLACEMENT:-0-7}
export ROBOCASA_SIM_PPO_MICRO_BATCH=${ROBOCASA_SIM_PPO_MICRO_BATCH:-32}
export ROBOCASA_SIM_PPO_WORLD_SIZE=${ROBOCASA_SIM_PPO_WORLD_SIZE:-8}
export ROBOCASA_FLOW_TRAIN_ENVS=${ROBOCASA_FLOW_TRAIN_ENVS:-128}
export ROBOCASA_FLOW_ROUNDS=${ROBOCASA_FLOW_ROUNDS:-1}
export ROBOCASA_FLOW_SAMPLING_RECORD=${ROBOCASA_FLOW_SAMPLING_RECORD:-full}
export ROBOCASA_FLOW_ACTOR_BATCH_DROP=${ROBOCASA_FLOW_ACTOR_BATCH_DROP:-'["forward_inputs.observation/image","forward_inputs.observation/wrist_image","forward_inputs.observation/extra_view_image","forward_inputs.chains"]'}
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl NCCL_P2P_DISABLE=1 FLUFFY_GPU_EXECUTION=baige
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((TASK_GPUS-1)))}
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 WANDB_MODE=disabled
test ! -e "$TASK_OUTPUT"
mkdir -p "$TASK_OUTPUT"
if [[ $TASK_SEGMENT == q1 ]]; then touch "$TASK_OUTPUT/CHECKPOINT_AND_EXIT"; fi
exec > >(tee "$TASK_OUTPUT/console.log") 2>&1
finish() {
 TASK_RC=$?
 trap - EXIT
 if [[ -n ${TASK_STORAGE_PID:-} ]]; then kill "$TASK_STORAGE_PID" 2>/dev/null || true; fi
 if [[ -n ${TASK_TELEMETRY_PID:-} ]]; then kill "$TASK_TELEMETRY_PID" 2>/dev/null || true; fi
 TASK_FINISHED=$(date +%s)
 printf '{"started_unix":%s,"finished_unix":%s,"allocated_gpus":%s,"observed_gpu_seconds":%s,"exit_code":%s}\n' \
  "$TASK_STARTED" "$TASK_FINISHED" "$TASK_GPUS" "$((TASK_GPUS*(TASK_FINISHED-TASK_STARTED)))" "$TASK_RC" > "$TASK_OUTPUT/clock.json"
 exit "$TASK_RC"
}
trap finish EXIT
test "$(git -C "$TASK_META" rev-parse HEAD)" = "$TASK_CODE_PIN"
test -z "$(git -C "$TASK_META" status --porcelain)"
test "$(git -C "$RLINF_PATH" rev-parse HEAD)" = "$RLINF_EXPECTED_COMMIT"
"$TASK_PY" -B - <<'PY'
import hashlib,json,os,pathlib,socket,time
root=pathlib.Path(os.environ['TASK_OUTPUT'])
assert socket.gethostname().startswith('job-')
cpu=pathlib.Path('/sys/fs/cgroup/cpu.max').read_text().split()
assert int(cpu[0])/int(cpu[1])>=int(os.environ['TASK_CPUS'])
assert int(pathlib.Path('/sys/fs/cgroup/memory.max').read_text())>=int(os.environ['TASK_RAM_GIB'])*(1<<30)
fs=os.statvfs(root)
assert fs.f_bavail*fs.f_frsize>=200*(1<<30), 'PFS free space below campaign stop floor'
shm=os.statvfs('/dev/shm');assert shm.f_blocks*shm.f_frsize>=int(os.environ['TASK_SHM_GIB'])*(1<<30)
(root/'write-probe').write_text('Flow SDE PPO/GRPO allocated project write probe\n')
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
 runtime=os.environ['RLINF_PATH'],runtime_commit='8f9bfa7661928ef8c17494b04f4d7bf3125634e1',
 config_name=os.environ.get('TASK_CONFIG','robocasa365_pi05_flow_sde_ppo_vlm_critic'),
 model_sha256=model,norm_sha256=norm,
 allocated_gpus=int(os.environ['TASK_GPUS']),CPU=int(os.environ['TASK_CPUS']),
 RAM_GiB=int(os.environ['TASK_RAM_GIB']),SHM_GiB=int(os.environ['TASK_SHM_GIB']),free_bytes=fs.f_bavail*fs.f_frsize,
 quota='Shared PFS2TiB; no per-project quota interface exposed',started_unix=time.time(),
 task=os.environ['ROBOCASA_FLOW_TASK'], segment=os.environ['TASK_SEGMENT'],
 batch_layout=dict(world=8, train_envs_per_rank=16, eval_envs_per_rank=8,
                   actor_ranks=8 if os.environ.get('ROBOCASA_SIM_PPO_ACTOR_PLACEMENT','0-7')=='0-7' else 2),
 nominal_controls_source='flow-contract.json'),indent=2)+'\n')
PY
"$TASK_PY" -B "$TASK_META/scripts/[train]gpu_probe.py" --expected-count "$TASK_GPUS" --purpose training \
 --baige-container --baige-exclusive-training --output "$TASK_OUTPUT/gpu-probe.json"
(while true; do
 nvidia-smi --query-gpu=timestamp,index,uuid,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
 sleep 15
done) > "$TASK_OUTPUT/gpu-memory.csv" &
TASK_TELEMETRY_PID=$!
(while sleep 30; do
 TASK_FREE=$(df -Pk "$TASK_OUTPUT" | awk 'NR==2 {print $4}')
 if (( TASK_FREE < 209715200 )); then touch "$TASK_OUTPUT/CHECKPOINT_AND_EXIT"; fi
done) > "$TASK_OUTPUT/storage-guard.log" 2>&1 &
TASK_STORAGE_PID=$!
timeout --signal=TERM --kill-after=60s "$TASK_WALL_SECONDS" "$TASK_PY" -B "$TASK_META/scripts/[train]robocasa_flow_ppo.py" --config-name "$TASK_CONFIG"
