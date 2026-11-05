#!/usr/bin/env bash
set -Eeuo pipefail
TASK_SOURCE=${1:?reviewed exact source}
TASK_OUT=${2:?fresh training output}
TASK_BRIDGE=${3:?completed bridge output}
TASK_PRIOR_GPU_SECONDS=${4:?conservative completed task GPU charge}
TASK_WORLD=${5:-4}
[[ $TASK_WORLD == 4 ]]
TASK_LIMIT=${6:-$((12000/TASK_WORLD))}
TASK_RESUME=${7:-}
TASK_ACCOUNTING=${8:-$TASK_OUT/action-accounting}
TASK_CAPACITY=${9:-}
TASK_MEMORY=${10:-$(dirname "$TASK_OUT")/memory02}
TASK_MEMORY_STRATEGY=${11:-per_scene_close}
[[ $TASK_MEMORY_STRATEGY == per_scene_close || $TASK_MEMORY_STRATEGY == native_batch_end ]]
TASK_MINIMUM_HOST_GIB=512;TASK_TRIM_AFTER_CLOSE=true
if [[ $TASK_MEMORY_STRATEGY == native_batch_end ]]; then
  TASK_MINIMUM_HOST_GIB=1024;TASK_TRIM_AFTER_CLOSE=false
fi
[[ $TASK_LIMIT =~ ^[0-9]+$ && $TASK_LIMIT -le 10800 ]]
TASK_CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
TASK_RUN_CODE=$TASK_CODE/outputs/robotwin-sidney-rlinf-ppo-20261105-r1
TASK_PYTHON=/mowenjie/fluffy-bassoon-2/envs/robotwin-vlrla-20261029-r1/bin/python
[[ $(uname -s) == Linux && $(hostname) == job-* ]]
[[ $(git -C "$TASK_CODE" rev-parse HEAD) == "$TASK_SOURCE" ]]
[[ -z $(git -C "$TASK_CODE" status --porcelain --untracked-files=no) ]]
[[ ! -e $TASK_OUT && $TASK_PRIOR_GPU_SECONDS =~ ^[0-9]+$ ]]
# Measured256-scene reset/rollout latency requires a longer continuation.
# Twenty additional GPU-hours including the host-memory retry;
# this remains within the repository-authorized50GPUh task scope.
[[ $((TASK_PRIOR_GPU_SECONDS+TASK_WORLD*(TASK_LIMIT+60))) -le 72000 ]]
mkdir -p "$TASK_OUT"
exec > "$TASK_OUT/worker.log" 2>&1
export TASK_SOURCE TASK_BRIDGE
export REPO_PATH=/mowenjie/fluffy-bassoon-2/sources/robotwin-baseline-smoke-20261028-r1/RLinf
export ROBOTWIN_PATH=/mowenjie/fluffy-bassoon-2/sources/robotwin-baseline-smoke-20261028-r1/RoboTwin
export ROBOTWIN_ASSETS_PATH=$ROBOTWIN_PATH ASSETS_PATH=$ROBOTWIN_PATH
export EMBODIED_PATH=$REPO_PATH/examples/embodiment
export PYTHONPATH=$TASK_CODE/src:$REPO_PATH:$ROBOTWIN_PATH
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export ROBOT_PLATFORM=ALOHA MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export FLUFFY_GPU_EXECUTION=baige NCCL_P2P_DISABLE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_ALLOC_CONF=expandable_segments:True
if [[ ${CUDA_VISIBLE_DEVICES+x} != x ]]; then export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((TASK_WORLD-1))); fi
export NATIVE_PPO_WORLD=$TASK_WORLD NATIVE_PPO_PLACEMENT=0-$((TASK_WORLD-1)) NATIVE_PPO_MODE=train
export NATIVE_PPO_OUTPUT=$TASK_OUT NATIVE_PPO_ACCOUNTING_ROOT=$TASK_ACCOUNTING
export NATIVE_PPO_ATTEMPT=$(basename "$TASK_OUT")
export NATIVE_PPO_MODEL=/mowenjie/fluffy-bassoon-2/cache/robotwin-sidney-rlinf-20261105-r1
export NATIVE_PPO_INITIALIZER_REVISION=e49e2ab6c11f07511573b67261bd129e88d0a416
TASK_STARTED=$(date +%s)
TASK_MEMORY_PID=
export NATIVE_PPO_DEADLINE=$((TASK_STARTED+TASK_LIMIT-60))
finish() {
  TASK_RC=$?
  trap - EXIT
  if [[ -n $TASK_MEMORY_PID ]]; then kill "$TASK_MEMORY_PID" 2>/dev/null || true; fi
  TASK_ENDED=$(date +%s)
  printf '{"exit_code":%s,"started_unix":%s,"ended_unix":%s,"worker_gpu_seconds":%s,"prior_bridge_gpu_seconds":%s}\n' \
    "$TASK_RC" "$TASK_STARTED" "$TASK_ENDED" "$((TASK_WORLD*(TASK_ENDED-TASK_STARTED)))" "$TASK_PRIOR_GPU_SECONDS" > "$TASK_OUT/EXIT.json"
}
trap finish EXIT
"$TASK_PYTHON" "$TASK_CODE/scripts/[train]gpu_probe.py" --baige-container \
  --baige-exclusive-training --purpose training --expected-count "$TASK_WORLD" --output "$TASK_OUT/gpu-probe.json"
export LYCO_BAIGE_GPU_MAP=$TASK_OUT/gpu-probe.json
"$TASK_PYTHON" "$TASK_RUN_CODE/host_memory.py" "$TASK_OUT" --minimum-gib "$TASK_MINIMUM_HOST_GIB" --once
"$TASK_PYTHON" "$TASK_RUN_CODE/host_memory.py" "$TASK_OUT" --minimum-gib "$TASK_MINIMUM_HOST_GIB" > "$TASK_OUT/host-memory.log" 2>&1 &
TASK_MEMORY_PID=$!
[[ $(git -C "$REPO_PATH" rev-parse HEAD) == c605b9d55ed93a180119b099e6f836abff293b08 ]]
[[ $(git -C "$ROBOTWIN_PATH" rev-parse HEAD) == dca9ec682688821c42944eacd7f5dd9bfb15396f ]]
[[ -z $(git -C "$REPO_PATH" status --porcelain --untracked-files=no) ]]
[[ -z $(git -C "$ROBOTWIN_PATH" status --porcelain --untracked-files=no) ]]
mkdir -p /root/.sapien/physx
if [[ ! -e /root/.sapien/physx/105.1-physx-5.3.1.patch0 ]]; then
  ln -s /mowenjie/fluffy-bassoon/inputs/maniskill-scene-assets-20261028-r1/physx/105.1-physx-5.3.1.patch0 /root/.sapien/physx/105.1-physx-5.3.1.patch0
fi
export TASK_CAPACITY TASK_MEMORY TASK_MEMORY_STRATEGY
TASK_MICRO=$("$TASK_PYTHON" - <<'PY'
import hashlib,json,os
from pathlib import Path
bridge=Path(os.environ['TASK_BRIDGE'])
result=json.loads((bridge/'rlinf/RESULT.json').read_text())
assert result['status']=='complete' and result['optimizer_updates']==0
assert result['episode']['complete'] and result['episode']['task']=='place_phone_stand'
assert result['execution_microbatch'] in (8,16,32)
receipt=json.loads((bridge/'REUSED-INPUTS.json').read_text())['conversion']
cache=Path(os.environ['NATIVE_PPO_MODEL'])
for filename,key in [('model.safetensors','model_sha256'),('sidney-processors.json','processors_sha256')]:
    h=hashlib.sha256()
    with (cache/filename).open('rb') as handle:
        for block in iter(lambda:handle.read(8*1024**2),b''):h.update(block)
    assert h.hexdigest()==receipt[key]
Path(os.environ['NATIVE_PPO_OUTPUT'],'bridge-admission.json').write_text(json.dumps(result,indent=2)+'\n')
# Qualification uses the same64-scene native cameras/physics. Include model,
# Ray and pending CPU rollout/checkpoint memory beyond its model-free RSS.
# These are conservative admission allowances, not a measured full-phase peak;
# host-memory telemetry remains required throughout the scientific retry.
memory_path=Path(os.environ['TASK_MEMORY'])/'RESULT.json'
memory=json.loads(memory_path.read_text())
assert memory['scenes']==64 and not memory['model_loaded'] and memory['optimizer_updates']==0
for event in memory['events']:
    assert all(event[key] for key in ('allocator_observation_identity',
        'python_numpy_torch_rng_identity','scene_identity','motor_input_identity'))
strategy=os.environ['TASK_MEMORY_STRATEGY']
if strategy=='per_scene_close':
    assert memory['status']=='complete' and len(memory['events']) in (2,3)
    assert memory['paired_reset_rgb_bit_identity']
    assert memory['paired_reset_max_abs']['states']<=1e-6
    assert memory['paired_motor_prefix_state_max_abs']<=1e-6 and memory['paired_motor_prefix_reward_identity']
    selected_round=memory.get('per_close_round_index',2)
else:
    # Use the already completed native batch-end mode, without the experimental
    # per-scene callback. Preserve the failed diagnostic's global status.
    assert memory_path.parent.name=='memory02'
    selected_round=1
    native=next(r for r in memory['snapshots'] if r['stage']=='round1/after_reset')
    assert native['native_malloc_trim_reset_interval']==1 and not native['trim_after_each_scene_close']
    assert any(r['stage']=='round1/chunk0' and r['physical_controls']==3200 for r in memory['snapshots'])
    assert any(r['round']==1 for r in memory['events'])
selected=[row for row in memory['snapshots'] if row['stage'].startswith(f'round{selected_round}/')]
peak=max(max(row['smaps']['Rss'],row.get('reset_peak',{}).get('rss_bytes',0)) for row in selected)
headroom_allowances=dict(environment_runtime_per_rank_bytes=10*2**30,
    other_workers_bytes=80*2**30,pending_rollout_and_checkpoint_bytes=64*2**30)
bound=4*(peak+headroom_allowances['environment_runtime_per_rank_bytes'])+sum(
    headroom_allowances[key] for key in ('other_workers_bytes','pending_rollout_and_checkpoint_bytes'))
limit=json.loads(Path(os.environ['NATIVE_PPO_OUTPUT'],'host-memory.json').read_text())['limit_bytes']
assert bound<.95*limit,f'Qualified full-phase host-memory bound exceeds Ray threshold: {bound/2**30:.1f}GiB'
Path(os.environ['NATIVE_PPO_OUTPUT'],'memory-admission.json').write_text(json.dumps(dict(
    qualification=str(memory_path),sha256=hashlib.sha256(memory_path.read_bytes()).hexdigest(),
    strategy=strategy,qualification_global_status=memory['status'],selected_completed_round=selected_round,
    scene_count_per_rank=64,measured_peak_rss_bytes=peak,allowances=headroom_allowances,
    projected_full_phase_bytes=bound,cgroup_limit_bytes=limit,ray_memory_guard_unchanged=True),indent=2)+'\n')
capacity=os.environ.get('TASK_CAPACITY')
if capacity:
    selection=Path(capacity)/'SELECTION.json'
    if selection.exists():
        qualified=json.loads(selection.read_text())
        assert qualified['status']=='complete' and qualified['global_batch_size']==2048 and qualified['world_size']==4
        micro=qualified['selected_micro_batch_size'];assert micro in (64,128,256,512)
        assert qualified['selection_basis']=='maximum measured throughput with memory headroom'
        for rank in range(4):
            row=json.loads((Path(capacity)/f'micro-{micro}/rank-{rank:02d}.json').read_text())
            assert row['status']=='complete' and len(row['updates'])==4
        print(micro)
    else:
        qualified=json.loads((Path(capacity)/'RESULT.json').read_text())
        assert qualified['status']=='complete' and qualified['optimizer_updates']==0
        assert qualified['execution_microbatch']==64 and qualified['backward']['frozen_vlm_has_no_gradients']
        assert qualified['backward']['logprob_max_abs_error']<=1e-3
        print(64)
    Path(os.environ['NATIVE_PPO_OUTPUT'],'capacity-admission.json').write_text(json.dumps(qualified,indent=2)+'\n')
else:print(result['execution_microbatch'])
PY
)
cd "$TASK_CODE"
"$TASK_PYTHON" -m pytest -q tests/test_robotwin_noise.py > "$TASK_OUT/environment-tests.log" 2>&1
"$TASK_PYTHON" -m pytest -q tests/test_rlinf_native_ppo.py \
  -k 'unresolved_calls or original_rollout or worker_rng or sampler_and_eval_rng or actor_sidecar_requires' > "$TASK_OUT/core-tests.log" 2>&1
# Compose/validate the actual source configuration inside the allocation before
# model initialization. It keeps global2048 and only tunes execution microbatch.
"$TASK_PYTHON" "$TASK_RUN_CODE/validate_config.py" "$TASK_CODE" "$TASK_MICRO" "$TASK_MEMORY_STRATEGY"
"$TASK_PYTHON" - <<'PY'
import hashlib,json,os
from pathlib import Path
root=Path(os.environ['NATIVE_PPO_ACCOUNTING_ROOT']);world=int(os.environ['NATIVE_PPO_WORLD'])
rows=[]
for rank in range(world):
    files=[];charged=0
    for path in sorted((root/f'rank-{rank:02d}').glob('*.json')):
        raw=path.read_bytes();value=json.loads(raw)
        assert value['rank_action_ceiling'] in (1000000//world,1200000//world)
        amount=sum(call.get('executed_or_conservative',call['requested']) for call in value['calls'])
        charged+=amount;files.append(dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),charged=amount))
    assert charged+(819200+30720)//world<=1200000//world
    rows.append(dict(rank=rank,charged_before_attempt=charged,previous_files=files))
Path(os.environ['NATIVE_PPO_OUTPUT'],'action-admission.json').write_text(json.dumps(dict(
    maximum_actions=1200000,previous_maximum_actions=1000000,old_files_unchanged=True,ranks=rows),indent=2)+'\n')
PY
TASK_RECOMPUTE=false
if [[ $TASK_MICRO != $((256/TASK_WORLD)) ]]; then TASK_RECOMPUTE=true; fi
TASK_RESUME_ARGS=()
if [[ -n $TASK_RESUME ]]; then
  "$TASK_PYTHON" - "$TASK_RESUME" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]);receipt=json.loads((root/'pilot-boundary.json').read_text())
assert receipt['completed'] and receipt['global_step'] in (0,1)
assert len(receipt['workers'])==12
for row in receipt['actor_checkpoint_files']:
    p=Path(row['path']);assert p.is_file() and p.stat().st_size==row['bytes']
for role in ('actor','rollout','env'):
    for rank in range(4):assert (root/'pilot'/f'{role}-{rank:02d}.pt').is_file()
PY
  TASK_RESUME_ARGS+=("runner.resume_dir=$TASK_RESUME")
fi
timeout --signal=INT --kill-after=20s "$((TASK_LIMIT-120))" "$TASK_PYTHON" "$TASK_RUN_CODE/train_entry.py" \
  --config-path "$TASK_CODE/configs/native_ppo" --config-name sidney_phone_two \
  "actor.micro_batch_size=$TASK_MICRO" "algorithm.recompute_logprobs=$TASK_RECOMPUTE" \
  "native_pilot.memory_strategy=$TASK_MEMORY_STRATEGY" "native_pilot.trim_after_scene_close=$TASK_TRIM_AFTER_CLOSE" \
  "${TASK_RESUME_ARGS[@]}" > "$TASK_OUT/train.log" 2>&1
