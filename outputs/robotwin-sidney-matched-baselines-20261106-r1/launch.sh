#!/usr/bin/env bash
set -Eeuo pipefail
TASK_SOURCE=${1:?reviewed source commit}
TASK_OUT=${2:?fresh output identity}
TASK_PHASE=${3:?qualification or ppo or grpo}
TASK_PLAN=${4:?admitted funding/storage contract}
TASK_LIMIT=${5:?maximum worker seconds}
[[ $TASK_PHASE == optimizer_probe || $TASK_PHASE == gradient_probe || $TASK_PHASE == qualification || $TASK_PHASE == ppo || $TASK_PHASE == grpo ]]
[[ $TASK_LIMIT =~ ^[0-9]+$ ]]
TASK_CONFIG_MODE=$TASK_PHASE
TASK_MINIMUM_HOST_GIB=1024
if [[ $TASK_PHASE == gradient_probe || $TASK_PHASE == optimizer_probe ]]; then
  [[ $TASK_LIMIT -le 840 ]]; TASK_ALGORITHM=grpo;TASK_CONFIG_MODE=qualification
  TASK_MINIMUM_HOST_GIB=256
elif [[ $TASK_PHASE == qualification ]]; then
  [[ $TASK_LIMIT -le 2640 ]]; TASK_ALGORITHM=grpo
else
  [[ $TASK_LIMIT -le 10200 ]]; TASK_ALGORITHM=$TASK_PHASE
fi
[[ $(uname -s) == Linux && $(hostname) == job-* ]]
TASK_CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
TASK_RUN=$TASK_CODE/outputs/robotwin-sidney-matched-baselines-20261106-r1
TASK_SHARED=$TASK_CODE/outputs/robotwin-sidney-rlinf-ppo-20261105-r1
TASK_PYTHON=/mowenjie/fluffy-bassoon-2/envs/robotwin-vlrla-20261029-r1/bin/python
[[ $(git -C "$TASK_CODE" rev-parse HEAD) == "$TASK_SOURCE" ]]
[[ -z $(git -C "$TASK_CODE" status --porcelain --untracked-files=no) ]]
[[ ! -e $TASK_OUT && -f $TASK_PLAN ]]
mkdir -p "$TASK_OUT"
exec > "$TASK_OUT/worker.log" 2>&1
export TASK_LIMIT TASK_CODE TASK_SOURCE
export REPO_PATH=/mowenjie/fluffy-bassoon-2/sources/robotwin-baseline-smoke-20261028-r1/RLinf
export ROBOTWIN_PATH=/mowenjie/fluffy-bassoon-2/sources/robotwin-baseline-smoke-20261028-r1/RoboTwin
export ROBOTWIN_ASSETS_PATH=$ROBOTWIN_PATH ASSETS_PATH=$ROBOTWIN_PATH
export EMBODIED_PATH=$REPO_PATH/examples/embodiment
export PYTHONPATH=$TASK_CODE/src:$REPO_PATH:$ROBOTWIN_PATH:$TASK_RUN:$TASK_SHARED
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export ROBOT_PLATFORM=ALOHA MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export FLUFFY_GPU_EXECUTION=baige NCCL_P2P_DISABLE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_ALLOC_CONF=expandable_segments:True
if [[ ${CUDA_VISIBLE_DEVICES+x} != x ]]; then export CUDA_VISIBLE_DEVICES=0,1,2,3; fi
export NATIVE_PPO_WORLD=4 NATIVE_PPO_PLACEMENT=0-3 NATIVE_PPO_MODE=train
export NATIVE_PPO_OUTPUT=$TASK_OUT NATIVE_PPO_ACCOUNTING_ROOT=$TASK_OUT/action-accounting
export NATIVE_PPO_ATTEMPT=train
export NATIVE_PPO_MODEL=/mowenjie/fluffy-bassoon-2/cache/robotwin-sidney-rlinf-20261105-r1
export NATIVE_PPO_INITIALIZER_REVISION=e49e2ab6c11f07511573b67261bd129e88d0a416
TASK_STARTED=$(date +%s)
export NATIVE_PPO_DEADLINE=$((TASK_STARTED+TASK_LIMIT-60))
TASK_MEMORY_PID=
finish() {
  TASK_RC=$?
  trap - EXIT
  if [[ -n $TASK_MEMORY_PID ]]; then kill "$TASK_MEMORY_PID" 2>/dev/null || true; fi
  TASK_ENDED=$(date +%s)
  printf '{"exit_code":%s,"started_unix":%s,"ended_unix":%s,"worker_gpu_seconds":%s}\n' \
    "$TASK_RC" "$TASK_STARTED" "$TASK_ENDED" "$((4*(TASK_ENDED-TASK_STARTED)))" > "$TASK_OUT/EXIT.json"
}
trap finish EXIT
"$TASK_PYTHON" "$TASK_CODE/scripts/[train]gpu_probe.py" --baige-container \
  --baige-exclusive-training --purpose training --expected-count 4 --output "$TASK_OUT/gpu-probe.json"
export LYCO_BAIGE_GPU_MAP=$TASK_OUT/gpu-probe.json
"$TASK_PYTHON" "$TASK_SHARED/host_memory.py" "$TASK_OUT" --minimum-gib "$TASK_MINIMUM_HOST_GIB" --once
"$TASK_PYTHON" "$TASK_SHARED/host_memory.py" "$TASK_OUT" --minimum-gib "$TASK_MINIMUM_HOST_GIB" > "$TASK_OUT/host-memory.log" 2>&1 &
TASK_MEMORY_PID=$!
[[ $(git -C "$REPO_PATH" rev-parse HEAD) == c605b9d55ed93a180119b099e6f836abff293b08 ]]
[[ $(git -C "$ROBOTWIN_PATH" rev-parse HEAD) == dca9ec682688821c42944eacd7f5dd9bfb15396f ]]
[[ -z $(git -C "$REPO_PATH" status --porcelain --untracked-files=no) ]]
[[ -z $(git -C "$ROBOTWIN_PATH" status --porcelain --untracked-files=no) ]]
mkdir -p /root/.sapien/physx
if [[ ! -e /root/.sapien/physx/105.1-physx-5.3.1.patch0 ]]; then
  ln -s /mowenjie/fluffy-bassoon/inputs/maniskill-scene-assets-20261028-r1/physx/105.1-physx-5.3.1.patch0 /root/.sapien/physx/105.1-physx-5.3.1.patch0
fi
"$TASK_PYTHON" "$TASK_RUN/admission.py" "$TASK_PLAN" "$TASK_PHASE"
if [[ $TASK_PHASE == qualification || $TASK_PHASE == gradient_probe || $TASK_PHASE == optimizer_probe ]]; then
  "$TASK_PYTHON" "$TASK_RUN/numerical_qualification.py" "$TASK_OUT"
  for TASK_METHOD in ppo grpo; do
    "$TASK_PYTHON" "$TASK_RUN/configuration.py" validate "$TASK_METHOD" scientific > "$TASK_OUT/config-$TASK_METHOD-scientific.json"
  done
fi
"$TASK_PYTHON" "$TASK_RUN/configuration.py" validate "$TASK_ALGORITHM" "$TASK_CONFIG_MODE" > "$TASK_OUT/config-validation.json"
TASK_PARENT_OUT=$TASK_OUT
TASK_ENTRY=$TASK_RUN/paired_entry.py
if [[ $TASK_PHASE == gradient_probe || $TASK_PHASE == optimizer_probe ]]; then
  TASK_ENTRY=$TASK_RUN/gradient_probe.py
  export TASK_FIXTURE=/mowenjie/fluffy-bassoon-2/outputs/robotwin-sidney-matched-baselines-20261106-r1/qual02
  export TASK_PROBE_MODE=train
  if [[ $TASK_PHASE == optimizer_probe ]]; then export TASK_OPTIMIZER_DIAGNOSTIC=1; fi
fi
if [[ $TASK_PHASE == qualification || $TASK_PHASE == gradient_probe || $TASK_PHASE == optimizer_probe ]]; then
  export NATIVE_PPO_OUTPUT=$TASK_PARENT_OUT/train
  mkdir "$NATIVE_PPO_OUTPUT"
fi
mapfile -t TASK_OVERRIDES < <("$TASK_PYTHON" "$TASK_RUN/configuration.py" "$TASK_ALGORITHM" "$TASK_CONFIG_MODE")
[[ ${#TASK_OVERRIDES[@]} -ge 18 ]]
TASK_REMAINING=$((NATIVE_PPO_DEADLINE-$(date +%s)-60))
[[ $TASK_REMAINING -gt 240 ]]
timeout --signal=INT --kill-after=20s "$TASK_REMAINING" "$TASK_PYTHON" "$TASK_ENTRY" \
  --config-path "$TASK_CODE/configs/native_ppo" --config-name sidney_phone_two \
  "${TASK_OVERRIDES[@]}" > "$TASK_PARENT_OUT/train.log" 2>&1
if [[ $TASK_PHASE == qualification || $TASK_PHASE == gradient_probe || $TASK_PHASE == optimizer_probe ]]; then
  TASK_RESUME=$TASK_PARENT_OUT/train/sidney-paired-grpo/checkpoints/global_step_1
  # This scheduler container is exclusive to this qualification. The completed
  # first process has saved its full boundary; release its Ray runtime before
  # starting an independent native checkpoint-loading process in the same job.
  "$TASK_PYTHON" -m ray.scripts.scripts stop --force > "$TASK_PARENT_OUT/ray-cleanup.log" 2>&1
  export NATIVE_PPO_OUTPUT=$TASK_PARENT_OUT/restore NATIVE_PPO_ATTEMPT=restore
  mkdir "$NATIVE_PPO_OUTPUT"
  if [[ $TASK_PHASE == gradient_probe || $TASK_PHASE == optimizer_probe ]]; then
    export TASK_PROBE_MODE=restore TASK_PROBE_CHECKPOINT=$TASK_PARENT_OUT/train/checkpoint1
    if [[ $TASK_PHASE == optimizer_probe ]]; then export TASK_PROBE_CHECKPOINT=$TASK_PARENT_OUT/train/checkpoint0; fi
    mapfile -t TASK_OVERRIDES < <("$TASK_PYTHON" "$TASK_RUN/configuration.py" grpo qualification)
  else
    mapfile -t TASK_OVERRIDES < <("$TASK_PYTHON" "$TASK_RUN/configuration.py" grpo qualification "$TASK_RESUME")
  fi
  [[ ${#TASK_OVERRIDES[@]} -ge 18 ]]
  TASK_REMAINING=$((NATIVE_PPO_DEADLINE-$(date +%s)-60))
  [[ $TASK_REMAINING -gt 120 ]]
  TASK_RESTORE_STARTED=$(date +%s)
  timeout --signal=INT --kill-after=20s "$TASK_REMAINING" "$TASK_PYTHON" "$TASK_ENTRY" \
    --config-path "$TASK_CODE/configs/native_ppo" --config-name sidney_phone_two \
    "${TASK_OVERRIDES[@]}" > "$TASK_PARENT_OUT/restore.log" 2>&1
  printf '{"seconds":%s,"includes_fresh_workers_model_native_DCP_and_all_worker_sidecars":true}\n' \
    "$(($(date +%s)-TASK_RESTORE_STARTED))" > "$TASK_PARENT_OUT/recovery-timing.json"
fi
if [[ $TASK_PHASE == ppo || $TASK_PHASE == grpo ]]; then
  # Verify this arm's actual trained actor/Adam boundary in a new process while
  # preserving the declared simulator cases, SGD count and 170-minute ceiling.
  "$TASK_PYTHON" -m ray.scripts.scripts stop --force > "$TASK_PARENT_OUT/ray-cleanup-final.log" 2>&1
  export TASK_PROBE_MODE=restore
  export TASK_FIXTURE=/mowenjie/fluffy-bassoon-2/outputs/robotwin-sidney-matched-baselines-20261106-r1/qual02
  export TASK_PROBE_CHECKPOINT=$TASK_PARENT_OUT/sidney-paired-$TASK_ALGORITHM/checkpoints/global_step_2
  export NATIVE_PPO_OUTPUT=$TASK_PARENT_OUT/final-restore NATIVE_PPO_ATTEMPT=final-restore
  mkdir "$NATIVE_PPO_OUTPUT"
  mapfile -t TASK_OVERRIDES < <("$TASK_PYTHON" "$TASK_RUN/configuration.py" "$TASK_ALGORITHM" scientific)
  TASK_REMAINING=$((NATIVE_PPO_DEADLINE-$(date +%s)-60))
  [[ $TASK_REMAINING -gt 120 ]]
  TASK_RESTORE_STARTED=$(date +%s)
  timeout --signal=INT --kill-after=20s "$TASK_REMAINING" "$TASK_PYTHON" "$TASK_RUN/gradient_probe.py" \
    --config-path "$TASK_CODE/configs/native_ppo" --config-name sidney_phone_two \
    "${TASK_OVERRIDES[@]}" > "$TASK_PARENT_OUT/final-restore.log" 2>&1
  printf '{"seconds":%s,"actor_only":true,"extra_optimizer_updates":0,"extra_simulator_controls":0}\n' \
    "$(($(date +%s)-TASK_RESTORE_STARTED))" > "$TASK_PARENT_OUT/final-restore-timing.json"
fi

