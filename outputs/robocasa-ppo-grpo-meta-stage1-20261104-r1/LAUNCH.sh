#!/usr/bin/env bash
# Execute inside an exclusive Baige scheduler container after budget approval.
set -Eeuo pipefail
: "${TASK_META:?clean deployed main checkout}" "${TASK_CODE_PIN:?verified source commit}"
TASK_ARM=${1:?ppo or grpo}
case "$TASK_ARM" in
  ppo) export TASK_CONFIG=robocasa365_pi05_flow_sde_ppo_vlm_critic ;;
  grpo) export TASK_CONFIG=robocasa365_pi05_flow_sde_grpo ;;
  *) exit 2 ;;
esac
TASK_RUN=/mowenjie/fluffy-bassoon-2/outputs/robocasa-ppo-grpo-meta-stage1-20261104-r1
export TASK_OUTPUT=$TASK_RUN/$TASK_ARM/attempt-01
export ROBOCASA_SIM_PPO_ACCOUNTING=$TASK_RUN/$TASK_ARM/accounting
export TASK_ATTEMPT=$TASK_ARM-attempt-01 TASK_SEGMENT=stage1
export TASK_GPUS=8 TASK_CPUS=168 TASK_RAM_GIB=1280 TASK_SHM_GIB=32
export TASK_WALL_SECONDS=14400
export ROBOCASA_FLOW_TASK=OpenDrawer
export ROBOCASA_FLOW_TRAIN_SEED=2100000 ROBOCASA_FLOW_EVAL_SEED=610100
export ROBOCASA_FLOW_MAX_EPOCHS=40 ROBOCASA_FLOW_UPDATE_EPOCHS=1
export ROBOCASA_FLOW_VAL_INTERVAL=10 ROBOCASA_FLOW_SAVE_INTERVAL=10
export ROBOCASA_FLOW_NOISE_LEVEL=0.3 ROBOCASA_FLOW_IGNORE_LAST=true
export ROBOCASA_FLOW_GROUP_SIZE=8
export ROBOCASA_FLOW_MAX_ACTIONS=4100000
export ROBOCASA_FLOW_TRAIN_ENVS=128 ROBOCASA_FLOW_ROUNDS=1
export ROBOCASA_FLOW_SAMPLING_RECORD=full
export ROBOCASA_SIM_PPO_GLOBAL_ESCROW=true
export ROBOCASA_FLOW_ACTOR_BATCH_DROP='[]'
# Preserve the qualified Stage 1 actor layout when current-main defaults change.
export ROBOCASA_SIM_PPO_ACTOR_PLACEMENT=0-1
export ROBOCASA_SIM_PPO_MICRO_BATCH=16
exec bash "$TASK_META/outputs/robocasa-ppo-grpo-meta-pipeline-20261104-r1/COMMAND.sh"
