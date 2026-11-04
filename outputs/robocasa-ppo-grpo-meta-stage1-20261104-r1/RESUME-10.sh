#!/usr/bin/env bash
set -Eeuo pipefail
: "${TASK_META:?}" "${TASK_CODE_PIN:?}" "${GRPO_BEFORE_SHA:?}"
run=/mowenjie/fluffy-bassoon-2/outputs/robocasa-ppo-grpo-meta-stage1-20261104-r1
export ROBOCASA_SIM_PPO_RESUME=$run/grpo/attempt-01/OpenDrawer/checkpoints/global_step_10
export ROBOCASA_SIM_PPO_BEFORE_REFERENCE=$run/GRPO-BEFORE-RESUME10.json
export ROBOCASA_SIM_PPO_BEFORE_SHA256=$GRPO_BEFORE_SHA
export ROBOCASA_FLOW_FRESH_EVAL_PROCESSES=true
export TASK_IDENTITY=attempt-02 TASK_WALL_SECONDS=8880
exec bash "$TASK_META/outputs/robocasa-ppo-grpo-meta-stage1-20261104-r1/LAUNCH.sh" grpo
