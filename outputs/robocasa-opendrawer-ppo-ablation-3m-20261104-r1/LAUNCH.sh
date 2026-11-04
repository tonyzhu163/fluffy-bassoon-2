#!/usr/bin/env bash
set -Eeuo pipefail
root=/mowenjie/fluffy-bassoon-2
run=$root/outputs/robocasa-opendrawer-ppo-ablation-3m-20261104-r1
export TASK_META=$root/sources/opendrawer-ablation-3m-engine-r1
export TASK_CODE_PIN=dcd125e15f0aa03dae172b93a2efef067a9b1f91
export MATCHED_PPO_ARM=${1:?E C or D}
export TASK_ATTEMPT=$MATCHED_PPO_ARM-r1 TASK_SEGMENT=matched3m
export TASK_OUTPUT=$run/$MATCHED_PPO_ARM/attempt-01
export ROBOCASA_SIM_PPO_ACCOUNTING=$run/$MATCHED_PPO_ARM/accounting
export TASK_CONFIG=robocasa365_pi05_flow_sde_ppo_vlm_critic
export TASK_GPUS=8 TASK_CPUS=168 TASK_RAM_GIB=1280 TASK_SHM_GIB=32
export ROBOCASA_FLOW_TASK=OpenDrawer ROBOCASA_FLOW_TRAIN_SEED=2100000 ROBOCASA_FLOW_EVAL_SEED=610100
export ROBOCASA_FLOW_MAX_EPOCHS=40 ROBOCASA_FLOW_UPDATE_EPOCHS=1
export ROBOCASA_FLOW_VAL_INTERVAL=10 ROBOCASA_FLOW_SAVE_INTERVAL=10
export ROBOCASA_FLOW_NOISE_LEVEL=0.3 ROBOCASA_FLOW_IGNORE_LAST=true ROBOCASA_ABLATION_PREFIX=true
export ROBOCASA_FLOW_MAX_ACTIONS=4100000 ROBOCASA_FLOW_TRAIN_ENVS=128 ROBOCASA_FLOW_ROUNDS=1
export ROBOCASA_FLOW_SAMPLING_RECORD=full ROBOCASA_SIM_PPO_GLOBAL_ESCROW=true
export ROBOCASA_FLOW_ACTOR_BATCH_DROP='[]' ROBOCASA_SIM_PPO_MICRO_BATCH=16
input=$run/input-prefix-r1
export TASK_WALL_SECONDS=10320
case "$MATCHED_PPO_ARM" in
  E) export ROBOCASA_FLOW_UPDATE_EPOCHS=4 TASK_WALL_SECONDS=13920 ;;
  C) export ROBOCASA_ABLATION_PREFIX=false; input=$run/input-suffix-r1 ;;
  D) export ROBOCASA_FLOW_IGNORE_LAST=false ;;
  *) exit 2 ;;
esac
export ROBOCASA_SIM_PPO_RESUME=$input/global_step_0
export ROBOCASA_SIM_PPO_BEFORE_REFERENCE=$input/BEFORE.json
test "$(git -C "$TASK_META" rev-parse HEAD)" = "$TASK_CODE_PIN"
test -z "$(git -C "$TASK_META" status --porcelain)"
if [[ "${2:-}" == --config ]]; then
  export RLINF_PATH=$root/sources/critic-selected-a1229239-r1
  export EMBODIED_PATH=$RLINF_PATH/examples/embodiment
  export ROBOCASA_SIM_PPO_OUTPUT=$TASK_OUTPUT ROBOCASA_SIM_PPO_WORLD_SIZE=8 TASK_DEADLINE=1
  export ROBOCASA_PI05_TORCH_MODEL=$root/inputs/robocasa365-public-pi05-c484-torch-20261101-r2
  export ROBOCASA_PI05_NATIVE_NORM=$root/inputs/robocasa365-public-pi05-c484-75000-20261101-r1/assets/norm_stats.json
  export PYTHONPATH=$TASK_META/src:$RLINF_PATH
  exec "$root/setups/robocasa365-native-20261101-r1/env/bin/python" -B "$TASK_META/scripts/[train]robocasa_flow_ppo.py" --config-name "$TASK_CONFIG" --cfg job --resolve
fi
export ROBOCASA_SIM_PPO_BEFORE_SHA256
ROBOCASA_SIM_PPO_BEFORE_SHA256=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["before_reference_sha256"])' "$input/PREPARED.json")
echo "$ROBOCASA_SIM_PPO_BEFORE_SHA256  $ROBOCASA_SIM_PPO_BEFORE_REFERENCE" | sha256sum -c -
test ! -e "$TASK_OUTPUT"
exec bash "$TASK_META/outputs/robocasa-ppo-grpo-meta-pipeline-20261104-r1/COMMAND.sh"
