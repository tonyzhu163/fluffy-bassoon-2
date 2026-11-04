#!/usr/bin/env bash
set -Eeuo pipefail
: "${SOURCE5_CODE_DIR:?}" "${SOURCE5_CODE_COMMIT:?}"
ROOT=/mowenjie/fluffy-bassoon-2
RUN=$ROOT/outputs/robocasa-eight-hour-babysit-20261104-r1
TARGET=${1:?PickPlaceCounterToCabinet or TurnOnSinkFaucet}
export TASK_META=$SOURCE5_CODE_DIR TASK_CODE_PIN=$SOURCE5_CODE_COMMIT
export TASK_ATTEMPT=source5-40-r1 TASK_SEGMENT=source5-40
export TASK_OUTPUT=$RUN/source5/$TARGET/attempt-01
export ROBOCASA_SIM_PPO_ACCOUNTING=$RUN/source5/$TARGET/accounting
case "$TARGET" in
  PickPlaceCounterToCabinet) export ROBOCASA_FLOW_TRAIN_SEED=2200000 ROBOCASA_FLOW_EVAL_SEED=620100 ;;
  TurnOnSinkFaucet) export ROBOCASA_FLOW_TRAIN_SEED=2800000 ROBOCASA_FLOW_EVAL_SEED=680100 ;;
  *) exit 2 ;;
esac
export ROBOCASA_FLOW_TASK=$TARGET
export TASK_CONFIG=robocasa365_pi05_flow_sde_ppo_reuse4
export TASK_GPUS=8 TASK_CPUS=168 TASK_RAM_GIB=1280 TASK_SHM_GIB=32 TASK_WALL_SECONDS=13200
export ROBOCASA_SIM_PPO_ACTOR_PLACEMENT=0-1 ROBOCASA_SIM_PPO_ROLLOUT_PLACEMENT=0-7
export ROBOCASA_SIM_PPO_MICRO_BATCH=16 ROBOCASA_SIM_PPO_WORLD_SIZE=8
export ROBOCASA_FLOW_MAX_EPOCHS=40 ROBOCASA_FLOW_UPDATE_EPOCHS=4
export ROBOCASA_FLOW_VAL_INTERVAL=10 ROBOCASA_FLOW_SAVE_INTERVAL=10
export ROBOCASA_FLOW_NOISE_LEVEL=0.3 ROBOCASA_FLOW_IGNORE_LAST=true
export ROBOCASA_FLOW_MAX_ACTIONS=4100000 ROBOCASA_FLOW_TRAIN_ENVS=128 ROBOCASA_FLOW_ROUNDS=1
export ROBOCASA_FLOW_SAMPLING_RECORD=full ROBOCASA_SIM_PPO_GLOBAL_ESCROW=true
export ROBOCASA_FLOW_ACTOR_BATCH_DROP='[]'
export RLINF_PATH=$ROOT/sources/critic-selected-a1229239-r1
export PYTHONPATH=$TASK_META/src:$RLINF_PATH
export ROBOCASA_SIM_PPO_OUTPUT=$TASK_OUTPUT TASK_DEADLINE=1
export ROBOCASA_PI05_TORCH_MODEL=$ROOT/inputs/robocasa365-public-pi05-c484-torch-20261101-r2
export ROBOCASA_PI05_NATIVE_NORM=$ROOT/inputs/robocasa365-public-pi05-c484-75000-20261101-r1/assets/norm_stats.json
if [[ ${2:-} == --config ]]; then
  exec "$ROOT/setups/robocasa365-native-20261101-r1/env/bin/python" -B "$TASK_META/scripts/[train]robocasa_flow_ppo.py" --config-name "$TASK_CONFIG" --cfg job --resolve
fi
python3 - "$RUN/SOURCE5-BUDGET.json" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
if not p.exists():raise SystemExit('New70GPU-hour source budget and endpoint-gate decision remain unapproved')
d=json.loads(p.read_text())
assert d.get('approved') is True and d.get('GPU_hours_ceiling')==70
assert d.get('independent_endpoint_gate_authorized') is True
PY
exec bash "$TASK_META/outputs/robocasa-ppo-grpo-meta-pipeline-20261104-r1/COMMAND.sh"
