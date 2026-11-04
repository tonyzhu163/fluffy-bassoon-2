#!/usr/bin/env bash
set -Eeuo pipefail
: "${TASK_META:?}" "${TASK_CODE_PIN:?}"
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_RUN=$TASK_ROOT/outputs/robocasa-flow-sde-critic-parity-20261104-r1
TASK_PY=$TASK_ROOT/setups/robocasa365-native-20261101-r1/env/bin/python
export TASK_RUNTIME=$TASK_ROOT/sources/flow10m-0988b5f9
export TASK_VENDOR=$TASK_ROOT/sources/critic-selected-a1229239-r1
export TASK_VENDOR_PIN=8f9bfa7661928ef8c17494b04f4d7bf3125634e1
export TASK_ATTEMPT=critic-fixed-r2 TASK_SEGMENT=comparison CRITIC_COMPARISON_ARM=fixed
export TASK_OUTPUT=$TASK_RUN/comparison-fixed-r2
export TASK_GPUS=8 TASK_CPUS=168 TASK_RAM_GIB=1280 TASK_SHM_GIB=32 TASK_WALL_SECONDS=1200
export ROBOCASA_FLOW_TASK=PickPlaceDrawerToCounter
export ROBOCASA_FLOW_TRAIN_SEED=2300000 ROBOCASA_FLOW_EVAL_SEED=630100
export ROBOCASA_FLOW_ACTOR_BATCH_DROP='["forward_inputs.observation/image","forward_inputs.observation/wrist_image","forward_inputs.observation/extra_view_image","forward_inputs.chains"]'
export ROBOCASA_SIM_PPO_GLOBAL_ESCROW=true
export ROBOCASA_SIM_PPO_ACCOUNTING=$TASK_RUN/accounting-fixed-r2
export ROBOCASA_SIM_PPO_RESUME=$TASK_RUN/comparison-input-r1/global_step_0
export ROBOCASA_SIM_PPO_BEFORE_REFERENCE=$TASK_RUN/comparison-input-r2/BEFORE-REFERENCE.json
export ROBOCASA_SIM_PPO_BEFORE_SHA256=$("$TASK_PY" -c 'import json,sys;print(json.load(open(sys.argv[1]))["before_reference_sha256"])' "$TASK_RUN/comparison-input-r2/RESTORED.json")
"$TASK_PY" -B - "$TASK_RUN" <<'PY'
import json,pathlib,hashlib,sys
r=pathlib.Path(sys.argv[1]);d=json.loads((r/'comparison-input-r2/RESTORED.json').read_text())
assert d['completed'] and d['canonical_ranks']==8 and len(d['files'])==24
for row in d['files']:assert hashlib.sha256(pathlib.Path(row['target']).read_bytes()).hexdigest()==row['sha256']
budget=json.loads((r/'RECOVERY-ADMISSION.json').read_text());assert budget['completed']
assert budget['prior_conservative_gpu_seconds']+10560+1200<=21600
PY
bash "$TASK_META/outputs/robocasa-flow-sde-critic-parity-20261104-r1/COMPARE-ARM.sh"
