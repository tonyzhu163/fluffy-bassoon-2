#!/usr/bin/env bash
set -Eeuo pipefail
TASK_SOURCE=${1:?exact reviewed project commit}
TASK_OUT=${2:?fresh PPO output}
TASK_PRIOR_GPU_SECONDS=${3:?conservative qualification and failed-attempt charge}
TASK_CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
TASK_RUN_CODE=$TASK_CODE/outputs/robotwin-ppo-source-pilot-20261105-r1
TASK_PYTHON=/mowenjie/fluffy-bassoon-2/envs/robotwin-vlrla-20261029-r1/bin/python
[[ $(uname -s) == Linux && $(hostname) == job-* ]]
[[ $(git -C "$TASK_CODE" rev-parse HEAD) == "$TASK_SOURCE" ]]
[[ -z $(git -C "$TASK_CODE" status --porcelain --untracked-files=no) ]]
[[ ! -e $TASK_OUT ]]
[[ $TASK_PRIOR_GPU_SECONDS =~ ^[0-9]+$ ]]
# Eight GPUs for 25 minutes, plus 60 seconds/GPU scheduler overhead.
[[ $((TASK_PRIOR_GPU_SECONDS+8*1560)) -le 14400 ]]
mkdir -p "$TASK_OUT"
export REPO_PATH=/mowenjie/fluffy-bassoon-2/sources/robotwin-baseline-smoke-20261028-r1/RLinf
export ROBOTWIN_PATH=/mowenjie/fluffy-bassoon-2/sources/robotwin-baseline-smoke-20261028-r1/RoboTwin
export ROBOTWIN_ASSETS_PATH=$ROBOTWIN_PATH EMBODIED_PATH=$REPO_PATH/examples/embodiment
export PYTHONPATH=$TASK_CODE/src:$REPO_PATH:$ROBOTWIN_PATH
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export ROBOT_PLATFORM=ALOHA MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NCCL_P2P_DISABLE=1
export NATIVE_PPO_WORLD=8 NATIVE_PPO_PLACEMENT=0-7 NATIVE_PPO_MODE=train
export NATIVE_PPO_OUTPUT=$TASK_OUT NATIVE_PPO_ACCOUNTING_ROOT=$TASK_OUT/action-accounting
export NATIVE_PPO_ATTEMPT=attempt-0001
export NATIVE_PPO_MODEL=/mowenjie/fluffy-bassoon-2/cache/robotwin-ppo-pilot-20261105-r1-restore02/pi05-motus-effc6e1a-r1
export NATIVE_PPO_INITIALIZER_REVISION=effc6e1a99bd4d0edf9447e8ba313def67649cfe
TASK_STARTED=$(date +%s)
export NATIVE_PPO_DEADLINE=$((TASK_STARTED+1440))
finish() {
  TASK_RC=$?
  trap - EXIT
  TASK_ENDED=$(date +%s)
  printf '{"exit_code":%s,"started_unix":%s,"ended_unix":%s,"worker_gpu_seconds":%s,"prior_gpu_seconds":%s}\n' \
    "$TASK_RC" "$TASK_STARTED" "$TASK_ENDED" "$((8*(TASK_ENDED-TASK_STARTED)))" "$TASK_PRIOR_GPU_SECONDS" > "$TASK_OUT/EXIT.json"
}
trap finish EXIT
"$TASK_PYTHON" - <<'PY'
import json
from pathlib import Path
root = Path('/mowenjie/fluffy-bassoon-2/outputs/robotwin-ppo-source-pilot-20261105-r1')
q = json.loads((root/'qual01/native/RESULT.json').read_text())
assert q['status'] == 'complete' and q['frozen_unchanged'] and q['native_parity']['passed']
assert q['task'] == 'place_phone_stand' and q['optimizer_updates'] == 0
ready = json.loads((root/'restoration02/MOTOR-READY.json').read_text())
assert ready['status'] == 'restored' and ready['write_probe']
model = Path(ready['model_path'])
expected = json.loads((model/'download-receipt.json').read_text())
actual = {r['path']: r for r in ready['files']}
for row in expected['files']:
    assert actual[row['file']]['sha256'] == row['sha256']
    assert actual[row['file']]['bytes'] == row['bytes']
PY
"$TASK_PYTHON" "$TASK_CODE/scripts/[train]gpu_probe.py" --baige-container \
  --baige-exclusive-training --purpose training --expected-count 8 --output "$TASK_OUT/gpu-probe.json"
timeout --signal=INT --kill-after=20s 1380 "$TASK_PYTHON" "$TASK_RUN_CODE/train_entry.py" \
  --config-path "$TASK_CODE/configs/native_ppo" --config-name adjust_bottle_published_batch_two \
  env.train.task_config.task_name=place_phone_stand env.eval.task_config.task_name=place_phone_stand \
  "++actor.model.openpi_data.norm_stats_path=$NATIVE_PPO_MODEL/assets/pi0.5_clean_randomize_joint_training/norm_stats.json" \
  ++actor.model.openpi_data.adapt_to_pi=true ++actor.model.openpi_data.extra_delta_transform=true
