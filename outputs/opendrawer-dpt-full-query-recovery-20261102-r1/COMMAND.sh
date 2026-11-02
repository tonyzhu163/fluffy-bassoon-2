#!/usr/bin/env bash
set -Eeuo pipefail
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_RUN=$TASK_ROOT/outputs/opendrawer-dpt-full-query-recovery-20261102-r1
TASK_CODE=$TASK_ROOT/worktrees/opendrawer-paired-eval-20261102-r1
TASK_PY=$TASK_ROOT/setups/robocasa365-native-20261101-r1/env/bin/python
TASK_ATTEMPT=dpt-full-final-query-r1
TASK_STARTED=$(date +%s)
TASK_DEADLINE=$((TASK_STARTED+3600))
TASK_NODE=$TASK_RUN/attempts/$TASK_ATTEMPT/node-00
export TASK_ATTEMPT TASK_READOUT_OUTPUT=$TASK_RUN TASK_READOUT_FRESH_ID=1
export TASK_READOUT_METHODS='["dpt_full"]'
export TASK_READOUT_CASE_BANK=$TASK_ROOT/outputs/opendrawer-ad-dpt-paired-cases-20261102-r1
export PYTHONPATH=$TASK_ROOT/worktrees/opendrawer-ad-dpt-20261102-r1/src:$TASK_ROOT/sources/robocasa365-public-pi05-nnx-leaves-20261101-r1/RLinf
export RLINF_PATH=$TASK_ROOT/sources/robocasa365-public-pi05-nnx-leaves-20261101-r1/RLinf
export ROBOCASA_ASSETS_PATH=$TASK_ROOT/setups/robocasa365-native-20261101-r1/assets
export OPENPI_DATA_HOME=$TASK_ROOT/setups/robocasa365-native-20261101-r1/cache/.cache/openpi
export ROBOCASA_NATIVE_OUTPUT=$TASK_NODE/native
export ROBOCASA_PI05_TORCH_MODEL=$TASK_ROOT/inputs/robocasa365-public-pi05-c484-torch-20261101-r2
export ROBOCASA_PI05_NATIVE_NORM=$TASK_ROOT/inputs/robocasa365-public-pi05-c484-75000-20261101-r1/assets/norm_stats.json
export FLUFFY_GPU_EXECUTION=baige NCCL_P2P_DISABLE=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
mkdir -p "$TASK_RUN/attempts/$TASK_ATTEMPT"
mkdir "$TASK_NODE"
exec > >(tee "$TASK_NODE/console.log") 2>&1
finish() {
 TASK_RC=$?
 trap - EXIT
 TASK_FINISHED=$(date +%s)
 printf '{"started_unix":%s,"finished_unix":%s,"allocated_gpus":1,"observed_gpu_seconds":%s,"exit_code":%s}\n' \
  "$TASK_STARTED" "$TASK_FINISHED" "$((TASK_FINISHED-TASK_STARTED))" "$TASK_RC" > "$TASK_NODE/clock.json"
 exit "$TASK_RC"
}
trap finish EXIT
TASK_ACTUAL_BINDING=$(sha256sum "$TASK_RUN/binding.json")
test "${TASK_ACTUAL_BINDING%% *}" = d4a6609f2e905c9a48261487512ee5b741d0d8bd54b9bdbbca087b3df44f7c28
"$TASK_PY" -B "$TASK_ROOT/worktrees/opendrawer-ad-dpt-20261102-r1/scripts/[train]gpu_probe.py" \
 --expected-count 1 --purpose evaluation --baige-container --output "$TASK_NODE/gpu-probe.json"
export LYCO_BAIGE_GPU_MAP=$TASK_NODE/gpu-probe.json
export LYCO_BAIGE_ISOLATED_UUID=$($TASK_PY -c 'import json,os; print(json.load(open(os.environ["LYCO_BAIGE_GPU_MAP"]))["devices"][0]["uuid"])')
export MUJOCO_EGL_DEVICE_ID=0
timeout --signal=TERM --kill-after=60s 3660 "$TASK_PY" -B \
 "$TASK_CODE/outputs/opendrawer-ad-dpt-k1-id-ood-20261102-r1/READOUT.py" --worker 2 --deadline "$TASK_DEADLINE"
