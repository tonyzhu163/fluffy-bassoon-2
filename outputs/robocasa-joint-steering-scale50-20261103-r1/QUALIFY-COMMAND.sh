#!/usr/bin/env bash
set -Eeuo pipefail
: "${TASK_META:?}" "${TASK_CODE_PIN:?}" "${TASK_QUAL_RUN:?}" "${RANK:?}" "${WORLD_SIZE:?}" "${MASTER_ADDR:?}"
test "$RANK" = 0
test "$WORLD_SIZE" = 1
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_PY=$TASK_ROOT/setups/robocasa365-native-20261101-r1/env/bin/python
export TASK_QUAL_STARTED=$(date +%s)
export TASK_PRIOR_QUAL_GPU_SECONDS=${TASK_PRIOR_QUAL_GPU_SECONDS:-0}
export TASK_QUAL_SUBMITTED_UNIX=${TASK_QUAL_SUBMITTED_UNIX:-$TASK_QUAL_STARTED}
TASK_QUAL_CAMPAIGN_END=$((TASK_QUAL_SUBMITTED_UNIX+2400-TASK_PRIOR_QUAL_GPU_SECONDS))
test "$TASK_PRIOR_QUAL_GPU_SECONDS" -ge 0
test "$TASK_QUAL_CAMPAIGN_END" -gt "$((TASK_QUAL_STARTED+90))"
test ! -e "$TASK_QUAL_RUN"
mkdir -p "$TASK_QUAL_RUN"
exec > >(tee "$TASK_QUAL_RUN/job-console.log") 2>&1
finish() {
  TASK_QUAL_RC=$?
  trap - EXIT
  TASK_QUAL_FINISHED=$(date +%s)
  printf '{"started_unix":%s,"finished_unix":%s,"allocated_gpus":1,"process_gpu_seconds":%s,"exit_code":%s,"allocation_release_not_included":true}\n' \
    "$TASK_QUAL_STARTED" "$TASK_QUAL_FINISHED" "$((TASK_QUAL_FINISHED-TASK_QUAL_STARTED))" "$TASK_QUAL_RC" > "$TASK_QUAL_RUN/clock.json"
  exit "$TASK_QUAL_RC"
}
trap finish EXIT
export RLINF_PATH=$TASK_ROOT/sources/robocasa365-public-pi05-nnx-leaves-20261101-r1/RLinf
export PYTHONPATH=$TASK_META/src:$RLINF_PATH
export ROBOCASA_ASSETS_PATH=$TASK_ROOT/setups/robocasa365-native-20261101-r1/assets
export OPENPI_DATA_HOME=$TASK_ROOT/setups/robocasa365-native-20261101-r1/cache/.cache/openpi
export FLUFFY_GPU_EXECUTION=baige MUJOCO_GL=egl PYOPENGL_PLATFORM=egl NCCL_P2P_DISABLE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
touch "$TASK_QUAL_RUN/.write-probe"
"$TASK_PY" -B - "$TASK_QUAL_RUN/PROFILE.json" <<'PY'
import json, os, pathlib, shutil, subprocess, sys
from fluffy_bassoon.meta_rl.joint_steering_ppo import TASKS, cases_for
root = pathlib.Path('/mowenjie/fluffy-bassoon-2')
code = pathlib.Path(os.environ['TASK_META'])
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=code, text=True).strip() == os.environ['TASK_CODE_PIN']
assert shutil.disk_usage(root).free > 128 * 2**30
training = cases_for('train', 1)
calibration = [dict(next(row for row in training if row['task'] == task),
    bank='joint_steering_calibration', native_seed=next(row for row in training if row['task'] == task)['native_seed'] + 800_000) for task in TASKS]
stress = [dict(row, bank='joint_steering_qualification', native_seed=row['native_seed'] + 900_000) for row in training[:48]]
profile = dict(code=str(code), source_commit=os.environ['TASK_CODE_PIN'],
    runtime_path=os.environ['RLINF_PATH'], runtime_commit='8a1233c3fae6f5905adfda7684c455842b0ecfb8',
    model_directory=str(root/'inputs/robocasa365-public-pi05-c484-torch-20261101-r2'),
    model_sha256='661bf172a2ae9dfca75216697eeaa543408bcbdfa53c98132759b93e0e8993a5',
    model_config_sha256='1c85bc28303a2144a1cc3af492b1b6d91f9cdcd94b30f04212505ffb5118f1b0',
    norm_path=str(root/'inputs/robocasa365-public-pi05-c484-75000-20261101-r1/assets/norm_stats.json'),
    norm_sha256='4aed1af411bd0e0f49d2b0e6d6832b11b8682917231a07173fbc30fd493bbdee',
    allocated_start_unix=int(os.environ['TASK_QUAL_SUBMITTED_UNIX']),
    prior_qualification_gpu_seconds=int(os.environ['TASK_PRIOR_QUAL_GPU_SECONDS']),
    calibration_cases=calibration, stress_cases=stress)
pathlib.Path(sys.argv[1]).write_text(json.dumps(profile, indent=2)+'\n')
PY
TASK_QUAL_SCRIPT=$TASK_META/outputs/robocasa-joint-steering-scale50-20261103-r1/QUALIFY.py
run_phase() {
  TASK_PHASE=$1
  TASK_PROFILE=$2
  TASK_LEFT=$((TASK_QUAL_CAMPAIGN_END-30-$(date +%s)))
  test "$TASK_LEFT" -gt 30
  timeout --signal=TERM --kill-after=15s "${TASK_LEFT}s" "$TASK_PY" -B "$TASK_QUAL_SCRIPT" \
    --phase "$TASK_PHASE" --arm both --profile "$TASK_PROFILE" --output "$TASK_QUAL_RUN/$TASK_PHASE" \
    --expected-hostname "$MASTER_ADDR" --deadline-unix "$((TASK_QUAL_CAMPAIGN_END-60))"
}
run_phase calibrate "$TASK_QUAL_RUN/PROFILE.json"
"$TASK_PY" -B - "$TASK_QUAL_RUN" <<'PY'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]); calibration=root/'calibrate/CALIBRATION.json'
assert json.loads(calibration.read_text())['completed']
profile=json.loads((root/'PROFILE.json').read_text())
profile['calibration_reference']=dict(path=str(calibration),sha256=hashlib.sha256(calibration.read_bytes()).hexdigest())
(root/'STRESS-PROFILE.json').write_text(json.dumps(profile,indent=2)+'\n')
PY
run_phase stress "$TASK_QUAL_RUN/STRESS-PROFILE.json"
