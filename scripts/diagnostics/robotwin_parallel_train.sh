#!/usr/bin/env bash
# One exclusive GPU per method; immutable parent caps compose the three ledgers.
set -euo pipefail
TASK_COMMIT=${1:?40-hex source commit required}
TASK_RUN=${2:?fresh robotwin-wholetrial run identity required}
TASK_METHOD=${3:?ppo, sac or ppo_reset required}
TASK_PARENT_MANIFEST=${4:?immutable parent campaign manifest required}
TASK_RESUME=${5:-}
TASK_SLICE_MINUTES=${6:-780}
[[ "$TASK_COMMIT" =~ ^[0-9a-f]{40}$ ]]
[[ "$TASK_RUN" =~ ^robotwin-wholetrial-[a-zA-Z0-9_-]+$ ]]
[[ "$TASK_METHOD" =~ ^(ppo|sac|ppo_reset)$ ]]
[[ "$TASK_SLICE_MINUTES" =~ ^[0-9]+$ ]]
(( TASK_SLICE_MINUTES >= 30 && TASK_SLICE_MINUTES <= 780 ))
TASK_DRIVER_SECONDS=$(( TASK_SLICE_MINUTES * 60 - 600 ))
if (( TASK_DRIVER_SECONDS > 43200 )); then TASK_DRIVER_SECONDS=43200; fi
TASK_TIMEOUT_SECONDS=$(( TASK_SLICE_MINUTES * 60 - 180 ))
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_WORKTREE=$TASK_ROOT/worktrees/$TASK_RUN
TASK_OUTPUT=$TASK_ROOT/outputs/$TASK_RUN
TASK_PYTHON=$TASK_ROOT/envs/robotwin-vlrla-20261029-r1/bin/python
TASK_RLINF=$TASK_ROOT/sources/robotwin-baseline-smoke-20261028-r1/RLinf
TASK_ROBOTWIN=$TASK_ROOT/sources/robotwin-baseline-smoke-20261028-r1/RoboTwin
TASK_MODEL=$TASK_ROOT/cache/robotwin/pi05-motus-effc6e1a-r1
TASK_NORM=$TASK_MODEL/assets/pi0.5_clean_randomize_joint_training/norm_stats.json
test -f "$TASK_PARENT_MANIFEST"
test ! -e "$TASK_OUTPUT"
mkdir -p "$TASK_OUTPUT"
exec > >(tee "$TASK_OUTPUT/console.log") 2>&1
trap 'TASK_EXIT=$?; printf "%s\n" "$TASK_EXIT" > "$TASK_OUTPUT/exit-code.txt"' EXIT
export PATH="$(dirname "$TASK_PYTHON"):$PATH"
export PYTHONPATH="$TASK_WORKTREE/src:$TASK_RLINF:$TASK_ROBOTWIN"
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 PYTHONDONTWRITEBYTECODE=1 FLUFFY_GPU_EXECUTION=baige
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME=/mowenjie/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1
export JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MPLBACKEND=Agg
export OPENPI_DATA_HOME=/mowenjie/fluffy-bassoon/inputs/rlinf-openpi-tokenizer-befaa248-20261028-r1
export ROBOT_PLATFORM=ALOHA ASSETS_PATH="$TASK_ROBOTWIN" ROBOTWIN_PATH="$TASK_ROBOTWIN"
export EMBODIED_PATH="$TASK_RLINF/examples/embodiment" REPO_PATH="$TASK_RLINF"
export FB2_ROBOTWIN_RENDER_RECORDS="$TASK_OUTPUT/renderer-bindings"
mkdir -p /root/.sapien/physx
if [ ! -e /root/.sapien/physx/105.1-physx-5.3.1.patch0 ]; then
    ln -s /mowenjie/fluffy-bassoon/inputs/maniskill-scene-assets-20261028-r1/physx/105.1-physx-5.3.1.patch0 /root/.sapien/physx/105.1-physx-5.3.1.patch0
fi
cd "$TASK_WORKTREE"
test "$(git rev-parse HEAD)" = "$TASK_COMMIT"
test -z "$(git status --porcelain)"
"$TASK_PYTHON" scripts/diagnostics/robotwin_parallel_campaign.py admit-worker \
    --manifest "$TASK_PARENT_MANIFEST" --source-commit "$TASK_COMMIT" \
    --method "$TASK_METHOD" --run-id "$TASK_RUN" --robotwin-root "$TASK_ROBOTWIN" \
    --output "$TASK_OUTPUT/parent-admission.json"
"$TASK_PYTHON" "scripts/[train]gpu_probe.py" --expected-count 1 \
    --output "$TASK_OUTPUT/gpu-probe.json" --purpose training --baige-container
export LYCO_BAIGE_GPU_MAP="$TASK_OUTPUT/gpu-probe.json"
timeout 180 "$TASK_PYTHON" -m pytest -q \
    tests/test_robotwin_noise.py tests/test_robotwin_assets.py \
    tests/test_robotwin_fast_observations.py tests/test_vla_noise_ppo.py \
    tests/test_vla_noise_sac.py tests/test_vla_noise_ppo_review.py \
    tests/test_robotwin_campaign_budget.py tests/test_robotwin_corpus.py \
    tests/test_robotwin_campaign_driver.py tests/test_robotwin_parallel_campaign.py

# Existing source-only qualification transfers a scalar LR, never trained
# weights. Resumes use their own copied receipt and the same immutable pin.
TASK_CALIBRATION=$TASK_ROOT/outputs/robotwin-staged-20261030-r2/optimizer-calibration
if [ -n "$TASK_RESUME" ]; then
    TASK_CALIBRATION=$("$TASK_PYTHON" - "$TASK_RESUME" <<'PYRESUME'
from pathlib import Path
import sys
path=Path(sys.argv[1]).resolve()
if path.is_file(): path=path.parent
if path.name=='experiment': path=path.parent
print(path/'optimizer-calibration')
PYRESUME
)
fi
"$TASK_PYTHON" - "$TASK_CALIBRATION" "$TASK_OUTPUT/optimizer-calibration" <<'PYCALIBRATION'
from pathlib import Path
import hashlib,json,shutil,sys
source=Path(sys.argv[1]); target=Path(sys.argv[2])
result_path=source/'result.json'; result=json.loads(result_path.read_text())
assert hashlib.sha256(result_path.read_bytes()).hexdigest()=='821995144fe49d5503e07ac05b1932ba729ce19d64e101740e06b0fa11ac3f61'
admission=json.loads((source/'admission.json').read_text())
assert admission['qualified'] is True and admission['learning_rate']==1e-6
assert admission['calibration_sha256']==hashlib.sha256(result_path.read_bytes()).hexdigest()
assert result['float32_matmul_precision']=='high' and result['executed_new_robot_actions']==0
candidate=next(row for row in result['candidates'] if row['learning_rate']==1e-6)
assert candidate['status']=='complete' and candidate['provisional_stable'] is True
assert candidate['optimizer_steps']==5 and candidate['post_divergence']['all_finite']
assert candidate['post_divergence']['mean_joint_analytic_kl']<=.02
assert candidate['post_divergence']['post_clip_fraction']<=.2
target.mkdir(exist_ok=False)
for filename in ('result.json','admission.json'): shutil.copy2(source/filename,target/filename)
(target/'reuse.json').write_text(json.dumps(dict(source=str(source),weights_transferred=False,
    learning_rate=1e-6,new_recipe='benchmark-native complete K4 trials; existing numerical gate is not a learning-performance guarantee'),indent=2)+'\n')
PYCALIBRATION
TASK_CAMPAIGN_ID=$("$TASK_PYTHON" -c 'import json,sys;print(json.load(open(sys.argv[1]))["worker"]["campaign_id"])' "$TASK_OUTPUT/parent-admission.json")
TASK_LEDGER_ROOT=$("$TASK_PYTHON" -c 'import json,sys;print(json.load(open(sys.argv[1]))["worker"]["ledger_root"])' "$TASK_OUTPUT/parent-admission.json")
TASK_ACTION_CAP=$("$TASK_PYTHON" -c 'import json,sys;print(json.load(open(sys.argv[1]))["worker"]["maximum_actions"])' "$TASK_OUTPUT/parent-admission.json")
TASK_NATIVE_EPISODES=0
if [ "$TASK_METHOD" = ppo ]; then TASK_NATIVE_EPISODES=1; fi
TASK_ARGS=(--source-commit "$TASK_COMMIT" --model-path "$TASK_MODEL" \
    --norm-stats "$TASK_NORM" --robotwin-root "$TASK_ROBOTWIN" --rlinf-root "$TASK_RLINF" \
    --output "$TASK_OUTPUT/experiment" --seed 20261030 --source-steps 1100000 \
    --source-iterations 100 --source-eval-iterations 10 25 50 100 \
    --benchmark-episode-limits --ppo-trials-per-update 6 \
    --ppo-epochs 5 --ppo-minibatch-trials 16 --learning-rate 1e-6 --ppo-max-update-kl .1 \
    --train-episodes 4 --eval-episodes 8 --methods "$TASK_METHOD" \
    --native-horizon 32 --episode-limit 600 \
    --eval-tasks adjust_bottle move_stapler_pad --source-validation-episodes 4 \
    --native-episodes "$TASK_NATIVE_EPISODES" --task-config-name demo_clean \
    --sac-warmup 500 --sac-batch-size 256 --sac-utd 20 \
    --campaign-root "$TASK_LEDGER_ROOT" --campaign-id "$TASK_CAMPAIGN_ID" \
    --campaign-max-actions "$TASK_ACTION_CAP" --record-corpus --final-observation-only \
    --capture-qualification "$TASK_ROOT/outputs/robotwin-capture-qual-20261029-r1/qualification/qualification.json" \
    --wall-seconds "$TASK_DRIVER_SECONDS" --boundary-reserve-seconds 300)
if [ -n "$TASK_RESUME" ]; then TASK_ARGS+=(--resume "$TASK_RESUME"); fi
timeout --signal=TERM --kill-after=30s "$TASK_TIMEOUT_SECONDS" "$TASK_PYTHON" \
    scripts/diagnostics/robotwin_noise_compare.py "${TASK_ARGS[@]}"
