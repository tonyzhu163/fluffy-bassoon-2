#!/usr/bin/env bash
# Staged source learning, after fixed-source optimizer calibration passes.
set -euo pipefail
TASK_COMMIT=${1:?40-hex source commit required}
TASK_RUN=${2:?new output identity required}
TASK_RESUME=${3:-}
TASK_SLICE_MINUTES=${4:-120}
[[ "$TASK_COMMIT" =~ ^[0-9a-f]{40}$ ]]
[[ "$TASK_RUN" =~ ^robotwin-staged-[a-zA-Z0-9-]+$ ]]
[[ "$TASK_SLICE_MINUTES" =~ ^[0-9]+$ ]]
(( TASK_SLICE_MINUTES >= 8 && TASK_SLICE_MINUTES <= 120 ))
# Keep the original scientific recipe while fitting a measured remaining
# allocation slice. Reserve tests/startup and a clean driver shutdown.
TASK_DRIVER_SECONDS=$(( TASK_SLICE_MINUTES * 60 - 300 ))
TASK_TIMEOUT_SECONDS=$(( TASK_SLICE_MINUTES * 60 - 180 ))
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_WORKTREE=$TASK_ROOT/worktrees/$TASK_RUN
TASK_OUTPUT=$TASK_ROOT/outputs/$TASK_RUN
TASK_PYTHON=$TASK_ROOT/envs/robotwin-vlrla-20261029-r1/bin/python
TASK_RLINF=$TASK_ROOT/sources/robotwin-baseline-smoke-20261028-r1/RLinf
TASK_ROBOTWIN=$TASK_ROOT/sources/robotwin-baseline-smoke-20261028-r1/RoboTwin
TASK_MODEL=$TASK_ROOT/cache/robotwin/pi05-motus-effc6e1a-r1
TASK_NORM=$TASK_MODEL/assets/pi0.5_clean_randomize_joint_training/norm_stats.json
test ! -e "$TASK_OUTPUT"
mkdir -p "$TASK_OUTPUT"
exec > >(tee "$TASK_OUTPUT/console.log") 2>&1
trap 'TASK_EXIT=$?; printf "%s\n" "$TASK_EXIT" > "$TASK_OUTPUT/exit-code.txt"' EXIT
export PATH="$(dirname "$TASK_PYTHON"):$PATH"
export PYTHONPATH="$TASK_WORKTREE/src:$TASK_RLINF:$TASK_ROBOTWIN"
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 PYTHONDONTWRITEBYTECODE=1 FLUFFY_GPU_EXECUTION=baige
# Baige may isolate one device with NVIDIA_VISIBLE_DEVICES while leaving the
# CUDA variable unset. Preserve an existing allocation list; otherwise select
# the sole container-local device, as in the qualified project launchers.
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
"$TASK_PYTHON" "scripts/[train]gpu_probe.py" --expected-count 1 \
    --output "$TASK_OUTPUT/gpu-probe.json" --purpose training --baige-container
export LYCO_BAIGE_GPU_MAP="$TASK_OUTPUT/gpu-probe.json"
timeout 120 "$TASK_PYTHON" -m pytest -q \
    tests/test_robotwin_noise.py tests/test_robotwin_assets.py \
    tests/test_robotwin_fast_observations.py tests/test_vla_noise_ppo.py \
    tests/test_vla_noise_sac.py tests/test_vla_noise_ppo_review.py \
    tests/test_robotwin_campaign_budget.py tests/test_robotwin_corpus.py \
    tests/test_robotwin_campaign_driver.py
# Keep source-only numerical qualification inside this SAME approved slice.
# It transfers only a learning rate; main learners start from fresh weights.
if [ -n "$TASK_RESUME" ]; then
    # Resume the same qualified recipe; never repeat source SGD calibration.
    "$TASK_PYTHON" - "$TASK_RESUME" "$TASK_OUTPUT/optimizer-calibration" <<'PYCOPY'
from pathlib import Path
import shutil,sys
previous=Path(sys.argv[1]).resolve()
if previous.is_file(): previous=previous.parent
if previous.name=='experiment': previous=previous.parent
source=previous/'optimizer-calibration'/'result.json'
if not source.is_file(): raise SystemExit('Resume requires its original optimizer qualification receipt')
target=Path(sys.argv[2]); target.mkdir(parents=True,exist_ok=False)
shutil.copy2(source,target/'result.json')
PYCOPY
else
timeout --signal=TERM --kill-after=5s 50 "$TASK_PYTHON" \
    scripts/diagnostics/robotwin_saved_ppo_lr.py --source-commit "$TASK_COMMIT" \
    --input "$TASK_ROOT/outputs/robotwin-noise-matched-20261029-r4/experiment/ppo-recovery.pt" \
    --input-sha256 722618543005ffcac5591593f67f2f8eaa3bd9fcc1be5cebf66995e9487306a9 \
    --reference-post "$TASK_ROOT/outputs/robotwin-saved-ppo-20261030-r2/diagnostic/diagnostic-recovery.pt" \
    --reference-post-sha256 c2f1fe3c7e21f1151ab453adab2e2520b5dafe2146936b8f31741091804bdcfb \
    --learning-rates 1e-6 3e-7 --output "$TASK_OUTPUT/optimizer-calibration" --wall-seconds 45
fi
TASK_PPO_LR=$("$TASK_PYTHON" - "$TASK_OUTPUT/optimizer-calibration/result.json" <<'PYQUAL'
import hashlib,json,math,sys
from pathlib import Path
path=Path(sys.argv[1]); result=json.loads(path.read_text())
assert result['float32_matmul_precision']=='high'
assert result['executed_new_robot_actions']==0 and result['main_comparison_advanced'] is False
assert result['candidate_rates']==[1e-6,3e-7]
admitted=[]
for candidate in result['candidates']:
    if not candidate.get('provisional_stable'): continue
    post=candidate['post_divergence']
    assert candidate['status']=='complete' and candidate['optimizer_steps']==5
    assert post['all_finite'] and post['mean_joint_analytic_kl']<=.02 and post['post_clip_fraction']<=.2
    assert candidate['post_reconstruction']['mean_max_error']<=1e-5
    assert candidate['post_reconstruction']['log_prob_max_error']<=2e-4
    admitted.append(candidate)
if not admitted:
    raise SystemExit('No PPO setting passed the declared source-only gate; stop before new robot actions')
selected=max(admitted,key=lambda row:row['learning_rate'])
receipt=dict(qualified=True,selection='largest predeclared learning rate passing the same source-only gate',
    learning_rate=selected['learning_rate'],epochs=5,ppo_trials_per_main_update=8,
    main_weights='fresh independent initialization, not the diagnostic trained weights',
    calibration_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),calibration=str(path),
    candidate_recovery_sha256=selected['recovery_sha256'],post_divergence=selected['post_divergence'],
    ppo_max_update_kl=.1)
(path.parent/'admission.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(selected['learning_rate'])
PYQUAL
)
TASK_ARGS=(--source-commit "$TASK_COMMIT" --model-path "$TASK_MODEL" \
    --norm-stats "$TASK_NORM" --robotwin-root "$TASK_ROBOTWIN" --rlinf-root "$TASK_RLINF" \
    --output "$TASK_OUTPUT/experiment" --source-steps 128000 --source-milestones 32000 64000 128000 \
    --ppo-trials-per-update 8 --ppo-epochs 5 --learning-rate "$TASK_PPO_LR" \
    --ppo-max-update-kl 0.1 --train-episodes 4 --eval-episodes 8 \
    --methods ppo sac ppo_reset --native-horizon 32 --episode-limit 400 \
    --eval-tasks adjust_bottle move_stapler_pad \
    --source-validation-episodes 4 --native-episodes 1 --task-config-name demo_clean \
    --sac-warmup 500 --sac-batch-size 256 --sac-utd 20 \
    --campaign-root "$TASK_ROOT/campaigns/robotwin-one-million-20261029-r1" \
    --campaign-id robotwin-one-million-20261029-r1 --campaign-max-actions 1000000 \
    --campaign-prior-receipts "$TASK_WORKTREE/outputs/robotwin-campaign-1m-20261029-r1/prior-receipts.json" \
    --record-corpus --final-observation-only \
    --capture-qualification "$TASK_ROOT/outputs/robotwin-capture-qual-20261029-r1/qualification/qualification.json" \
    --wall-seconds "$TASK_DRIVER_SECONDS" --boundary-reserve-seconds 240)
if [ -n "$TASK_RESUME" ]; then TASK_ARGS+=(--resume "$TASK_RESUME"); fi
timeout --signal=TERM --kill-after=30s "$TASK_TIMEOUT_SECONDS" "$TASK_PYTHON" \
    scripts/diagnostics/robotwin_noise_compare.py "${TASK_ARGS[@]}"
