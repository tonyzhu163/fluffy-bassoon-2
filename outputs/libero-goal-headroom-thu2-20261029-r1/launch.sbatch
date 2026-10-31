#!/bin/bash
#SBATCH --account=team1
#SBATCH --partition=RTX3090
#SBATCH --gres=gpu:3090:1
#SBATCH --exclude=node02
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --job-name=fb2-libero-goal-headroom
#SBATCH --output=/cluster/nvme5/zhujiayang/fluffy-bassoon-2/outputs/libero-goal-headroom-thu2-20261029-r1/slurm-%j.out
# Usage: sbatch launch.sbatch <40-hex source commit>. node02 is excluded because a
# foreign non-Slurm process occupied one of its GPUs on 2026-10-28 (coordination.md).
set -euo pipefail
export TASK_SOURCE=${1:?source commit}
srun --ntasks=1 --cpu-bind=none bash <<'STEP'
set -euo pipefail
run=libero-goal-headroom-thu2-20261029-r1
root=/cluster/nvme5/zhujiayang/fluffy-bassoon-2
worktree=$root/worktrees/$run
out=$root/outputs/$run
env=$root/envs/rlinf-pi05-libero-e1a82eb-20261029-r1
model=$root/inputs/rlinf-pi05-libero130-fullshot-sft-6222623f-20261029-r1
trap 'printf "%s\n" "$?" > "$out/exit-code.txt"' EXIT
[[ $TASK_SOURCE =~ ^[0-9a-f]{40}$ ]]
cd "$worktree"
test "$(git rev-parse HEAD)" = "$TASK_SOURCE"
test -z "$(git status --porcelain)"
printf 'source=%s\njob=%s\nstep=%s\nnode=%s\nstart=%s\n' "$TASK_SOURCE" "$SLURM_JOB_ID" "$SLURM_STEP_ID" \
  "$SLURMD_NODENAME" "$(date -u +%FT%TZ)" > "$out/allocation.txt"
export PYTHONPATH=$worktree/src PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export FLUFFY_GPU_EXECUTION=slurm LIBERO_CONFIG_PATH=$env/libero-config
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
"$env/bin/python" - "$out" <<'CHECK'
import json, os, sys, tempfile
out = sys.argv[1]
stat = os.statvfs(out)
free = stat.f_bavail * stat.f_frsize
if free < 20 * 1024**3:
    raise SystemExit("Less than 20 GiB free on project NVMe")
with tempfile.TemporaryFile(dir=out) as f:
    f.write(b"probe"); f.flush(); os.fsync(f.fileno())
json.dump(dict(free_bytes=free, write_probe_passed=True), open(os.path.join(out, "storage.json"), "w"), indent=2)
CHECK
"$env/bin/python" 'scripts/[train]gpu_probe.py' --expected-count 1 --purpose training --output "$out/gpu-probe.json"
CUDA_VISIBLE_DEVICES='' "$env/bin/python" -m pytest -q -p no:cacheprovider tests/test_libero_goal_headroom.py \
  tests/test_gpu_execution.py > "$out/tests.log" 2>&1
common=(--model-path "$model" --norm-stats "$model/physical-intelligence/libero/norm_stats.json"
  --expected-model-sha256 4d9089c941793f170b625c2ed0ac7a3aa09b6f103e52dbbc82e67301529d6683)
"$env/bin/python" scripts/diagnostics/libero_goal_headroom.py model-check --device cuda "${common[@]}" \
  --output-dir "$out/model-check" > "$out/model-check.log" 2>&1
"$env/bin/python" scripts/diagnostics/libero_goal_headroom.py run "${common[@]}" --source-commit "$TASK_SOURCE" \
  --batch-size 34 --max-elapsed-seconds 2700 --output-dir "$out/run" > "$out/run.log" 2>&1
STEP
