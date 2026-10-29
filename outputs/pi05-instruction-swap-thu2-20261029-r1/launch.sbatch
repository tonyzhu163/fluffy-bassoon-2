#!/bin/bash
# Usage: sbatch launch.sbatch <source-commit> <rl|sft>
# Instruction-swap check, one checkpoint per job: CPU tests, single-object baseline,
# multi-object explicit/swapped/vague conditions. Frozen weights, no training.
#SBATCH --job-name=fb2-instr-swap
#SBATCH --partition=RTX3090
#SBATCH --gres=gpu:3090:1
#SBATCH --exclude=node01
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/cluster/nvme5/zhujiayang/fluffy-bassoon-2/outputs/pi05-instruction-swap-thu2-20261029-r1/slurm-%j.out
set -euo pipefail
export TASK_SOURCE=${1:?source commit} TASK_CKPT=${2:?rl or sft} TASK_START=$(date +%s)
[[ $TASK_SOURCE =~ ^[0-9a-f]{40}$ && $TASK_CKPT =~ ^(rl|sft)$ ]]
srun --ntasks=1 --cpu-bind=none bash <<'STEP'
set -euo pipefail
RUN=pi05-instruction-swap-thu2-20261029-r1
ROOT=/cluster/nvme5/zhujiayang/fluffy-bassoon-2; OLD=/cluster/nvme5/zhujiayang/fluffy-bassoon
W=$ROOT/worktrees/$RUN; O=$ROOT/outputs/$RUN/$TASK_CKPT-job$SLURM_JOB_ID; I=$ROOT/inputs/$RUN
mkdir "$O"
trap 'printf "%s\n" "$?" > "$O/exit-code.txt"' EXIT
cd "$W"
test "$(git rev-parse HEAD)" = "$TASK_SOURCE"
test -z "$(git status --porcelain)"
printf 'source=%s\njob=%s\nnode=%s\ncuda_visible=%s\ncheckpoint=%s\n' "$TASK_SOURCE" "$SLURM_JOB_ID" \
  "$SLURMD_NODENAME" "${CUDA_VISIBLE_DEVICES:-}" "$TASK_CKPT" > "$O/allocation.txt"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 FLUFFY_GPU_EXECUTION=slurm
export PYTHONPATH="$W/src"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=$OLD/cache/huggingface WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false
export MS_ASSET_DIR=$OLD/cache/rlinf/maniskill-data MANISKILL_ASSET_DIR=$OLD/cache/rlinf/maniskill_assets
export OPENPI_DATA_HOME=$OLD/cache/rlinf/openpi_tokenizer
export OMP_NUM_THREADS=4 XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cpu
P=$OLD/envs/rlinf-pi05-maniskill-e1a82eb-r1/bin/python
RLINF=$OLD/cache/src/RLinf-e1a82eb
CUDA_VISIBLE_DEVICES='' "$P" -m pytest -q -p no:cacheprovider tests/test_maniskill_instruction_swap.py \
  tests/test_maniskill_pi05.py tests/test_rlinf_checkpoint_compat.py > "$O/tests.log" 2>&1
if [[ $TASK_CKPT == rl ]]; then
  M=$OLD/cache/rlinf/RLinf-Pi05-ManiSkill-25Main-RL-FlowNoise/checkpoints/global_step_150/actor
  NORM=$M/assets/global_step_150/meta/norm_stats.json
  SHA=0adb5b40f064760ff245f53ae0a4e078b86680b65f77d1f5991a173543edd5a1
  # RL checkpoint stores legacy noise/value head names: load through the patched copy (patches/).
  if [[ ! -e $I/rlinf-e1a82ebf-flow-noise ]]; then
    mkdir -p "$I"
    "$P" scripts/diagnostics/prepare_rlinf_flow_noise_copy.py --source-commit "$TASK_SOURCE" \
      --original-root "$RLINF" --output-root "$I/rlinf-e1a82ebf-flow-noise" > "$O/rlinf-copy.log" 2>&1
  fi
  RLINF=$I/rlinf-e1a82ebf-flow-noise
else
  M=$OLD/cache/rlinf/RLinf-Pi05-ManiSkill-25Main-SFT
  NORM=$M/physical-intelligence/maniskill/norm_stats.json
  SHA=9877be1633a9bc89834d00ef890e1a4cd34a91c155d710b97c92fbab8a5caa89
fi
COMMON=(--rlinf-root "$RLINF" --model-path "$M" --norm-stats "$NORM" --expected-model-sha256 "$SHA"
  --source-commit "$TASK_SOURCE" --checkpoint-label "$TASK_CKPT" --num-envs 32 --seed 2910290000)
S=scripts/diagnostics/maniskill_pi05_instruction_swap.py
"$P" "$S" --mode single "${COMMON[@]}" --objects 0,9,12,13 --per-object 16 \
  --max-elapsed-seconds 600 --output-dir "$O/single" > "$O/single.log" 2>&1
# Leave ~6 minutes of the 60-minute allocation for checkpoint hashing and the final fingerprint.
REMAIN=$(( TASK_START + 3240 - $(date +%s) ))
"$P" "$S" --mode multi "${COMMON[@]}" --pairs 0:9,12:13,0:13,9:12 --per-pair 24 \
  --max-elapsed-seconds "$REMAIN" --output-dir "$O/multi" > "$O/multi.log" 2>&1
"$P" "$S" --mode summarize --single-dir "$O/single" --multi-dir "$O/multi" --output-dir "$O" > "$O/summarize.log" 2>&1
STEP
