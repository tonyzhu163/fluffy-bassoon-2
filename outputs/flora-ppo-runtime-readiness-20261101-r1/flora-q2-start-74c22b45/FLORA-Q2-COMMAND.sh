#!/usr/bin/env bash
set -Eeuo pipefail
TASK_STARTED=${TASK_STARTED:-$(date +%s)}
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_SOURCE=dd77304841e32fb436e62bbc4dcb76f44d602546
TASK_WT=$TASK_ROOT/worktrees/flora-ppo-fsdp-precision-dd773048-20261101-r1
TASK_INPUT=$TASK_ROOT/inputs/flora-ppo-toolchain-repair-r2-20261101-r1
TASK_RUN=flora-ppo-official-inference-q2-20261101-r1
TASK_PY=/mowenjie/fluffy-bassoon-2/envs/flora-rewarder-129a2ada-py312-20261101-r1/bin/python
TASK_OUT=$TASK_ROOT/outputs/$TASK_RUN
TASK_CLOCK=$TASK_ROOT/outputs/flora-ppo-official-inference-q1-20261101-r1/clock
export TASK_STARTED TASK_ROOT TASK_SOURCE TASK_RUN TASK_WT TASK_INPUT TASK_PY TASK_OUT TASK_CLOCK
cd "$TASK_WT"
test "$(git rev-parse HEAD)" = "$TASK_SOURCE"
test -z "$(git status --porcelain)"
test ! -e "$TASK_OUT"
mkdir -p "$TASK_OUT/tmp" "$TASK_OUT/cache"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH=$TASK_WT/src
TASK_TICKET=$TASK_INPUT/FLORA-Q1-ADMISSION.json
export TASK_TICKET
TASK_ATTEMPT=$("$TASK_PY" -B - <<'CLOCK'
import hashlib,json,os
from pathlib import Path
from fluffy_bassoon.meta_rl.vla_source40_ppo_worker import begin
p=Path(os.environ['TASK_TICKET']);ticket=json.loads(p.read_text())
saved=json.loads((Path(os.environ['TASK_CLOCK'])/'worker-time.json').read_text());c=saved['contract']
assert len(saved['attempts'])==1 and saved['attempts'][0]['status']=='finished' and saved['attempts'][0]['charged_seconds']==446 and saved['attempts'][0]['exit_code']==1
assert c['source_commit']==os.environ['TASK_SOURCE'] and c['run_id']=='flora-ppo-official-inference-q1-20261101-r1' and c['ticket_sha256']==hashlib.sha256(p.read_bytes()).hexdigest() and c['actions']==0
child=Path(os.environ['TASK_INPUT'])/'FLORA-TOOLCHAIN-R2-ADMISSION.json';admission=json.loads(child.read_text());assert admission['same_clock']['maximum_new_inclusive_gpu_seconds']==3000
print(begin(os.environ['TASK_CLOCK'],c,slice_seconds=2940,worker_seconds=3600,resume=True,now=int(os.environ['TASK_STARTED'])))
CLOCK
)
export TASK_ATTEMPT
exec > >(tee "$TASK_OUT/console.log") 2>&1
task_cleanup() {
 trap - EXIT INT TERM
 PYTHONPATH=$TASK_WT/src "$TASK_PY" -B - <<'CLOSE' || true
import json,os
from fluffy_bassoon.meta_rl.vla_source40_ppo_worker import finish
print(json.dumps(finish(os.environ['TASK_CLOCK'],os.environ['TASK_ATTEMPT'],int(os.environ['TASK_RC']))))
CLOSE
 printf '%s\n' "$TASK_RC" > "$TASK_OUT/exit-code.txt"
 exit "$TASK_RC"
}
export TASK_RC=0
trap 'TASK_RC=$?;export TASK_RC;task_cleanup' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
export FLUFFY_GPU_EXECUTION=baige OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
export TMPDIR=/tmp/flq2 VLLM_CACHE_ROOT=$TASK_OUT/cache/vllm HF_HOME=$TASK_OUT/cache/huggingface
export PYTHONPATH=$TASK_WT/src:$TASK_ROOT/deps/nvidia-flora-129a2ada-20261101-r1/repo
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# Explicit isolated toolkit binding BEFORE the first Torch/FlashInfer import.
export CUDA_HOME=$TASK_ROOT/envs/flora-rewarder-129a2ada-py312-20261101-r1/lib/python3.12/site-packages/nvidia/cu13
export CUDA_PATH=$CUDA_HOME CUDACXX=$CUDA_HOME/bin/nvcc
export PATH=$CUDA_HOME/bin:$PATH LD_LIBRARY_PATH=$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}
mkdir -m 700 -p "$TMPDIR"
TASK_REMAINING=$((TASK_STARTED+2820-$(date +%s)));test "$TASK_REMAINING" -gt 120
timeout --signal=INT --kill-after=30s "$TASK_REMAINING" "$TASK_PY" -B "$TASK_INPUT/FLORA-Q2-WORKLOAD.py"
