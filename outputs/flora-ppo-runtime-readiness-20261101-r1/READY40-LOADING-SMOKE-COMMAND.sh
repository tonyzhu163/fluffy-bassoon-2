#!/usr/bin/env bash
set -Eeuo pipefail
TASK_STARTED=${TASK_STARTED:-$(date +%s)}
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_SOURCE=a3b6111ac79e19ec12c99a3df6dcae8d201f4b53
TASK_RUN=flora-ppo-ready40-loading-smoke-q1-20261101-r1
TASK_WT=$TASK_ROOT/worktrees/flora-ppo-ready40-cpu-q1-20261101-r1
TASK_OUT=$TASK_ROOT/outputs/$TASK_RUN
TASK_INPUT=$TASK_ROOT/inputs/flora-ppo-runtime-readiness-20261101-r1
TASK_RLINF=$TASK_ROOT/sources/robotwin-baseline-smoke-20261028-r1/RLinf
TASK_PY=/mowenjie/fluffy-bassoon/envs/maniskill-pi05-rlinf-e1a82ebf-20261028-r1/bin/python
export TASK_STARTED TASK_ROOT TASK_SOURCE TASK_RUN TASK_WT TASK_OUT TASK_INPUT TASK_RLINF TASK_PY
cd "$TASK_WT"
test "$(git rev-parse HEAD)" = "$TASK_SOURCE"
test -z "$(git status --porcelain)"
test ! -e "$TASK_OUT"
mkdir -p "$TASK_OUT/tmp" "$TASK_OUT/cache"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH=$TASK_WT/src
TASK_ATTEMPT=$("$TASK_PY" -B - <<'PY'
import hashlib,json,os
from pathlib import Path
from fluffy_bassoon.meta_rl.vla_source40_ppo_worker import begin
p=Path(os.environ['TASK_INPUT'])/'READY40-PYTORCH-LOADING-SMOKE-TICKET.json';ticket=json.loads(p.read_text())
assert ticket['resource']['total_inclusive_gpu_seconds']==3600 and ticket['interactions']['maximum_actions_including_settling']==415
c=dict(phase_id='flora-ppo-comparison-20261101-r1',source_commit=os.environ['TASK_SOURCE'],run_id=os.environ['TASK_RUN'],ticket_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),maximum_actions=415,maximum_episodes=1,output_bytes=16<<30,initialization='RLinf40_45ccfcc4')
print(begin(Path(os.environ['TASK_OUT'])/'clock',c,slice_seconds=3540,worker_seconds=3600,now=int(os.environ['TASK_STARTED'])))
PY
)
export TASK_ATTEMPT
exec > >(tee "$TASK_OUT/console.log") 2>&1
task_cleanup() {
 trap - EXIT INT TERM
 PYTHONPATH=$TASK_WT/src "$TASK_PY" -B - <<'PY' || true
import json,os
from fluffy_bassoon.meta_rl.vla_source40_ppo_worker import finish
print(json.dumps(finish(os.environ['TASK_OUT']+'/clock',os.environ['TASK_ATTEMPT'],int(os.environ['TASK_RC']))))
PY
 printf '%s\n' "$TASK_RC" > "$TASK_OUT/exit-code.txt"
 exit "$TASK_RC"
}
export TASK_RC=0
trap 'TASK_RC=$?;export TASK_RC;task_cleanup' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
export FLUFFY_GPU_EXECUTION=baige OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false
export OPENPI_DATA_HOME=/mowenjie/fluffy-bassoon/cache/openpi-215abfb TMPDIR=$TASK_OUT/tmp
export PYTHONPATH=$TASK_WT/src:$TASK_RLINF:/mowenjie/fluffy-bassoon/third_party/openpi-215abfb/src:/mowenjie/fluffy-bassoon/third_party/openpi-215abfb/packages/openpi-client/src
export EMBODIED_PATH=$TASK_RLINF/examples/embodiment
export FLORA_PPO_ARM=S FLORA_PPO_PHASE=smoke FLORA_PPO_WORLD=1 FLORA_PPO_PLACEMENT=0 FLORA_PPO_EVAL_SLOTS=1 FLORA_PPO_ONLY_EVAL=true
export FLORA_PPO_MODEL=$TASK_ROOT/inputs/rlinf-pi05-libero40-sft-45ccfcc4-20261101-r1
export FLORA_PPO_NORM=$FLORA_PPO_MODEL/physical-intelligence/libero/norm_stats.json
export FLORA_PPO_OUTPUT=$TASK_OUT/experiment FLORA_PPO_EXCHANGE=$TASK_ROOT/outputs/flora-ppo-comparison-20261101-r1/exchange
export FLORA_PPO_MONITORING=$TASK_ROOT/outputs/flora-ppo-comparison-20261101-r1/monitoring
export FLORA_PPO_ACCOUNTING=$TASK_ROOT/outputs/flora-ppo-comparison-20261101-r1/common-actions
export FLORA_PPO_PROFILE=$TASK_OUT/profile.json FLORA_PPO_PROFILE_SHA=unbound FLORA_PPO_ATTEMPT=$TASK_RUN
TASK_REMAINING=$((TASK_STARTED+3420-$(date +%s)));test "$TASK_REMAINING" -gt 120
exec 3>&1
# One original absolute deadline covers probe, strict load and the single episode.
timeout --signal=INT --kill-after=30s "$TASK_REMAINING" "$TASK_PY" -B "$TASK_INPUT/READY40-LOADING-SMOKE-WORKLOAD.py"
