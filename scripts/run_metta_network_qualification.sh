#!/bin/bash
set -euo pipefail
TASK_SOURCE=${1:?exact source commit}
TASK_RUN=${2:?fresh output identity}
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_WORKTREE=$TASK_ROOT/worktrees/$TASK_RUN
TASK_OUTPUT=$TASK_ROOT/outputs/$TASK_RUN/qualification
TASK_PYTHON=$TASK_ROOT/envs/metaworld-baselines-20261028-r1/bin/python
[[ $TASK_SOURCE =~ ^[0-9a-f]{40}$ ]]
[[ $TASK_RUN =~ ^metta-network-qual-baige-20261029-r[0-9]+$ ]]
[[ $(hostname) =~ ^job-[a-z0-9]+-master-0$ ]]
cd "$TASK_WORKTREE"
test "$(git rev-parse HEAD)" = "$TASK_SOURCE"
test -z "$(git status --porcelain)"
mkdir -p "$(dirname "$TASK_OUTPUT")"
mkdir "$TASK_OUTPUT"
exec > >(tee "$TASK_OUTPUT/console.log") 2>&1
TASK_STARTED=$(date +%s)
trap 'TASK_EXIT=$?; printf "%s\n" "$TASK_EXIT" > "$TASK_OUTPUT/exit-code.txt"; printf "%s\n" "$(($(date +%s)-TASK_STARTED))" > "$TASK_OUTPUT/allocation-active-seconds.txt"' EXIT
export PYTHONPATH="$TASK_WORKTREE/src" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export FLUFFY_GPU_EXECUTION=baige CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MUJOCO_GL=egl
export FLUFFY_META_RL_CUDA_QUALIFICATION=1 FLUFFY_METAWORLD_REAL_QUALIFICATION=1
export FLUFFY_METTA_DEFAULT_CAPACITY=1 FLUFFY_METTA_CAPACITY_OUTPUT="$TASK_OUTPUT/default-capacity.json"
"$TASK_PYTHON" 'scripts/[train]gpu_probe.py' --expected-count 1 --purpose training \
  --baige-container --output "$TASK_OUTPUT/gpu-probe.json"
timeout 1050 "$TASK_PYTHON" -m pytest -q -s -p no:cacheprovider tests/test_metta.py \
  tests/test_meta_rl_checkpoint.py --junitxml="$TASK_OUTPUT/tests.xml" \
  > "$TASK_OUTPUT/qualification.log" 2>&1
"$TASK_PYTHON" - "$TASK_SOURCE" "$TASK_OUTPUT" <<'PY'
import datetime,json,sys,xml.etree.ElementTree as ET
from pathlib import Path
out=Path(sys.argv[2]);suite=ET.parse(out/'tests.xml').getroot()
cases=suite.findall('.//testcase')
assert cases and not suite.findall('.//failure') and not suite.findall('.//error')
assert (out/'default-capacity.json').exists()
receipt=dict(source_commit=sys.argv[1],utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
             tests=len(cases),skipped=len(suite.findall('.//skipped')),status='passed',
             scope='Independent METTA network math, gradients, causal resets/summaries, prefix PPO, exact recovery, real ML10 and default capacity. No long training.')
(out/'summary.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
PY
