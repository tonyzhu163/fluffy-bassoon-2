#!/usr/bin/env bash
set -Eeuo pipefail
CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
COMMIT=$(git -C "$CODE" rev-parse HEAD)
test -z "$(git -C "$CODE" status --porcelain)"
ROOT=/mowenjie/fluffy-bassoon-2
RUN=$ROOT/outputs/steering-effect-refit-20261106-r1
PY=$ROOT/setups/robocasa365-native-20261101-r1/env/bin/python
ATTEMPT=${STEERING_ATTEMPT:?Require a fresh allocation attempt}
export RLINF_PATH=$ROOT/sources/critic-selected-a1229239-r1
export PYTHONPATH=$CODE/src:$RLINF_PATH
export OPENPI_DATA_HOME=$ROOT/setups/robocasa365-native-20261101-r1/cache/.cache/openpi
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NCCL_P2P_DISABLE=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl FLUFFY_GPU_EXECUTION=baige PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
mkdir -p "$RUN/probes" "$RUN/logs"
test ! -e "$RUN/logs/$ATTEMPT.log"
exec > >(tee "$RUN/logs/$ATTEMPT.log") 2>&1
"$PY" - <<'PY'
from pathlib import Path
import json,os,shutil
run=Path('/mowenjie/fluffy-bassoon-2/outputs/steering-effect-refit-20261106-r1')
ledger=json.loads((run/'ALLOCATIONS.json').read_text())
assert ledger['combined_hard_ceiling_GPU_hours']==150
assert ledger['parent_conservative_GPU_hours']==140.2711111111111
assert ledger['new_committed_GPU_hours']<=9.72888888888889
assert ledger['stage']=='diagnostic' and ledger['maximum_current_allocation_GPU_hours']==1
assert shutil.disk_usage(run).free>286*2**30
probe=run/'.write-probe'
with probe.open('wb') as f:f.write(b'probe');f.flush();os.fsync(f.fileno())
probe.unlink()
print(json.dumps(dict(stage='diagnostic',source_only=True,free_GiB=shutil.disk_usage(run).free/2**30)))
PY
PROBE=$RUN/probes/$ATTEMPT.json
"$PY" -B "$CODE/scripts/[train]gpu_probe.py" --expected-count 1 --purpose evaluation --baige-container --output "$PROBE"
timeout --signal=TERM --kill-after=30s 3120 "$PY" -B "$CODE/outputs/steering-effect-refit-20261106-r1/DIAGNOSE.py" \
    --code "$CODE" --commit "$COMMIT" --probe "$PROBE" --output "$RUN/diagnostics/$ATTEMPT"
