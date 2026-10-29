#!/usr/bin/env bash
# One exclusive Baige GPU; sources, environments and artifacts already staged.
set -euo pipefail
TASK_SOURCE=${1:?exact source commit}
TASK_RUN=${2:?fresh run ID}
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_WORKTREE=$TASK_ROOT/worktrees/$TASK_RUN
TASK_OUTPUT=$TASK_ROOT/outputs/$TASK_RUN
TASK_PYTHON=$TASK_ROOT/envs/dsrl-can-20261029-r2/bin/python
TASK_DSRL=$TASK_ROOT/deps/dsrl-can-20261029-r1/repo
TASK_ROBOMIMIC=$TASK_ROOT/deps/robomimic-compat-20261029-r1/repo
TASK_INPUT=$TASK_ROOT/inputs/dsrl-can-8d21b9c-20261029-r1
[[ $TASK_SOURCE =~ ^[0-9a-f]{40}$ ]]
[[ $TASK_RUN =~ ^robomimic-can-dsrl-baige-20261029-qual-r[0-9]+$ ]]
[[ $(hostname) =~ ^job-[a-z0-9]+-master-0$ ]]
test -n "${KUBERNETES_SERVICE_HOST:-}"
test -n "${NVIDIA_VISIBLE_DEVICES:-}"
cd "$TASK_WORKTREE"
test "$(git rev-parse HEAD)" = "$TASK_SOURCE"
test -z "$(git status --porcelain)"
mkdir "$TASK_OUTPUT"
exec > >(tee "$TASK_OUTPUT/console.log") 2>&1
trap 'TASK_EXIT=$?; printf "%s\n" "$TASK_EXIT" > "$TASK_OUTPUT/exit-code.txt"' EXIT
export PYTHONPATH="$TASK_WORKTREE/src:$TASK_DSRL:$TASK_DSRL/dppo:$TASK_DSRL/stable-baselines3"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export FLUFFY_GPU_EXECUTION=baige CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export NUMBA_CACHE_DIR="$TASK_ROOT/cache/dsrl-can-numba-r1"
export WANDB_MODE=disabled
mkdir -p "$NUMBA_CACHE_DIR"
CUDA_VISIBLE_DEVICES='' "$TASK_PYTHON" -m pytest -q -p no:cacheprovider \
  tests/test_can_dsrl.py tests/test_can_dsrl_compare.py > "$TASK_OUTPUT/tests.log" 2>&1
TASK_COMMON=(--dsrl-root "$TASK_DSRL" --robomimic-root "$TASK_ROBOMIMIC" --checkpoint "$TASK_INPUT/state_5000.pt"
  --normalization "$TASK_INPUT/normalization.npz" --source-commit "$TASK_SOURCE"
  --checkpoint-sha256 61851045e6b516807826e3bda4270c9e4a086023f2dd85af00eadb59d9b98a1b
  --normalization-sha256 a4bb04c498625bfad0ee6faae21674c0a06879cd9f9614bbd2da41a7d0dc1c1a)
"$TASK_PYTHON" -m fluffy_bassoon.meta_rl.can_dsrl_run --method qualify "${TASK_COMMON[@]}" \
  --output "$TASK_OUTPUT/qualification" --max-seconds 600
"$TASK_PYTHON" -m fluffy_bassoon.meta_rl.can_dsrl_run --method native "${TASK_COMMON[@]}" \
  --output "$TASK_OUTPUT/native" --eval-trials 8 --attempts 4 --max-seconds 500
"$TASK_PYTHON" - "$TASK_OUTPUT" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
q = json.loads((p/'qualification/summary.json').read_text())
rows = json.loads((p/'native/evaluation.json').read_text())['native']
success = [x for row in rows for trial in row['successes'] for x in trial]
result = dict(qualified=q['passed'], native_success=sum(success)/len(success),
              native_episodes=len(success), training_launched=False,
              next_action='Review learning headroom and measured simulator/update cost before bounded pilot')
(p/'admission.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result), flush=True)
PY
