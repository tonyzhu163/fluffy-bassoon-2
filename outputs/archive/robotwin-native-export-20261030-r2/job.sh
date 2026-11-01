set -euo pipefail
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_RUN=robotwin-native-export-20261030-r2
TASK_PIN=9c27e027ffed1f66a5d56c79e142e7a55faba717
TASK_CODE=$TASK_ROOT/worktrees/$TASK_RUN
TASK_OUT=$TASK_ROOT/outputs/$TASK_RUN
TASK_PYTHON=$TASK_ROOT/envs/robotwin-vlrla-20261029-r1/bin/python
TASK_PREFIX=$TASK_ROOT/exports/robotwin-native-prefix-20261030-r1
TASK_SOURCE=$TASK_ROOT/outputs/robotwin-wholetrial-sac-20261030-r2/experiment/corpus
TASK_SHA=f2fb39571cc3d0e809dfdc454f7f6df7f48e585fc972fd4c2fb8c83d36e63a42
test ! -e "$TASK_OUT"
mkdir -p "$TASK_OUT"
exec > >(tee "$TASK_OUT/console.log") 2>&1
trap 'TASK_EXIT=$?; printf "%s\n" "$TASK_EXIT" > "$TASK_OUT/exit-code.txt"' EXIT
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$TASK_CODE/src"
cd "$TASK_CODE"
test "$(git rev-parse HEAD)" = "$TASK_PIN"
test -z "$(git status --porcelain)"
TASK_HOST=$(hostname)
"$TASK_PYTHON" - "$TASK_OUT" "$TASK_PIN" <<'PYRUNTIME'
from pathlib import Path
import json,platform,socket,sys,time,torch
assert sys.platform=='linux'
assert torch.cuda.device_count()==0
Path(sys.argv[1], 'runtime.json').write_text(json.dumps(dict(hostname=socket.gethostname(),source_commit=sys.argv[2],platform=platform.platform(),started_unix=time.time(),allocated_cpus=4,allocated_gpus=0,additional_simulator_actions=0,additional_vla_queries=0,sgd_steps=0),indent=2)+'\n')
PYRUNTIME
timeout 90 "$TASK_PYTHON" -m pytest -q -x --tb=short -p no:cacheprovider --basetemp "/dev/shm/$TASK_RUN" tests/test_robotwin_history_corpus.py tests/test_vla_noise_ppo.py tests/test_vla_noise_sac.py > "$TASK_OUT/tests.log" 2>&1
cat "$TASK_OUT/tests.log"
timeout 145 "$TASK_PYTHON" -m fluffy_bassoon.meta_rl.robotwin_history_corpus --corpus-root "$TASK_SOURCE" --source-recovery "$TASK_PREFIX/sac-source-recovery.pt" --source-sha256 "$TASK_SHA" --source-method sac --teacher-recovery "$TASK_PREFIX/sac-source-recovery.pt" --teacher-sha256 "$TASK_SHA" --teacher-method sac --objectives ad dpt --output "$TASK_OUT/export" --device cpu --threads 1 --source-commit "$TASK_PIN" --expected-hostname "$TASK_HOST" > "$TASK_OUT/export.log" 2>&1
cat "$TASK_OUT/export.log"
