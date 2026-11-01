#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_RUN=robotwin-native-evaluation-cpu-20261030-r1
TASK_PIN=82c4d8c006db064096549538e9cbab0c2f05facd
TASK_CODE=$TASK_ROOT/worktrees/robotwin-history-live-20261030-r1
TASK_OUT=$TASK_ROOT/outputs/$TASK_RUN
TASK_PYTHON=$TASK_ROOT/envs/robotwin-vlrla-20261029-r1/bin/python
test ! -e "$TASK_OUT"
mkdir -p "$TASK_OUT"
exec > >(tee "$TASK_OUT/console.log") 2>&1
trap 'TASK_EXIT=$?; printf "%s\n" "$TASK_EXIT" > "$TASK_OUT/exit-code.txt"' EXIT
export CUDA_VISIBLE_DEVICES='' PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$TASK_CODE/src"
export FB2_EXPECTED_SOURCE_COMMIT=$TASK_PIN FB2_EXPECTED_HOSTNAME=$(hostname)
cd "$TASK_CODE"
test "$(git rev-parse HEAD)" = "$TASK_PIN"
test -z "$(git status --porcelain)"
cp "${BASH_SOURCE[0]}" "$TASK_OUT/launch-script.sh"
"$TASK_PYTHON" -c 'import json,socket;print(json.dumps(dict(hostname=socket.gethostname(),allocated_gpus=0,allocated_cpus=4)))' > "$TASK_OUT/allocation-context.json"
test "$(sha256sum outputs/robotwin-native-evaluation-integration-20261030-r1/focused-cpu.sh | cut -d' ' -f1)" = f6bd1b649e6bf713210c40e66c567d4ba33e793021b95209ba2ad4f6eb9490c4
timeout 230 bash outputs/robotwin-native-evaluation-integration-20261030-r1/focused-cpu.sh > "$TASK_OUT/tests-and-admission.log" 2>&1
cat "$TASK_OUT/tests-and-admission.log"
"$TASK_PYTHON" - "$TASK_OUT" <<'PYRECEIPT'
from pathlib import Path
import json,sys
out=Path(sys.argv[1]);text=(out/'tests-and-admission.log').read_text();start=text.rfind('\n{');assert start>=0
receipt=json.loads(text[start+1:]);assert len(receipt['checked'])==5 and receipt['additional_robot_actions']==0 and receipt['simulator_constructed'] is False
(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
PYRECEIPT
