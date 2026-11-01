#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_RUN=robotwin-execution-control-qual-20261031-r1
TASK_OUT=$TASK_ROOT/outputs/$TASK_RUN
TASK_PYTHON=$TASK_ROOT/envs/robotwin-vlrla-20261029-r1/bin/python
TASK_CONTROL=$TASK_ROOT/worktrees/robotwin-execution-control-code-20261031-r2
TASK_GUARD=$TASK_ROOT/worktrees/robotwin-parent-guard-qual-20261030-r1
TASK_PAIR=$TASK_ROOT/worktrees/robotwin-trained-pair-launcher-20261031-r1
test ! -e "$TASK_OUT"
mkdir -p "$TASK_OUT"
exec > >(tee "$TASK_OUT/console.log") 2>&1
trap 'TASK_EXIT=$?; printf "%s\n" "$TASK_EXIT" > "$TASK_OUT/exit-code.txt"' EXIT
export CUDA_VISIBLE_DEVICES='' PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cp "${BASH_SOURCE[0]}" "$TASK_OUT/launch-script.sh"
"$TASK_PYTHON" - "$TASK_OUT" <<'PYHOST'
from pathlib import Path
import json,os,signal,socket,sys,torch
assert sys.platform=='linux' and socket.gethostname().startswith('job-') and torch.cuda.device_count()==0
assert hasattr(os,'pidfd_open') and hasattr(signal,'pidfd_send_signal')
Path(sys.argv[1],'allocation-context.json').write_text(json.dumps(dict(hostname=socket.gethostname(),allocated_gpus=0,allocated_cpus=4,memory_gib=16,real_source_process_signals=0,additional_robot_actions=0,additional_vla_queries=0,additional_production_sgd=0))+'\n')
PYHOST
phase() {
 TASK_LABEL=$1; TASK_CODE=$2; TASK_PIN=$3; TASK_TIMEOUT=$4; shift 4
 cd "$TASK_CODE"
 test "$(git rev-parse HEAD)" = "$TASK_PIN"
 test -z "$(git status --porcelain)"
 export PYTHONPATH="$TASK_CODE/src"
 printf '%s\n' "$TASK_PIN" > "$TASK_OUT/$TASK_LABEL-source.txt"
 timeout --signal=INT --kill-after=10s "$TASK_TIMEOUT" "$TASK_PYTHON" -m pytest -q -x --tb=short -p no:cacheprovider \
  --basetemp "/dev/shm/$TASK_RUN-$TASK_LABEL" --junitxml "$TASK_OUT/$TASK_LABEL.xml" "$@" \
  > "$TASK_OUT/$TASK_LABEL.log" 2>&1
 cat "$TASK_OUT/$TASK_LABEL.log"
}
phase owner-control "$TASK_CONTROL" c62be99615e797fbec108e1f1b5e232e4a591494 120 \
 outputs/robotwin-trained-handoff-control-20261030-r1/tests/test_control.py
phase panel-guard "$TASK_GUARD" c166171b847ada9a20a9629a07d5a0100b517097 120 \
 tests/test_robotwin_parent_guard.py tests/test_robotwin_history_live.py tests/test_robotwin_history_evaluate.py
phase trained-pair "$TASK_PAIR" c353970a316837b13277ec65587e58e5cb25ce56 240 \
 tests/test_robotwin_trained_pair_plan.py tests/test_robotwin_trained_pair_preflight.py \
 tests/test_robotwin_trained_pair_runtime.py tests/test_robotwin_trained_pair_audit.py
"$TASK_PYTHON" - "$TASK_OUT" <<'PYRECEIPT'
from pathlib import Path
import json,sys,xml.etree.ElementTree as ET
out=Path(sys.argv[1]);rows=[]
for phase in ('owner-control','panel-guard','trained-pair'):
 root=ET.parse(out/f'{phase}.xml').getroot();suites=list(root) if root.tag=='testsuites' else [root]
 tests=sum(int(x.attrib.get('tests',0)) for x in suites)
 assert tests>0 and all(int(x.attrib.get(k,0))==0 for x in suites for k in ('failures','errors','skipped'))
 rows.append(dict(phase=phase,source_commit=(out/f'{phase}-source.txt').read_text().strip(),tests=tests,seconds=sum(float(x.attrib.get('time',0)) for x in suites),passed=True))
 receipt=dict(completed=True,stages=rows,allocation=json.loads((out/'allocation-context.json').read_text()),qualification_only=True,live_source_control_performed=False,synthetic_fixture_sgd_present=True)
(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
PYRECEIPT
