#!/usr/bin/env bash
# Frozen native source-task screen: the first five existing eval-bank scenes.
set -Eeuo pipefail
TASK_SOURCE=${1:?exact reviewed project commit}
TASK_OUT=${2:?fresh screen output}
TASK_CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
[[ $(uname -s) == Linux && $(hostname) == job-* ]]
[[ $(git -C "$TASK_CODE" rev-parse HEAD) == "$TASK_SOURCE" ]]
[[ -z $(git -C "$TASK_CODE" status --porcelain --untracked-files=no) ]]
[[ ! -e $TASK_OUT ]]
mkdir -p "$TASK_OUT"
TASK_STARTED=$(date +%s)
finish() {
  TASK_RC=$?
  trap - EXIT
  TASK_ENDED=$(date +%s)
  printf '{"exit_code":%s,"started_unix":%s,"ended_unix":%s,"worker_gpu_seconds":%s}\n' \
    "$TASK_RC" "$TASK_STARTED" "$TASK_ENDED" "$((TASK_ENDED-TASK_STARTED))" > "$TASK_OUT/EXIT.json"
}
trap finish EXIT
# The scheduler also enforces30min. This group deadline retains exit evidence.
timeout --signal=TERM --kill-after=15s 1650 bash -c '
  set -Eeuo pipefail
  for task in handover_mic place_a2b_left place_phone_stand; do
    bash "$1/outputs/robotwin-sidney-qualification-20261105-r1/qualify.sh" "$2" "$3/$task" "$task" 5
  done
' screen "$TASK_CODE" "$TASK_SOURCE" "$TASK_OUT"
python3 - "$TASK_OUT" "$TASK_SOURCE" <<'PY'
import json, sys
from pathlib import Path
root=Path(sys.argv[1]); rows=[]
for task in ('handover_mic','place_a2b_left','place_phone_stand'):
    result=json.loads((root/task/'native/RESULT.json').read_text())
    assert result['status']=='complete' and result['episode_count']==5
    assert result['optimizer_updates']==0
    rows.append(dict(task=task,successes=result['successes'],episodes=5,
                     seeds=[e['actual_seed'] for e in result['episodes']],
                     controls=[e['physical_controls'] for e in result['episodes']]))
(root/'SCREEN.json').write_text(json.dumps(dict(status='complete',source_commit=sys.argv[2],
    tasks=rows,episodes=15,optimizer_updates=0,
    limitation='Three source tasks with existing banks; not the full six-task VLRLA screen or a comparison with Motus.'),indent=2)+'\n')
PY
