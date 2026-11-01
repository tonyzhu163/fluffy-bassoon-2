#!/usr/bin/env bash
# Phase payload ONLY after the existing parent's actual0GPU composite attestation.
# Parent owns whole shared deadline, timeout/cleanup, receipt clock and other test component.
set -Eeuo pipefail
: "${TASK_CODE:?}" "${TASK_PYTHON:?}" "${TASK_RECEIPTS:?}" "${TASK_DEADLINE_UNIX:?}"
: "${TASK_PROFILE:?}" "${TASK_PROFILE_SHA256:?}" "${TASK_DEMOS:?}" "${TASK_DEMOS_SHA256:?}"
cd "$TASK_CODE"
"$TASK_PYTHON" -m pytest -q -p no:cacheprovider tests/test_robotwin_prelift_factorial.py \
  --junitxml="$TASK_RECEIPTS/prelift-factorial.junit.xml"
"$TASK_PYTHON" - "$TASK_PROFILE" "$TASK_PROFILE_SHA256" "$TASK_DEMOS" "$TASK_DEMOS_SHA256" "$TASK_DEADLINE_UNIX" "$TASK_RECEIPTS/preflight04-factorial.json" <<'PY'
import hashlib,json,math,sys,time
from pathlib import Path
deadline=float(sys.argv[5])
def guard():
    if sys.platform!='linux' or not math.isfinite(deadline) or time.time()>=deadline:raise TimeoutError('Parent original composite CPU deadline')
guard()
def bound(path,sha):
    guard();p=Path(path)
    if p.is_symlink() or hashlib.sha256(p.read_bytes()).hexdigest()!=sha:raise ValueError('Exact bound profile/demos required')
    return json.loads(p.read_text())
profile=bound(sys.argv[1],sys.argv[2]);demos=bound(sys.argv[3],sys.argv[4])
from fluffy_bassoon.meta_rl.robotwin_prelift_probe import preflight04
result=preflight04(profile,demos,guard=guard);guard()
with Path(sys.argv[6]).open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
PY
