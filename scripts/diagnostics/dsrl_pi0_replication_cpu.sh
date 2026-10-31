#!/usr/bin/env bash
# Qualified runtime only; actual0GPU/4CPU/32GiB allocation. No simulator/VLA.
set -Eeuo pipefail
TASK_SOURCE=${1:?exact runner source}
TASK_RUN=${2:?fresh qualification identity}
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_PREP=dsrl-pi0-task57-20261101-prep-r3
TASK_CODE=$(pwd)
TASK_PY=$TASK_ROOT/envs/$TASK_PREP/bin/python
TASK_OUT=$TASK_ROOT/outputs/$TASK_RUN
test "$(git rev-parse HEAD)" = "$TASK_SOURCE"
test -z "$(git status --porcelain)"
test ! -e "$TASK_OUT"
mkdir -p "$TASK_OUT"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONPATH=$TASK_CODE/src:$TASK_ROOT/deps/$TASK_PREP/repo
export DSRL_OFFICIAL_ROOT=$TASK_ROOT/deps/$TASK_PREP/repo
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export LIBERO_CONFIG_PATH=$TASK_ROOT/cache/$TASK_PREP/libero-config
"$TASK_PY" - "$TASK_OUT" "$TASK_SOURCE" <<'PY'
import glob, json, pathlib, socket, sys
out, source = pathlib.Path(sys.argv[1]), sys.argv[2]
assert sys.platform == 'linux' and socket.gethostname().startswith('job-')
assert not glob.glob('/dev/nvidia[0-9]*'), 'CPU qualification must expose no physical GPU devices'
value=dict(hostname=socket.gethostname(),source_commit=source,physical_gpu_devices=[],
    cpu_max=pathlib.Path('/sys/fs/cgroup/cpu.max').read_text(),
    memory_max=pathlib.Path('/sys/fs/cgroup/memory.max').read_text())
(out/'allocation.json').write_text(json.dumps(value,indent=2)+'\n')
PY
export CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu
exec > >(tee "$TASK_OUT/console.log") 2>&1
TASK_RC=0
timeout --signal=TERM --kill-after=15s 480 "$TASK_PY" -m pytest -q -p no:cacheprovider \
  -k populated --junitxml="$TASK_OUT/qualification.xml" tests/test_dsrl_pi0_replication.py \
  > "$TASK_OUT/qualification.log" 2>&1 || TASK_RC=$?
"$TASK_PY" - "$TASK_OUT" "$TASK_CODE" "$TASK_SOURCE" "$TASK_RC" <<'PY'
import hashlib,json,pathlib,sys
out,code,source,rc=pathlib.Path(sys.argv[1]),pathlib.Path(sys.argv[2]),sys.argv[3],int(sys.argv[4])
paths=['src/fluffy_bassoon/meta_rl/dsrl_pi0_replication.py','tests/test_dsrl_pi0_replication.py']
receipt=dict(source_commit=source,exit_code=rc,passed=rc==0,
    carried_qualification_source='6effe419b1190be1dd8b3d12e765c5ddcae7528b',
    new_check='populated SAC/optimizer/replay/prefetch roundtrip, next update and same-episode alternating slots',
    file_sha256={name:hashlib.sha256((code/name).read_bytes()).hexdigest() for name in paths})
(out/'qualification-summary.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt),flush=True)
PY
printf '%s\n' "$TASK_RC" > "$TASK_OUT/exit-code.txt"
exit "$TASK_RC"
