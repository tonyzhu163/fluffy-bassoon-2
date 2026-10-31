#!/usr/bin/env bash
# Existing exclusive1GPU allocation only; one fresh seed, no automatic retries.
set -Eeuo pipefail
TASK_STARTED=$(date +%s)
TASK_SOURCE=${1:?exact reviewed runner source}
TASK_RUN=${2:?fresh scientific run identity}
TASK_QUAL=${3:?allocated full qualification output directory}
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_PREP=dsrl-pi0-task57-20261101-prep-r5
TASK_RUNTIME=dsrl-pi0-task57-20261101-prep-r4
TASK_ENV=$TASK_ROOT/envs/$TASK_RUNTIME
TASK_REPO=$TASK_ROOT/deps/$TASK_RUNTIME/repo
TASK_OUT=$TASK_ROOT/outputs/$TASK_RUN
TASK_CODE=$(pwd)
TASK_PY=$TASK_ENV/bin/python
TASK_PROBE=/mowenjie/fluffy-bassoon/envs/lerobot-v044-8fff0fde/bin/python
test "$(git rev-parse HEAD)" = "$TASK_SOURCE"
test -z "$(git status --porcelain)"
test ! -e "$TASK_OUT"
mkdir -p "$TASK_OUT"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONPATH=$TASK_CODE/src:$TASK_REPO:$TASK_REPO/openpi/src:$TASK_REPO/openpi/packages/openpi-client/src
exec > >(tee "$TASK_OUT/console.log") 2>&1
task_cleanup() {
  TASK_RC=$?
  trap - EXIT
  "$TASK_PY" - "$TASK_OUT" "$TASK_STARTED" "$TASK_SOURCE" "$TASK_RC" <<'PY'
import json,math,pathlib,sys,time
out,started,source,rc=pathlib.Path(sys.argv[1]),float(sys.argv[2]),sys.argv[3],int(sys.argv[4])
ended=time.time()
value=dict(source_commit=source,started_unix=started,ended_unix=ended,exit_code=rc,
    charged_gpu_seconds=math.ceil(ended-started)+60,maximum_gpu_seconds=36000,
    scheduler_allowance_seconds=60,allocation_charged_once=True)
(out/'worker-time.json').write_text(json.dumps(value,indent=2)+'\n')
PY
  printf '%s\n' "$TASK_RC" > "$TASK_OUT/exit-code.txt"
  exit "$TASK_RC"
}
trap task_cleanup EXIT
"$TASK_PY" - "$TASK_ROOT/outputs/$TASK_PREP" "$TASK_QUAL" "$TASK_CODE" <<'PY'
import hashlib,json,pathlib,sys,xml.etree.ElementTree as ET
prep,qual,code=map(pathlib.Path,sys.argv[1:])
def checked_xml(path,count):
    root=ET.parse(path).getroot()
    suites=[root] if root.tag=='testsuite' else list(root.iter('testsuite'))
    assert sum(int(s.get('tests',0)) for s in suites)==count
    assert all(int(s.get(key,0))==0 for s in suites for key in ('failures','errors','skipped'))
assert prep.resolve()==qual.resolve()
assert (prep/'exit-code.txt').read_text().strip()=='0'
checked_xml(qual/'qualification.xml',9)
value=json.loads((qual/'qualification-summary.json').read_text())
assert value['passed'] is True and value['exit_code']==0 and value['full_suite'] is True
for name,digest in value['file_sha256'].items():
    assert hashlib.sha256((code/name).read_bytes()).hexdigest()==digest
PY
export FLUFFY_GPU_EXECUTION=baige
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=disabled
"$TASK_PROBE" 'scripts/[train]gpu_probe.py' --expected-count 1 --purpose training \
  --baige-container --baige-exclusive-training --output "$TASK_OUT/gpu-probe.json"
export LYCO_BAIGE_GPU_MAP=$TASK_OUT/gpu-probe.json
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0
export OPENPI_DATA_HOME=$TASK_ROOT/cache/$TASK_PREP
export LIBERO_CONFIG_PATH=$TASK_ROOT/cache/$TASK_RUNTIME/libero-config
export XDG_CACHE_HOME=$TASK_ROOT/cache/$TASK_RUN
TASK_CHECKPOINT=$(cat "$TASK_ROOT/outputs/$TASK_PREP/checkpoint-path.txt")
TASK_REMAINING=$(( TASK_STARTED + 35880 - $(date +%s) ))
test "$TASK_REMAINING" -gt 600
timeout --signal=INT --kill-after=20s "$TASK_REMAINING" "$TASK_PY" \
  -m fluffy_bassoon.meta_rl.dsrl_pi0_replication \
  --dsrl-root "$TASK_REPO" --checkpoint "$TASK_CHECKPOINT" \
  --checkpoint-receipt "$TASK_ROOT/outputs/$TASK_PREP/checkpoint-receipt.json" \
  --source-commit "$TASK_SOURCE" --output "$TASK_OUT/experiment" \
  --deadline-unix "$((TASK_STARTED + 35640))"
