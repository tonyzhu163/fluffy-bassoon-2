#!/usr/bin/env bash
# Allocated Baige 0GPU/8CPU/64GiB only. No models, simulator or scheduler calls.
set -Eeuo pipefail
TASK_STARTED=$(date +%s)
TASK_SOURCE=${1:?exact reviewed combined source}
TASK_RUN=${2:?fresh immutable CPU identity}
TASK_CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
TASK_PYTHON=/mowenjie/fluffy-bassoon/envs/lerobot-v044-8fff0fde/bin/python
TASK_TEST=/mowenjie/fluffy-bassoon-2/envs/pytest-overlay-20261029-r1
TASK_OUTPUT=/mowenjie/fluffy-bassoon-2/outputs/$TASK_RUN
case "$TASK_RUN" in *[!a-zA-Z0-9-]*|'') exit 2;; esac
cd "$TASK_CODE"
test "$(git rev-parse HEAD)" = "$TASK_SOURCE"
test -z "$(git status --porcelain)"
test ! -e "$TASK_OUTPUT"
mkdir "$TASK_OUTPUT"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export PYTHONPATH=$TASK_TEST:$TASK_CODE/src
exec > >(tee "$TASK_OUTPUT/console.log") 2>&1
task_cleanup() {
  TASK_RC=$?
  trap - EXIT
  "$TASK_PYTHON" - "$TASK_OUTPUT" "$TASK_STARTED" "$TASK_SOURCE" "$TASK_RC" <<'PY'
import json, math, sys, time
from pathlib import Path
root, started, source, rc = Path(sys.argv[1]), float(sys.argv[2]), sys.argv[3], int(sys.argv[4])
clock = dict(format='task57_zero_gpu_cpu_clock_v1', source_commit=source,
    started_unix=started, ended_unix=time.time(), status='finished', exit_code=rc,
    maximum_cpu_worker_seconds=1800,
    charged_cpu_worker_seconds=math.ceil(time.time()-started)+60,
    charged_gpu_seconds=0 if (root/'zero-gpu-attestation.json').is_file() else None,
    simulator_actions=0, optimizer_updates=0)
(root/'cpu-worker-clock.json').write_text(json.dumps(clock, indent=2)+'\n')
PY
  printf '%s\n' "$TASK_RC" > "$TASK_OUTPUT/exit-code.txt"
  exit "$TASK_RC"
}
trap task_cleanup EXIT
# Deliberately do not mask CUDA until real allocation visibility is attested.
TASK_REMAINING=$((1740 - $(date +%s) + TASK_STARTED))
test "$TASK_REMAINING" -gt 60
timeout --signal=INT --kill-after=20s "$TASK_REMAINING" "$TASK_PYTHON" - \
  "$TASK_OUTPUT" "$TASK_CODE" "$TASK_SOURCE" "$TASK_STARTED" <<'PY'
import hashlib, json, os, socket, subprocess, sys, time
from pathlib import Path
root, code, source, started = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], float(sys.argv[4])
assert sys.platform == 'linux' and socket.gethostname().startswith('job-')
assert sys.version_info[:2] == (3, 12)
assert Path(sys.prefix).resolve() == Path('/mowenjie/fluffy-bassoon/envs/lerobot-v044-8fff0fde').resolve()
import torch
from fluffy_bassoon.meta_rl.vla_source40_cpu_qualification import cgroup_resources, require_zero_gpu
attestation = require_zero_gpu(torch)
resources = cgroup_resources()
resources['requested'] = dict(gpu_count=0, cpu_cores=8, memory_bytes=64*1024**3)
resources['cgroup_matches_request'] = dict(
    cpu=resources['effective_cpu_quota_cores'] == 8,
    memory=resources['memory_limit_bytes'] == 64*1024**3)
attestation.update(resources=resources, hostname=socket.gethostname(), source_commit=source)
(root/'zero-gpu-attestation.json').write_text(json.dumps(attestation, indent=2)+'\n')
assert all(resources['cgroup_matches_request'].values()), 'Actual CPU allocation differs'
tests = ['tests/test_vla_target_online_dsrl_task57.py',
         'tests/test_vla_target_online_dsrl_task57_recovery.py']
rc = 1
try:
    with (root/'tests.log').open('x') as log:
        rc = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
            '--junitxml='+str(root/'junit.xml'), *tests], cwd=code,
            env=dict(os.environ), stdout=log, stderr=subprocess.STDOUT,
            timeout=max(1, started+1680-time.time()), check=False).returncode
finally:
    log_path = root/'tests.log'
    summary = dict(format='task57_zero_gpu_cpu_qualification_v1', source_commit=source,
        runtime=sys.executable, hostname=socket.gethostname(), passed=rc == 0, exit_code=rc,
        tests=tests, test_log_sha256=hashlib.sha256(log_path.read_bytes()).hexdigest()
        if log_path.is_file() else None, model_construction=False, simulator_actions=0,
        optimizer_updates=0, charged_gpu_seconds=0, started_unix=started, ended_unix=time.time())
    (root/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
raise SystemExit(rc)
PY
