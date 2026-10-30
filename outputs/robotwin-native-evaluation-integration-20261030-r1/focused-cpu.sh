#!/usr/bin/env bash
# Run only inside the execution owner's allocated Linux CPU job.
set -euo pipefail
: "${FB2_EXPECTED_SOURCE_COMMIT:?Set the exact clean integrated commit}"
: "${FB2_EXPECTED_HOSTNAME:?Set the actual allocated job hostname}"
test "$(uname -s)" = Linux
test "$(hostname)" = "$FB2_EXPECTED_HOSTNAME"
case "$(hostname)" in job-*) ;; *) exit 2 ;; esac
TASK_CODE=$(git rev-parse --show-toplevel)
test "$(git rev-parse HEAD)" = "$FB2_EXPECTED_SOURCE_COMMIT"
test -z "$(git status --porcelain)"
TASK_PYTHON=/mowenjie/fluffy-bassoon-2/envs/robotwin-vlrla-20261029-r1/bin/python
export PYTHONPATH="$TASK_CODE/src"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
"$TASK_PYTHON" -m pytest -q \
  tests/test_robotwin_history_evaluate.py \
  tests/test_robotwin_history.py \
  tests/test_robotwin_history_corpus.py \
  tests/test_vla_history.py \
  tests/test_vla_noise_ppo.py \
  tests/test_vla_noise_ppo_review.py \
  tests/test_vla_noise_sac.py \
  tests/test_robotwin_noise.py \
  tests/test_robotwin_campaign_budget.py \
  tests/test_robotwin_corpus.py \
  tests/test_robotwin_campaign_driver.py \
  tests/test_robotwin_fast_observations.py \
  tests/test_robotwin_assets.py
"$TASK_PYTHON" - <<'PYADMIT'
import hashlib
import json
from pathlib import Path
import socket
import torch
from fluffy_bassoon.meta_rl.robotwin_history_evaluate import admit_plan

torch.set_num_threads(4)
backend=Path('src/fluffy_bassoon/rl/robotwin_noise.py')
assert hashlib.sha256(backend.read_bytes()).hexdigest()=='eacc64b4daf8f78aeb5e023a2c4d2a2a040eacfd2e662335c118334c2ab50ff3'
path=Path('outputs/robotwin-native-evaluation-plan-20261030-r1/plan.json')
assert hashlib.sha256(path.read_bytes()).hexdigest()=='b911dd972cc30e6f6b02cacf831eafa84738bd66ebd032824bb5fc94d340287c'
receipt=admit_plan(json.loads(path.read_text()))
receipt.update(hostname=socket.gethostname(),simulator_constructed=False,allocated_gpus=0)
print(json.dumps(receipt,indent=2),flush=True)
PYADMIT
