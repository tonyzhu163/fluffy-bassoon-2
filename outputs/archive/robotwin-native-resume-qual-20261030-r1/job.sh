#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_RUN=robotwin-native-resume-qual-20261030-r1
TASK_PIN=90a234de059eb6d8dc681d668b03839739236e88
TASK_LINEAGE=db0184089fa88a5f96c153f9b80029bb91b77def
TASK_CODE=$TASK_ROOT/worktrees/$TASK_RUN
TASK_INPUTS=${1:?Git-pinned initial manifest directory required}
TASK_OUT=$TASK_ROOT/outputs/$TASK_RUN
TASK_PYTHON=$TASK_ROOT/envs/robotwin-vlrla-20261029-r1/bin/python
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
cp "${BASH_SOURCE[0]}" "$TASK_OUT/launch-script.sh"
timeout 120 "$TASK_PYTHON" -m pytest -q -x --tb=short -p no:cacheprovider \
 --basetemp "/dev/shm/$TASK_RUN" tests/test_robotwin_noise.py tests/test_robotwin_assets.py \
 tests/test_robotwin_fast_observations.py tests/test_vla_noise_ppo.py tests/test_vla_noise_sac.py \
 tests/test_vla_noise_ppo_review.py tests/test_robotwin_campaign_budget.py tests/test_robotwin_corpus.py \
 tests/test_robotwin_campaign_driver.py tests/test_robotwin_parallel_campaign.py tests/test_robotwin_trial_resume.py \
 > "$TASK_OUT/tests.log" 2>&1
cat "$TASK_OUT/tests.log"
timeout 100 "$TASK_PYTHON" - "$TASK_INPUTS" "$TASK_OUT" "$TASK_PIN" "$TASK_LINEAGE" <<'PYQUAL'
from pathlib import Path
import json,hashlib,socket,sys,time,gc,torch
from fluffy_bassoon.meta_rl.robotwin_trial_resume import load_trial_continuation,artifact_digest
inputs,out,pin,lineage=Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3],sys.argv[4]
assert sys.platform=='linux' and socket.gethostname().startswith('job-') and torch.cuda.device_count()==0
rows=[]
for method in ('ppo','sac','ppo_reset'):
 p=inputs/f'{method}-initial-90a234d.json';sha=hashlib.sha256(p.read_bytes()).hexdigest()
 expected=(p.with_suffix('.json.sha256')).read_text().split()[0] if p.with_suffix('.json.sha256').exists() else (p.with_suffix('.sha256')).read_text().split()[0]
 assert sha==expected
 manifest=json.loads(p.read_text());source=Path(manifest['checkpoint']['path'])
 joint=torch.load(source,map_location='cpu',weights_only=False)
 saved=joint['states'][method]
 before=artifact_digest(saved['learner'])
 continuation=load_trial_continuation(p,sha,checkpoint_path=source,checkpoint=joint,execution_commit=pin,lineage_commit=lineage)
 assert artifact_digest(saved['learner'])==before
 assert continuation.start_episode=={'ppo':0,'sac':1,'ppo_reset':1}[method]
 assert continuation.prefix_steps=={'ppo':0,'sac':239,'ppo_reset':223}[method]
 assert continuation.prefix_chunks=={'ppo':0,'sac':8,'ppo_reset':7}[method]
 shapes={k:list(v.shape) for k,v in continuation.prefix_rollout.items()} if continuation.prefix_rollout else None
 rows.append(dict(method=method,source_checkpoint_sha256=manifest['checkpoint']['sha256'],manifest_sha256=sha,
     source_steps=saved['source_steps'],completed_trials=saved['completed_source_trials'],completed_rounds=saved['completed_source_iterations'],
     learner_updates=saved['learner']['updates'],learner_primitive_steps=saved['learner']['primitive_steps'],
     prefix_steps=continuation.prefix_steps,prefix_chunks=continuation.prefix_chunks,start_episode=continuation.start_episode,
     next_seed=continuation.trial['seeds'][continuation.start_episode],prefix_rollout_shapes=shapes,
     pending_rollouts=len(saved.get('pending_rollouts',[])),learner_unchanged=True))
 del joint,saved,continuation;gc.collect()
receipt=dict(completed=True,hostname=socket.gethostname(),source_commit=pin,lineage_source_commit=lineage,
 allocated_gpus=0,allocated_cpus=4,additional_simulator_actions=0,additional_vla_queries=0,additional_production_sgd_updates=0,synthetic_fixture_updates_present=True,manifests=rows)
(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
PYQUAL
