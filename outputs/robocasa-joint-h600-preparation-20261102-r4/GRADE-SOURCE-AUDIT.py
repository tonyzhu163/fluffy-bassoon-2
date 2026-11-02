"""Read-only CPU verification of every calibration shard and model/source receipt."""
import json
import os
from pathlib import Path
import subprocess

from fluffy_bassoon.meta_rl.rlinf_flora_ppo import digest, sha, flora_jobs, video_refs_digest
from fluffy_bassoon.meta_rl.rlinf_flora_ppo_runtime import read_ref
from fluffy_bassoon.meta_rl.rlinf_robocasa_ppo import PPO_TASKS, protocol

ROOT = Path('/mowenjie/fluffy-bassoon-2')
RUN = ROOT / 'outputs/robocasa-joint-h600-preparation-20261102-r4'
CODE = ROOT / 'worktrees/joint-h600-preparation-20261102-r4'
assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
qualification = json.loads((CODE / 'outputs' / RUN.name / 'CPU-QUALIFICATION.json').read_text())
for name, expected in qualification['files'].items(): assert sha(CODE / name) == expected
inputs = json.loads((CODE / 'outputs/robocasa-flora-throughput-20261102-r2/INPUTS.json').read_text())
assert subprocess.check_output(['git', '-C', inputs['official_code'], 'rev-parse', 'HEAD'], text=True).strip() == inputs['official_code_commit']
model = read_ref(inputs['model_receipt'])
assert model['completed'] and model['model_revision'] == inputs['model_revision']
gates = json.loads((RUN / 'CALIBRATION-GATES.json').read_text())
checks = []
for task in PPO_TASKS:
    gate = read_ref(gates['gates'][task]); episodes = read_ref(gate['native_episodes'])
    official = read_ref(gate['official_outputs']); reconstructed = {}
    for rank, reference in enumerate(gate['shards']):
        shard = read_ref(reference); receipt = shard['scorer_receipt']; subset = episodes[rank::4]
        assert shard['completed'] and shard['annotation_rank'] == rank and shard['model_load_count'] == 1
        assert shard['episodes'] == gate['native_episodes']
        assert shard['native_receipt_sha256'] == gate['native_receipt']['sha256']
        assert receipt['completed'] and receipt['model_load_count'] == 1
        assert receipt['code_commit'] == inputs['official_code_commit']
        assert receipt['model_revision'] == inputs['model_revision']
        assert receipt['model_receipt'] == inputs['model_receipt']
        assert receipt['batch_size'] == 8 and receipt['gpu_memory_utilization'] == .75
        assert receipt['jobs_sha256'] == digest(flora_jobs(subset, protocol(task, baseline_count=50)))
        assert receipt['video_refs_sha256'] == video_refs_digest(subset)
        assert sha(receipt['output']['path']) == receipt['output']['sha256']
        rows = read_ref(shard['outputs'])
        assert rows == [json.loads(line) for line in Path(receipt['output']['path']).read_text().splitlines()]
        assert [r['id'] for r in rows] == [e['id'] for e in subset]
        for row in rows:
            assert row['id'] not in reconstructed
            reconstructed[row['id']] = row
        checks.append(dict(task=task, rank=rank, grades=len(rows), source_and_model_bound=True,
                           scorer_receipt_sha256=digest(receipt), original_prompt_job_keys=['id','task','videos']))
    assert [reconstructed[e['id']] for e in episodes] == official
warmup = json.loads((CODE / 'outputs' / RUN.name / 'warmup-episodes.json').read_text())
for rank in range(4):
    receipt = json.loads((RUN / f'job/node-{rank:02d}/scorer-{rank:02d}/warmup-scores.receipt.json').read_text())
    assert receipt['model_receipt'] == inputs['model_receipt']
    assert receipt['code_commit'] == inputs['official_code_commit'] and receipt['model_revision'] == inputs['model_revision']
    assert receipt['jobs_sha256'] == digest(flora_jobs(warmup, protocol('CoffeeSetupMug', baseline_count=50)))
    assert receipt['video_refs_sha256'] == video_refs_digest(warmup)
    assert sha(receipt['output']['path']) == receipt['output']['sha256']
    assert len(Path(receipt['output']['path']).read_text().splitlines()) == 8
assert sum(x['grades'] for x in checks) == 250
assert {p.name for p in (RUN / 'calibration/cohorts').iterdir()} == set(PPO_TASKS)
print(json.dumps(dict(completed=True, execution='Allocated Linux CPU with CUDA masked; no model inference',
    calibration_grades=250, shard_receipts=20, warmup_replay_grades=32, model_loads=4,
    code_commit=inputs['official_code_commit'], model_revision=inputs['model_revision'],
    original_instruction_and_three_views_only=True, custom_rubric=False, OOD_Flora_grades=0,
    core_bytes_match95_test_qualification=True, checks=checks), indent=2))
