"""Read-only JSON/hash check; no models, arrays, tests or journal traversal."""
import hashlib
import json
from pathlib import Path
import time

root = Path('/mowenjie/fluffy-bassoon-2/outputs/dsrl-pi05-headroom-20261101-s0-r1')
case = root / 'experiment/case-learn-44'
def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()
def ref(path):
    path = Path(path)
    return dict(path=str(path), sha256=digest(path), bytes=path.stat().st_size)

index_bytes = (case / 'index.json').read_bytes()
index = json.loads(index_bytes)
clock = json.loads((root / 'worker-time.json').read_text())
attempt = clock['attempts'][-1]
receipt = dict(format='pi05_headroom_learning_light_check_v1', checked_unix=time.time(),
    job_id='job-w8deahieli8q', source_commit='b6655659d8bff2c6b5d14eaf05d63f4647176d4f',
    attempt=attempt, index_sha256=hashlib.sha256(index_bytes).hexdigest(), index=index,
    probe=ref(Path(attempt['path']) / 'gpu-probe.json'),
    migration=ref(root / 'screen-to-learning-source.json'),
    recovery_payload_deserialized=False, exhaustive_journal_reconciliation=False)
for retry in range(2):
    state_bytes = (case / 'state.json').read_bytes()
    model = json.loads(state_bytes)['model']
    actual = ref(case / model['path'])
    next_model = json.loads((case / 'state.json').read_text())['model']
    if next_model == model and actual['sha256'] == model['sha256']:
        receipt['stable_recovery'] = dict(pointer=model, actual=actual, stable=True)
        break
else:
    receipt['stable_recovery'] = dict(stable=False, reason='Rotating live slot changed during bounded read')
panels = index['panels']
native = {row['case']['case_id']: row for row in panels[0]['receipts']}
identity_keys = ('initial_state', 'scene_seed', 'episode_random_seed',
    'initial_state_vector_sha256', 'initial_observation_sha256')
receipt['panel_summaries'] = []
for panel in panels:
    rows = panel['receipts']
    receipt['panel_summaries'].append(dict(name=panel['name'], queries=len(rows),
        successes=sum(row['success'] for row in rows),
        primitive_steps=sum(row['primitive_steps'] for row in rows),
        settling_steps=sum(row['settling_steps'] for row in rows),
        optimizer_updates=panel['actual_optimizer_updates'],
        replay_isolated=all(row['query_data_added_to_replay'] is False for row in rows),
        matched_native_initials=all(all(row[k] == native[row['case']['case_id']][k]
            for k in identity_keys) for row in rows),
        immutable_recovery=ref(case / panel['recovery_path'])))
if index['training']['receipts']:
    row = index['training']['receipts'][-1]
    actual = ref(case / row['trace'])
    receipt['newest_training_trace'] = dict(receipt=row, actual=actual,
        hash_matches=actual['sha256'] == row['trace_sha256'])
receipt['verification_limits'] = 'Producer cumulative costs retained; no full journal forest scan, tensor deserialization, model execution, or saved-array numerical analysis.'
print(json.dumps(receipt, indent=2))
