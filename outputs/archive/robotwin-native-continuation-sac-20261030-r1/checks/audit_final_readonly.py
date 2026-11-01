"""Read-only saved-artifact audit; no project/model/NumPy imports or controls."""
from pathlib import Path
import ast
import datetime
import hashlib
import json
import math
import struct
import zipfile

BASE = Path('/mowenjie/fluffy-bassoon-2/outputs/robotwin-native-continuation-sac-20261030-r1')
OUT = BASE / 'slices/0061/experiment'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


summary = read(OUT / 'summary.json')
supervisor = read(BASE / 'supervisor-summary.json')
children = read(BASE / 'children.json')
assert supervisor['status'] == summary['status'] == 'complete'
assert supervisor['failure'] is None and supervisor['next_manifest'] is None
assert (BASE / 'exit-code.txt').read_text().strip() == '0'
assert len(children) == 61
assert all(x['raw_exit_code'] == x['exit_code'] == 0 and not x['timed_out'] for x in children)

refs = [children[-1]['checkpoint']] + [
    dict(path=x['target_recovery'], sha256=x['target_recovery_sha256'])
    for x in summary['target'] if x.get('stage_source_iteration') == 100
]
checkpoints = []
for ref in refs:
    path = Path(ref['path'])
    assert sha(path) == ref['sha256']
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None
        members = len(archive.infolist())
    checkpoints.append(dict(path=str(path), sha256=ref['sha256'], bytes=path.stat().st_size,
                            ZIP_members=members, all_CRCs_passed=True))

stages = {key: dict(status=v['status'], source_steps=v['source_steps'],
                   source_updates=v['source_updates'], source_rng_restored=v['source_rng_restored'],
                   units=len(v['evaluation_units'])) for key, v in summary['stages'].items()}
assert len(stages) == 4
assert all(v['status'] == 'complete' and v['source_rng_restored'] and v['units'] == 10
           for v in stages.values())

counts = dict(episodes=0, chunks=0, actual_actions=0)
phases = {}
for reset_path in (OUT / 'corpus/episodes').glob('*/reset.json'):
    reset = read(reset_path)
    phase = reset['context']['phase']
    if phase not in ('source_validation', 'target'):
        continue
    manifest = read(reset_path.with_name('manifest.json'))
    assert manifest['pending_decision'] is None
    counts['episodes'] += 1
    actions = 0
    chunks = sorted((reset_path.parent / 'chunks').glob('*.json'))
    for chunk_path in chunks:
        chunk = read(chunk_path)
        assert chunk['status'] == 'complete'
        ref = chunk['native_result']['arrays']
        path = OUT / 'corpus' / ref['path']
        assert sha(path) == ref['sha256']
        with zipfile.ZipFile(path) as archive, archive.open('valid_action_mask.npy') as stream:
            assert stream.read(6) == b'\x93NUMPY'
            version = stream.read(2)
            size_bytes = 2 if version == b'\x01\x00' else 4
            size = struct.unpack('<H' if size_bytes == 2 else '<I', stream.read(size_bytes))[0]
            header = ast.literal_eval(stream.read(size).decode())
            raw = stream.read()
        assert header['descr'] in ('|b1', '?', '|?')
        assert len(raw) == math.prod(header['shape']) and all(x in (0, 1) for x in raw)
        actual = sum(raw)
        assert actual == chunk['native_result']['executed_primitive_steps']
        actions += actual
        counts['chunks'] += 1
    assert actions == manifest['executed_primitive_steps']
    assert len(chunks) == manifest['completed_chunks']
    counts['actual_actions'] += actions
    phases[phase] = phases.get(phase, 0) + actions
assert counts['episodes'] == 56

ledger = summary['campaign_budget']
assert ledger['pending_actions'] == ledger['uncertain_actions'] == 0
assert not ledger['unresolved_decisions']
assert ledger['charged_actions'] == ledger['actual_actions'] == 1088029
assert ledger['header']['total_actions'] == 1194400
snapshot = Path(supervisor['existing_ledger_root']) / 'snapshot.json'
motor = read(OUT / 'frozen-motor.json')
assert motor['unchanged'] and motor['before'] == motor['after']
source = summary['source']['sac']
assert source['allocation_source_steps'] == 0
assert source['updates'] == 638500 and source['optimizer_steps'] == 1915500

result = dict(
    status='passed_final_checkpoint_native_mask_and_closed_stage_audit',
    checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    summary_ref=dict(path=str(OUT / 'summary.json'), sha256=sha(OUT / 'summary.json')),
    supervisor_ref=dict(path=str(BASE / 'supervisor-summary.json'), sha256=sha(BASE / 'supervisor-summary.json')),
    children_count=len(children), all_child_raw_exits_zero=True,
    verified_native_cleanup_faults=supervisor['verified_native_cleanup_faults'],
    checkpoints=checkpoints, stages=stages, actual_eval_masks=counts,
    actual_eval_actions_by_phase=phases, frozen_motor=motor,
    closed_ledger={k: ledger[k] for k in ('charged_actions', 'actual_actions', 'pending_actions',
                  'uncertain_actions', 'by_phase', 'by_method', 'decision_count',
                  'journal_events', 'journal_sha256')},
    canonical_snapshot_ref=dict(path=str(snapshot), sha256=sha(snapshot), bytes=snapshot.stat().st_size),
    canonical_snapshot_keys=list(read(snapshot)), final_source=source,
    terminal_summary_elapsed_seconds=supervisor['elapsed_seconds'],
    limitations='Read-only SHA/ZIP/JSON/native-mask inspection; no tensor deserialization/model construction/tests. '
                'Full RGB payloads not rehashed. Loader/Adam/RNG semantics qualified in allocated production; '
                'source_rng_restored true after every stage.')
print(json.dumps(result, indent=2))
