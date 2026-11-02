"""Check downloaded artifact bytes and JSON; does not execute model code."""
from pathlib import Path
import hashlib
import json
import zipfile

root = Path(__file__).resolve().parent
archive = Path('/Users/mowenjie/Downloads/drawer-target-r1-evidence.zip')
source = '7da6f7fe863a08f2065bbcfa1ec3fafb76ae36be'
sha = lambda b: hashlib.sha256(b).hexdigest()
read = lambda p: json.loads(p.read_text())
with zipfile.ZipFile(archive) as z:
    manifest = json.loads(z.read('TRANSFER-MANIFEST.json'))
    assert manifest['source_commit'] == source
    for name, digest in manifest['files'].items():
        assert not Path(name).is_absolute() and '..' not in Path(name).parts
        data = z.read(name)
        assert sha(data) == digest, name
        path = root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            assert path.read_bytes() == data, name
        else:
            path.write_bytes(data)
    (root/'TRANSFER-MANIFEST.json').write_bytes(z.read('TRANSFER-MANIFEST.json'))
checks = read(root/'CPU-CHECKS.json')
assert checks['completed'] and checks['new_outputs'] == 74
assert sha((root/'scoring/PROMPTS.json').read_bytes()) == sha((root/'PREPARED-PROMPTS.json').read_bytes()) == checks['prompts_sha256']
prepared = read(root/'scoring/PROMPTS.json')
messages = {p['id']: p['messages'] for p in prepared}
assert len(messages) == len(prepared) == 74
rows = []
receipts = list((root/'scoring').glob('*.receipt.json'))
for path in receipts:
    stem = path.name.removesuffix('.receipt.json')
    receipt = read(path)
    out = root/'scoring'/(stem+'.jsonl')
    audit = root/'scoring'/(stem+'.prompts.json')
    result = [json.loads(line) for line in out.read_text().splitlines()]
    actual = read(audit)
    assert receipt['completed'] and receipt['labels'] == len(result) == len(actual)
    assert sha(out.read_bytes()) == receipt['sha256']
    assert sha(audit.read_bytes()) == receipt['prompts_sha256']
    for row, prompt in zip(result, actual):
        assert row['id'] == prompt['id']
        assert prompt['messages_sha256'] == sha(json.dumps(messages[row['id']], sort_keys=True).encode())
        assert len(prompt['rendered_sha256']) == 64
        uid = row['id'].rsplit('/', 1)[0]
        control = int(uid.rsplit('-', 1)[1])
        assert row['sampled_frames'] == checks['expected_sampled_frames'][uid]
        assert row['observed_through'] == control and row['prefix_frame_count'] == control+1
    rows.extend(result)
assert len(receipts) == 11 and len(rows) == len({r['id'] for r in rows}) == 74
for arm, digest in checks['criteria_sha256'].items():
    assert sha((root/(arm+'.json')).read_bytes()) == digest
paired = read(root/'PAIRED-SAMPLE.json')
assert len(paired) == 35 and sum(len(p['labels']) for p in paired) == 105
result = read(root/'RESULT.json')
assert result['completed'] and len(result['anchors']) == 4 and all(a['equal'] for a in result['anchors'])
charge = read(root/'terminal-charge.json')
assert charge['source_commit'] == source and charge['exit_code'] == 0 and charge['charged_gpu_seconds'] <= 900
report = dict(completed=True, archive_sha256=sha(archive.read_bytes()), source_commit=source,
    verified_files=len(manifest['files']), receipts=len(receipts), unique_outputs=len(rows),
    actual_rendered_prompt_audits=len(rows), paired_labels=105, anchor_repeats=4,
    all_anchors_match=True, prepared_actual_messages_match=True,
    official_sampled_frame_counts_match=True, local_check='Artifact bytes and JSON only; no model computation')
(root/'COPIED-HASHES.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report))
for arm, metric in result['metrics'].items():
    print(arm, 'false5', metric['false_full_scores'], 'post5', metric['post_success_full_scores'],
        'raw_timed', metric['raw']['correctly_timed_episodes'],
        'confirmed4', metric['two_probe4']['correctly_timed_episodes'],
        'false4', metric['two_probe4']['false_accepted_episodes'],
        'confirmed5', metric['two_probe5']['correctly_timed_episodes'], 'passes', metric['passes_development_screen'])
    for case in metric['two_probe4']['cases']:
        print(case['episode_id'].split('-case-')[1][:6], case['native_first_success'], case['first_accepted'])
print('CHARGE', charge)
