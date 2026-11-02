"""Package completed evidence for download and independent hash verification."""
import hashlib
import json
from pathlib import Path
import platform
import zipfile

assert platform.system() == 'Linux'
root = Path('/mowenjie/fluffy-bassoon-2/outputs/robocasa-flora-drawer-chunk-rubric-20261103-r1')
result = json.loads((root/'RESULT.json').read_text())
charge = json.loads((root/'terminal-charge.json').read_text())
assert result['completed'] and charge['exit_code'] == 0
assert json.loads((root/'scoring/COMPLETED.json').read_text())['new_outputs'] == 74
paths = sorted(set([*root.glob('*.json'), *(root/'scoring').glob('*.json'),
    *(root/'scoring').glob('*.jsonl'), root/'allocation/gpu-probe.json', root/'allocation/scorer.log']))
manifest = dict(completed=True, source_commit=charge['source_commit'],
    files={str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths})
archive = Path('/root/workspace/drawer-chunk-r1-evidence.zip')
with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as output:
    for path in paths:
        output.write(path, str(path.relative_to(root)))
    output.writestr('TRANSFER-MANIFEST.json', json.dumps(manifest, indent=2)+'\n')
print('EVIDENCE_ARCHIVE', archive.name, archive.stat().st_size,
    hashlib.sha256(archive.read_bytes()).hexdigest(), 'FILES', len(paths))
print('ANCHORS', result['anchors'])
for arm, metric in result['metrics'].items():
    print(arm, 'false', metric['false_full_scores'], 'raw_timed', metric['raw']['correctly_timed_episodes'], 'confirmed_timed', metric['two_probe_confirmation']['correctly_timed_episodes'], 'confirmed_false', metric['two_probe_confirmation']['false_accepted_episodes'], 'passes', metric['passes_development_screen'])
    for case in metric['two_probe_confirmation']['cases']:
        print(case['episode_id'].split('-case-')[1].split('-')[0], case['native_first_success'], case['first_accepted'])
print('CHARGE', charge)
