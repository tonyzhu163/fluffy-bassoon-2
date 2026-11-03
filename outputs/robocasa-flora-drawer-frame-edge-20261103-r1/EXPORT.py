"""Package completed evidence for download and independent hash verification."""
import hashlib
import json
from pathlib import Path
import platform
import zipfile

assert platform.system() == 'Linux'
root = Path('/mowenjie/fluffy-bassoon-2/outputs/robocasa-flora-drawer-frame-edge-20261103-r1')
result = json.loads((root/'RESULT.json').read_text())
charge = json.loads((root/'terminal-charge.json').read_text())
assert result['completed'] and charge['exit_code'] == 0
assert json.loads((root/'scoring/COMPLETED.json').read_text())['new_outputs'] == 40
paths = sorted(set([*root.glob('*.json'), *(root/'scoring').glob('*.json'),
    *(root/'scoring').glob('*.jsonl'), root/'allocation/gpu-probe.json', root/'allocation/scorer.log']))
manifest = dict(completed=True, source_commit=charge['source_commit'],
    files={str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths})
archive = Path('/root/workspace/drawer-frame-edge-r1-evidence.zip')
with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as output:
    for path in paths:
        output.write(path, str(path.relative_to(root)))
    output.writestr('TRANSFER-MANIFEST.json', json.dumps(manifest, indent=2)+'\n')
print('EVIDENCE_ARCHIVE', archive.name, archive.stat().st_size,
    hashlib.sha256(archive.read_bytes()).hexdigest(), 'FILES', len(paths))
print('ANCHORS', result['anchors'])
for arm, metric in result['metrics'].items():
    print(arm, 'false', metric['false_full_scores'], 'post', metric['post_success_full_scores'],
        'confirmed5', metric['correctly_timed_confirmed_episodes'],
        'false_episodes', metric['false_accepted_episodes'],
        'closed_again', metric['closed_again_full_scores'], 'passes', metric['passes_development_screen'])
    for case in metric['cases']:
        print(case['episode_id'].split('-case-')[1].split('-')[0], case['native_first_success'], case['first_accepted'])
print('CHARGE', charge)
