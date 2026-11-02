"""Package completed evidence for download and independent hash verification."""
import hashlib
import json
from pathlib import Path
import platform
import zipfile

assert platform.system() == 'Linux'
root = Path('/mowenjie/fluffy-bassoon-2/outputs/robocasa-flora-first-success-rubric-20261103-r3')
result = json.loads((root/'RESULT.json').read_text())
charge = json.loads((root/'terminal-charge.json').read_text())
assert result['completed'] and charge['exit_code'] == 0
assert json.loads((root/'scoring/COMPLETED.json').read_text())['new_outputs'] == 74
paths = sorted(set([*root.glob('*.json'), *(root/'scoring').glob('*.json'),
    *(root/'scoring').glob('*.jsonl'), root/'allocation/gpu-probe.json', root/'allocation/scorer.log']))
manifest = dict(completed=True, source_commit=charge['source_commit'],
    files={str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths})
archive = Path('/root/workspace/drawer-r3-evidence-eada02cb.zip')
with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as output:
    for path in paths:
        output.write(path, str(path.relative_to(root)))
    output.writestr('TRANSFER-MANIFEST.json', json.dumps(manifest, indent=2)+'\n')
print('EVIDENCE_ARCHIVE', archive.name, archive.stat().st_size,
    hashlib.sha256(archive.read_bytes()).hexdigest(), 'FILES', len(paths))
print('ANCHORS', result['anchors'])
for arm, metric in result['metrics'].items():
    print(arm, 'false', metric['pre_success_or_wrong_target_full_scores'],
        'timed_successes', metric['correctly_timed_successful_episodes'],
        'passes', metric['passes_development_screen'])
    for case in metric['cases']:
        print(case['id'].split('/')[-1], case['native_first_success'], case['first_sampled_full_score'])
print('CHARGE', charge)
