"""Package completed run evidence for verified transfer through the console UI."""
import base64
import gzip
import hashlib
import json
from pathlib import Path
import platform

assert platform.system() == 'Linux'
root = Path('/mowenjie/fluffy-bassoon-2/outputs/robocasa-flora-first-success-rubric-20261103-r2')
assert json.loads((root/'RESULT.json').read_text())['completed']
assert json.loads((root/'terminal-charge.json').read_text())['exit_code'] == 0
paths = [p for p in root.glob('*.json') if p.is_file()]
paths += [p for p in (root/'scoring').glob('*') if p.suffix in ('.json', '.jsonl')]
paths += [root/'allocation/gpu-probe.json']
paths = sorted(set(paths))
files = {str(p.relative_to(root)): base64.b64encode(p.read_bytes()).decode() for p in paths}
hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
log = root/'allocation/scorer.log'
wrapper = dict(files=files, sha256=hashes,
    retained_log=dict(path=str(log), sha256=hashlib.sha256(log.read_bytes()).hexdigest(), bytes=log.stat().st_size))
packed = base64.b64encode(gzip.compress(json.dumps(wrapper).encode(), mtime=0)).decode()
parts = [packed[i:i+8000] for i in range(0, len(packed), 8000)]
for i, part in enumerate(parts):
    (Path('/root/workspace')/f'drawer-r2-evidence-part{i:02}.txt').write_text(f'PART{i:02}:{part}:ENDPART{i:02}\n')
print('TRANSFER', dict(files=len(paths), encoded_chars=len(packed), parts=len(parts),
    packed_sha256=hashlib.sha256(packed.encode()).hexdigest()))
for arm, metrics in json.loads((root/'RESULT.json').read_text())['metrics'].items():
    print(arm, 'pre/wrong', metrics['pre_success_or_wrong_target_full_scores'],
        'post', metrics['post_success_full_scores'])
    for case in metrics['cases']:
        print(case['id'].split('/')[-1], 'native', case['native_first_success'],
            'first', case['first_sampled_full_score'], 'wrong', case['premature_or_wrong_target'])
