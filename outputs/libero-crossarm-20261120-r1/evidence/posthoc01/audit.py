"""Read a completed pilot and measure post hoc separability of the frozen reader's support embeddings."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from learner import _support, load

run = Path(sys.argv[1])
read = lambda p: json.loads(p.read_text())
status = read(run / 'status.json')
assert status['status'] == 'complete' and status['paired_cases'] == 4
checkpoint = run / 'reader-adapter.pt'
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == status['checkpoint_sha256']
frozen = load(checkpoint)
labels = ('Panda', 'Sawyer', 'UR5e')

@torch.inference_mode()
def embedding(file):
    with np.load(file) as t:
        history = _support(t['commands'], t['motion'], frozen._support_size)
    return frozen._network.encode(torch.from_numpy(history[None])).numpy()[0]

refs = {body: sorted((run / 'calibration_check').glob(f'*/{body}/prescribed/tape.npz')) for body in labels}
assert all(len(files) == 2 for files in refs.values())
centres = {body: np.stack([embedding(f) for f in files]).mean(0) for body, files in refs.items()}
rows, cases = [], []
for pair in read(run / 'pairs.json'):
    traces = {}
    for tag in ('support', 'donor'):
        s = pair[tag]
        z = embedding(run / s['tape'])
        dist = {body: float(np.linalg.norm(z - centre)) for body, centre in centres.items()}
        rows.append(dict(path=s['tape'], body=s['body'], nearest=min(dist, key=dist.get), distances=dist))
    assert pair['support']['commands_sha256'] == pair['donor']['commands_sha256']
    arms = pair['arms']
    baseline = arms['unadapted']
    for name, record in arms.items():
        assert record['reset'] == baseline['reset'] and record['opening_sha256'] == baseline['opening_sha256']
        assert record['terminated'] or record['truncated']
        with np.load(run / record['tape']) as t:
            assert len(t['commands']) == len(t['rewards']) == record['controls']
            assert np.isclose(t['rewards'].sum(), record['native_return'])
            assert bool((t['rewards'] > 0).any()) == record['native_success']
            traces[name] = t['commands'][:5, :6].copy()
    assert all(v.shape == (5, 6) for v in traces.values())
    rms = lambda a, b: float(np.sqrt(np.mean((traces[a] - traces[b])**2)))
    cases.append(dict(case=pair['case'], prompt=baseline['reset']['prompt'],
                      outcomes={n: r['native_success'] for n, r in arms.items()},
                      controls={n: r['controls'] for n, r in arms.items()},
                      opening_rms_vs_learned={n: rms('learned', n) for n in arms},
                      opening_rms_vs_calibrated={n: rms('action_oracle', n) for n in arms}))
for name, counts in status['counts'].items():
    assert counts['successes'] == sum(c['outcomes'][name] for c in cases)
print(json.dumps(dict(checkpoint=status['checkpoint_sha256'],
                      method='Euclidean distance to the mean of two labelled, held-apart 16-transition calibration-check histories per body (Goal 4 and Goal 5, start 9). Frozen reader, no classifier tuning.',
                      reference_tapes={b: [str(f.relative_to(run)) for f in fs] for b, fs in refs.items()},
                      queries=rows, nearest_correct=sum(r['body']==r['nearest'] for r in rows), cases=cases), indent=2))
