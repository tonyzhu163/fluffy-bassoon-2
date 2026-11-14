"""Protect complete resume state when retiring superseded PPO checkpoints."""
import hashlib
import json

import pytest

from fluffy_bassoon.rlinf.checkpoint_retention import retain_boundaries


def boundary(root, step):
    path = root / f'global_step_{step}'
    path.mkdir(parents=True)
    rows = []
    for name in ('model-optimizer-rng', 'worker-sampler-rng'):
        file = path / name
        file.write_bytes(f'{name}-{step}'.encode())
        rows.append(dict(path=str(file), bytes=file.stat().st_size,
                         sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
    data = dict(completed=True, global_step=step, actor_checkpoint_files=rows[:1], workers=rows[1:])
    # Native worker receipts carry path/hash, while DCP entries also carry size.
    data['workers'][0].pop('bytes')
    (path / 'pilot-boundary.json').write_text(json.dumps(data))
    return path


def test_retention_preserves_initial_eval_latest_two_and_incomplete(tmp_path):
    for step in (0, 1, 5, 10, 15, 20, 25, 30):
        boundary(tmp_path, step)
    (tmp_path / 'global_step_35').mkdir()
    assert retain_boundaries(tmp_path, current_step=30, keep_latest=2, protected_steps=(10, 20)) == [1, 5, 15]
    assert {p.name for p in tmp_path.glob('global_step_*')} == {
        'global_step_0', 'global_step_10', 'global_step_20', 'global_step_25', 'global_step_30', 'global_step_35'}
    assert json.loads((tmp_path / 'retired-boundaries/global_step_5.json').read_text())['removed']


@pytest.mark.parametrize('damage', ['corrupt_newest', 'extra_old_file', 'outside_inventory'])
def test_invalid_boundary_never_deletes_fallback(tmp_path, damage):
    paths = [boundary(tmp_path, s) for s in (0, 1, 5, 10)]
    if damage == 'corrupt_newest':
        (paths[-1] / 'worker-sampler-rng').write_bytes(b'broken')
    elif damage == 'extra_old_file':
        (paths[1] / 'unregistered').write_bytes(b'must retain')
    else:
        receipt = paths[1] / 'pilot-boundary.json'
        data = json.loads(receipt.read_text())
        data['workers'][0]['path'] = str(paths[0] / 'worker-sampler-rng')
        receipt.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        retain_boundaries(tmp_path, current_step=10, keep_latest=2)
    assert all(p.exists() for p in paths)
