"""Verify a private policy-0 copy and its original paired baseline on Baige."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path('/mowenjie/fluffy-bassoon-2/outputs')
SOURCE = ROOT / 'robocasa-ppo-grpo-meta-stage1-20261104-r1/ppo/attempt-01'
TARGET = ROOT / 'robocasa-opendrawer-ppo-ablation-3m-20261104-r1/input-prefix-r1'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(16 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    import torch

    assert shutil.disk_usage(ROOT).free >= 230 * (1 << 30)
    TARGET.mkdir(exist_ok=False)
    original = SOURCE / 'OpenDrawer/checkpoints/global_step_0'
    local = TARGET / 'global_step_0'
    shutil.copytree(original, local)
    boundary = json.loads((local / 'pilot-boundary.json').read_text())
    assert boundary['completed'] and boundary['global_step'] == 0
    assert len(boundary['workers']) == 18 and len(boundary['actor_checkpoint_files']) == 4
    for item in boundary['workers'] + boundary['actor_checkpoint_files']:
        private = local / Path(item['path']).relative_to(original)
        assert sha(private) == item['sha256'], private
    states = {}
    for path in sorted((local / 'pilot').glob('*.pt')):
        state = torch.load(path, map_location='cpu', weights_only=False)
        if state['role'] == 'actor':
            assert state['optimizer_steps'] == 0
        assert state['paired_eval_rng'] is not None
        states[path.stem] = {k: state.get(k) for k in ('role', 'rank', 'version', 'iteration', 'optimizer_steps')}
    canonical = TARGET / 'canonical-eval'
    shutil.copytree(SOURCE / 'canonical-eval', canonical)
    references = []
    episodes = []
    for rank in range(8):
        sub = f'rank-{rank:02d}'
        row = json.loads((canonical / sub / 'complete.json').read_text())
        assert sha(canonical / sub / 'initial.pt') == row['snapshot_sha256']
        path = SOURCE / 'environment' / sub / 'before-i000-round00-episodes.json'
        receipt = json.loads(path.read_text())
        assert receipt['completed'] and receipt['rank'] == rank
        references.append(dict(path=str(path), sha256=sha(path)))
        episodes.extend(receipt['rows'])
    assert sorted(r['seed'] for r in episodes) == list(range(610100, 610164))
    assert sum(r['success_once'] for r in episodes) == 34
    reference = dict(completed=True, cases=64, checkpoint=str(local),
        checkpoint_boundary_sha256=sha(local / 'pilot-boundary.json'), canonical_directory=str(canonical),
        metrics=json.loads((SOURCE / 'before-evaluation.json').read_text())['metrics'], episode_refs=references)
    assert reference['metrics']['success_once'] == 34 / 64
    out = TARGET / 'BEFORE.json'
    out.write_text(json.dumps(reference, indent=2)+'\n')
    (TARGET / 'PREPARED.json').write_text(json.dumps(dict(verified_files=22, states=states,
        completed=True, source=str(original), checkpoint=str(local), boundary_sha256=reference['checkpoint_boundary_sha256'],
        before_reference=str(out), before_reference_sha256=sha(out), SGD=0, controls=0), indent=2)+'\n')


if __name__ == '__main__':
    main()
