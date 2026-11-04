"""Bind evaluation weights and restore the historical PPO40 from its verified archive."""
import os
from pathlib import Path
import subprocess
import sys

import torch

HERE = Path(__file__).resolve().parent
ROOT = Path('/mowenjie/fluffy-bassoon-2/outputs')
STAGE = ROOT / 'robocasa-ppo-grpo-meta-stage1-20261104-r1/ppo/attempt-01'
SWEEP = 'robocasa-flow-sde-checkpoint-selection-20261104-r1'
sys.path.insert(0, str(HERE.parent / SWEEP))
from sweep import read, sha, write


def boundary(step, root=STAGE):
    path = root / f'OpenDrawer/checkpoints/global_step_{step}/pilot-boundary.json'
    data = read(path)
    assert data['completed'] and data['global_step'] == step
    weights, = [r for r in data['actor_checkpoint_files'] if r['path'].endswith('/full_weights.pt')]
    return dict(weights, marker=str(path), marker_sha256=sha(path))


def load(row):
    path = Path(row['path'])
    assert sha(path) == row['sha256'] and path.stat().st_size == row['bytes']
    return torch.load(path, mmap=True, map_location='cpu', weights_only=True)


def same(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(
        a.reshape(-1).view(torch.uint8), b.reshape(-1).view(torch.uint8))


def main(staging):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    torch.set_num_threads(2)
    staging.mkdir(exist_ok=False, parents=True)
    fs = os.statvfs(staging)
    assert fs.f_bavail * fs.f_frsize > 232 << 30
    manifest = HERE.parent / SWEEP / 'SWEEP-PREPARED-R2.json'
    row = next(r for r in read(manifest)['tasks'] if r['task'] == 'OpenDrawer')
    original = {c['step']: c for c in row['checkpoints'] if c['step'] in (0, 40)}
    models = {}
    for step, label in [(0, 'sft'), (40, 'old')]:
        cp = original[step]
        assert sha(cp['marker']) == cp['marker_sha256']
        path = Path(cp['path'])
        ready_path = ROOT / SWEEP / f'OpenDrawer/staging-r1/dev/{step}/READY.json'
        if not path.is_file() and ready_path.is_file():
            ready = read(ready_path)
            if ready['completed'] and ready['checkpoint']['sha256'] == cp['sha256'] and Path(ready['path']).is_file():
                path = Path(ready['path'])
        if not path.is_file():
            checkpoint = Path(cp['marker']).parent
            cold = read(checkpoint / 'COLD-STORAGE.json')
            assert cold['tar_compare'] and cold['full_BOS_readback'] and Path(cold['source']) == checkpoint
            archive = Path(cold['archive'])
            assert archive.stat().st_size == cold['archive_bytes']
            member = checkpoint.name + '/actor/model_state_dict/full_weights.pt'
            path = staging / (label + '.pt')
            with path.open('xb') as stream:
                subprocess.run(['tar','-I','zstd','-xOf',str(archive),member],stdout=stream,check=True)
        if path.parent != staging:
            private = staging / (label + '.pt')
            os.link(path, private)
            path = private
        models[label] = dict(cp, path=str(path))
        assert sha(path) == cp['sha256']
    ref, before, trained = load(models['sft']), load(boundary(0)), load(boundary(40))
    assert ref.keys() == before.keys() == trained.keys()
    heads = [k for k in ref if k.startswith('value_head.')]
    actor = sorted(ref.keys() - set(heads))
    assert actor and any(ref[k].shape != trained[k].shape for k in heads)
    assert all(same(ref[k], before[k]) for k in actor)
    assert all(ref[k].shape == trained[k].shape and ref[k].dtype == trained[k].dtype for k in actor)
    converted = staging / 'new-ppo40-eval.pt'
    torch.save({k: ref[k] if k in heads else trained[k] for k in ref}, converted)
    check = torch.load(converted, mmap=True, map_location='cpu', weights_only=True)
    assert check.keys() == trained.keys() and all(same(check[k], trained[k]) for k in actor)
    assert all(same(check[k], ref[k]) for k in heads)
    models['new'] = dict(path=str(converted), sha256=sha(converted), source=boundary(40))
    resets = [STAGE / f'environment/rank-{b:02d}/before-i000-round00-initial.pt' for b in range(8)]
    assert sha(row['source_config']) == row['source_config_sha256']
    assert sha(row['initial_parity_fixture']) == row['initial_parity_fixture_sha256']
    write(staging / 'READY.json', dict(models=models, source_config=row['source_config'],
        source_config_sha256=row['source_config_sha256'], fixture=row['initial_parity_fixture'],
        fixture_sha256=row['initial_parity_fixture_sha256'], initial_stage1=boundary(0),
        initial_actor_tensors_equal=len(actor), converted_actor_tensors_equal=len(actor),
        substituted_inference_keys=heads, source_sweep_sha256=sha(manifest),
        resets=[dict(path=str(p), sha256=sha(p)) for p in resets]))


def add_grpo(staging):
    """Bind the fourth policy only after the immutable GRPO final checkpoint."""
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    torch.set_num_threads(2)
    fs = os.statvfs(staging)
    assert fs.f_bavail * fs.f_frsize > 214 << 30
    root = ROOT / 'robocasa-ppo-grpo-meta-stage1-20261104-r1/grpo/attempt-01'
    assert read(root/'clock.json')['exit_code'] == 0
    assert read(root/'after-evaluation.json')['global_step'] == 40
    assert read(root/'progress.json')['completed_iterations'] == 40
    base = staging/'READY.json'
    record = read(base)
    ref = load(record['models']['sft'])
    start, final = boundary(0, root), boundary(40, root)
    initial, trained = load(start), load(final)
    actor = {k for k in ref if not k.startswith('value_head.')}
    assert initial.keys() == trained.keys() == actor
    assert all(same(ref[k], initial[k]) for k in actor)
    assert all(ref[k].dtype == trained[k].dtype and ref[k].shape == trained[k].shape for k in actor)
    target = staging/'grpo40-eval.pt'
    with target.open('xb') as stream:
        torch.save({k: trained[k] if k in actor else ref[k] for k in ref}, stream)
    check = torch.load(target, mmap=True, map_location='cpu', weights_only=True)
    assert check.keys() == ref.keys() and all(same(check[k], trained[k] if k in actor else ref[k]) for k in ref)
    record['models']['grpo'] = dict(path=str(target), sha256=sha(target), source=final,
        initial_source=start, initialized_actor_tensors_equal=len(actor),
        converted_actor_tensors_equal=len(actor), inference_only_keys=sorted(ref.keys()-actor))
    record['three_policy_ready_sha256'] = sha(base)
    write(staging/'READY-FOUR.json', record)


if __name__ == '__main__':
    if len(sys.argv) > 2 and sys.argv[2] == 'add-grpo':
        add_grpo(Path(sys.argv[1]))
    else:
        main(Path(sys.argv[1]))
