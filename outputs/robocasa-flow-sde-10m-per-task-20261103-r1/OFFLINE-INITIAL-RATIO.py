"""Fixed B16 saved-chain likelihood readout; no sampling, simulator or SGD.

Prepare on allocated Linux CPU before corpus tiering. Run on one allocated GPU
after native source closure. Only private extracted model weights are removed.
"""
import argparse
import gc
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

TASKS = ['OpenDrawer', 'CloseToasterOvenDoor', 'PickPlaceDrawerToCounter',
         'CoffeeSetupMug', 'TurnOffStove']
RUNTIME = '0988b5f9e9bf31ac3db673cc2860959611b9ae12'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(16 << 20), b''):
            h.update(b)
    return h.hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def tensor_sha(value):
    import torch
    return hashlib.sha256(value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def prepare(root, output):
    import torch
    torch.set_num_threads(2)
    assert root.name == 'robocasa-flow-sde-10m-per-task-20261103-r1'
    assert output.resolve().is_relative_to(root.resolve())
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for task in TASKS:
        q1 = root / task / ('q1-r2' if task == 'OpenDrawer' else 'q1-r1')
        manifest_path = q1 / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        assert manifest['source'] == RUNTIME and manifest['allocated_gpus'] == 8
        assert manifest['batch_layout'] == dict(world=8, train_envs_per_rank=16, eval_envs_per_rank=8, actor_ranks=2)
        assert json.loads((q1 / 'clock.json').read_text())['exit_code'] == 0
        cfg_file = q1 / 'resolved-config.json'
        cfg = json.loads(cfg_file.read_text())
        assert cfg['env']['train']['total_num_envs'] == 128
        assert cfg['env']['train']['rollout_epoch'] == 1
        batch_path = q1 / 'training-rollouts/iteration-0001-policy-0000-rank-00.pt'
        batch_hash = sha(batch_path)
        batch = torch.load(batch_path, map_location='cpu', weights_only=False)['rollout_batch']
        T, B = batch['prev_logprobs'].shape[:2]
        assert (T, B) == (12, 64)
        flat = {k: v.reshape(T * B, *v.shape[2:]) for k, v in batch['forward_inputs'].items()}
        old = batch['prev_logprobs'].reshape(T * B, 50, 12)
        chain_index = {tensor_sha(row): i for i, row in enumerate(flat['chains'])}
        assert len(chain_index) == T * B
        found = None
        for sampling_path in sorted((q1 / 'sampling/rank-00').glob('*.pt.gz')):
            with gzip.open(sampling_path, 'rb') as f:
                record = torch.load(f, map_location='cpu', weights_only=False)
            if record['mode'] != 'train' or record['terminal_value_only']:
                continue
            assert record['producing_policy_iteration'] == 0
            indices = [chain_index.get(tensor_sha(row)) for row in record['forward_sampling_fields']['chains']]
            if all(i is not None for i in indices):
                found = (sampling_path, record, indices)
                break
        assert found is not None
        sampling_path, record, indices = found
        assert len(indices) == len(set(indices)) == 16
        index = torch.tensor(indices)
        assert torch.equal(old[index], record['old_logprobs'])
        inputs = {k: v[index].clone() for k, v in flat.items()}
        target = output / (task + '.pt')
        torch.save(dict(forward_inputs=inputs, old_logprobs=old[index].clone()), target)
        row = dict(task=task, fixture=str(target), fixture_sha256=sha(target),
                   input_tensor_sha256={k: tensor_sha(v) for k, v in inputs.items()},
                   old_logprobs_sha256=tensor_sha(old[index]), indices=indices,
                   config=str(cfg_file), config_sha256=sha(cfg_file),
                   manifest=str(manifest_path), manifest_sha256=sha(manifest_path),
                   actor_batch=str(batch_path), actor_batch_sha256=batch_hash,
                   sampling=str(sampling_path), sampling_sha256=sha(sampling_path),
                   collection=1, producing_policy=0, batch_size=16,
                   fixed_original_rollout_cohort=True)
        rows.append(row)
        write(output / 'PREPARED.json', dict(completed=False, root=str(root), runtime=RUNTIME,
              tasks=rows, new_controls=0, SGD=0, model_calls=0))
        del inputs, batch, flat, old, record
        gc.collect()
    write(output / 'PREPARED.json', dict(completed=True, root=str(root), runtime=RUNTIME,
          tasks=rows, new_controls=0, SGD=0, model_calls=0))


def checkpoint_catalog(root, task):
    steps = {}
    for marker in (root / task).glob(f'*/{task}/checkpoints/global_step_*/pilot-boundary.json'):
        boundary = json.loads(marker.read_text())
        assert boundary['completed']
        matches = [r for r in boundary['actor_checkpoint_files']
                   if r['path'].endswith('/actor/model_state_dict/full_weights.pt')]
        assert len(matches) == 1
        row = dict(step=int(boundary['global_step']), marker=str(marker),
                   marker_sha256=sha(marker), **matches[0])
        step = row['step']
        if step in steps:
            assert row['sha256'] == steps[step]['sha256']
        else:
            steps[step] = row
    expected = [0, 1, 2] + list(range(10, 131, 10))
    assert sorted(steps) == expected, sorted(steps)
    return [steps[s] for s in expected]


def obtain_weights(row, output):
    path = Path(row['path'])
    if path.is_file():
        assert sha(path) == row['sha256']
        return path, False, dict(storage='local')
    checkpoint = Path(row['marker']).parent
    cold_path = checkpoint / 'COLD-STORAGE.json'
    cold = json.loads(cold_path.read_text())
    assert cold['tar_compare'] and cold['full_BOS_readback']
    assert cold['source'] == str(checkpoint)
    archive = Path(cold['archive'])
    assert archive.stat().st_size == cold['archive_bytes']
    size = cold['source_inventory']['actor/model_state_dict/full_weights.pt'][0]
    fs = os.statvfs(output)
    assert fs.f_bavail * fs.f_frsize > (200 << 30) + size
    target = output / 'extracted-full_weights.pt'
    assert not target.exists()
    member = f'{checkpoint.name}/actor/model_state_dict/full_weights.pt'
    with target.open('xb') as f:
        subprocess.run(['tar', '-I', 'zstd', '-xOf', str(archive), member], stdout=f, check=True)
    assert sha(target) == row['sha256']
    return target, True, dict(storage='verified_cold_member', archive=str(archive),
             original_archive_sha256=cold['sha256'], cold_receipt_sha256=sha(cold_path),
             extracted_member=member, extracted_model_sha256=row['sha256'])


def run(prepared, task, output):
    import torch
    from omegaconf import OmegaConf
    from rlinf.models.embodiment.openpi import get_model
    from rlinf.models.embodiment.openpi.checkpoint import _normalize_state_dict, _convert_official_openpi_keys
    from rlinf.utils.ckpt_convertor.openpi._core import as_state_dict
    from fluffy_bassoon.meta_rl.rlinf_robocasa_flow_ppo import flow_logprob_parity
    assert torch.cuda.device_count() == 1
    torch.set_num_threads(4)
    admission = json.loads(prepared.read_text())
    assert admission['completed'] and admission['runtime'] == RUNTIME
    root = Path(admission['root'])
    production = root / task / 'production-r1'
    clock = json.loads((production / 'clock.json').read_text())
    progress = json.loads((production / 'progress.json').read_text())
    after = json.loads((production / 'after-evaluation.json').read_text())
    assert clock['exit_code'] == 0 and clock['allocated_gpus'] == 8
    assert progress['completed_iterations'] == 130 and not progress['operator_checkpoint_exit']
    assert after['completed'] and after['global_step'] == 130
    fixture = next(r for r in admission['tasks'] if r['task'] == task)
    assert sha(fixture['fixture']) == fixture['fixture_sha256']
    assert sha(fixture['config']) == fixture['config_sha256']
    assert sha(fixture['manifest']) == fixture['manifest_sha256']
    data = torch.load(fixture['fixture'], map_location='cpu', weights_only=False)
    assert {k: tensor_sha(v) for k, v in data['forward_inputs'].items()} == fixture['input_tensor_sha256']
    assert tensor_sha(data['old_logprobs']) == fixture['old_logprobs_sha256']
    assert data['old_logprobs'].shape == (16, 50, 12)
    catalog = checkpoint_catalog(root, task)
    assert output.resolve().is_relative_to(root.resolve())
    output.mkdir(parents=True, exist_ok=False)
    cfg = OmegaConf.create(json.loads(Path(fixture['config']).read_text()))
    assert cfg.actor.model.openpi.noise_method == 'flow_sde'
    assert float(cfg.actor.model.openpi.noise_level) == 0.3
    model = get_model(cfg.actor.model, torch_dtype=torch.bfloat16).to('cuda:0')
    selected = {k: v.to('cuda:0') for k, v in data['forward_inputs'].items()}
    initial = None
    rows = []
    for row in catalog:
        tick = time.monotonic()
        path, temporary, storage = obtain_weights(row, output)
        state = _convert_official_openpi_keys(_normalize_state_dict(as_state_dict(
            torch.load(path, map_location='cpu', weights_only=False))), path)
        model.load_state_dict(state, strict=True)
        del state
        if temporary:
            path.unlink()
        gc.collect()
        model.set_global_step(row['step'])
        model.eval()
        with torch.no_grad():
            result = model(forward_inputs=selected, compute_logprobs=True,
                           compute_values=True, compute_entropy=False, use_cache=False)
        logprobs = result['logprobs'].detach().float().cpu()
        assert logprobs.shape == (16, 50, 12) and bool(torch.isfinite(logprobs).all())
        if row['step'] == 0:
            parity = flow_logprob_parity(logprobs, data['old_logprobs'])
            assert parity['passed'], parity
            initial = logprobs.clone()
        delta = logprobs.double() - initial.double()
        joint = delta.sum(dim=(1, 2))
        rows.append(dict(checkpoint=row, storage=storage,
            mean_chunk_log_ratio_to_initial=joint.mean().item(),
            mean_coordinate_log_ratio_to_initial=delta.mean().item(),
            per_sample_chunk_log_ratio_to_initial=joint.tolist(),
            seconds=time.monotonic() - tick))
        value = dict(completed=False, task=task, runtime=RUNTIME,
            prepared=str(prepared), prepared_sha256=sha(prepared), fixture=fixture,
            initial_parity=parity, batch_size=16, chunk=50, executed_action_dimensions=12,
            metric='Mean log p_checkpoint minus log p_initial on one fixed original B16 policy-zero saved-chain cohort; its negative is a small-sample conditional-transition KL estimate, not a full-action marginal KL or performance gate.',
            rows=rows, new_controls=0, SGD=0, model_forward_calls=len(rows))
        write(output / 'RESULT.json', value)
        print(json.dumps(rows[-1]), flush=True)
        del result
    value['completed'] = True
    write(output / 'RESULT.json', value)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='mode', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('run')
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--task', choices=TASKS, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare(args.root, args.output)
    else:
        run(args.prepared, args.task, args.output)


if __name__ == '__main__':
    main()
