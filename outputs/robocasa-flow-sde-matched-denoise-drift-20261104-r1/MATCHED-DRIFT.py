"""Cross-task drift on cohorts matched by Flow-SDE denoising index.

The fixed-B16 replay used one cohort per task, each at a single denoising
index (OpenDrawer 0 ... Drawer→Counter 9). Log-density changes scale with the
inverse transition variance, so those raw numbers are not comparable across
tasks. Here every task is scored on the same set of indices, 16 policy-zero
samples per index, under every retained checkpoint.

  prepare  Linux CPU: pick matched samples from the retained iteration-1 actor
           batches (policy zero, both actor ranks) of each task.
  stage    dev CPU (BOS mount): expose one verified checkpoint at a time.
  run      one allocated GPU: score all matched batches under each checkpoint.

No simulator controls, sampling or SGD.
"""
import argparse
import gc
import importlib.util
import json
from pathlib import Path
import time

HERE = Path(__file__).resolve()
SOURCE_RUN = 'robocasa-flow-sde-10m-per-task-20261103-r1'
_spec = importlib.util.spec_from_file_location(
    'offline_initial_ratio', HERE.parents[1] / SOURCE_RUN / 'OFFLINE-INITIAL-RATIO.py')
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)
TASKS, RUNTIME, sha, write, tensor_sha = base.TASKS, base.RUNTIME, base.sha, base.write, base.tensor_sha
ORDER = ['PickPlaceDrawerToCounter', 'OpenDrawer', 'CoffeeSetupMug', 'CloseToasterOvenDoor',
         'TurnOffStove']
assert sorted(ORDER) == sorted(TASKS)
NUM_STEPS, NOISE_LEVEL, PER_INDEX, SEED = 10, 0.3, 16, 20261104


def transition_std():
    """Flow-SDE transition std per denoising index; matches the native sampler
    (verified in ../robocasa-flow-sde-dimension-drift-20261104-r1/COHORT-READOUT.json)."""
    import torch
    times = torch.linspace(1, 0, NUM_STEPS + 1, dtype=torch.float64)
    denominator = torch.where(times == 1, times[1], times)
    return (NOISE_LEVEL * torch.sqrt((times[:-1] - times[1:]) * times[:-1] / (1 - denominator[:-1]))).tolist()


def sample_index(denoise_inds):
    """One denoising index per sample; fail loudly if the saved layout differs."""
    flat = denoise_inds.reshape(denoise_inds.shape[0], -1)
    assert bool((flat == flat[:, :1]).all()), 'denoise_inds vary within a sample'
    return flat[:, 0].long()


def select(indices, generator):
    """Up to PER_INDEX positions for each index, seeded and order-free."""
    import torch
    chosen, available = {}, {}
    for k in range(NUM_STEPS):
        positions = torch.nonzero(indices == k).flatten()
        available[k] = int(positions.numel())
        if positions.numel() >= PER_INDEX:
            order = torch.randperm(positions.numel(), generator=generator)[:PER_INDEX]
            chosen[k] = positions[order].sort().values
    return chosen, available


def prepare(root, output):
    import torch
    torch.set_num_threads(2)
    assert root.name == SOURCE_RUN
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for task in ORDER:
        q1 = root / task / ('q1-r2' if task == 'OpenDrawer' else 'q1-r1')
        cfg_file = q1 / 'resolved-config.json'
        cfg = json.loads(cfg_file.read_text())
        assert cfg['actor']['model']['openpi']['noise_method'] == 'flow_sde'
        assert float(cfg['actor']['model']['openpi']['noise_level']) == NOISE_LEVEL
        assert int(cfg['actor']['model']['num_steps']) == NUM_STEPS
        flats, olds, sources = [], [], []
        for rank in range(2):
            path = q1 / f'training-rollouts/iteration-0001-policy-0000-rank-{rank:02d}.pt'
            assert path.is_file(), f'{path} absent: restore it from the Q1 corpus archive first'
            batch = torch.load(path, map_location='cpu', weights_only=False)['rollout_batch']
            T, B = batch['prev_logprobs'].shape[:2]
            flats.append({k: v.reshape(T * B, *v.shape[2:]) for k, v in batch['forward_inputs'].items()})
            olds.append(batch['prev_logprobs'].reshape(T * B, 50, 12))
            sources.append(dict(path=str(path), sha256=sha(path), shape=[T, B]))
            del batch
        flat = {k: torch.cat([f[k] for f in flats]) for k in flats[0]}
        old = torch.cat(olds)
        assert old.shape[0] == 1536, old.shape
        indices = sample_index(flat['denoise_inds'])
        chosen, available = select(indices, torch.Generator().manual_seed(SEED))
        batches = []
        for k, positions in chosen.items():
            inputs = {name: v[positions].clone() for name, v in flat.items()}
            assert bool((sample_index(inputs['denoise_inds']) == k).all())
            target = output / f'{task}-index{k}.pt'
            torch.save(dict(forward_inputs=inputs, old_logprobs=old[positions].clone()), target)
            batches.append(dict(denoise_index=k, fixture=str(target), fixture_sha256=sha(target),
                                positions=positions.tolist(),
                                input_tensor_sha256={n: tensor_sha(v) for n, v in inputs.items()}))
        rows.append(dict(task=task, config=str(cfg_file), config_sha256=sha(cfg_file),
                         actor_batches=sources, available_per_index=available, batches=batches))
        write(output / 'PREPARED.json', dict(completed=False, root=str(root), runtime=RUNTIME,
              per_index=PER_INDEX, seed=SEED, tasks=rows))
        del flat, old, flats, olds
        gc.collect()
    common = sorted(set.intersection(*[{b['denoise_index'] for b in r['batches']} for r in rows]))
    assert common, 'No denoising index has 16 policy-zero samples in every task'
    write(output / 'PREPARED.json', dict(completed=True, root=str(root), runtime=RUNTIME,
          per_index=PER_INDEX, seed=SEED, common_indices=common, tasks=rows,
          new_controls=0, SGD=0, model_calls=0))


def stage(root, output, result_root):
    """Same one-private-model-at-a-time bridge as the source run, bound to this script."""
    assert output.resolve().is_relative_to(root.resolve().parent)
    output.mkdir(parents=True, exist_ok=False)
    receipts = []
    for task in ORDER:
        for row in base.checkpoint_catalog(root, task):
            directory = output / task / str(row['step'])
            directory.mkdir(parents=True, exist_ok=False)
            path, temporary, storage = base.obtain_weights(row, directory)
            write(directory / 'READY.json', dict(completed=True, task=task, checkpoint=row,
                  path=str(path), temporary=temporary, storage=storage,
                  result_root=str(result_root), script_sha256=sha(HERE)))
            consumed = result_root / task / f"CONSUMED-{row['step']}.json"
            deadline = time.monotonic() + 6 * 3600
            while not consumed.exists():
                if time.monotonic() > deadline:
                    raise TimeoutError('GPU consumption receipt absent; private model retained')
                time.sleep(5)
            receipt = json.loads(consumed.read_text())
            assert receipt['task'] == task and receipt['step'] == row['step']
            assert receipt['model_sha256'] == row['sha256'] and receipt['strict_load_passed']
            assert receipt['ready_sha256'] == sha(directory / 'READY.json')
            if temporary:
                assert path.parent == directory and sha(path) == row['sha256']
                path.unlink()
            receipts.append(dict(task=task, step=row['step'], private_copy_reclaimed=temporary))
            write(output / 'PROGRESS.json', dict(completed=False, receipts=receipts))
    write(output / 'PROGRESS.json', dict(completed=True, receipts=receipts,
          new_controls=0, SGD=0, model_calls=0))


def summarize(deltas, std):
    """deltas: {index: [16, 50, 12] float64}. Raw and variance-normalized drift."""
    per_index = {}
    for k, delta in sorted(deltas.items()):
        chunk = delta.sum(dim=(1, 2))
        scale = (std[k] / std[0]) ** 2          # rescale to index-0 noise
        per_index[str(k)] = dict(
            mean_chunk_log_ratio=chunk.mean().item(),
            per_sample_chunk_log_ratio=chunk.tolist(),
            variance_normalized_mean_chunk_log_ratio=chunk.mean().item() * scale,
            per_dimension_chunk_log_ratio=delta.sum(dim=1).mean(dim=0).tolist())
    values = [v['variance_normalized_mean_chunk_log_ratio'] for v in per_index.values()]
    return dict(per_index=per_index,
                mean_variance_normalized_over_indices=sum(values) / len(values))


def run(prepared, task, output, staging):
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
    entry = next(r for r in admission['tasks'] if r['task'] == task)
    common = admission['common_indices']
    assert sha(entry['config']) == entry['config_sha256']
    fixtures = {}
    for b in entry['batches']:
        if b['denoise_index'] not in common:
            continue
        assert sha(b['fixture']) == b['fixture_sha256']
        data = torch.load(b['fixture'], map_location='cpu', weights_only=False)
        assert {n: tensor_sha(v) for n, v in data['forward_inputs'].items()} == b['input_tensor_sha256']
        fixtures[b['denoise_index']] = data
    std = transition_std()
    catalog = base.checkpoint_catalog(root, task)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'deltas').mkdir()
    cfg = OmegaConf.create(json.loads(Path(entry['config']).read_text()))
    model = get_model(cfg.actor.model, torch_dtype=torch.bfloat16).to('cuda:0')
    initial, parity, rows = {}, {}, []
    for row in catalog:
        tick = time.monotonic()
        ready_path = staging / task / str(row['step']) / 'READY.json'
        deadline = time.monotonic() + 4 * 3600
        while not ready_path.exists():
            if time.monotonic() > deadline:
                raise TimeoutError('Verified PFS model staging is not ready')
            time.sleep(5)
        ready = json.loads(ready_path.read_text())
        assert ready['completed'] and ready['task'] == task and ready['checkpoint'] == row
        assert ready['script_sha256'] == sha(HERE)
        path = Path(ready['path'])
        assert sha(path) == row['sha256']
        state = _convert_official_openpi_keys(_normalize_state_dict(as_state_dict(
            torch.load(path, map_location='cpu', weights_only=False))), path)
        model.load_state_dict(state, strict=True)
        del state
        write(output / f"CONSUMED-{row['step']}.json", dict(task=task, step=row['step'],
              model_sha256=row['sha256'], strict_load_passed=True, ready_sha256=sha(ready_path)))
        gc.collect()
        model.set_global_step(row['step'])
        model.eval()
        deltas = {}
        for k, data in fixtures.items():
            selected = {n: v.to('cuda:0') for n, v in data['forward_inputs'].items()}
            with torch.no_grad():
                result = model(forward_inputs=selected, compute_logprobs=True,
                               compute_values=True, compute_entropy=False, use_cache=False)
            logprobs = result['logprobs'].detach().float().cpu()
            assert logprobs.shape == (16, 50, 12) and bool(torch.isfinite(logprobs).all())
            if row['step'] == 0:
                parity[str(k)] = flow_logprob_parity(logprobs, data['old_logprobs'])
                assert parity[str(k)]['passed'], (k, parity[str(k)])
                initial[k] = logprobs.clone()
            deltas[k] = logprobs.double() - initial[k].double()
            del result, selected
        torch.save(deltas, output / 'deltas' / f"step-{row['step']:03d}.pt")
        rows.append(dict(step=row['step'], model_sha256=row['sha256'],
                         seconds=time.monotonic() - tick, **summarize(deltas, std)))
        value = dict(completed=False, task=task, runtime=RUNTIME, common_indices=common,
                     transition_std=std, initial_parity=parity, rows=rows,
                     new_controls=0, SGD=0, model_forward_calls=len(rows) * len(fixtures))
        write(output / 'RESULT.json', value)
        print(json.dumps(dict(task=task, step=row['step'],
              normalized=rows[-1]['mean_variance_normalized_over_indices'])), flush=True)
    value['completed'] = True
    write(output / 'RESULT.json', value)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='mode', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('stage')
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--result-root', type=Path, required=True)
    p = sub.add_parser('run')
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--task', choices=TASKS, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--staging', type=Path, required=True)
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare(args.root, args.output)
    elif args.mode == 'stage':
        stage(args.root, args.output, args.result_root)
    else:
        run(args.prepared, args.task, args.output, args.staging)


if __name__ == '__main__':
    main()
