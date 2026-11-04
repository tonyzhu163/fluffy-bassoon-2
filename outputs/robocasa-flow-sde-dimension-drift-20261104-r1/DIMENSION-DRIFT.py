"""Per-dimension likelihood drift and per-update PPO metrics for the 10M sources.

Reads only retained artifacts of robocasa-flow-sde-10m-per-task-20261103-r1.
No sampling, simulator controls or SGD.

  metrics  CPU: join every updates/iteration-*-rank-*.json into one CSV per run.
  stage    dev CPU (BOS mount): expose one verified checkpoint at a time.
  run      one allocated GPU: score the fixed B16 cohort under each checkpoint and
           keep the full [16, 50, 12] log-ratio tensor, not just its sum.
"""
import argparse
import csv
import gc
import importlib.util
import json
import math
from pathlib import Path
import time

HERE = Path(__file__).resolve()
SOURCE_RUN = 'robocasa-flow-sde-10m-per-task-20261103-r1'
_spec = importlib.util.spec_from_file_location(
    'offline_initial_ratio', HERE.parents[1] / SOURCE_RUN / 'OFFLINE-INITIAL-RATIO.py')
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)
TASKS, RUNTIME, sha, write, tensor_sha = base.TASKS, base.RUNTIME, base.sha, base.write, base.tensor_sha
# Env action layout from configs/native_ppo/robocasa365_pi05_flow_sde_10m.yaml.
BINARIZED = {6: 'gripper', 11: 'base_mode'}
# Stage and GPU run must walk tasks in the same order, or each waits on the other.
ORDER = ['PickPlaceDrawerToCounter', 'CoffeeSetupMug', 'TurnOffStove', 'OpenDrawer',
         'CloseToasterOvenDoor']
assert sorted(ORDER) == sorted(TASKS)


def metrics(root, output):
    """One row per (task, segment, iteration, rank) with every logged scalar."""
    assert root.name == SOURCE_RUN
    output.mkdir(parents=True, exist_ok=False)
    summary = {}
    for task in TASKS:
        rows = []
        for path in sorted((root / task).glob('*/updates/iteration-*-rank-*.json')):
            value = json.loads(path.read_text())
            if not value.get('completed'):
                continue
            row = dict(task=task, segment=path.parents[1].name,
                       iteration=value['producing_policy_iteration'] + 1, rank=value['rank'],
                       optimizer_steps_after=value['optimizer_steps_after'])
            row.update(value['metrics'])
            rows.append(row)
        # A retried segment may repeat an iteration; keep the production copy.
        best = {}
        for row in rows:
            key = (row['iteration'], row['rank'])
            if key not in best or row['segment'].startswith('production'):
                best[key] = row
        rows = [best[k] for k in sorted(best)]
        assert {(r['iteration'], r['rank']) for r in rows} == {
            (iteration, rank) for iteration in range(1, 131) for rank in (0, 1)
        }, f'Incomplete update/rank coverage for {task}'
        assert all(r['optimizer_steps_after'] == 24 * r['iteration'] for r in rows)
        assert all(math.isfinite(v) for r in rows for v in r.values() if isinstance(v, float))
        iterations = sorted({r['iteration'] for r in rows})
        summary[task] = dict(rows=len(rows), first=iterations[0] if iterations else None,
                             last=iterations[-1] if iterations else None,
                             missing=sorted(set(range(1, 131)) - set(iterations)))
        if rows:
            fields = sorted({k for r in rows for k in r},
                            key=lambda k: (k not in ('task', 'segment', 'iteration', 'rank'), k))
            with (output / f'{task}.csv').open('x', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
    write(output / 'METRICS.json', dict(completed=True, root=str(root), tasks=summary,
          new_controls=0, SGD=0, model_calls=0))


def stage(root, output, result_root):
    """Same one-private-model-at-a-time bridge as the source run, bound to this script."""
    diagnostic = root.parent / HERE.parent.name
    assert output.resolve() == (diagnostic / 'staging').resolve()
    assert result_root.resolve() == (diagnostic / 'gpu').resolve()
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
            receipts.append(dict(task=task, step=row['step'], private_copy_reclaimed=temporary,
                            ready_sha256=sha(directory / 'READY.json'), consumed_sha256=sha(consumed)))
            write(output / 'PROGRESS.json', dict(completed=False, receipts=receipts))
    write(output / 'PROGRESS.json', dict(completed=True, receipts=receipts,
          new_controls=0, SGD=0, model_calls=0))


def describe(tensor):
    """Per last-dimension mean/std/min/max of a [..., D] tensor."""
    flat = tensor.double().reshape(-1, tensor.shape[-1])
    return dict(shape=list(tensor.shape), mean=flat.mean(0).tolist(), std=flat.std(0).tolist(),
                min=flat.min(0).values.tolist(), max=flat.max(0).values.tolist())


def decompose(delta):
    """delta: [16, 50, 12] float64 log p_checkpoint - log p_initial."""
    per_dim = delta.sum(dim=1).mean(dim=0)          # mean over samples of per-chunk sum, per dim
    total = per_dim.sum().item()
    absolute = per_dim.abs().sum().item()
    negative = (-per_dim.clamp(max=0)).sum().item()
    return dict(
        mean_chunk_log_ratio_to_initial=delta.sum(dim=(1, 2)).mean().item(),
        per_sample_chunk_log_ratio_to_initial=delta.sum(dim=(1, 2)).tolist(),
        per_dimension_chunk_log_ratio=per_dim.tolist(),
        per_dimension_share=[(v / total) if total else 0.0 for v in per_dim.tolist()],
        binarized_dimensions_share=(sum(per_dim[i].item() for i in BINARIZED) / total) if total else 0.0,
        binarized_absolute_dimension_share=(sum(abs(per_dim[i].item()) for i in BINARIZED) / absolute) if absolute else 0.0,
        binarized_negative_dimension_share=(sum(max(-per_dim[i].item(), 0.0) for i in BINARIZED) / negative) if negative else 0.0,
        absolute_dimension_mass=absolute, negative_dimension_mass=negative,
        arm_dimension_log_ratio=per_dim[:6].sum().item(),
        arm_absolute_dimension_share=per_dim[:6].abs().sum().item() / absolute if absolute else 0.0,
        per_timestep_log_ratio=delta.sum(dim=2).mean(dim=0).tolist(),
        per_sample_per_dimension=delta.sum(dim=1).tolist())


def run(prepared, reference, task, output, staging):
    import torch
    from omegaconf import OmegaConf
    from rlinf.models.embodiment.openpi import get_model
    from rlinf.models.embodiment.openpi.checkpoint import _normalize_state_dict, _convert_official_openpi_keys
    from rlinf.utils.ckpt_convertor.openpi._core import as_state_dict
    from fluffy_bassoon.meta_rl.rlinf_robocasa_flow_ppo import flow_logprob_parity
    assert torch.cuda.device_count() == 1
    torch.set_num_threads(4)
    admission = json.loads(prepared.read_text())
    recorded = json.loads((HERE.parents[1] / SOURCE_RUN / 'OFFLINE-INITIAL-RATIO-PREPARED-R2.json').read_text())
    assert admission == recorded, 'Prepared fixtures differ from the recorded R2 admission'
    assert admission['completed'] and admission['runtime'] == RUNTIME
    root = Path(admission['root'])
    production = root / task / 'production-r1'
    assert json.loads((production / 'clock.json').read_text())['exit_code'] == 0
    assert json.loads((production / 'progress.json').read_text())['completed_iterations'] == 130
    fixture = next(r for r in admission['tasks'] if r['task'] == task)
    assert sha(fixture['fixture']) == fixture['fixture_sha256']
    assert sha(fixture['config']) == fixture['config_sha256']
    assert sha(fixture['manifest']) == fixture['manifest_sha256']
    data = torch.load(fixture['fixture'], map_location='cpu', weights_only=False)
    assert {k: tensor_sha(v) for k, v in data['forward_inputs'].items()} == fixture['input_tensor_sha256']
    assert tensor_sha(data['old_logprobs']) == fixture['old_logprobs_sha256']
    assert data['old_logprobs'].shape == (16, 50, 12)
    previous = {r['checkpoint']['step']: r['mean_chunk_log_ratio_to_initial']
                for r in json.loads((reference / task / 'RESULT.json').read_text())['rows']}
    reference_value = json.loads((reference / task / 'RESULT.json').read_text())
    assert reference_value['completed'] and reference_value['runtime'] == RUNTIME
    assert reference_value['fixture'] == fixture
    catalog = base.checkpoint_catalog(root, task)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'deltas').mkdir()
    cfg = OmegaConf.create(json.loads(Path(fixture['config']).read_text()))
    assert cfg.actor.model.openpi.noise_method == 'flow_sde'
    assert float(cfg.actor.model.openpi.noise_level) == 0.3
    inputs = data['forward_inputs']
    cohort = dict(denoise_inds=inputs['denoise_inds'].tolist(),
                  action=describe(inputs['action']), model_action=describe(inputs['model_action']))
    model = get_model(cfg.actor.model, torch_dtype=torch.bfloat16).to('cuda:0')
    selected = {k: v.to('cuda:0') for k, v in inputs.items()}
    initial, rows = None, []
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
        assert Path(ready['result_root']).resolve() == output.parent.resolve()
        path = Path(ready['path'])
        assert path == Path(row['path']) or (path.parent == ready_path.parent and
                                            path.name == 'extracted-full_weights.pt')
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
        torch.save(delta, output / 'deltas' / f"step-{row['step']:03d}.pt")
        parts = decompose(delta)
        # Same weights/inputs as R3; GPU reductions are not bitwise deterministic.
        old = previous[row['step']]
        parts['reproduces_r3'] = abs(parts['mean_chunk_log_ratio_to_initial'] - old) <= 1e-2 * max(1.0, abs(old))
        assert parts['reproduces_r3'], f'R3 likelihood mismatch at {task} step {row["step"]}'
        rows.append(dict(step=row['step'], model_sha256=row['sha256'], r3_mean_chunk=old,
                         seconds=time.monotonic() - tick, **parts))
        value = dict(completed=False, task=task, runtime=RUNTIME, fixture=fixture['fixture'],
                     initial_parity=parity, binarized_dimensions=BINARIZED, cohort=cohort,
                     metric='Signed conditional-transition log ratios on one initial-policy B16 saved-chain cohort; not full-action marginal KL or evidence of unchanged executed actions.',
                     rows=rows, new_controls=0, SGD=0, model_forward_calls=len(rows))
        write(output / 'RESULT.json', value)
        print(json.dumps(dict(step=row['step'], mean=parts['mean_chunk_log_ratio_to_initial'],
              binarized_share=parts['binarized_dimensions_share'],
              reproduces_r3=parts['reproduces_r3'])), flush=True)
        del result
    value['completed'] = True
    write(output / 'RESULT.json', value)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='mode', required=True)
    p = sub.add_parser('metrics')
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('stage')
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--result-root', type=Path, required=True)
    p = sub.add_parser('run')
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--task', choices=TASKS, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--staging', type=Path, required=True)
    args = parser.parse_args()
    if args.mode == 'metrics':
        metrics(args.root, args.output)
    elif args.mode == 'stage':
        stage(args.root, args.output, args.result_root)
    else:
        run(args.prepared, args.reference, args.task, args.output, args.staging)


if __name__ == '__main__':
    main()
