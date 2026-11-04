"""512-case paired endpoint test: policy 0 vs policy 130, two pre-declared tasks.

Reuses the checkpoint-selection run's qualified eval-only workers
(native_eval.SelectionRollout / SelectionEnvironment / configuration) and its
helpers, without modifying that run. No training, no checkpoint selection.

  prepare  any CPU: bind the two tasks' step-0/step-130 rows from the adopted
           sweep manifest and fix seeds, bank order and statistics.
  stage    dev CPU with the BOS mount: expose both models once, keep them until
           the GPU result exists, then remove only this run's private copies.
  run      one scheduler GPU per task.
"""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import time

HERE = Path(__file__).resolve()
RUN = HERE.parent.name
SWEEP_RUN = 'robocasa-flow-sde-checkpoint-selection-20261104-r1'
SOURCE_RUN = 'robocasa-flow-sde-10m-per-task-20261103-r1'
TASKS = ['CloseToasterOvenDoor', 'OpenDrawer']        # pre-declared primary pair
STEPS = (0, 130)
CASES, BANK = 512, 8
BANKS = CASES // BANK
CASE_BASE, RNG_BASE, BOOT_BASE = 11_000_000, 17_000_000, 18_000_000
ALPHA = 0.05


def _sweep():
    spec = importlib.util.spec_from_file_location('sweep_helpers', HERE.parents[1] / SWEEP_RUN / 'sweep.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


helpers = _sweep()
sha, read, write, wait_file = helpers.sha, helpers.read, helpers.write, helpers.wait_file


def order(bank):
    """ABBA: 0,130 | 130,0 | 0,130 | ... — balanced over time, one reload per bank."""
    return STEPS if bank % 2 == 0 else STEPS[::-1]


def seeds_for(ordinal):
    return [CASE_BASE + 10_000 * ordinal + i for i in range(CASES)]


def prepare(sweep_manifest, output):
    manifest = read(sweep_manifest)
    assert manifest['completed'] and manifest['budget_approved']
    rows = []
    for task in TASKS:
        row = next(r for r in manifest['tasks'] if r['task'] == task)
        cps = [cp for cp in row['checkpoints'] if cp['step'] in STEPS]
        assert sorted(cp['step'] for cp in cps) == list(STEPS)
        seeds = seeds_for(row['ordinal'])
        used = set(row['DEV']['seeds']) | set(row['TEST']['seeds']) | set(range(10_030_000, 10_030_008))
        assert not used & set(seeds) and len(set(seeds)) == CASES
        rows.append(dict(task=task, ordinal=row['ordinal'], source_config=row['source_config'],
            source_config_sha256=row['source_config_sha256'],
            initial_parity_fixture=row['initial_parity_fixture'],
            initial_parity_fixture_sha256=row['initial_parity_fixture_sha256'],
            checkpoints=cps, seeds=seeds,
            bank_order=[list(order(b)) for b in range(BANKS)],
            rng_seeds=[RNG_BASE + 10_000 * row['ordinal'] + b for b in range(BANKS)],
            bootstrap_seed=BOOT_BASE + row['ordinal']))
    value = dict(completed=True, run=RUN, source_runtime=manifest['source_runtime'],
        sweep_manifest=f'{SWEEP_RUN}/{Path(sweep_manifest).name}', sweep_manifest_sha256=sha(sweep_manifest),
        tasks=rows, cases=CASES, bank_size=BANK,
        statistics=dict(
            primary='update130 versus policy 0, first success in 600 controls, paired by case',
            test='exact two-sided McNemar on discordant cases; reject only with wins > losses',
            multiplicity=f'Holm across the {len(TASKS)} pre-declared tasks at family alpha {ALPHA}',
            interval='paired case bootstrap 95%, descriptive, not simultaneous',
            declared_before_data=True,
            not_a_teacher_gate_change='Original 64-case gates stay as recorded; admission uses this per human decision'),
        maximum_episodes=len(TASKS) * CASES * len(STEPS),
        maximum_policy_controls=len(TASKS) * CASES * len(STEPS) * 600)
    write(output, value)
    return value


def bind(prepared, task):
    value = read(prepared)
    assert value['completed'] and value['run'] == RUN and value['cases'] == CASES
    row = next(r for r in value['tasks'] if r['task'] == task)
    assert sha(row['source_config']) == row['source_config_sha256']
    assert sha(row['initial_parity_fixture']) == row['initial_parity_fixture_sha256']
    for cp in row['checkpoints']:
        assert sha(cp['marker']) == cp['marker_sha256']
    return value, row


def stage(prepared, task, directory, output, deadline):
    _, row = bind(prepared, task)
    assert not directory.exists()
    directory.mkdir(parents=True)
    spec = importlib.util.spec_from_file_location(
        'source_checkpoint_readout', HERE.parents[1] / SOURCE_RUN / 'OFFLINE-INITIAL-RATIO.py')
    source = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(source)
    private = []
    for cp in row['checkpoints']:
        dest = directory / str(cp['step'])
        dest.mkdir()
        path, temporary, storage = source.obtain_weights(cp, dest)
        assert sha(path) == cp['sha256']
        write(dest / 'READY.json', dict(completed=True, task=task, checkpoint=cp, path=str(path),
              private_copy=temporary, storage=storage, output=str(output),
              prepared_sha256=sha(prepared), staging_script_sha256=sha(HERE)))
        if temporary:
            private.append((path, cp['sha256']))
    result = wait_file(output / 'RESULT.json', deadline)
    assert result['completed'] and result['task'] == task
    for path, expected in private:
        assert path.name == 'extracted-full_weights.pt' and path.resolve().is_relative_to(directory.resolve())
        assert sha(path) == expected
        path.unlink()
    write(directory / 'RESULT.json', dict(completed=True, task=task, private_copies_reclaimed=len(private),
          source_files_changed=False, SGD=0, new_controls=0))


def paired(initial, policy, seed):
    """Exact two-sided McNemar and descriptive paired case-bootstrap interval."""
    import numpy as np
    assert set(initial) == set(policy)
    n_cases = len(initial)
    diffs = np.array([int(policy[k]) - int(initial[k]) for k in sorted(initial)])
    wins, losses = int((diffs == 1).sum()), int((diffs == -1).sum())
    n = wins + losses
    p = min(1., 2 * sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / 2**n) if n else 1.
    counts = np.array([(diffs == x).sum() for x in [-1, 0, 1]])
    boot = np.random.default_rng(seed).multinomial(n_cases, counts / n_cases, size=10000)
    interval = np.quantile((boot[:, 2] - boot[:, 0]) / n_cases, [.025, .975]) * 100
    return dict(cases=n_cases, initial_successes=int(sum(initial.values())),
                policy_successes=int(sum(policy.values())),
                delta_percentage_points=float(diffs.mean() * 100), wins=wins, losses=losses,
                exact_McNemar_p=p, paired_bootstrap95_percentage_points=interval.tolist(),
                bootstrap_seed=seed, bootstrap_draws=10000, intervals_simultaneous=False)


def holm(results, alpha=ALPHA):
    """results: {task: paired(...)}. Step-down Holm; rejection also needs wins > losses."""
    ranked = sorted(results, key=lambda t: results[t]['exact_McNemar_p'])
    decisions, open_ = {}, True
    for i, task in enumerate(ranked):
        threshold = alpha / (len(ranked) - i)
        r = results[task]
        open_ = open_ and r['exact_McNemar_p'] < threshold      # step-down stops at first failure
        decisions[task] = dict(threshold=threshold, rejected_two_sided=bool(open_),
                               improvement=bool(open_ and r['wins'] > r['losses']))
    return decisions


def run(prepared, task, staging, output, deadline):
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from native_eval import configuration, SelectionEnvironment, SelectionRollout
    from rlinf.runners.embodied_eval_runner import EmbodiedEvalRunner
    from rlinf.scheduler import Cluster
    from rlinf.utils.placement import HybridComponentPlacement
    assert torch.cuda.device_count() == 1
    _, row = bind(prepared, task)
    cfg = configuration(dict(task=task, config=row['source_config'],
         config_sha256=row['source_config_sha256']), output, deadline,
         maximum_actions=CASES * len(STEPS) * 600 + 4800)
    write(output / 'resolved-config.json', OmegaConf.to_container(cfg, resolve=True))
    cluster = Cluster(cluster_cfg=cfg.cluster, distributed_log_dir=cfg.runner.per_worker_log_path)
    placement = HybridComponentPlacement(cfg, cluster)
    assert placement.get_world_size('env') == placement.get_world_size('rollout') == 1
    rollout = SelectionRollout.create_group(cfg).launch(cluster, name=cfg.rollout.group_name,
        placement_strategy=placement.get_strategy('rollout'))
    env = SelectionEnvironment.create_group(cfg).launch(cluster, name=cfg.env.group_name,
        placement_strategy=placement.get_strategy('env'))
    runner = EmbodiedEvalRunner(cfg, rollout, env)
    runner.init_workers()
    loaded = None

    def load(step, label):
        nonlocal loaded
        cp = next(cp for cp in row['checkpoints'] if cp['step'] == step)
        ready = wait_file(staging / str(step) / 'READY.json', deadline - 300)
        assert ready['completed'] and ready['task'] == task and ready['checkpoint'] == cp
        assert ready['output'] == str(output) and ready['prepared_sha256'] == sha(prepared)
        assert ready['staging_script_sha256'] == sha(HERE)
        if loaded != cp['sha256']:
            records = rollout.select_checkpoint(ready['path'], cp['sha256'], step, label, 1).wait()
            assert len(records) == 1 and records[0]['strict_load']
            loaded = cp['sha256']
        return cp

    load(0, 'preflight-000')
    parity = rollout.initial_likelihood_parity(row['initial_parity_fixture'],
                                               row['initial_parity_fixture_sha256']).wait()
    assert len(parity) == 1 and parity[0]['passed']
    write(output / 'initial-parity.json', parity)
    panels = []

    def bank(step, ordinal):
        if time.time() > deadline - 300:
            raise TimeoutError('Preserve completed bank receipts before scheduler deadline')
        fs = os.statvfs(output)
        assert fs.f_bavail * fs.f_frsize > 200 << 30, 'PFS floor reached; stop before another bank'
        seeds = row['seeds'][ordinal * BANK:(ordinal + 1) * BANK]
        assert seeds == list(range(seeds[0], seeds[0] + BANK))
        phase = f'test-c{step:03d}-b{ordinal:03d}'
        bank_output = output / 'banks' / f'bank-{ordinal:03d}'
        rng_seed = row['rng_seeds'][ordinal]
        rollout.select_sampling_scope(phase, rng_seed).wait()
        env.select_bank(phase, step, seeds[0], str(bank_output)).wait()
        started = time.monotonic()
        metrics = runner.evaluate()
        episode_path = bank_output / 'environment/rank-00' / f'{phase}-i{step:03d}-round00-episodes.json'
        episodes = read(episode_path)
        assert episodes['completed'] and [r['seed'] for r in episodes['rows']] == seeds
        initial_path = episode_path.with_name(episode_path.name.replace('-episodes.json', '-initial.json'))
        report = read(initial_path)['report']
        assert report['matched'] and report['physical_state_exact'] and report['public_state_exact']
        sampler_paths = sorted((output / 'sampler').glob(phase + '-*.json'))
        assert len(sampler_paths) == 12
        sampling = [read(p) for p in sampler_paths]
        assert all(r['batch_size'] == 8 and r['actions_shape'] == [8, 50, 12] for r in sampling)
        rng_reference = bank_output / 'SAMPLER-RNG-REFERENCE.json'
        rng_hashes = [dict(before=r['before_rng_sha256'], after=r['after_rng_sha256']) for r in sampling]
        if rng_reference.exists():
            assert read(rng_reference)['rng_hashes'] == rng_hashes, 'Common sampler RNG changed'
        else:
            write(rng_reference, dict(rng_seed=rng_seed, rng_hashes=rng_hashes))
        assert all(r['native_counts']['controls'] - r['native_counts']['reset_controls'] == 600
                   for r in episodes['rows'])
        record = dict(step=step, bank=ordinal, seeds=seeds, rng_seed=rng_seed,
            seconds=time.monotonic() - started,
            first_success=sum(r['success_once'] for r in episodes['rows']),
            success_at_end=sum(r['success_at_end'] for r in episodes['rows']),
            episodes=episodes['rows'], episode_path=str(episode_path), episode_sha256=sha(episode_path),
            initial_report_sha256=sha(initial_path), images_exact=report['images_exact'],
            sampler_sha256={p.name: sha(p) for p in sampler_paths},
            metrics={k: np.asarray(v).tolist() for k, v in metrics.items()})
        write(output / 'panels' / f'{phase}.json', record)
        panels.append(record)
        write(output / 'PROGRESS.json', dict(completed=False, task=task, banks_completed=len(panels),
            episodes_completed=BANK * len(panels), policy_controls=4800 * len(panels),
            latest=dict(step=step, bank=ordinal), SGD=0))
        print(json.dumps(dict(task=task, step=step, bank=ordinal,
              first_success=record['first_success'], banks_completed=len(panels))), flush=True)

    for ordinal in range(BANKS):
        for step in row['bank_order'][ordinal]:
            load(step, f'test-{step:03d}-b{ordinal:03d}')
            bank(step, ordinal)
    outcomes = {}
    for step in STEPS:
        # Pair by case seed, not episode id: ids carry the phase/policy.
        outcomes[step] = {r['seed']: bool(r['success_once'])
            for p in panels if p['step'] == step for r in p['episodes']}
        assert len(outcomes[step]) == CASES
    released = env.release_environments().wait()
    runner.metric_logger.finish()
    calls = read(output.parent / 'accounting/rank-00' / (output.name + '.json'))['calls']
    assert calls and all(r['resolved'] for r in calls)
    controls = sum(r['executed_or_conservative'] for r in calls)
    expected = 600 * CASES * len(STEPS)
    assert controls == expected
    result = dict(completed=True, task=task,
        first_successes={str(s): sum(v.values()) for s, v in outcomes.items()},
        final_vs_initial=paired(outcomes[0], outcomes[130], row['bootstrap_seed']),
        prepared_sha256=sha(prepared), bank_receipts=len(panels), valid_episodes=expected // 600,
        policy_controls=expected, all_controls=controls, unresolved_intents=0, SGD=0,
        fixed_eight_lane_batches=True, common_sampler_rng=True, abba_order=True, released=released,
        initial_parity=parity, rendering_variability_allowed=True,
        raw_receipt_sha256={p.name: sha(p) for p in sorted((output / 'panels').glob('*.json'))},
        finished_unix=time.time())
    write(output / 'RESULT.json', result)
    print(json.dumps(result), flush=True)


def combine(prepared, outputs, destination):
    """CPU: Holm across both pre-declared tasks once every task RESULT exists."""
    value = read(prepared)
    results = {}
    for task in TASKS:
        r = read(Path(outputs) / task / 'gpu-r1' / 'RESULT.json')
        assert r['completed'] and r['prepared_sha256'] == sha(prepared)
        results[task] = r['final_vs_initial']
    write(destination, dict(completed=True, tasks=results, holm=holm(results),
          family_alpha=ALPHA, statistics=value['statistics']))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='mode', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--sweep-manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    for name in ('stage', 'run'):
        p = sub.add_parser(name)
        p.add_argument('--prepared', type=Path, required=True)
        p.add_argument('--task', choices=TASKS, required=True)
        p.add_argument('--staging', type=Path, required=True)
        p.add_argument('--output', type=Path, required=True)
        p.add_argument('--deadline', type=float, required=True)
    p = sub.add_parser('combine')
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--outputs', type=Path, required=True)
    p.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare(args.sweep_manifest, args.output)
    elif args.mode == 'stage':
        stage(args.prepared, args.task, args.staging, args.output, args.deadline)
    elif args.mode == 'run':
        run(args.prepared, args.task, args.staging, args.output, args.deadline)
    else:
        combine(args.prepared, args.outputs, args.destination)


if __name__ == '__main__':
    main()
