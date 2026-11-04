"""Native fixed-B8 DEV selection and independent TEST; no training.

`stage` is CPU-only on the development endpoint with the BOS mount. `run`
requires exactly one scheduler GPU and the unchanged qualified source runtime.
"""
import argparse
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import subprocess
import time

RUN = 'robocasa-flow-sde-checkpoint-selection-20261104-r1'
SOURCE_RUN = 'robocasa-flow-sde-10m-per-task-20261103-r1'
TASKS = ['OpenDrawer', 'CloseToasterOvenDoor', 'PickPlaceDrawerToCounter',
         'CoffeeSetupMug', 'TurnOffStove']
STEPS = [0, 1, 2] + list(range(10, 131, 10))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(16 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def bind_matrix(path, task):
    matrix = read(path)
    assert matrix['completed'] and matrix['maximum_episodes'] == 7040
    assert matrix['source_runtime'] == '0988b5f9e9bf31ac3db673cc2860959611b9ae12'
    row = next(r for r in matrix['tasks'] if r['task'] == task)
    assert sorted(r['step'] for r in row['checkpoints']) == STEPS
    assert sorted(row['DEV']['checkpoint_order']) == STEPS
    assert row['DEV']['cases'] == 64 and row['TEST']['cases'] == 128
    assert len(set(row['DEV']['seeds'])) == 64 and len(set(row['TEST']['seeds'])) == 128
    assert not set(row['DEV']['seeds']) & set(row['TEST']['seeds'])
    assert sha(row['source_config']) == row['source_config_sha256']
    assert sha(row['initial_parity_fixture']) == row['initial_parity_fixture_sha256']
    for cp in row['checkpoints']:
        assert sha(cp['marker']) == cp['marker_sha256']
    return matrix, row


def wait_file(path, deadline):
    path = Path(path)
    while not path.is_file():
        if time.time() > deadline:
            raise TimeoutError(f'No verified staging/selection receipt before deadline: {path}')
        time.sleep(3)
    return read(path)


def stage(matrix_path, task, directory, output, deadline):
    """One prefetched DEV model plus policy zero; retain TEST models for reloads."""
    _, row = bind_matrix(matrix_path, task)
    assert directory.name.startswith('staging-') and output.parent.name == task
    assert directory.parent == output.parent
    assert not directory.exists()
    directory.mkdir(parents=True)
    source_script = Path(__file__).resolve().parents[1] / SOURCE_RUN / 'OFFLINE-INITIAL-RATIO.py'
    spec = importlib.util.spec_from_file_location('source_checkpoint_readout', source_script)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    private, ready_rows = [], {}

    def prepare(cp, mode):
        dest = directory / mode / str(cp['step'])
        dest.mkdir(parents=True, exist_ok=False)
        path, temporary, storage = helper.obtain_weights(cp, dest)
        assert sha(path) == cp['sha256']
        value = dict(completed=True, task=task, checkpoint=cp, path=str(path),
                     private_copy=temporary, storage=storage, output=str(output),
                     matrix_sha256=sha(matrix_path), staging_script_sha256=sha(__file__))
        write(dest / 'READY.json', value)
        if temporary:
            private.append(path)
        ready_rows[(mode, cp['step'])] = (dest / 'READY.json', value)
        return dest / 'READY.json', value

    def consumed(cp, receipt_name):
        receipt = wait_file(output / receipt_name, deadline)
        assert receipt['step'] == cp['step'] and receipt['model_sha256'] == cp['sha256']
        assert receipt['strict_load'] and receipt['matrix_sha256'] == sha(matrix_path)
        return receipt

    cp0 = next(cp for cp in row['checkpoints'] if cp['step'] == 0)
    zero_ready, zero = prepare(cp0, 'dev')
    consumed(cp0, 'MODEL-LOADED-preflight-000.json')
    for step in row['DEV']['checkpoint_order']:
        cp = next(cp for cp in row['checkpoints'] if cp['step'] == step)
        ready_path, value = (zero_ready, zero) if step == 0 else prepare(cp, 'dev')
        receipt = consumed(cp, f'MODEL-LOADED-dev-{step:03d}.json')
        assert receipt['ready_sha256'] == sha(ready_path)
        # The GPU has strictly copied weights into its model. Only this run's
        # verified private extracted copy may be removed; source paths stay.
        if step != 0 and value['private_copy']:
            path = Path(value['path'])
            assert path.parent == ready_path.parent and path.name == 'extracted-full_weights.pt'
            assert sha(path) == cp['sha256']
            path.unlink()
            private.remove(path)
        write(directory / 'PROGRESS.json', dict(completed=False, latest_DEV_loaded=step,
              source_files_changed=False, private_models_retained=[str(p) for p in private]))
    lock_path = output / 'SELECTION-LOCK.json'
    lock = wait_file(lock_path, deadline)
    assert lock['completed'] and lock['task'] == task and lock['matrix_sha256'] == sha(matrix_path)
    for step in lock['test_steps']:
        cp = next(cp for cp in row['checkpoints'] if cp['step'] == step)
        if step == 0:
            target = directory / 'test/0/READY.json'
            write(target, dict(zero, selection_sha256=sha(lock_path)))
        else:
            ready_path, value = prepare(cp, 'test')
            write(ready_path, dict(value, selection_sha256=sha(lock_path)))
    result = wait_file(output / 'RESULT.json', deadline)
    assert result['completed'] and result['task'] == task
    assert result['selection_sha256'] == sha(lock_path)
    for path in private:
        assert path.name == 'extracted-full_weights.pt' and path.resolve().is_relative_to(directory.resolve())
        cp_sha = next(v['checkpoint']['sha256'] for _, v in ready_rows.values() if v['path'] == str(path))
        assert sha(path) == cp_sha
        path.unlink()
    write(directory / 'RESULT.json', dict(completed=True, task=task,
          gpu_result_sha256=sha(output / 'RESULT.json'), private_copies_reclaimed=True,
          source_files_changed=False, SGD=0, new_controls=0))


def paired(initial, policy, seed):
    """Exact paired binary test and descriptive paired case-bootstrap interval."""
    import math
    import numpy as np
    assert set(initial) == set(policy) and len(initial) == 128
    diffs = np.array([int(policy[k]) - int(initial[k]) for k in sorted(initial)])
    wins, losses = int((diffs == 1).sum()), int((diffs == -1).sum())
    n = wins + losses
    p = min(1., 2 * sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / 2**n) if n else 1.
    counts = np.array([(diffs == x).sum() for x in [-1, 0, 1]])
    boot = np.random.default_rng(seed).multinomial(128, counts / 128, size=10000)
    interval = np.quantile((boot[:, 2] - boot[:, 0]) / 128, [.025, .975]) * 100
    return dict(cases=128, initial_successes=sum(initial.values()), policy_successes=sum(policy.values()),
                delta_percentage_points=float(diffs.mean() * 100), wins=wins, losses=losses,
                exact_McNemar_p=p, paired_bootstrap95_percentage_points=interval.tolist(),
                bootstrap_seed=seed, bootstrap_draws=10000, intervals_simultaneous=False)


def run(matrix_path, task, staging, output, deadline):
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from native_eval import configuration, SelectionEnvironment, SelectionRollout
    from rlinf.runners.embodied_eval_runner import EmbodiedEvalRunner
    from rlinf.scheduler import Cluster
    from rlinf.utils.placement import HybridComponentPlacement
    assert torch.cuda.device_count() == 1
    matrix, row = bind_matrix(matrix_path, task)
    approval = read(matrix_path.parent / 'BUDGET-APPROVAL.json')
    assert approval['approved'] and approval['total_gpu_hours'] == 22
    assert approval['sweep_manifest_sha256'] == sha(matrix_path)
    cfg = configuration(dict(task=task, config=row['source_config'],
         config_sha256=row['source_config_sha256']), output, deadline, maximum_actions=1000000)
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

    def load(step, mode, label):
        nonlocal loaded
        cp = next(cp for cp in row['checkpoints'] if cp['step'] == step)
        ready_path = staging / mode / str(step) / 'READY.json'
        ready = wait_file(ready_path, deadline - 300)
        assert ready['completed'] and ready['task'] == task and ready['checkpoint'] == cp
        assert ready['output'] == str(output) and ready['matrix_sha256'] == sha(matrix_path)
        assert ready['staging_script_sha256'] == sha(__file__)
        if mode == 'test':
            assert ready['selection_sha256'] == sha(output / 'SELECTION-LOCK.json')
        if loaded != cp['sha256']:
            records = rollout.select_checkpoint(ready['path'], cp['sha256'], step, label, 1).wait()
            assert len(records) == 1 and records[0]['strict_load']
            loaded = cp['sha256']
        receipt = dict(step=step, model_sha256=cp['sha256'], strict_load=True,
             ready_sha256=sha(ready_path), matrix_sha256=sha(matrix_path))
        write(output / f'MODEL-LOADED-{label}.json', receipt)
        return cp

    # Initial likelihood qualification is mandatory regardless of randomized
    # DEV order and does not consume a validation or test case.
    load(0, 'dev', 'preflight-000')
    parity = rollout.initial_likelihood_parity(row['initial_parity_fixture'],
                                          row['initial_parity_fixture_sha256']).wait()
    assert len(parity) == 1 and parity[0]['passed']
    write(output / 'initial-parity.json', parity)
    panels = []

    def bank(panel, step, ordinal):
        if time.time() > deadline - 300:
            raise TimeoutError('Preserve completed bank receipts before scheduler deadline')
        fs = os.statvfs(output)
        assert fs.f_bavail * fs.f_frsize > 200 << 30, 'PFS floor reached; stop before another bank'
        seeds = row[panel]['seeds'][ordinal * 8:(ordinal + 1) * 8]
        assert len(seeds) == 8 and seeds == list(range(seeds[0], seeds[0]+8))
        phase = f'{panel.lower()}-c{step:03d}-b{ordinal:02d}'
        bank_output = output / 'banks' / panel.lower() / f'bank-{ordinal:02d}'
        rng_seed = (13000000 if panel == 'DEV' else 14000000) + row['ordinal'] * 10000 + ordinal
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
        assert all(r['native_counts']['controls']-r['native_counts']['reset_controls'] == 600
                   for r in episodes['rows'])
        record = dict(panel=panel, step=step, bank=ordinal, seeds=seeds, rng_seed=rng_seed,
            seconds=time.monotonic()-started, first_success=sum(r['success_once'] for r in episodes['rows']),
            success_at_end=sum(r['success_at_end'] for r in episodes['rows']),
            episodes=episodes['rows'], episode_path=str(episode_path), episode_sha256=sha(episode_path),
            initial_report_sha256=sha(initial_path), images_exact=report['images_exact'],
            sampler_sha256={p.name: sha(p) for p in sampler_paths},
            metrics={k: np.asarray(v).tolist() for k,v in metrics.items()})
        write(output / 'panels' / f'{phase}.json', record)
        panels.append(record)
        write(output / 'PROGRESS.json', dict(completed=False, task=task, banks_completed=len(panels),
            episodes_completed=8*len(panels), policy_controls=4800*len(panels),
            latest=dict(panel=panel, step=step, bank=ordinal), SGD=0))
        print(json.dumps(dict(task=task, panel=panel, step=step, bank=ordinal,
              first_success=record['first_success'], banks_completed=len(panels))), flush=True)

    for step in row['DEV']['checkpoint_order']:
        load(step, 'dev', f'dev-{step:03d}')
        for ordinal in range(8):
            bank('DEV', step, ordinal)
    scores = {step: sum(r['first_success'] for r in panels if r['panel']=='DEV' and r['step']==step)
              for step in STEPS}
    winner = min(STEPS, key=lambda step: (-scores[step], step))
    test_steps = sorted(set([0, winner, 130]))
    selected = next(cp for cp in row['checkpoints'] if cp['step'] == winner)
    lock = dict(completed=True, task=task, selected_step=winner, selected_model_sha256=selected['sha256'],
        DEV_first_successes={str(k): v for k,v in scores.items()}, DEV_cases=64,
        tie_break='earliest checkpoint including zero', test_steps=test_steps,
        locked_unix=time.time(), matrix_sha256=sha(matrix_path),
        DEV_receipt_sha256={p.name: sha(p) for p in sorted((output/'panels').glob('dev-*.json'))})
    assert len(lock['DEV_receipt_sha256']) == 128
    write(output / 'SELECTION-LOCK.json', lock)
    permutations = list(itertools.permutations(test_steps))
    for ordinal in range(16):
        for step in permutations[ordinal % len(permutations)]:
            load(step, 'test', f'test-{step:03d}-b{ordinal:02d}')
            bank('TEST', step, ordinal)
    outcomes = {}
    for step in test_steps:
        outcomes[step] = {r['episode_id']: bool(r['success_once'])
            for p in panels if p['panel']=='TEST' and p['step']==step for r in p['episodes']}
        assert len(outcomes[step]) == 128
    released = env.release_environments().wait()
    runner.metric_logger.finish()
    journal = output.parent / 'accounting/rank-00' / (output.name + '.json')
    calls = read(journal)['calls']
    assert calls and all(r['resolved'] for r in calls)
    controls = sum(r['executed_or_conservative'] for r in calls)
    expected = 600 * (1024 + 128*len(test_steps))
    assert controls == expected
    result = dict(completed=True, task=task, selected_step=winner, DEV_first_successes=scores,
        TEST_first_successes={str(k):sum(v.values()) for k,v in outcomes.items()},
        selected_vs_initial=paired(outcomes[0],outcomes[winner],15000000+row['ordinal']),
        final_vs_initial=paired(outcomes[0],outcomes[130],16000000+row['ordinal']),
        selection_sha256=sha(output/'SELECTION-LOCK.json'), matrix_sha256=sha(matrix_path),
        bank_receipts=len(panels), valid_episodes=expected//600, policy_controls=expected,
        all_controls=controls, reset_controls=0, unresolved_intents=0, SGD=0,
        fixed_eight_lane_batches=True, common_sampler_rng=True, released=released,
        initial_parity=parity, rendering_variability_allowed=True,
        original_teacher_gates_unchanged=True,
        raw_receipt_sha256={p.name:sha(p) for p in sorted((output/'panels').glob('*.json'))},
        finished_unix=time.time())
    write(output/'RESULT.json',result)
    print(json.dumps(result),flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['stage', 'run'])
    parser.add_argument('--matrix', type=Path, required=True)
    parser.add_argument('--task', choices=TASKS, required=True)
    parser.add_argument('--staging', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--deadline', type=float, required=True)
    args = parser.parse_args()
    if args.mode == 'stage':
        stage(args.matrix,args.task,args.staging,args.output,args.deadline)
    else:
        run(args.matrix,args.task,args.staging,args.output,args.deadline)


if __name__ == '__main__':
    main()
