"""Read the preregistered final native teacher gates; no model/simulator calls."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

TASKS = ['OpenDrawer', 'CloseToasterOvenDoor', 'PickPlaceDrawerToCounter',
         'CoffeeSetupMug', 'TurnOffStove']


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(16 << 20), b''):
            digest.update(data)
    return digest.hexdigest()


def read(path, evidence):
    data = Path(path).read_bytes()
    evidence[str(path)] = hashlib.sha256(data).hexdigest()
    return json.loads(data)


def episode_rows(paths, phase, iteration, per_rank, evidence):
    assert len(paths) == 8
    by_rank = {}
    for path in paths:
        value = read(path, evidence)
        assert value['completed'] and value['phase'] == phase and value['iteration'] == iteration
        assert value['rank'] not in by_rank and len(value['rows']) == per_rank
        for row in value['rows']:
            counts = row['native_counts']
            assert counts['controls'] - counts['reset_controls'] == 600
            assert isinstance(row['success_once'], bool) and isinstance(row['success_at_end'], bool)
        by_rank[value['rank']] = value
    assert set(by_rank) == set(range(8))
    return by_rank


def rate(rows, key):
    successes = sum(int(row[key]) for row in rows)
    return dict(successes=successes, episodes=len(rows), rate=successes / len(rows))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--task', choices=TASKS, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    assert args.root.resolve() == Path('/mowenjie/fluffy-bassoon-2/outputs/robocasa-flow-sde-10m-per-task-20261103-r1')
    assert args.output.resolve().is_relative_to(args.root.resolve())
    base = args.root / args.task
    production = base / 'production-r1'
    q1 = base / ('q1-r2' if args.task == 'OpenDrawer' else 'q1-r1')
    evidence = {}
    for run in [q1, production]:
        manifest = read(run / 'manifest.json', evidence)
        assert manifest['source'] == '0988b5f9e9bf31ac3db673cc2860959611b9ae12'
        assert manifest['allocated_gpus'] == 8
        assert manifest['batch_layout'] == dict(world=8, train_envs_per_rank=16,
                                               eval_envs_per_rank=8, actor_ranks=2)
    clock = read(production / 'clock.json', evidence)
    progress = read(production / 'progress.json', evidence)
    after_metrics = read(production / 'after-evaluation.json', evidence)
    assert clock['exit_code'] == 0 and progress['completed_iterations'] == 130
    assert not progress['operator_checkpoint_exit']
    assert after_metrics['completed'] and after_metrics['global_step'] == 130
    contract = read(production / 'flow-contract.json', evidence)
    assert contract['horizon'] == 600 and contract['iterations'] == 130
    all_ids, iterations = set(), []
    maximum_chunk_gap = maximum_coordinate_gap = 0.
    for collection in range(1, 131):
        run = q1 if collection == 1 else production
        groups = episode_rows(sorted((run / 'environment').glob(
            f'*/training-i{collection-1:03d}-*-episodes.json')), 'training', collection-1, 16, evidence)
        rows = [row for rank in range(8) for row in groups[rank]['rows']]
        expected_seeds = set(range(contract['train_seed']+(collection-1)*128,
                                  contract['train_seed']+collection*128))
        assert {row['seed'] for row in rows} == expected_seeds
        for row in rows:
            assert row['episode_id'] == f"{args.task}/{row['seed']}" and row['episode_id'] not in all_ids
            all_ids.add(row['episode_id'])
        for rank in range(2):
            update = read(run / 'updates' / f'iteration-{collection:03d}-rank-{rank:02d}.json', evidence)
            parity = read(run / 'logprob-parity' / f'iteration-{collection:03d}-rank-{rank:02d}.json', evidence)
            assert update['completed'] and update['optimizer_steps_after'] == collection*24
            assert update['optimizer_steps_before'] == (collection-1)*24
            assert update['frozen_vlm_unchanged'] and all(math.isfinite(v) for v in update['losses'].values())
            assert parity['passed'] and parity['finite'] and parity['before_backward']
            assert parity['optimizer_steps'] == (collection-1)*24
            maximum_chunk_gap = max(maximum_chunk_gap, parity['max_abs_chunk_log_ratio'])
            maximum_coordinate_gap = max(maximum_coordinate_gap, parity['max_abs_coordinate_log_ratio'])
        iterations.append(dict(collection=collection, first_success=rate(rows, 'success_once'),
                               success_at_end=rate(rows, 'success_at_end')))
    first = sum(row['first_success']['successes'] for row in iterations[:10])
    last = sum(row['first_success']['successes'] for row in iterations[-10:])
    train_pass = last-first >= 128
    before_ref = read(base / 'BEFORE-RESUME.json', evidence)
    assert before_ref['completed'] and before_ref['cases'] == 64
    for item in before_ref['episode_refs']:
        assert sha(item['path']) == item['sha256']
    before_groups = episode_rows([Path(row['path']) for row in before_ref['episode_refs']],
        'before', 0, 8, evidence)
    after_groups = episode_rows(sorted((production / 'environment').glob(
        '*/after-i130-*-episodes.json')), 'after', 130, 8, evidence)
    before, after, pairing = {}, {}, {}
    for rank in range(8):
        a, b = before_groups[rank], after_groups[rank]
        assert [row['episode_id'] for row in a['rows']] == [row['episode_id'] for row in b['rows']]
        for group in [a, b]:
            assert [row['seed'] for row in group['rows']] == list(range(
                contract['eval_seed']+rank*8, contract['eval_seed']+(rank+1)*8))
            assert all(row['episode_id'] == f"{args.task}/{row['seed']}" for row in group['rows'])
            assert sha(group['initial']['path']) == group['initial']['sha256']
        old_pair = read(Path(a['initial']['path']).with_suffix('.json'), evidence)
        new_pair = read(Path(b['initial']['path']).with_suffix('.json'), evidence)
        assert sha(old_pair['canonical']) == sha(new_pair['canonical'])
        marker = read(new_pair['canonical'], evidence)
        assert sha(Path(new_pair['canonical']).parent / 'initial.pt') == marker['snapshot_sha256']
        report = new_pair['report']
        assert report['matched'] and report['physical_state_exact'] and report['public_state_exact']
        assert not report['observations_modified']
        pairing[rank] = report
        before.update({row['episode_id']: row for row in a['rows']})
        after.update({row['episode_id']: row for row in b['rows']})
    assert len(before) == len(after) == 64 and set(before) == set(after)
    assert not set(before) & all_ids
    wins = sum(not before[key]['success_once'] and after[key]['success_once'] for key in before)
    losses = sum(before[key]['success_once'] and not after[key]['success_once'] for key in before)
    discordant = wins + losses
    p = min(1., 2*sum(math.comb(discordant, k) for k in range(min(wins, losses)+1))/2**discordant)
    eval_pass = wins > losses and p < .05
    before_rate, after_rate = rate(list(before.values()), 'success_once'), rate(list(after.values()), 'success_once')
    assert abs(before_rate['rate']-before_ref['metrics']['success_once']) < 1e-6
    assert abs(after_rate['rate']-after_metrics['metrics']['success_once']) < 1e-6
    cp = production / args.task / 'checkpoints/global_step_130'
    boundary = read(cp / 'pilot-boundary.json', evidence)
    assert boundary['completed'] and boundary['global_step'] == 130 and len(boundary['workers']) == 18
    assert not (cp / 'COLD-STORAGE.json').exists()
    for row in boundary['workers'] + boundary['actor_checkpoint_files']:
        assert Path(row['path']).resolve().is_relative_to(cp.resolve()) and sha(row['path']) == row['sha256']
    result = dict(completed=True, task=args.task, checked_unix=time.time(), source='0988b5f9',
        training=dict(iterations=130, episodes=16640, controls=9984000, optimizer_steps=3120,
            first10_successes=first, last10_successes=last, window_episodes=1280,
            improvement_percentage_points=(last-first)/1280*100, required_percentage_points=10, passed=train_pass),
        paired_evaluation=dict(cases=64, controls_per_panel=38400, before=before_rate, after=after_rate,
            before_success_at_end=rate(list(before.values()), 'success_at_end'),
            after_success_at_end=rate(list(after.values()), 'success_at_end'),
            improved_cases=wins, regressed_cases=losses, exact_two_sided_mcnemar_p=p, passed=eval_pass),
        admitted_teacher=train_pass and eval_pass, corpus_role='admitted_teacher' if train_pass and eval_pass else 'reference_source',
        physical_pairing=pairing, render_differences='Recorded; exact pixels and bitwise kernels are not required',
        maximum_initial_chunk_likelihood_gap=maximum_chunk_gap,
        maximum_initial_coordinate_likelihood_gap=maximum_coordinate_gap,
        frozen_vlm_unchanged_all_updates=True, finite_losses_all_updates=True,
        checkpoint=dict(path=str(cp), boundary_sha256=evidence[str(cp/'pilot-boundary.json')], all_worker_and_model_hashes_verified=True),
        iteration_rates=iterations, evidence_sha256=evidence,
        remaining=['Complete all five full corpus joins, raw hashes, archives and accounting before any student fitting',
                   'Offline fixed-bank source-to-initial log-ratio reporting as specified in PLAN.md'])
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ['task','training','paired_evaluation','admitted_teacher','corpus_role']}), flush=True)


if __name__ == '__main__':
    main()
