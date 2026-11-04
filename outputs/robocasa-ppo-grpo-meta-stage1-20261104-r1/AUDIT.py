"""Read-only Stage 1 artifact audit; run on allocated Linux CPU, never a test suite."""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import time


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def audit(root, *, snapshots=False, previous=None):
    refs = []

    def read(path):
        data = json.loads(path.read_text())
        refs.append(dict(path=str(path), sha256=digest(path)))
        return data

    contract = read(root / 'flow-contract.json')
    manifest = read(root / 'manifest.json')
    progress = read(root / 'progress.json') if (root / 'progress.json').exists() else {}
    probe = read(root / 'gpu-probe.json')
    assert probe['status'] == 'passed' and len(probe['devices']) == 8
    assert contract['iterations'] == 40 and contract['update_epochs'] == 1
    assert contract['optimizer_steps_per_update'] == 6 and contract['eval_panels'] == 5
    sources = [root] if previous is None else [previous, root]
    resume_step = None
    if previous is not None:
        cfg = read(root / 'resolved-config.json')
        source_boundary = Path(cfg['runner']['resume_dir'])
        assert source_boundary.parent.parent.parent == previous
        source_record = read(source_boundary / 'pilot-boundary.json')
        resume_step = source_record['global_step']
        reference = Path(cfg['native_pilot']['before_reference'])
        before = read(reference)
        assert digest(reference) == cfg['native_pilot']['before_reference_sha256']
        assert before['pending_evaluation_step'] == resume_step == 10
        assert before['checkpoint_boundary_sha256'] == digest(source_boundary / 'pilot-boundary.json')
        prior_contract = read(previous / 'flow-contract.json')
        assert all(contract[k] == v for k, v in prior_contract.items())
        prior_probe = read(previous / 'gpu-probe.json')
        assert prior_probe['status'] == 'passed' and len(prior_probe['devices']) == 8
        replay = read(root / 'evaluations' / f'step-{resume_step:03}.json')
        assert replay['completed'] and replay['global_step'] == resume_step
        assert read(previous / 'clock.json')['exit_code'] != 0

    def paths(pattern):
        for src in sources:
            for path in sorted(src.glob(pattern)):
                # The failed, partial DEV10 was charged but replayed in full.
                if previous is not None and src == previous and path.name.startswith(f'eval_step_{resume_step:03}-'):
                    continue
                yield path

    updates = collections.defaultdict(dict)
    for path in paths('updates/*.json'):
        row = read(path)
        iteration = row['producing_policy_iteration'] + 1
        if previous is not None:
            assert (iteration <= resume_step) == path.is_relative_to(previous)
        assert row['completed'] and row['frozen_vlm_unchanged']
        assert row['optimizer_steps_after'] - row['optimizer_steps_before'] == 6
        assert all(math.isfinite(v) for v in row['losses'].values())
        assert all(math.isfinite(v) for v in row['metrics'].values())
        assert row['rank'] not in updates[iteration]
        updates[iteration][row['rank']] = row
    parity = []
    for path in paths('logprob-parity/*.json'):
        row = read(path)
        assert row['passed'] and row['before_backward']
        if contract['add_value_head']:
            assert row['value_head_enabled'] and row['value']['passed']
        parity.append(dict(path=str(path), **row))
    assert len({Path(row['path']).name for row in parity}) == len(parity)
    episodes = collections.defaultdict(dict)
    for path in paths('environment/rank-*/*-episodes.json'):
        row = read(path)
        assert row['completed']
        key = (row['phase'], row['iteration'])
        for episode in row['rows']:
            episode_id = episode['episode_id']
            assert episode_id not in episodes[key], (key, episode_id)
            counts = episode['native_counts']
            assert counts['controls'] - counts['reset_controls'] == 600
            episodes[key][episode_id] = episode
    panels = []
    baseline = episodes.get(('before', 0), {})
    train_seeds = set()
    group_outcomes = collections.Counter()
    for (phase, step), rows in sorted(episodes.items(), key=lambda item: (item[0][1], item[0][0])):
        expected = 128 if phase == 'training' else 64
        if len(rows) != expected:
            continue  # A live panel can have only some worker receipts published.
        seeds = collections.Counter(row['seed'] for row in rows.values())
        if phase == 'training':
            assert len(seeds) == contract['scenes_per_iteration']
            assert set(seeds.values()) == {contract['group_size']}
            assert not train_seeds.intersection(seeds)
            train_seeds.update(seeds)
            successful = collections.Counter(row['seed'] for row in rows.values() if row['success_once'])
            group_outcomes.update(successful[seed] for seed in seeds)
        panel = dict(phase=phase, policy_iteration=step, episodes=len(rows),
                     successes=sum(row['success_once'] for row in rows.values()),
                     end_successes=sum(row['success_at_end'] for row in rows.values()))
        panel['success_rate'] = panel['successes'] / panel['episodes']
        if phase != 'training' and phase != 'before':
            assert len(baseline) == 64 and set(rows) == set(baseline)
            wins = sum(row['success_once'] and not baseline[key]['success_once'] for key, row in rows.items())
            losses = sum(not row['success_once'] and baseline[key]['success_once'] for key, row in rows.items())
            n = wins + losses
            p = min(1.0, 2 * sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / 2**n) if n else 1.0
            panel.update(paired_wins=wins, paired_losses=losses,
                         exact_two_sided_McNemar_p=p,
                         gain_pp=100 * (panel['success_rate'] - sum(r['success_once'] for r in baseline.values()) / 64))
        panels.append(panel)
    physical = []
    for path in paths('environment/rank-*/*-initial.json'):
        row = read(path)
        report = row['report']
        assert report['matched'] and report['physical_state_exact'] and report['public_state_exact']
        physical.append(dict(path=str(path), report=report))
    group_snapshots = 0
    if snapshots and contract['group_size'] > 1:
        import torch
        from fluffy_bassoon.meta_rl.rlinf_robocasa_flow_ppo import verify_grpo_group_reset
        for path in paths('environment/rank-*/training-*-initial.pt'):
            snapshot = torch.load(path, map_location='cpu', weights_only=False)
            verify_grpo_group_reset(snapshot, contract['group_size'])
            refs.append(dict(path=str(path), sha256=digest(path)))
            group_snapshots += 1
    clock = read(root / 'clock.json') if (root / 'clock.json').exists() else None
    terminal = (root / 'after-evaluation.json').exists() and clock is not None and clock['exit_code'] == 0
    complete = terminal and progress.get('completed_iterations') == 40
    if complete:
        assert set(updates) == set(range(1, 41))
        assert all(set(rows) == {0, 1} for rows in updates.values())
        assert len(parity) == 80
        for i, ranks in updates.items():
            assert all(r['optimizer_steps_before'] == (i - 1) * 6 and
                       r['optimizer_steps_after'] == i * 6 for r in ranks.values())
        assert len([p for p in panels if p['phase'] == 'training']) == 40
        assert {p['policy_iteration'] for p in panels if p['phase'] != 'training'} == {0, 10, 20, 30, 40}
        assert len(physical) == 40  # Eight ranks at each of five DEV panels.
        if snapshots and contract['group_size'] > 1:
            assert group_snapshots == 320
    train = [p for p in panels if p['phase'] == 'training']
    dev = [p for p in panels if p['phase'] != 'training']
    first10 = sum(p['successes'] for p in train[:10]) / 1280 if len(train) >= 10 else None
    last10 = sum(p['successes'] for p in train[-10:]) / 1280 if len(train) == 40 else None
    kls = [r['metrics']['actor/approx_kl'] for ranks in updates.values() for r in ranks.values()]
    clips = [r['metrics']['actor/clip_fraction'] for ranks in updates.values() for r in ranks.values()]
    final_dev = next((p for p in dev if p['phase'] == 'after'), None)
    result = dict(completed=complete, checked_unix=time.time(), root=str(root),
                  source=manifest['source'], contract=contract, progress=progress,
                  completed_update_rank_pairs=sum(map(len, updates.values())),
                  parity_receipts=len(parity), max_approx_kl=max(kls, default=None),
                  max_clip_fraction=max(clips, default=None), TRAIN=train, DEV64=dev,
                  training_first10_rate=first10, training_last10_rate=last10,
                  training_gain_pp=100 * (last10 - first10) if last10 is not None else None,
                  final_DEV_gain_pp=final_dev['gain_pp'] if final_dev else None,
                  DEV_peak_successes=max((p['successes'] for p in dev), default=None),
                  physical_reset_receipts=len(physical), GRPO_raw_group_snapshots_rechecked=group_snapshots,
                  clock=clock, references=refs,
                  interpretation='Exploratory Stage 1 gate; full curves still require collapse/clipping review. No independent TEST or student admission is implied.')
    if previous is not None:
        result['previous_root'] = str(previous)
        result['group_counts_by_winning_members'] = dict(sorted(group_outcomes.items()))
        result['max_abs_chunk_log_ratio'] = max(r['max_abs_chunk_log_ratio'] for r in parity)
        result['KL_over_threshold'] = [
            dict(update=i, rank=rank, approx_kl=r['metrics']['actor/approx_kl'])
            for i, ranks in sorted(updates.items()) for rank,r in sorted(ranks.items())
            if r['metrics']['actor/approx_kl'] > 0.03]
        if complete:
            charged = collections.Counter()
            unresolved = 0
            ledgers = sorted(Path(cfg['native_pilot']['accounting_root']).glob('rank-*/grpo-attempt-*.json'))
            assert len(ledgers) == 16
            for path in ledgers:
                row = read(path)
                assert row['maximum_all_in_actions'] == contract['maximum_actions']
                for call in row['calls']:
                    charged[call['phase']] += call.get('executed_or_conservative', call['requested'])
                    unresolved += int(not call['resolved'])
            assert sum(charged.values()) <= contract['maximum_actions']
            assert charged['training'] == contract['TRAIN_policy_controls']
            result['action_ledger'] = dict(total=sum(charged.values()), by_phase=charged,
                                           unresolved_calls=unresolved)
            boundaries = []
            for step in (0, 1, 10, 20, 30, 40):
                parent = previous if step <= resume_step else root
                path = parent / f'OpenDrawer/checkpoints/global_step_{step}/pilot-boundary.json'
                boundary = read(path)
                assert boundary['completed'] and boundary['global_step'] == step
                assert len(boundary['workers']) == 18 and len(boundary['actor_checkpoint_files']) == 4
                if step == 40:
                    for item in boundary['workers'] + boundary['actor_checkpoint_files']:
                        assert digest(Path(item['path'])) == item['sha256']
                boundaries.append(dict(step=step, path=str(path), sha256=digest(path)))
            result['boundaries'] = boundaries
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--verify-group-snapshots', action='store_true')
    parser.add_argument('--previous-root', type=Path)
    args = parser.parse_args()
    report = audit(args.root, snapshots=args.verify_group_snapshots, previous=args.previous_root)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key not in {'references', 'TRAIN', 'DEV64'}}, indent=2))
