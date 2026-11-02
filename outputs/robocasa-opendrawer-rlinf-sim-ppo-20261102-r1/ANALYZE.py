"""Allocated-CPU closeout of the existing eighteen-update run; no model calls."""
import csv
import datetime
import json
import math
from pathlib import Path

ROOT = Path('/mowenjie/fluffy-bassoon-2/outputs')
OUT = ROOT / 'robocasa-opendrawer-rlinf-sim-ppo-20261102-r1'
NEW = ROOT / 'robocasa-opendrawer-flow-sde-20261102-r4'
OLD = ROOT / 'robocasa-opendrawer-flow-sde-20261102-r3'


def read(path):
    return json.loads(Path(path).read_text())


def checked_episode_rows(paths):
    rows = [row for path in paths for row in read(path)['rows']]
    assert all(row['native_counts']['controls'] == 400 and
               row['native_counts']['reset_controls'] == 0 for row in rows)
    return rows


def paired(before, after, key):
    a = {row['seed']: bool(row[key]) for row in before}
    b = {row['seed']: bool(row[key]) for row in after}
    assert len(a) == len(b) == 64 and set(a) == set(b)
    wins = sum(b[seed] and not a[seed] for seed in a)
    losses = sum(a[seed] and not b[seed] for seed in a)
    n = wins + losses
    p = min(1., 2 * sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / 2**n) if n else 1.
    return dict(before=sum(a.values()), after=sum(b.values()), cases=64,
                after_only=wins, before_only=losses, exact_McNemar_two_sided_p=p,
                gain_percentage_points=100 * (sum(b.values()) - sum(a.values())) / 64)


def main():
    clock = read(NEW / 'clock.json')
    assert clock['exit_code'] == 0 and read(NEW / 'after-evaluation.json')['global_step'] == 18
    reference = read(OUT / 'BEFORE-RECOVERY.json')
    before = checked_episode_rows([row['path'] for row in reference['episode_refs']])
    after = checked_episode_rows(sorted((NEW / 'environment').glob('*/after-*-episodes.json')))
    assert sorted(row['seed'] for row in before) == list(range(410100, 410164))
    assert sorted(row['seed'] for row in after) == list(range(410100, 410164))
    pairs = [read(path) for path in (NEW / 'environment').glob('*/after-*-initial.json')]
    assert len(pairs) == 8 and all(p['report']['matched'] and p['report']['physical_state_exact']
                                  and p['report']['public_state_exact'] for p in pairs)
    rows, archives = [], []
    for iteration in range(1, 19):
        source = OLD if iteration <= 6 else NEW
        episodes = checked_episode_rows(sorted((source / 'environment').glob(
            f'*/training-i{iteration-1:03d}-*-episodes.json')))
        assert len(episodes) == 128
        start = 210100 + (iteration - 1) * 128
        assert sorted(row['seed'] for row in episodes) == list(range(start, start + 128))
        workers = [read(source / f'updates/iteration-{iteration:03d}-rank-{rank:02d}.json')
                   for rank in range(2)]
        assert all(u['optimizer_steps_after'] == iteration * 16 and
                   u['optimizer_steps_after'] - u['optimizer_steps_before'] == 16 and
                   u['frozen_vlm_unchanged'] and all(math.isfinite(v) for v in u['losses'].values())
                   for u in workers)
        metrics = workers[0]['metrics']
        row = dict(iteration=iteration, episodes=128,
                   success_once=sum(e['success_once'] for e in episodes),
                   success_at_end=sum(e['success_at_end'] for e in episodes),
                   approx_kl=metrics['actor/approx_kl'],
                   clip_fraction=metrics['actor/clip_fraction'],
                   explained_variance=metrics['critic/explained_variance'],
                   value_loss=metrics['critic/value_loss'], grad_norm=metrics['actor/grad_norm'],
                   optimizer_steps=iteration * 16, frozen_vlm_unchanged=True)
        rows.append(row)
        for rank in range(2):
            archive = read(source / 'training-rollouts' /
                           f'iteration-{iteration:04d}-policy-{iteration-1:04d}-rank-{rank:02d}.json')
            assert archive['completed'] and Path(archive['path']).is_file()
            assert not archive['evaluation_data']
            archives.append(archive)
    first = sum(row['success_once'] for row in rows[:3])
    last = sum(row['success_once'] for row in rows[-3:])
    comparison = {key: paired(before, after, key) for key in ('success_once', 'success_at_end')}
    recorded = read(NEW / 'after-evaluation.json')['metrics']
    assert all(abs(recorded[key] - comparison[key]['after'] / 64) < 1e-8 for key in comparison)
    assert recorded['num_trajectories'] == 64
    charged, resolved, unresolved = 0, 0, 0
    journals = []
    for path in sorted((OUT / 'accounting').glob('rank-*/*.json')):
        data = read(path)
        done = sum(row.get('executed_or_conservative', 0) for row in data['calls'] if row['resolved'])
        pending = sum(row['requested'] for row in data['calls'] if not row['resolved'])
        resolved += done
        unresolved += pending
        journals.append(dict(path=str(path), resolved_controls=done, unresolved_controls_charged=pending))
    charged = resolved + unresolved
    assert charged <= 1_000_000
    train_pass = (last - first) / 384 >= .10
    eval_pass = (comparison['success_once']['after'] > comparison['success_once']['before'] and
                 comparison['success_once']['exact_McNemar_two_sided_p'] < .05)
    result = dict(completed=True,
        checked_Shanghai=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        cluster='Baige', job='job-h3d8va4slqg5', source='9ed1d19d42a346c6330c4d7f55303a48d21158ad',
        runtime='8a1233c3fae6f5905adfda7684c455842b0ecfb8', iterations=18, optimizer_steps=288,
        TRAIN_episodes=2304, TRAIN_H50_decisions=18432, before_cases=64, after_cases=64,
        TRAIN_first3_success=first, TRAIN_last3_success=last, TRAIN_three_bank_denominator=384,
        TRAIN_gain_percentage_points=100 * (last - first) / 384,
        TRAIN_ge10_point_gate_pass=train_pass, heldout_paired_gain_gate_pass=eval_pass,
        declared_joint_learning_gate_pass=train_pass and eval_pass,
        evaluation=comparison, per_iteration=rows, full_18_training_archives=archives,
        paired_initial_reports=pairs, VLM_frozen_all_updates=True,
        behavior_KL_proxy=dict(minimum=min(r['approx_kl'] for r in rows),
            maximum=max(r['approx_kl'] for r in rows), final=rows[-1]['approx_kl'],
            semantics='-masked_mean(new_logp - rollout_behavior_logp), averaged over native minibatches/epochs/ranks; not original-SFT drift',
            original_SFT_divergence_measured=False, cumulative_sum_is_not_KL=True,
            active_kl_beta=0, KL_early_stop=False, recipe_changed=False),
        controls=dict(resolved=resolved, unresolved_charged=unresolved, all_in_charged=charged,
            maximum=1_000_000, full_train_plus_before_after_nominal=972800,
            interrupted_old_prefix_charge=2400, journals=journals),
        new_job_process_clock=clock, prior_closed_attempts_conservative_GPU_seconds=5290,
        resource_limit_note='Final scheduler terminal/release and all-in GPU-time bound recorded independently in SCHEDULER-CLOSEOUT.json.',
        scientific_limit='One OpenDrawer simulator-reward action-expert learner; no Flora, new task, extra evaluation or extra source iteration.')
    (OUT / 'RESULT.json').write_text(json.dumps(result, indent=2) + '\n')
    with (OUT / 'TRAIN-DIAGNOSTICS.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    plot(result)
    print(json.dumps({key: value for key, value in result.items() if key not in
                      ('per_iteration', 'full_18_training_archives', 'paired_initial_reports')}, indent=2))


def plot(result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    rows = result['per_iteration']; x = [row['iteration'] for row in rows]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(x, [100 * row['success_once'] / 128 for row in rows], 'o-', color='#2468a2', label='First success')
    ax.plot(x, [100 * row['success_at_end'] / 128 for row in rows], '--', color='#7b8894', label='Success at end')
    baseline = result['TRAIN_first3_success'] / 384 * 100
    ax.axhline(baseline, color='#555555', lw=1, label='First 3 mean')
    ax.axhline(baseline + 10, color='#a85332', lw=1, ls=':', label='+10 point gate')
    ax.set(title='A  TRAIN success (128 fresh cases / update)', xlabel='PPO update', ylabel='Success (%)', ylim=(35, 85))
    ax.legend(fontsize=8, loc='best')
    ax = axes[0, 1]
    ax.plot(x, [row['approx_kl'] for row in rows], 'o-', color='#2468a2', label='Behavior-KL proxy')
    ax.plot(x, [row['clip_fraction'] for row in rows], 's--', color='#bb733c', label='Clip fraction')
    ax.set(title='B  Native minibatch diagnostics', xlabel='PPO update', ylabel='Reported scalar')
    ax.legend(fontsize=8)
    ax = axes[1, 0]
    ax.plot(x, [row['explained_variance'] for row in rows], 'o-', color='#398668')
    ax.axhline(0, color='#aaaaaa', lw=.8)
    ax.set(title='C  Critic explained variance', xlabel='PPO update', ylabel='Explained variance')
    ax = axes[1, 1]
    keys = ['success_once', 'success_at_end']; positions = np.arange(2)
    for offset, label, field, color in [(-.18, 'Before', 'before', '#9cabb7'), (.18, 'After update 18', 'after', '#2468a2')]:
        counts = [result['evaluation'][key][field] for key in keys]
        bars = ax.bar(positions + offset, [100 * c / 64 for c in counts], width=.34, label=label, color=color)
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f'{count}/64', ha='center', fontsize=9)
    primary = result['evaluation']['success_once']
    ax.set(title=f"D  Same 64 held-out cases (primary exact p={primary['exact_McNemar_two_sided_p']:.3g})",
           ylabel='Success (%)', xticks=positions, xticklabels=['First success', 'Success at end'], ylim=(0, 100))
    ax.legend(fontsize=8)
    for ax in axes.flat:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.15); ax.set_axisbelow(True)
    fig.suptitle('OpenDrawer pi0.5 action-expert Flow-SDE PPO — simulator reward', fontsize=13)
    fig.savefig(OUT / 'learning-and-drift.png', dpi=180)
    fig.savefig(OUT / 'learning-and-drift.pdf')
    plt.close(fig)


if __name__ == '__main__':
    main()
