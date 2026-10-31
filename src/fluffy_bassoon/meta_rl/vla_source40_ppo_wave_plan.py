"""Predeclare frozen final-source PPO deployment after terminal source validation.

Metadata/hash operations only. No outcome-dependent checkpoint or case choice,
allocation, reservation, model construction or writes to source-validation runs.
"""
import argparse
import fcntl
import hashlib
import json
import math
from pathlib import Path

from .vla_source40_noise_plan import build_plan as original_cases, content_hash


FORMAT = 'source40_frozen_ppo_target_wave_plan_v1'
LEDGER_FORMAT = 'source40_frozen_ppo_target_wave_budget_v1'
ZERO_METHODS = ('native', 'ppo_full', 'ppo_trained_episode')
ONE_METHODS = ('ppo_full', 'ppo_episode_reset', 'ppo_trained_episode')
PANEL = tuple(range(0, 90, 10))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def terminal_validation(root, index_sha):
    root = Path(root).resolve()
    with (root / 'launcher.lock').open('r') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError('Source validation is still active; no PPO target admission') from error
        index_path, contract_path = root / 'experiment/index.json', root / 'experiment/contract.json'
        if (root / 'exit-code.txt').read_text().strip() != '0' or sha(index_path) != index_sha:
            raise ValueError('Require the exact successful terminal source-validation index')
        index, clock = read(index_path), read(root / 'worker-time.json')
        contract, receipts = index['contract'], index.get('receipts', [])
        if (index.get('format') != 'source40_frozen_ppo_validation_v1' or index.get('completed') is not True
                or index.get('weights_unchanged') is not True or index.get('optimizer_updates') != 0
                or contract != read(contract_path) or len(receipts) != 32
                or {(r['slot'], r['trial']) for r in receipts} != {(s, t) for s in range(16) for t in range(2)}
                or any(r['optimizer_updates'] != 0 or not math.isfinite(r['replay_error']) or r['replay_error'] > 1e-4
                    or len(r['receipt']['attempts']) != 4 for r in receipts)
                or not 0 < index.get('total_action_charge', 0) <= 56000
                or clock.get('worker_seconds') != 14400 or not clock.get('attempts')
                or any(r['status'] != 'finished' or type(r.get('charged_seconds')) is not int or r['charged_seconds'] < 0 for r in clock['attempts'])
                or clock['attempts'][-1]['exit_code'] != 0
                or sum(r['charged_seconds'] for r in clock['attempts']) > 14400):
            raise ValueError('Source validation must complete all32 K4 trials with frozen weights/replay parity and its original time/action limits')
        manifest = root / 'models.json'
        if sha(manifest) != contract['models_sha256'] or set(contract.get('models', {})) != {'full', 'episode'}:
            raise ValueError('Source-validation model manifest differs')
        exports = {}
        for mode in ('full', 'episode'):
            spec, path = contract['models'][mode], root / 'experiment' / f'frozen-{mode}.pt'
            if sha(path) != spec['policy_sha256'] or spec.get('actual_source_tasks') != 40:
                raise ValueError('Successful source validation must bind both actual final all40 policy exports')
            exports[mode] = dict(path=str(path), sha256=spec['policy_sha256'], tensor_sha256=spec['tensor_sha256'],
                source_updates=spec['source_updates'], source_counters=spec['source_counters'],
                recovery_sha256=spec['recovery_sha256'], terminal_lineage=spec['terminal_lineage'])
        recovery = root / 'experiment/validation-recovery.pt'
        if sha(recovery) != index['latest_recovery_sha256']:
            raise ValueError('Terminal source-validation full recovery checksum differs')
        return dict(format='source40_successful_terminal_validation_v1', root=str(root), index_sha256=index_sha,
            contract_sha256=sha(contract_path), models_manifest_path=str(manifest), models_manifest_sha256=sha(manifest),
            validation_recovery_sha256=sha(recovery), validation_physical_actions=index['total_action_charge'],
            validation_gpu_seconds=sum(r['charged_seconds'] for r in clock['attempts']),
            schedule_sha256=contract['schedule_sha256'], exports=exports,
            selection='Fixed final R3 source policies before validation; qualification is not performance-based checkpoint selection')


def build_plan(*, worktree, source_commit, source_validation):
    if source_validation.get('format') != 'source40_successful_terminal_validation_v1' or set(source_validation.get('exports', {})) != {'full', 'episode'}:
        raise ValueError('Bind the successful terminal source-validation receipt and both frozen exports')
    base = original_cases(worktree=worktree, source_commit=source_commit)
    by_id = {e['case']['task_id']: e for e in base['cases']}
    # PANEL IDs are all even. Put them first, so case striding allocates five
    # versus four panel cases instead of placing all nine on one worker.
    order = list(PANEL) + [i for i in range(90) if i not in PANEL]
    cases = []
    for task in order:
        entry = by_id[task]
        groups = []
        for phase, methods, budget in (('A', ZERO_METHODS, 0), ('B', ONE_METHODS if task in PANEL else (), 1)):
            for method in methods:
                groups.append(dict(group_id=f'{entry["case"]["case_id"]}/{phase}/{method}/k{budget}',
                    case=entry['case'], phase=phase, method=method, source_condition='native' if method == 'native' else 'source_ppo',
                    updates=None, support_budgets=[budget], episodes=budget + 1, maximum_actions=(budget + 1) * 410))
        cases.append(dict(case=entry['case'], simulator=entry['simulator'], groups=groups,
            zero_shot_reuse=dict(ppo_episode_reset=groups[1]['group_id']),
            nominal_actions=sum(g['maximum_actions'] for g in groups), nominal_episodes=sum(g['episodes'] for g in groups)))
    workers = [dict(worker=i, case_ids=[e['case']['case_id'] for e in cases[i::2]],
        nominal_actions=sum(e['nominal_actions'] for e in cases[i::2]), nominal_episodes=sum(e['nominal_episodes'] for e in cases[i::2]),
        maximum_total_actions=70000, worker_seconds=3600) for i in (0, 1)]
    if [w['nominal_actions'] for w in workers] != [67650, 65190]:
        raise ValueError('PPO target case assignment no longer fits the intended equal70k worker caps')
    value = dict(format=FORMAT, contract=dict(source_commit=source_commit, worktree=str(Path(worktree).resolve()),
        source_validation=source_validation, zero_methods=list(ZERO_METHODS), one_methods=list(ONE_METHODS),
        panel_ids=list(PANEL), horizon=400, settle_steps=10, maximum_total_actions=140000, maximum_gpu_seconds=7200,
        optimizer_updates=0, qualification_extra_resets=0, target_data_in_source_pretraining=False,
        policy_selection='Both immutable final source policies that completed source-local validation; no target checkpoint selection',
        case_order='Fixed nine panel IDs first, then remaining numeric IDs; case[i::2] balances5/4 K1 cases before outcomes',
        zero_shot_reuse='Same-weight ppo_episode_reset K0 references ppo_full K0; separately trained reset is never substituted'),
        cases=cases, groups=[g for e in cases for g in e['groups']], workers=workers,
        counts=dict(cases=90, groups=297, physical_episodes=324, nominal_actions_including_settling=132840,
            action_cap=140000, gpu_seconds=7200))
    value['ppo_wave_sha256'] = content_hash(value)
    return value


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('worktree', 'validation-root', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--source-commit', required=True)
    p.add_argument('--validation-index-sha256', required=True)
    args = p.parse_args(argv)
    proof = terminal_validation(args.validation_root, args.validation_index_sha256)
    value = build_plan(worktree=args.worktree, source_commit=args.source_commit, source_validation=proof)
    with args.output.open('x') as stream:
        stream.write(json.dumps(value, indent=2) + '\n')
    print(json.dumps(dict(ppo_wave_sha256=value['ppo_wave_sha256'], counts=value['counts'], workers=value['workers'])))


if __name__ == '__main__':
    main()
