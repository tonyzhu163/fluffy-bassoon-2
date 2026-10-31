"""Predeclare contemporaneous native/zero/unbounded-tied/clipped-tied noise controls on all90.

Metadata only; no model imports, outcomes, reservations or allocations.
"""
import argparse
import hashlib
import json
from pathlib import Path


FORMAT = 'source40_libero90_noise_controls_plan_v1'
LEDGER_FORMAT = 'libero90_noise_controls_budget_v1'
METHODS = ('native', 'zero', 'tied_gaussian', 'tied_gaussian_clipped')
PI05_RECEIPT = 'ec064cce32f5003e2e3979c5712d7d6385018f6bfcc59cb2161105881e70d170'
LIBERO_COMMIT = '87edbd1b823e464bc66557d213cc45e9f88aa543'
NOISE = dict(
    native='NumPy default_rng(episode_seed).standard_normal((10,32)).astype(float32); independent rows; unbounded',
    zero='float32 zeros(32), repeated across native H10; no clipping/rescaling',
    tied_gaussian_clipped='NumPy default_rng(episode_seed).standard_normal(32).astype(float32), then clip[-1,1], repeated across native H10; no rescaling',
    tied_gaussian='NumPy default_rng(episode_seed).standard_normal(32).astype(float32), repeated across native H10; unbounded N(0,I), no clipping/rescaling')


def content_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def build_plan(*, worktree, source_commit):
    if len(source_commit) != 40 or any(c not in '0123456789abcdef' for c in source_commit):
        raise ValueError('Require the exact reviewed source SHA')
    root, old = Path('/mowenjie/fluffy-bassoon-2'), Path('/mowenjie/fluffy-bassoon')
    worktree = Path(worktree).resolve()
    libero = root / 'deps/libero-dsrl-87edbd1-20261029-r1/repo'
    openpi = old / 'third_party/openpi-215abfb'
    cases = []
    for task in range(90):
        case = dict(benchmark='libero90', case_id=f'libero90:{task:02d}/state-0/seed-123',
            suite='libero_90', task_id=task, initial_state=0, scene_seed=123)
        groups = [dict(group_id=f'{case["case_id"]}/{method}/k0', case=case, method=method,
            source_condition='noise_control', updates=None, support_budgets=[0], episodes=1,
            maximum_actions=410) for method in METHODS]
        cases.append(dict(case=case, groups=groups,
            simulator=dict(python='/mowenjie/envs/wam-libero-plus-py310/bin/python',
                pythonpath=':'.join(map(str, (worktree / 'src', libero, openpi / 'packages/openpi-client/src', old / 'envs/pi05-plus-client-deps'))),
                module='fluffy_bassoon.meta_rl.vla_libero', arguments=['--mode', 'env-server', '--suite', 'libero_90',
                    '--task-id', str(task), '--horizon', '400', '--fresh-environment-on-reset'],
                horizon=400, settle_steps=10, expected_metadata=dict(suite='libero_90', task_id=task, horizon=400,
                    settle_steps=10, libero_commit=LIBERO_COMMIT, fresh_environment_on_reset=True))))
    workers = [dict(worker=i, case_ids=[entry['case']['case_id'] for entry in cases[i::2]],
        nominal_actions=73800, maximum_total_actions=76000, worker_seconds=3600) for i in range(2)]
    code = Path(__file__).resolve().parent
    provenance = {name: hashlib.sha256((code / name).read_bytes()).hexdigest()
        for name in ('vla_source40_corpus.py', 'vla_libero.py', 'vla_source40_evaluate.py')}
    value = dict(format=FORMAT, contract=dict(source_commit=source_commit, worktree=str(worktree),
        frozen_motor_receipt_sha256=PI05_RECEIPT, noise_distributions=dict(NOISE),
        source_warmup='OnlineSourceSimulator.choose: standard_normal(32).astype(float32) while learner.updates==0; collector run_episode bounded=False; repeat over H10',
        post_update_distinction='Source SAC and learned students clip steering[-1,1]; unbounded tied matches warmup, while clipped tied isolates the deployment bound',
        source_recipe_files_sha256=provenance, target_selection='All LIBERO90 tasks0..89, state0,scene_seed123; no outcomes or semantic filter',
        deployment_optimizer_updates=0, support_budget=0, methods=list(METHODS),
        maximum_total_actions=152000, maximum_gpu_seconds=7200, qualification_extra_resets=0,
        interpretation='Paired noise-interface diagnostic, not a learned checkpoint selection or source-data experiment'),
        cases=cases, groups=[g for entry in cases for g in entry['groups']], workers=workers,
        counts=dict(cases=90, groups=360, physical_episodes=360, nominal_actions_including_settling=147600,
            action_cap=152000, gpu_seconds=7200))
    value['noise_plan_sha256'] = content_hash(value)
    return value


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worktree', type=Path, required=True)
    p.add_argument('--source-commit', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(argv)
    value = build_plan(worktree=args.worktree, source_commit=args.source_commit)
    with args.output.open('x') as stream:
        stream.write(json.dumps(value, indent=2) + '\n')
    print(json.dumps(dict(noise_plan_sha256=value['noise_plan_sha256'], counts=value['counts'])))


if __name__ == '__main__':
    main()
