"""Reconcile closed native source episodes against all eight action journals.

Reservations are distinct from resolved executed controls. The original failed
OpenDrawer TRAIN collection stays charged and excluded from learning; its before
panel was reused and is counted exactly once. No simulator or model calls.
"""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import time

ROOT = Path('/mowenjie/fluffy-bassoon-2/outputs/robocasa-flow-sde-10m-per-task-20261103-r1')
TASKS = ['OpenDrawer', 'CloseToasterOvenDoor', 'PickPlaceDrawerToCounter',
         'CoffeeSetupMug', 'TurnOffStove']


def read(path, evidence):
    data = path.read_bytes()
    evidence[str(path)] = hashlib.sha256(data).hexdigest()
    return json.loads(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--task', choices=TASKS, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert args.root.resolve() == ROOT
    base = ROOT / args.task
    assert args.output.parent == base and not args.output.exists()
    evidence = {}
    production = base / 'production-r1'
    clock = read(production / 'clock.json', evidence)
    progress = read(production / 'progress.json', evidence)
    after = read(production / 'after-evaluation.json', evidence)
    assert clock['exit_code'] == 0 and progress['completed_iterations'] == 130
    assert not progress['operator_checkpoint_exit'] and after['completed'] and after['global_step'] == 130
    teacher = read(base / 'FINAL-READOUT-R1.json', evidence)
    assert teacher['completed'] and teacher['training']['controls'] == 9984000
    accepted_q1 = 'q1-r2' if args.task == 'OpenDrawer' else 'q1-r1'
    stage_names = [accepted_q1, 'production-r1']
    if args.task == 'OpenDrawer':
        stage_names.insert(0, 'q1-r1')
    actual_stages = {p.parent.name for p in base.glob('*/manifest.json')
                     if p.parent.name.startswith(('q1-', 'production-'))}
    assert actual_stages == set(stage_names)
    rows, attempts = [], set()
    totals = collections.Counter()
    for stage_name in stage_names:
        stage = base / stage_name
        manifest = read(stage / 'manifest.json', evidence)
        cfg = read(stage / 'resolved-config.json', evidence)
        clock = read(stage / 'clock.json', evidence)
        failed_train = args.task == 'OpenDrawer' and stage_name == 'q1-r1'
        assert manifest['task'] == args.task and manifest['allocated_gpus'] == 8
        assert manifest['hostname'].startswith('job-') and manifest['hostname'].endswith('-master-0')
        if not failed_train:
            assert clock['exit_code'] == 0
            assert manifest['source'] == '0988b5f9e9bf31ac3db673cc2860959611b9ae12'
        pilot = cfg['native_pilot']
        assert Path(pilot['accounting_root']) == base / 'accounting'
        assert pilot['world_size'] == 8 and pilot['maximum_actions'] == 12210000
        assert not pilot['global_action_escrow']
        attempt = pilot['attempt']
        assert isinstance(attempt, str) and '/' not in attempt and attempt not in attempts
        attempts.add(attempt)
        before = stage_name == 'q1-r1'
        is_production = stage_name == 'production-r1'
        expected = {'training': 1238400 if is_production else 9600,
                    'before': 4800 if before else 0, 'after': 4800 if is_production else 0}
        for rank in range(8):
            path = base / 'accounting' / f'rank-{rank:02d}' / (attempt + '.json')
            journal = read(path, evidence)
            assert journal['format'] == 'rlinf_native_ppo_actions_v1'
            assert journal['attempt'] == attempt and journal['rank_action_ceiling'] == 1526250
            controls, calls, reservations = collections.Counter(), collections.Counter(), collections.Counter()
            for call in journal['calls']:
                assert call['resolved'] is True
                assert isinstance(call['requested'], int) and call['requested'] > 0
                value = call['executed_or_conservative']
                assert isinstance(value, int) and 0 <= value <= call['requested']
                assert math.isfinite(call['started_unix']) and call['ended_unix'] >= call['started_unix']
                phase = call['phase']
                calls[phase] += 1
                reservations[phase] += call['requested']
                controls[phase] += value
                if phase in expected:
                    assert call['requested'] == value == (800 if phase == 'training' else 400)
                else:
                    assert phase in {p + '/native_constructor_reset' for p in expected}
                    assert value == 0
            for phase, count in expected.items():
                assert controls[phase] == count
                assert calls[phase] == count // (800 if phase == 'training' else 400)
                resets = calls[phase] // 12
                assert calls[phase + '/native_constructor_reset'] == resets
                assert reservations[phase + '/native_constructor_reset'] == resets * (2048 if phase == 'training' else 1024)
            assert sum(controls.values()) <= journal['rank_action_ceiling']
            totals['excluded_failed_TRAIN' if failed_train else 'valid_TRAIN'] += controls['training']
            totals['before'] += controls['before']
            totals['after'] += controls['after']
            totals['all_attempt_policy_controls'] += sum(controls.values())
            rows.append(dict(stage=stage_name, attempt=attempt, rank=rank,
                excluded_TRAIN=failed_train, controls=dict(controls), calls=dict(calls),
                requested_reservations=dict(reservations), unresolved=0))
    for rank in range(8):
        assert {p.stem for p in (base / 'accounting' / f'rank-{rank:02d}').glob('*.json')} == attempts
        assert sum(sum(r['controls'].values()) for r in rows if r['rank'] == rank) <= 1526250
    assert totals['valid_TRAIN'] == 9984000
    assert totals['before'] == totals['after'] == 38400
    assert totals['excluded_failed_TRAIN'] == (76800 if args.task == 'OpenDrawer' else 0)
    assert totals['all_attempt_policy_controls'] == 10060800 + totals['excluded_failed_TRAIN']
    result = dict(completed=True, task=args.task, checked_unix=time.time(),
        valid_TRAIN_episodes=16640, valid_TRAIN_controls=9984000,
        valid_evaluation_controls=76800, logical_optimizer_steps=3120,
        totals=dict(totals), unresolved_intents=0, constructor_reset_controls=0,
        accounting_root=str(base / 'accounting'), attempt_count=len(attempts),
        eight_rank_partition_ceiling=12210000, rows=rows, evidence_sha256=evidence,
        control_unit='One executed robot command; excludes reset physics ticks, denoising steps and action-vector dimensions.',
        new_controls=0, SGD=0, model_calls=0)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ['task', 'totals', 'unresolved_intents', 'constructor_reset_controls']}), flush=True)


if __name__ == '__main__':
    main()
