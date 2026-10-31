"""Fresh native-headroom selection, then ordinary DSRL on frozen pi0.5.

The native-only screen is executable independently of trainer preparation.
No task or checkpoint can change after the SHA-bound selection is frozen.
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import random
import re
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np

from . import vla_target_online_dsrl as onset
from . import vla_target_online_dsrl_task57 as qualified
from .vla_source40_ppo_worker import begin, finish
from .vla_source40_schedule import content_hash

FORMAT = 'pi05_native_headroom_online_dsrl_v1'
REVIEWED_MAIN = '4e422587484312bfe34d623c5b498ab303a96be0'
CANDIDATES = (0, 29, 33, 38, 44, 53, 59, 64, 70, 77, 79, 82)
MAX_ACTIONS, MAX_SECONDS, MAX_BYTES = 1000000, 57600, 64 * 1024 ** 3
MAX_JOB_SECONDS, SCREEN_SECONDS = 25200, 10800
MAX_UPDATES, PANEL_INTERVAL = 500000, 20000
DATA_ROLE = 'target-onlineDEV'
EVIDENCE = (
    'outputs/libero40-fixed100-target-plan-20261031-r1/case-results.csv',
    'outputs/libero40-native-crossrun-20261030-r1/README.md',
    'outputs/libero90-pi05-headroom-screen-20261101-r1/PLAN.md',
)


def target(task, state, draw=0, *, bank='query'):
    return dict(benchmark='libero90', suite='libero_90', task_id=task, initial_state=state,
        scene_seed=123, noise_draw=draw, bank=bank,
        case_id=f'libero90:{task:02d}/state-{state}/seed-123/draw-{draw}/{bank}')


def query_bank(task):
    return [target(task, state, draw) for draw in range(2) for state in range(40, 50)]


def plan(source):
    if not re.fullmatch('[0-9a-f]{40}', source):
        raise ValueError('Require exact reviewed source pin')
    repo = Path(__file__).resolve().parents[3]
    evidence = [dict(path=name, sha256=onset.sha256(repo / name)) for name in EVIDENCE]
    return dict(format=FORMAT, source_commit=source, reviewed_main_commit=REVIEWED_MAIN,
        candidate_ids=list(CANDIDATES), evidence=evidence,
        evidence_interpretation='Saved official40-motor development candidacy only; no pooled native rate, no semantic/task novelty filter',
        screening_cases=[target(task, state, bank='screen') for task in CANDIDATES for state in range(20)],
        screen_episodes_per_candidate=20, screen_distinct_state_ids_per_candidate=20,
        selection=dict(eligible_successes_inclusive=[2, 8], preferred_successes=4,
            ranking='abs(successes-4), then ascending task ID', maximum_confirmation_candidates=3,
            confirmation_successes_inclusive=[2, 8], no_eligible_or_confirmed='abort before any SGD'),
        confirmation_cases={str(task): query_bank(task) for task in CANDIDATES},
        confirmation_distinct_states=10, confirmation_noise_draws_per_state=2,
        physical_reset_seed=123, private_seed_convention='onset._seed(case,query,0); changes motor/noise RNG only, never physical reset seed',
        training_states=list(range(40)), learner_seed=0, entire_motor_frozen=True,
        motor_receipt_sha256=qualified.PI05_RECEIPT,
        motor_parameter_sha256=qualified.MOTOR_PARAMETER_SHA256,
        native_horizon=10, execute_steps=10, horizon=400, settling_steps=10,
        nominal_optimizer_stop=MAX_UPDATES, maximum_optimizer_updates=MAX_UPDATES + 780,
        learned_panel_interval=PANEL_INTERVAL, learned_panel_episodes=20,
        initial_controls=['native_full_gaussian', 'tied_unbounded_gaussian', 'initialized_bounded_sac'],
        final_controls=['native_full_gaussian'],
        recipe=dict(onset.RECIPE, departures='pi0.5 H10/ex10; actual-duration discount; development headroom selection; 20k-update/20-query panels'),
        maximum_total_actions=MAX_ACTIONS, maximum_phase_gpu_seconds=MAX_SECONDS,
        maximum_job_gpu_seconds=MAX_JOB_SECONDS, maximum_screen_gpu_seconds=SCREEN_SECONDS,
        maximum_output_bytes=MAX_BYTES, data_role=DATA_ROLE, target_only=True,
        source_assembly_admission=False, source_checkpoint_reused=False,
        all_in_planned_action_bound=871580, unassigned_action_margin=128420,
        selection_frozen_before_optimizer=True, no_task_or_checkpoint_switch_after_sgd=True,
        job_authority='Human authorizes each job under8h and independent parallel experiments; finite phase uses resumable slices, no campaign8h cap')


def ranked_candidates(receipts):
    counts = {task: [] for task in CANDIDATES}
    for row in receipts:
        case = row['case']
        if (case['task_id'] not in counts or case['initial_state'] not in range(20)
                or case['bank'] != 'screen' or case['noise_draw'] != 0):
            raise ValueError('Unexpected candidate screening case')
        counts[case['task_id']].append(row)
    for task, rows in counts.items():
        if len(rows) != 20 or {r['case']['initial_state'] for r in rows} != set(range(20)):
            raise ValueError('Selection requires all20 native episodes for every declared candidate')
    successes = {task: sum(bool(row['success']) for row in rows) for task, rows in counts.items()}
    order = sorted((task for task in CANDIDATES if 2 <= successes[task] <= 8),
        key=lambda task: (abs(successes[task] - 4), task))
    return order, [dict(task_id=task, successes=successes[task], episodes=20,
        distinct_state_ids=20,
        distinct_initial_state_hashes=len({row['initial_state_vector_sha256'] for row in counts[task]}),
        eligible=task in order) for task in CANDIDATES]


class NativeOnly:
    """No SAC actor, critic, replay records or optimizer is constructed to screen."""
    def __init__(self):
        self.agent, self.replay, self.updates = SimpleNamespace(_rng=onset._key(0)), [], 0
    def frozen_signature(self):
        return 'native-only-no-controller-no-optimizer'


def run_screen(motor, server_factory, *, root, contract, deadline, resume=False):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / 'screen-state.json'
    plan_path = root / 'plan.json'
    if plan_path.exists():
        if not resume or json.loads(plan_path.read_text()) != contract:
            raise ValueError('Only the original frozen screening plan may resume')
    else:
        onset.write_json(plan_path, contract)
    value = (json.loads(path.read_text()) if path.exists() else
        dict(format=FORMAT, phase='screen', screen_cursor=0, screen_receipts=[],
            ranked=None, rates=None, confirmation_index=0, confirmation_cursor=0,
            confirmations=[], active_confirmation=[], selected_task=None))
    learner = NativeOnly()
    servers = {}

    def save(reason=None):
        onset.write_json(path, value)
        result = dict(value, plan_sha256=onset.sha256(plan_path), stop_reason=reason,
            all_in_actions=onset.joint_charge(root), target_optimizer_updates=0,
            source_assembly_admission=False, data_role=DATA_ROLE,
            output_bytes=qualified.output_bytes(root))
        onset.write_json(root / 'screen-index.json', result)
        return result

    def episode(case, identity):
        if time.monotonic() + 90 >= deadline:
            return None
        if qualified.output_bytes(root) + 4 * 1024 ** 3 > MAX_BYTES:
            return None
        task = case['task_id']
        if task not in servers:
            for old in servers.values():
                old.close()
            servers.clear()
            servers[task] = server_factory(task)
        directory = root / f'case-{case["bank"]}-{task:02d}'
        sim = onset.JournaledSimulator(servers[task], directory, root,
            case_cap=MAX_ACTIONS, total_cap=MAX_ACTIONS)
        if not sim.reserve_episode():
            return None
        sim.role, sim.episode_id = 'query', identity
        rows, summary = qualified.native_query(sim, motor, learner, case)
        qualified._annotate(rows, episode_id=identity, state=case['initial_state'], updates=0)
        for row in rows:
            row.update(data_role=DATA_ROLE, bank=case['bank'], noise_draw=case['noise_draw'],
                physical_reset_seed=case['scene_seed'], private_noise_seed=summary['episode_random_seed'])
        summary.update(case=case, data_role=DATA_ROLE, native_only=True,
            source_assembly_admission=False, policy_signature=learner.frozen_signature())
        return onset.save_trace(directory, identity, rows, summary, contract)

    try:
        save()
        while value['phase'] == 'screen' and value['screen_cursor'] < len(contract['screening_cases']):
            case = contract['screening_cases'][value['screen_cursor']]
            receipt = episode(case, f'screen-{value["screen_cursor"]:03d}')
            if receipt is None:
                return save('whole_episode_time_action_or_storage_reservation')
            value['screen_receipts'].append(receipt)
            value['screen_cursor'] += 1
            save()
            print(json.dumps(dict(event='screen', completed=value['screen_cursor'],
                task=case['task_id'], state=case['initial_state'], success=receipt['success'])), flush=True)
        if value['phase'] == 'screen':
            value['ranked'], value['rates'] = ranked_candidates(value['screen_receipts'])
            value['phase'] = 'confirm' if value['ranked'] else 'aborted'
            with (root / 'screen-rates.csv').open('w') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(value['rates'][0]))
                writer.writeheader(); writer.writerows(value['rates'])
            save()
        while value['phase'] == 'confirm':
            if value['confirmation_index'] >= min(3, len(value['ranked'])):
                value['phase'] = 'aborted'
                break
            task = value['ranked'][value['confirmation_index']]
            cases = contract['confirmation_cases'][str(task)]
            while value['confirmation_cursor'] < len(cases):
                case = cases[value['confirmation_cursor']]
                receipt = episode(case, f'confirm-{task:02d}-{value["confirmation_cursor"]:02d}')
                if receipt is None:
                    return save('whole_confirmation_episode_reservation')
                value['active_confirmation'].append(receipt)
                value['confirmation_cursor'] += 1
                save()
            successes = sum(bool(row['success']) for row in value['active_confirmation'])
            value['confirmations'].append(dict(task_id=task, successes=successes, episodes=20,
                distinct_state_ids=10, noise_draws_per_state=2,
                distinct_initial_state_hashes=len({row['initial_state_vector_sha256'] for row in value['active_confirmation']}),
                receipts=value['active_confirmation']))
            if 2 <= successes <= 8:
                value['selected_task'], value['phase'] = task, 'selected'
                selection = dict(format=FORMAT, plan_sha256=onset.sha256(plan_path), task_id=task,
                    selected_before_any_sgd=True, optimizer_updates=0, data_role=DATA_ROLE,
                    query_bank=query_bank(task), screen_rates=value['rates'],
                    ranked_candidates=value['ranked'], confirmations=value['confirmations'],
                    all_in_actions_before_learning=onset.joint_charge(root),
                    development_selection=True, source_assembly_admission=False)
                frozen = root / 'frozen-selection.json'
                if frozen.exists() and json.loads(frozen.read_text()) != selection:
                    raise ValueError('Immutable selected task/query bank already differs')
                onset.write_json(frozen, selection)
                value['selection_sha256'] = onset.sha256(frozen)
            else:
                value['confirmation_index'] += 1
                value['confirmation_cursor'], value['active_confirmation'] = 0, []
            save()
        return save()
    finally:
        for server in servers.values():
            server.close()


def admit_learning_code(root, original, proposed, source, screen_source):
    """Explicit screen-to-trainer child pin; never rewrite the producer plan.

    This one migration is allowed before SGD. Later learning slices must use
    the admitted trainer pin. The original worker contract/cost ledger stays.
    """
    root = Path(root)
    if original == proposed:
        return
    if ({k: v for k, v in original.items() if k != 'source_commit'} !=
            {k: v for k, v in proposed.items() if k != 'source_commit'}
            or screen_source != original['source_commit']):
        raise ValueError('Explicit screen child may change code pin only, never frozen science or costs')
    admission_path = root / 'screen-to-learning-source.json'
    if admission_path.exists():
        admission = json.loads(admission_path.read_text())
        if admission['screen_source'] != screen_source or admission['learning_source'] != source:
            raise ValueError('Only the already admitted trainer code may resume')
        return
    experiment = root / 'experiment'
    selection_path = experiment / 'frozen-selection.json'
    plan_path = experiment / 'plan.json'
    selection = json.loads(selection_path.read_text())
    if (selection['optimizer_updates'] != 0 or not selection['selected_before_any_sgd']
            or selection['plan_sha256'] != onset.sha256(plan_path)):
        raise ValueError('Code child requires original frozen pre-SGD selection')
    for state in experiment.glob('case-learn-*/state.json'):
        if json.loads(state.read_text())['model']['updates'] > 0:
            raise ValueError('Cannot admit a different implementation after target SGD')
    ledger = json.loads((root / 'worker-time.json').read_text())
    onset.write_json(admission_path, dict(format='pi05_headroom_screen_child_v1',
        screen_source=screen_source, learning_source=source,
        original_screen_plan_sha256=onset.sha256(plan_path), frozen_selection_sha256=onset.sha256(selection_path),
        inherited_actual_actions=onset.joint_charge(experiment),
        inherited_gpu_seconds=sum(row.get('charged_seconds', row['reserved_seconds']) for row in ledger['attempts']),
        original_worker_contract_sha256=content_hash(original), source_changed_before_sgd=True,
        original_worker_contract_preserved=True, raw_data_copied_or_recounted=False))


def prepare_attempt(root, source, run, *, slice_seconds, started_unix=None, resume=False,
                    stage='screen', screen_source=None):
    if not re.fullmatch('[A-Za-z0-9-]+', run) or type(slice_seconds) is not int or not 600 <= slice_seconds <= MAX_JOB_SECONDS - 60:
        raise ValueError('Require immutable run and a job slice strictly below8h')
    contract = dict(plan(source), run_id=run)
    root = Path(root)
    ledger = root / 'worker-time.json'
    if ledger.exists():
        data = json.loads(ledger.read_text())
        if not resume or data['worker_seconds'] != MAX_SECONDS:
            raise ValueError('Only the original whole-phase source/cost clock may resume')
        if data['contract'] != contract:
            if stage != 'learn':
                raise ValueError('Screen source is immutable')
            admit_learning_code(root, data['contract'], contract, source, screen_source)
            contract = data['contract']
        prior = sum(row.get('charged_seconds', row['reserved_seconds']) for row in data['attempts'])
    else:
        if resume:
            raise ValueError('Cannot cold-restart the phase clock on resume')
        prior = 0
    if stage not in ('screen', 'learn'):
        raise ValueError('Unknown phase stage')
    actual = min(slice_seconds, MAX_SECONDS - prior - 60)
    if stage == 'screen' and ledger.exists():
        screen_charge = 0
        for row in data['attempts']:
            stage_file = Path(row['path']) / 'phase-stage.json'
            if not stage_file.exists() or json.loads(stage_file.read_text())['stage'] == 'screen':
                screen_charge += row.get('charged_seconds', row['reserved_seconds'])
        actual = min(actual, SCREEN_SECONDS - screen_charge - 60)
    elif stage == 'screen':
        actual = min(actual, SCREEN_SECONDS - 60)
    if actual < 600:
        raise ValueError('Finite phase time exhausted; preserve recovery')
    start = time.time() if started_unix is None else float(started_unix)
    attempt = begin(root, contract, slice_seconds=actual, worker_seconds=MAX_SECONDS,
        resume=resume, now=start)
    onset.write_json(attempt / 'phase-stage.json', dict(stage=stage))
    return dict(attempt=str(attempt), stop_unix=math.floor(start + actual), slice_seconds=actual)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='operation', required=True)
    clock = sub.add_parser('begin')
    clock.add_argument('--root', type=Path, required=True)
    clock.add_argument('--source', required=True); clock.add_argument('--run', required=True)
    clock.add_argument('--slice-seconds', type=int, required=True)
    clock.add_argument('--started-unix', type=float, required=True)
    clock.add_argument('--stage', choices=('screen', 'learn'), required=True)
    clock.add_argument('--screen-source')
    clock.add_argument('--resume', action='store_true')
    close = sub.add_parser('finish')
    close.add_argument('--root', type=Path, required=True); close.add_argument('--attempt', type=Path, required=True)
    close.add_argument('--exit-code', type=int, required=True)
    run = sub.add_parser('run')
    run.add_argument('--stage', choices=('screen', 'learn'), required=True)
    for name in ('output', 'checkpoint', 'openpi-root', 'dsrl-root', 'libero-root', 'simulator-python'):
        run.add_argument('--' + name, type=Path, required=True)
    for name in ('source-commit', 'expected-hostname', 'simulator-pythonpath'):
        run.add_argument('--' + name, required=True)
    run.add_argument('--selection-sha256')
    run.add_argument('--screen-source')
    run.add_argument('--stop-unix', type=float, required=True)
    run.add_argument('--port', type=int, default=8797)
    run.add_argument('--resume', action='store_true')
    args = ap.parse_args(argv)
    if args.operation == 'begin':
        value = prepare_attempt(args.root, args.source, args.run, slice_seconds=args.slice_seconds,
            started_unix=args.started_unix, resume=args.resume, stage=args.stage, screen_source=args.screen_source)
        print(f'{value["attempt"]}\t{value["stop_unix"]}\t{value["slice_seconds"]}')
        return
    if args.operation == 'finish':
        print(json.dumps(finish(args.root, args.attempt, args.exit_code))); return
    if sys.platform != 'linux' or socket.gethostname() != args.expected_hostname:
        raise ValueError('Models/simulator only on identified allocated Linux')
    from fluffy_bassoon.gpu_execution import validate_execution
    validate_execution(os.environ.get('FLUFFY_GPU_EXECUTION', 'baige'), os.environ)
    repo = Path(__file__).resolve().parents[3]
    if (subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip() != args.source_commit
            or subprocess.check_output(['git', 'status', '--porcelain'], cwd=repo, text=True).strip()):
        raise ValueError('Require exact clean reviewed headroom source')
    if onset.sha256(args.checkpoint / 'READY.json') != qualified.PI05_RECEIPT:
        raise ValueError('Require existing official OpenPI40 pi0.5 release')
    remaining = args.stop_unix - time.time()
    if not math.isfinite(remaining) or not 0 < remaining < MAX_JOB_SECONDS:
        raise ValueError('Require live per-job deadline below8h')
    deadline = time.monotonic() + remaining
    if args.stage == 'screen':
        deadline = min(deadline, time.monotonic() + SCREEN_SECONDS)
    args.output.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (args.output / 'driver.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    attempt = args.output / 'runtime-attempts' / str(time.time_ns())
    attempt.mkdir(parents=True)
    original = args.output
    args.output, args.model_config, args.feature_interface = attempt, 'pi05_libero', 'sac'
    motor = onset.FrozenOpenPI(args)
    args.output = original
    from .vla_motor_replay import parameter_signature
    before = parameter_signature(motor.policy._model)
    onset.write_json(attempt / 'motor-before.json', before)
    try:
        if before['sha256'] != qualified.MOTOR_PARAMETER_SHA256:
            raise ValueError('Entire frozen pi0.5 motor parameter hash differs')
        def server_factory(task):
            return onset.TargetServer(task_id=task, python=args.simulator_python,
                pythonpath=args.simulator_pythonpath, output=attempt / f'server-{task:02d}-{time.time_ns()}',
                port=args.port + task, libero_root=args.libero_root)
        contract = plan(args.source_commit)
        if args.stage == 'screen':
            result = run_screen(motor, server_factory, root=args.output, contract=contract,
                deadline=deadline, resume=args.resume)
        else:
            from .vla_target_online_dsrl_headroom_learning import run_selected
            result = run_selected(motor, server_factory, args, contract, deadline)
        onset.write_json(attempt / 'result.json', result)
    finally:
        after = parameter_signature(motor.policy._model)
        onset.write_json(attempt / 'motor-after.json', dict(before=before, after=after, equal=before == after))
        if before != after:
            raise RuntimeError('Frozen pi0.5 motor changed during headroom/DSRL phase')


if __name__ == '__main__':
    main()
