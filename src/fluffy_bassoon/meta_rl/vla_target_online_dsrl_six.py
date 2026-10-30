"""Canonical PixelSAC learning-onset curves on six predeclared LIBERO90 cases.

Frozen motor/noise interface and recorded4104D features match the meta arm.
The authors' actor still consumes image64/proprio8, not those prefix features.
Target experience is adaptation/evaluation data, excluded from source training.
"""
import argparse
import json
import math
import os
from pathlib import Path
import random
import socket
import subprocess
import sys
import time

import numpy as np

from . import vla_target_online_dsrl as onset
from .vla_source40_corpus import PI05_RECEIPT
from .vla_source40_schedule import content_hash


FORMAT = 'openpi_pi05_target_online_dsrl_six_onset_v1'
TASKS = (0, 10, 20, 30, 40, 50)
MILESTONES = tuple((episodes, episodes * 410) for episodes in (0, 1, 4, 8, 13, 16))
CASE_CAP = 9500
TOTAL_CAP = 64000
MAX_SECONDS = 7200
RECIPE = dict(onset.RECIPE,
    interface='Authors PixelSAC image64 single-camera +proprio8; recorded4104D native prefix is NOT actor input',
    recorded_feature_kind='openpi_final_prefix_visual_mean_language_mean_public_state_v2',
    representation_comparison='Unmatched actor representations: canonical pixel SAC versus prefix-conditioned history controllers',
    departures='Frozen pi0.5 H10/execute10, actual-duration discounts, predeclared target cases, separate frozen query milestones; bounded learning-onset slice')


def plan():
    return dict(format=FORMAT, cases=[onset.case(task) for task in TASKS],
        milestones=[dict(training_episodes=episodes, training_action_ceiling=actions)
            for episodes, actions in MILESTONES],
        maximum_actions_per_case=CASE_CAP, maximum_total_actions=TOTAL_CAP,
        maximum_gpu_seconds=MAX_SECONDS, target_only=True, source_assembly_admission=False,
        planning=dict(training_episodes=6 * 16, query_episodes=6 * len(MILESTONES),
            worst_case_training_actions=6 * 16 * 410,
            worst_case_query_actions=6 * len(MILESTONES) * 410,
            initial_qualification_actions=6 * 10,
            nominal_complete_matrix_actions=6 * ((16 + len(MILESTONES)) * 410 + 10),
            sum_individual_case_caps=6 * CASE_CAP,
            unassigned_joint_reserve=TOTAL_CAP - 6 * CASE_CAP),
        recipe=RECIPE,
        interpretation='Learning-onset/deployment-cost diagnostic; not converged ordinary DSRL or a full-suite score')


def validate_completed(index, case_contract, output):
    if (index.get('completed') is not True or index.get('case') != case_contract['case']
            or index.get('contract_sha256') != content_hash(case_contract)
            or index.get('training_episodes') != 16 or len(index.get('query_receipts', [])) != len(MILESTONES)
            or index.get('maximum_total_actions') != CASE_CAP or not index.get('target_only')
            or index.get('source_assembly_admission') is not False
            or onset.sha256(Path(output) / 'target-recovery.pkl') != index.get('latest_recovery_sha256')):
        raise ValueError('Completed six-case target receipt/recovery differs from its immutable contract')


def invocation_deadline(stop_unix, *, now_unix=None, now_monotonic=None):
    """A scheduler-slice stop is ephemeral, never part of the scientific contract."""
    now_unix = time.time() if now_unix is None else now_unix
    now_monotonic = time.monotonic() if now_monotonic is None else now_monotonic
    remaining = stop_unix - now_unix
    if not math.isfinite(remaining) or not 0 < remaining <= MAX_SECONDS:
        raise ValueError('Require an explicit live invocation stop within the original two-hour ceiling')
    return now_monotonic + remaining


def scientific_contract(source_commit, learner_seed):
    """Resume contract contains fixed science, never invocation clock values."""
    return dict(plan(), source_commit=source_commit, learner_seed=learner_seed,
        motor_receipt_sha256=PI05_RECEIPT, maximum_driver_seconds=MAX_SECONDS,
        fresh_actor_replay_optimizer_per_case=True, fixed_query_seed_across_milestones=True,
        query_data_added_to_replay=False, source_checkpoint_reused=False,
        prefix_features_recorded=True, prefix_features_are_actor_input=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('output', 'checkpoint', 'openpi-root', 'dsrl-root', 'libero-root', 'simulator-python'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('source-commit', 'expected-hostname', 'simulator-pythonpath'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--max-total-actions', type=int, required=True)
    parser.add_argument('--max-elapsed-seconds', type=float, default=MAX_SECONDS)
    parser.add_argument('--stop-unix', type=float, required=True)
    parser.add_argument('--learner-seed', type=int, default=0)
    parser.add_argument('--port', type=int, default=8779)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    started = time.monotonic()
    deadline = invocation_deadline(args.stop_unix)
    if sys.platform != 'linux' or socket.gethostname() != args.expected_hostname:
        raise ValueError('Run target-online DSRL only inside the explicitly identified allocated Linux host')
    if (args.max_total_actions != TOTAL_CAP or args.max_elapsed_seconds != MAX_SECONDS
            or not 0 <= args.learner_seed < 2**31):
        raise ValueError('Require the joint64k physical contract and at most2h driver ceiling')
    from fluffy_bassoon.gpu_execution import validate_execution
    validate_execution(os.environ.get('FLUFFY_GPU_EXECUTION', 'baige'), os.environ)
    root = Path(__file__).resolve().parents[3]
    if (subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip() != args.source_commit
            or subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True).strip()):
        raise ValueError('Require the exact clean reviewed six-case target baseline commit')
    if onset.sha256(args.checkpoint / 'READY.json') != PI05_RECEIPT:
        raise ValueError('Require the qualified official OpenPI40 motor')
    contract = scientific_contract(args.source_commit, args.learner_seed)
    args.output.mkdir(parents=True, exist_ok=args.resume)
    import fcntl
    lock = (args.output / 'run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.resume:
        if json.loads((args.output / 'contract.json').read_text()) != contract:
            raise ValueError('Six-case target baseline resume contract changed')
    else:
        onset.write_json(args.output / 'contract.json', contract)
    attempt = args.output / 'attempts' / str(time.time_ns())
    attempt.mkdir(parents=True)
    original_output = args.output
    args.model_config, args.feature_interface, args.output = 'pi05_libero', 'vla-prefix', attempt
    motor = onset.FrozenOpenPI(args)
    args.output = original_output
    if motor.feature_encoder is None or motor.feature_encoder.contract.get('observation_dim') != 4104:
        raise ValueError('The recorded native prefix must match the4104D meta-controller interface')
    contract_feature = motor.feature_encoder.contract
    feature_path = args.output / 'recorded-feature-contract.json'
    if feature_path.exists():
        if json.loads(feature_path.read_text()) != contract_feature:
            raise ValueError('Recorded prefix feature contract changed on target resume')
    else:
        onset.write_json(feature_path, contract_feature)
    results = []
    for target in contract['cases']:
        output = args.output / f'case-{target["task_id"]:02d}'
        case_contract = dict(contract_sha256=content_hash(contract), case=target, recipe=RECIPE,
            learner_seed=args.learner_seed, source_commit=args.source_commit, maximum_total_actions=CASE_CAP,
            motor_receipt_sha256=PI05_RECEIPT, libero_commit=onset.LIBERO_COMMIT,
            recorded_feature_contract=contract_feature, prefix_features_are_actor_input=False,
            horizon=400, settle_steps=10, fresh_environment_on_reset=True,
            target_only=True, source_assembly_admission=False)
        index_path = output / 'index.json'
        if index_path.exists() and json.loads(index_path.read_text()).get('completed') is True:
            result = json.loads(index_path.read_text())
            validate_completed(result, case_contract, output)
            results.append(result)
            continue
        if time.monotonic() + 120 >= deadline:
            break
        server = onset.TargetServer(task_id=target['task_id'], python=args.simulator_python,
            pythonpath=args.simulator_pythonpath, output=attempt / f'server-{target["task_id"]}',
            port=args.port, libero_root=args.libero_root)
        try:
            if (server.metadata.get('suite') != 'libero_90' or server.metadata.get('task_id') != target['task_id']
                    or server.metadata.get('horizon') != 400 or server.metadata.get('settle_steps') != 10
                    or server.metadata.get('libero_commit') != onset.LIBERO_COMMIT
                    or server.metadata.get('fresh_environment_on_reset') is not True):
                raise ValueError('Simulator differs from the predeclared native target case')
            output.mkdir(parents=True, exist_ok=True)
            qualification = onset.JournaledSimulator(server, output, args.output, case_cap=CASE_CAP, total_cap=TOTAL_CAP)
            initial = qualification.reset(target['initial_state'], target['scene_seed'])
            zeros = np.zeros((10, 32), np.float32)
            a = motor.infer(initial['observation'], server.metadata['instruction'], zeros)
            b = motor.infer(initial['observation'], server.metadata['instruction'], zeros.copy())
            if motor.horizon != 10 or not np.isfinite(a).all() or not np.allclose(a, b, rtol=0, atol=1e-5):
                raise RuntimeError('Frozen target motor failed native same-noise qualification')
            feature_qualification = motor.feature_encoder.qualify(initial['observation'], server.metadata['instruction'])
            onset.write_json(attempt / f'qualification-{target["task_id"]}.json',
                dict(same_noise_max_delta=float(np.abs(a-b).max()), prefix_features=feature_qualification,
                    nominal_state_sha256=initial['initial_state_vector_sha256'], settling_actions=10,
                    simulator={key: value for key, value in server.metadata.items() if key not in ('pid', 'hostname')}))
            np.random.seed(args.learner_seed)
            random.seed(args.learner_seed)
            learner, sampler = onset.OfficialSAC(args.dsrl_root, args.learner_seed), np.random.default_rng(args.learner_seed)
            result = onset.run_case(server, motor, learner, sampler, target=target, output=output, root=args.output,
                contract=case_contract, resume=(output / 'contract.json').exists(), deadline=deadline,
                milestones=MILESTONES, case_cap=CASE_CAP, total_cap=TOTAL_CAP, record_features=True)
            results.append(result)
            onset.write_json(args.output / 'progress.json', dict(completed_cases=sum(row['completed'] for row in results),
                cases=results, total_action_charge=onset.joint_charge(args.output)))
            if not result['completed']:
                break
        finally:
            server.close()
    value = dict(format=FORMAT, contract=contract, completed=len(results) == len(TASKS) and all(row['completed'] for row in results),
        cases=results, total_action_charge=onset.joint_charge(args.output), maximum_total_actions=TOTAL_CAP,
        target_only=True, source_assembly_admission=False, source_checkpoint_reused=False,
        recorded_feature_contract=contract_feature, prefix_features_are_actor_input=False,
        elapsed_seconds=time.monotonic() - started)
    onset.write_json(args.output / 'index.json', value)


if __name__ == '__main__':
    main()
