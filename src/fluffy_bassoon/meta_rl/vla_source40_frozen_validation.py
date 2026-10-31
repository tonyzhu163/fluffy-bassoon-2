"""Common-motor frozen source validation of separately trained PPO policies.

Eight fixed source goals, disjoint physical states40/41, two K4 trials per
policy/goal,3,500 physical actions per slot,56k joint. No optimizer is created.
"""
import argparse
from dataclasses import asdict
import gzip
import json
import math
import os
from pathlib import Path
import pickle
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

from .checkpoint import restore_rng, rng_state
from .vla_history import file_sha256
from .vla_source40_campaign import load_campaign
from .vla_source40_context_probe import policy_signature
from .vla_source40_corpus import PI05_RECEIPT, restore_motor_rng, save_motor_rng
from .vla_source40_multitask_ppo import OwnedSimulator, source_tasks
from .vla_source40_ppo import JournaledSimulator, episode_sink, physical_charge, write_json
from .vla_source40_ppo_lineage import terminal_source
from .vla_source40_schedule import SOURCE_SUITES, content_hash, load_schedule
from .vla_rl2 import RL2ActorCritic, RL2Config, collect_rl2_trial, export_rl2_policy
from .warmstart import atomic_save, checked_load, validate_weights


FORMAT = 'source40_frozen_ppo_validation_v1'
SLOT_CAP, TOTAL_CAP, WORKER_SECONDS = 3500, 56000, 14400
TRIAL_TIME_RESERVE = 360
MODES = ('full', 'episode')


def invocation_deadline(max_elapsed_seconds, stop_unix=None, *, now_unix=None, now_monotonic=None):
    """Keep setup/model loading inside the scheduler slice without changing its contract.

    The cumulative worker ledger remains authoritative. The optional absolute
    stop is invocation-local, so a shorter exact-source retry can retain the
    same scientific contract and finish at a complete trial boundary.
    """
    if (not math.isfinite(max_elapsed_seconds) or not 0 < max_elapsed_seconds <= WORKER_SECONDS
            or (stop_unix is not None and not math.isfinite(stop_unix))):
        raise ValueError('Require a finite invocation stop inside the approved four-hour ceiling')
    wall = time.time() if now_unix is None else now_unix
    monotonic = time.monotonic() if now_monotonic is None else now_monotonic
    remaining = max_elapsed_seconds if stop_unix is None else min(max_elapsed_seconds, max(0., stop_unix - wall))
    return monotonic + remaining


def plan(campaign, schedule):
    keys = [f'{suite}:{task:02d}' for suite in SOURCE_SUITES for task in (0, 1)]
    if campaign['source_validation_task_keys'] != keys:
        raise ValueError('Frozen validation requires the original eight fixed source goals')
    tasks = {row['task_key']: row for row in source_tasks(schedule)}
    slots = []
    for key in keys:
        cases = [dict(initial_state=40 + trial,
            scene_seed=int(content_hash(dict(task=key, validation_trial=trial, seed=2))[:8], 16) % (2**31 - 1))
            for trial in range(2)]
        for mode in MODES:
            selected = {name: tasks[key][name] for name in ('task_key', 'suite', 'task_id', 'bddl_sha256', 'initial_states_sha256')}
            slots.append(dict(task=selected, mode=mode, cases=cases, attempts=4, maximum_total_actions=SLOT_CAP))
    return slots


def source_admission(saved, mode, feature_contract, expected_tasks, *, schedule_sha256):
    contract = saved.get('contract', {})
    source = contract.get('source', {})
    runtime = saved.get('runtime', {})
    cursor = runtime.get('task_cycle', {})
    exposure = cursor.get('exposure', {})
    counters = saved.get('counters', {})
    config = RL2Config(**contract.get('config', {}))
    expected_config = RL2Config(observation_dim=4104, noise_dim=32, width=128, layers=2, heads=4, context_length=300)
    if (saved.get('version') != 1 or saved.get('boundary') != 'between_trials'
            or type(saved.get('iteration')) is not int or saved['iteration'] < 1
            or source.get('format') != 'openpi_pi05_source40_shared_ppo_v1'
            or source.get('source_only') is not True or source.get('source_data_from_targets') is not False
            or source.get('source_task_keys') != expected_tasks or source.get('schedule_sha256') != schedule_sha256
            or source.get('motor', {}).get('receipt_sha256') != PI05_RECEIPT
            or source.get('feature_contract') != feature_contract
            or source.get('maximum_total_actions') != 172000 or source.get('memory_mode') != mode
            or source.get('optimization', {}).get('learning_rate') != 1e-4
            or source.get('optimization', {}).get('epochs') != 2
            or contract.get('mode') != mode or contract.get('attempts') != 4
            or contract.get('train_states') != list(range(40)) or contract.get('heldout_states') != list(range(40, 50))
            or contract.get('optimizer') != dict(name='Adam', learning_rate=1e-4)
            or contract.get('ppo', {}).get('epochs') != 2 or config != expected_config
            or contract.get('resource_budget', {}).get('max_total_actions') != 172000
            or contract.get('resource_budget', {}).get('max_primitive_steps') != 172000
            or cursor.get('format') != 'source40_ppo_task_cursor_v1' or cursor.get('remaining') != 0
            or set(exposure) != set(expected_tasks) or not runtime.get('motor_rng')
            or not saved.get('environment_rng') or not saved.get('rng') or not saved.get('optimizer', {}).get('state')):
        raise ValueError('Require the complete final all40 source PPO recovery, expected architecture/mode and actual optimizer state')
    if (any(row.get('trials_started', 0) < 1 or row.get('episodes') != row['trials_started'] * 4
            or row.get('primitive_steps', 0) < 1 for row in exposure.values())
            or sum(row['trials_started'] for row in exposure.values()) != saved['iteration']
            or sum(row['primitive_steps'] for row in exposure.values()) != counters.get('environment_steps')
            or counters.get('settling_steps') != sum(row['episodes'] for row in exposure.values()) * 10
            or counters.get('optimizer_steps', 0) < saved['iteration'] * 2):
        raise ValueError('Final source recovery lacks complete actual all40 trial/update coverage')
    steps = [float(value['step']) for value in saved['optimizer']['state'].values() if 'step' in value]
    if not steps or any(not math.isfinite(value) or value <= 0 for value in steps):
        raise ValueError('Source optimizer did not actually advance')
    return config


def load_models(manifest_path, expected_sha256, *, motor, tasks, schedule_sha256, output, device):
    if file_sha256(manifest_path) != expected_sha256:
        raise ValueError('Frozen validation input manifest checksum differs')
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest.get('format') != 'source40_frozen_ppo_validation_models_v1' or set(manifest.get('models', {})) != set(MODES):
        raise ValueError('Require separately trained full and episode-reset source models')
    models, provenance = {}, {}
    for mode in MODES:
        spec = manifest['models'][mode]
        if 'terminal_root' not in spec or 'ledger_path' not in spec:
            raise ValueError('Require an explicit successful terminal R3 source lineage, not a partial checkpoint')
        lineage = terminal_source(spec['terminal_root'], mode, spec['ledger_path'])
        if spec != lineage:
            raise ValueError('Frozen source manifest differs from its terminal R3 lineage')
        saved = checked_load(spec['recovery_path'], spec['recovery_sha256'], weights_only=False)
        config = source_admission(saved, mode, motor.feature_encoder.contract, tasks, schedule_sha256=schedule_sha256)
        terminal = json.loads(Path(spec['terminal_index_path']).read_text())
        if (terminal['summary']['updates'] != saved['iteration'] or terminal['summary']['counters'] != saved['counters']
                or terminal['committed_task_exposure'] != saved['runtime']['task_cycle']
                or terminal['source_contract'] != saved['contract']['source']):
            raise ValueError('Final terminal index and complete source recovery differ')
        if file_sha256(spec['worker_time_path']) != spec['worker_time_sha256']:
            raise ValueError('Source worker-time receipt checksum differs')
        clock = json.loads(Path(spec['worker_time_path']).read_text())
        charge = sum(row.get('charged_seconds', row['reserved_seconds']) for row in clock['attempts'])
        if (clock.get('worker_seconds') != 21600 or charge > 21600
                or clock.get('contract', {}).get('source_commit') != saved['contract']['source']['source_commit']
                or clock['contract'].get('memory_mode') != mode or clock['contract'].get('maximum_total_actions') != 172000):
            raise ValueError('Source worker differs from the approved six-hour source contract')
        policy = RL2ActorCritic(config)
        validate_weights(policy, saved['policy'])
        policy.load_state_dict(saved['policy'], strict=True)
        policy = policy.to(device).eval().requires_grad_(False)
        path = Path(output) / f'frozen-{mode}.pt'
        metadata = dict(contract=saved['contract'], updates=saved['iteration'], counters=saved['counters'],
            full_recovery_sha256=spec['recovery_sha256'], terminal_lineage=lineage,
            selection='Fixed final complete source recovery; no validation selection')
        if path.exists():
            exported = checked_load(path, file_sha256(path))
            if exported.get('manifest') != metadata or any(not torch.equal(exported['policy'][key], value.detach().cpu())
                    for key, value in policy.state_dict().items()):
                raise ValueError('Frozen model export differs on exact validation resume')
            digest = file_sha256(path)
        else:
            digest = export_rl2_policy(path, policy, manifest=metadata)
        models[mode] = policy
        provenance[mode] = dict(recovery_path=spec['recovery_path'], recovery_sha256=spec['recovery_sha256'],
            policy_path=str(path), policy_sha256=digest, tensor_sha256=policy_signature(policy),
            source_commit=saved['contract']['source']['source_commit'], source_updates=saved['iteration'],
            source_counters=saved['counters'], reset_identity=saved['contract'].get('reset_identity'),
            terminal_lineage=lineage, source_rpc_cost=lineage['cumulative_physical_cost'],
            source_worker_time_charge_seconds=charge, actual_source_tasks=len(saved['runtime']['task_cycle']['exposure']))
    if asdict(models['full'].config) != asdict(models['episode'].config):
        raise ValueError('Frozen full/reset policy architectures differ')
    if file_sha256(manifest_path) != expected_sha256:
        raise ValueError('Model input manifest changed while loading')
    return models, provenance


def total_charge(output):
    return sum(physical_charge(path / 'execution-journal')['total_action_charge'] for path in Path(output).glob('slot-*'))


class RawSimulator(JournaledSimulator):
    def __init__(self, sim, journal, action_cap):
        super().__init__(sim, journal, action_cap)
        self.observation, self.noise = None, None
    def reset(self, initial_state, seed):
        result = super().reset(initial_state, seed)
        self.observation = result['observation']
        return result
    def step(self, actions):
        path = self.journal / f'raw-rpc-{time.time_ns()}.pkl.gz'
        # The inherited journal owns conservative physical accounting.
        with gzip.open(path, 'xb', compresslevel=1) as stream:
            pickle.dump(dict(observation=self.observation, proposed_actions=actions, executed_noise=self.noise,
                native_initial_noise=None if self.noise is None else np.repeat(self.noise[None], 10, axis=0)), stream, protocol=5)
        outcome = super().step(actions)
        with gzip.open(path.with_name(path.name + '.outcome'), 'xb', compresslevel=1) as stream:
            pickle.dump(outcome, stream, protocol=5)
        self.observation = outcome['observation']
        return outcome


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('output', 'models', 'campaign', 'schedule', 'checkpoint', 'openpi-root', 'simulator-python'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('models-sha256', 'source-commit', 'expected-hostname', 'simulator-pythonpath'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--max-total-actions', type=int, required=True)
    parser.add_argument('--max-elapsed-seconds', type=float, default=14400.)
    parser.add_argument('--invocation-stop-unix', type=float)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--port', type=int, default=8770)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--ppo-source-phase',choices=('legacy-r3','native-batch14'),default='legacy-r3')
    args = parser.parse_args(argv)
    if (sys.platform != 'linux' or socket.gethostname() != args.expected_hostname
            or args.max_total_actions != TOTAL_CAP or not 0 < args.max_elapsed_seconds <= WORKER_SECONDS):
        raise ValueError('Require allocated Linux and the approved56k/4h frozen-validation limits')
    # Includes all subsequent contract/model loading and qualification. The
    # launcher supplies a stop before its outer timeout, not a fresh4h clock.
    deadline = invocation_deadline(args.max_elapsed_seconds, args.invocation_stop_unix)
    started = time.monotonic()
    from fluffy_bassoon.gpu_execution import validate_execution
    validate_execution(os.environ.get('FLUFFY_GPU_EXECUTION', 'baige'), os.environ)
    root = Path(__file__).resolve().parents[3]
    if (subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip() != args.source_commit
            or subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True).strip()):
        raise ValueError('Require the exact clean reviewed frozen-validation commit')
    campaign, schedule = load_campaign(args.campaign), load_schedule(args.schedule)
    slots = plan(campaign, schedule)
    args.output.mkdir(parents=True, exist_ok=args.resume)
    import fcntl
    lock = (args.output / 'run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    attempt = args.output / 'attempts' / str(time.time_ns())
    attempt.mkdir(parents=True)
    from .vla_libero import FrozenOpenPI, run_episode
    motor = FrozenOpenPI(SimpleNamespace(**dict(vars(args), output=attempt,
        model_config='pi05_libero', feature_interface='vla-prefix')))
    if motor.metadata['receipt_sha256'] != PI05_RECEIPT or motor.horizon != 10:
        raise ValueError('Require the common qualified frozen OpenPI40 motor')
    loader=load_models
    if args.ppo_source_phase=='native-batch14':
        from .vla_source40_batch14_admission import load_models as loader
    models, provenance = loader(args.models, args.models_sha256, motor=motor,
        tasks=[row['task_key'] for row in source_tasks(schedule)], schedule_sha256=schedule['schedule_sha256'],
        output=args.output, device=args.device)
    if set(models)!=set(MODES):
        raise ValueError('Frozen source comparison requires separate full and episode source models')
    contract = dict(format=FORMAT, source_commit=args.source_commit, models_sha256=args.models_sha256,
        models=provenance, campaign_sha256=campaign['campaign_sha256'], schedule_sha256=schedule['schedule_sha256'],
        slots=slots, maximum_total_actions=TOTAL_CAP, maximum_seconds_per_invocation=WORKER_SECONDS,
        partition='source_validation', source_only=True, source_data_from_targets=False,
        training_gradient_data=False, optimizer_updates=0, model_selection='Fixed final source recovery before validation',
        physical_states=[40, 41], trials_per_slot=2, attempts_per_trial=4,
        motor=motor.metadata, feature_contract=motor.feature_encoder.contract,
        interpretation='Seen source goals on disjoint physical cases; no target90/Plus/Pro or novel-goal claim')
    recovery = args.output / 'validation-recovery.pt'
    if args.resume:
        if json.loads((args.output / 'contract.json').read_text()) != contract:
            raise ValueError('Frozen-validation resume contract changed')
        progress = torch.load(recovery, map_location='cpu', weights_only=False)
        if progress['contract_sha256'] != content_hash(contract):
            raise ValueError('Frozen-validation recovery is from another contract')
    else:
        write_json(args.output / 'contract.json', contract)
        progress = dict(contract_sha256=content_hash(contract), slot=0, trial=0, receipts=[],
            rng=rng_state(), motor_rng=save_motor_rng(motor.policy._rng))
        atomic_save(recovery, progress)
    server, connected = None, None
    stop = None
    try:
        for ordinal in range(progress['slot'], len(slots)):
            slot = slots[ordinal]
            task, mode = slot['task'], slot['mode']
            output = args.output / f'slot-{ordinal:02d}'
            output.mkdir(exist_ok=True)
            if time.monotonic() + TRIAL_TIME_RESERVE >= deadline:
                stop = 'complete_trial_time_reservation'
                break
            if connected != task['task_key']:
                if server is not None:
                    server.close()
                server = OwnedSimulator(task, python=args.simulator_python, pythonpath=args.simulator_pythonpath,
                    output=attempt / f'server-{ordinal:02d}', port=args.port)
                connected = task['task_key']
            sim = RawSimulator(server, output / 'execution-journal', SLOT_CAP)
            first = sim.reset(slot['cases'][0]['initial_state'], slot['cases'][0]['scene_seed'])['observation']
            zeros = np.zeros((10, 32), np.float32)
            a = motor.infer(first, sim.metadata['instruction'], zeros)
            b = motor.infer(first, sim.metadata['instruction'], zeros)
            if not np.allclose(a, b, rtol=0, atol=1e-5):
                raise RuntimeError('Common-motor same-noise qualification failed')
            write_json(output / f'qualification-{time.time_ns()}.json', dict(same_noise_max_delta=float(np.abs(a-b).max()),
                feature_parity=motor.feature_encoder.qualify(first, sim.metadata['instruction'])))
            restore_rng(progress['rng'])
            motor.policy._rng = restore_motor_rng(progress['motor_rng'])
            policy = models[mode]
            sink = episode_sink(output / 'raw', contract=dict(contract_sha256=content_hash(contract),
                partition='validation', optimizer_updates=0, frozen_policy_sha256=provenance[mode]['policy_sha256'], slot=slot))
            initial_trial = progress['trial'] if ordinal == progress['slot'] else 0
            for trial in range(initial_trial, 2):
                if (physical_charge(sim.journal)['total_action_charge'] + 1640 > SLOT_CAP
                        or total_charge(args.output) + 1640 > TOTAL_CAP
                        or time.monotonic() + TRIAL_TIME_RESERVE >= deadline):
                    stop = 'complete_trial_action_or_time_reservation'
                    break
                physical_case = slot['cases'][trial]
                episode_number = 0
                def runner(selected_sim, selected_motor, choose, **kwargs):
                    nonlocal episode_number
                    seed = int(content_hash(dict(task=task['task_key'], trial=trial, episode=episode_number, seed=2))[:8], 16)
                    torch.manual_seed(seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(seed)
                    import jax
                    selected_motor.policy._rng = jax.random.key(seed)
                    episode_number += 1
                    def recorded_choose(observation):
                        noise = choose(observation)
                        sim.noise = np.asarray(noise).copy()
                        return noise
                    return run_episode(selected_sim, selected_motor, recorded_choose, **kwargs)
                rollout, receipt = collect_rl2_trial(policy, sim, motor,
                    lambda observation: motor.feature_encoder(observation, sim.metadata['instruction']),
                    **physical_case, attempts=4, chunk=10, mode=mode, episode_runner=runner,
                    reset_identity=provenance[mode].get('reset_identity'),
                    episode_sink=lambda **episode: sink(trial=trial + 1,
                        behavior_update=provenance[mode]['source_updates'],
                        behavior_policy_sha256=provenance[mode]['policy_sha256'], **episode))
                with torch.no_grad():
                    replay, _, _ = policy.evaluate_sequence(rollout['observations'], rollout['raw_actions'], rollout['resets'])
                error = float((replay - rollout['log_probs']).abs().max())
                if error > 1e-4 or policy_signature(policy) != provenance[mode]['tensor_sha256']:
                    raise RuntimeError('Frozen policy likelihood replay or parameter identity failed')
                trace = output / f'trial-{trial:02d}-{time.time_ns()}.pt'
                atomic_save(trace, dict(rollout={key: value.cpu() for key, value in rollout.items()}, receipt=receipt,
                    frozen_policy_sha256=provenance[mode]['policy_sha256'], optimizer_updates=0))
                progress['receipts'].append(dict(slot=ordinal, trial=trial, task_key=task['task_key'], mode=mode,
                    physical_case=physical_case, receipt=receipt, replay_error=error,
                    trace=str(trace.relative_to(args.output)), trace_sha256=file_sha256(trace), optimizer_updates=0))
                progress.update(slot=ordinal + int(trial == 1), trial=0 if trial == 1 else trial + 1,
                    rng=rng_state(), motor_rng=save_motor_rng(motor.policy._rng))
                atomic_save(recovery, progress)
                write_json(args.output / 'progress.json', dict(completed_trials=len(progress['receipts']),
                    next_slot=progress['slot'], next_trial=progress['trial'], charged_total_actions=total_charge(args.output),
                    latest_recovery_sha256=file_sha256(recovery)))
            if stop:
                break
    finally:
        if server is not None:
            server.close()
    write_json(args.output / 'index.json', dict(format=FORMAT, contract=contract,
        completed=progress['slot'] == len(slots), receipts=progress['receipts'], stop_reason=stop,
        total_action_charge=total_charge(args.output), slot_costs=[physical_charge(args.output / f'slot-{ordinal:02d}' / 'execution-journal')
            for ordinal in range(len(slots))], latest_recovery_sha256=file_sha256(recovery),
        elapsed_seconds=time.monotonic() - started, optimizer_updates=0,
        invocation_stop_unix=args.invocation_stop_unix,
        weights_unchanged=all(policy_signature(models[mode]) == provenance[mode]['tensor_sha256'] for mode in MODES)))


if __name__ == '__main__':
    main()
