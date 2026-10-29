"""Bounded target comparison of one-source-goal AD scales and frozen source SAC.

Target0/1 are predeclared development cases. No deployment optimizer updates.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np

from .vla_history import FrozenHistoryController, file_sha256
from .vla_source40_corpus import PI05_RECEIPT, save_motor_rng
from .vla_source40_schedule import LIBERO_COMMIT, content_hash
from .vla_source40_evaluate import (CostLedger, TargetCase, evaluate_case, load_candidates)
from .vla_libero import FrozenOpenPI, OfficialSAC, SimulatorClient, write_json

MAX_DIAGNOSTIC_ACTIONS = 21046  # General21,056 reserve minus verified prior10 actions.


def comparison_plan():
    groups = []
    for task in (0, 1):
        case = dict(benchmark='libero90', case_id=f'libero90:{task:02d}/state-0/seed-123',
            suite='libero_90', task_id=task, initial_state=0, scene_seed=123)
        variants = [('native', None), ('frozen_source_sac', None)]
        variants += [(f'ad_{mode}', scale) for scale in (25, 100, 500) for mode in ('full', 'static')]
        if task == 0:
            variants.append(('ad_episode_reset', 500))
        groups += [dict(case=case, method=method, updates=updates, primary=True) for method, updates in variants]
    # Additional controls never replace a primary method/case group.
    for method, updates in [('ad_episode_reset', 500), ('tied_gaussian', None)]:
        for task in ((1,) if method == 'ad_episode_reset' else (0, 1)):
            groups.append(dict(case=dict(benchmark='libero90', case_id=f'libero90:{task:02d}/state-0/seed-123',
                suite='libero_90', task_id=task, initial_state=0, scene_seed=123),
                method=method, updates=updates, primary=False))
    return groups


def charged_actions(directory):
    total = 0
    for journal in Path(directory).glob('group-*/physical-journal'):
        costs = CostLedger(journal, max_primitive_steps=MAX_DIAGNOSTIC_ACTIONS, max_settling_steps=MAX_DIAGNOSTIC_ACTIONS).costs()
        total += sum(row[name] for row in costs.values() for name in
            ('primitive_steps', 'uncertain_primitive_steps_ceiling', 'settling_steps', 'uncertain_settling_steps_ceiling'))
    return total


class FrozenSAC:
    public_observation_input = True

    def __init__(self, learner):
        self.learner, self.policy, self.pending = learner, self, False

    def frozen_signature(self):
        from flax import serialization
        return hashlib.sha256(serialization.to_bytes(self.learner.agent._save_dict)).hexdigest()

    def set_episode_seed(self, seed):
        import jax
        self.learner.agent._rng = jax.random.PRNGKey(seed)

    def act(self, observation):
        if self.pending:
            raise RuntimeError('Consume the actual SAC outcome before another action')
        self.pending = True
        return self.learner.action(observation)

    def observe(self, *, executed_noise, reward, done):
        if not self.pending or np.asarray(executed_noise).shape != (32,):
            raise ValueError('Frozen SAC feedback must follow an actual executed noise action')
        self.pending = False

    def snapshot(self):
        return dict(rng=np.asarray(self.learner.agent._rng).copy(), pending=self.pending)

    def restore(self, saved):
        import jax.numpy as jnp
        if saved['pending']:
            raise ValueError('Frozen SAC recovery requires a complete episode boundary')
        self.learner.agent._rng = jnp.asarray(saved['rng'])
        self.pending = False


def load_scales(path, expected_sha256):
    if file_sha256(path) != expected_sha256:
        raise ValueError('Subset scale manifest checksum differs')
    value = json.loads(Path(path).read_text())
    if (value.get('format') != 'openpi_pi05_source40_subset_scales_v1'
            or value.get('completed') is not True or value.get('qualification_subset') is not True
            or value.get('seed') != 0 or {row['updates'] for row in value['scales']} != {25, 100, 500}
            or len(value['scales']) != 3
            or file_sha256(value['assembly_path']) != value['assembly_file_sha256']):
        raise ValueError('Require the explicit one-source subset and fixed25/100/500 update scales')
    return value


def load_sac(value, dsrl_root, motor):
    baseline = value['frozen_source_sac']
    if (file_sha256(baseline['checkpoint_path']) != baseline['checkpoint_sha256']
            or file_sha256(baseline['contract_path']) != baseline['contract_sha256']
            or file_sha256(baseline['index_path']) != baseline['index_sha256']):
        raise ValueError('Frozen source SAC checkpoint/contract checksum differs')
    assembly = json.loads(Path(value['assembly_path']).read_text())
    source = assembly['selected_sources']['train']
    if (len(source) != 1 or source[0]['index_sha256'] != baseline['index_sha256']
            or source[0]['selected_publication']['source_checkpoint_sha256'] != baseline['checkpoint_sha256']):
        raise ValueError('Frozen SAC must be the actual teacher of this one-goal source corpus')
    contract = json.loads(Path(baseline['contract_path']).read_text())
    if (contract.get('source_only') is not True or contract.get('source_data_from_targets') is not False
            or contract.get('partition') != 'train' or contract.get('task', {}).get('task_key') != baseline['task_key']
            or contract.get('motor', {}).get('receipt_sha256') != PI05_RECEIPT
            or contract.get('feature_contract') != motor.feature_encoder.contract):
        raise ValueError('SAC source provenance differs from the frozen OpenPI40 source contract')
    learner = OfficialSAC(dsrl_root, contract['learner_seed'])
    learner.restore(baseline['checkpoint_path'], np.random.default_rng(0), contract)
    if learner.updates != baseline['source_updates']:
        raise ValueError('SAC source update receipt differs')
    return FrozenSAC(learner)


class TargetServer:
    def __init__(self, *, task_id, python, pythonpath, output, port, libero_root):
        actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=libero_root, text=True).strip()
        if actual != LIBERO_COMMIT:
            raise ValueError('Target simulator must use the original pinned LIBERO source')
        mapping = ast.literal_eval(ast.parse((libero_root / 'libero/libero/benchmark/libero_suite_task_map.py').read_text()).body[0].value)
        name = mapping['libero_90'][task_id]
        bddl = libero_root / f'libero/libero/bddl_files/libero_90/{name}.bddl'
        expected_bddl = file_sha256(bddl)
        output.mkdir(parents=True, exist_ok=False)
        self.log, self.process, self.client = (output / 'simulator.log').open('w'), None, None
        try:
            self.process = subprocess.Popen([str(python), '-m', 'fluffy_bassoon.meta_rl.vla_libero',
                '--mode', 'env-server', '--output', str(output), '--port', str(port), '--suite', 'libero_90',
                '--task-id', str(task_id), '--horizon', '400', '--fresh-environment-on-reset'],
                env=dict(os.environ, PYTHONPATH=pythonpath), stdout=self.log, stderr=subprocess.STDOUT)
            started = time.monotonic()
            while not (output / 'simulator-ready.json').exists():
                if self.process.poll() is not None or time.monotonic() - started > 180:
                    raise RuntimeError(f'Target server failed to become ready; see {output}')
                time.sleep(.1)
            self.client = SimulatorClient(port)
            metadata = self.client.metadata
            if (metadata.get('libero_commit') != LIBERO_COMMIT or metadata.get('bddl_sha256') != expected_bddl
                    or metadata.get('horizon') != 400 or metadata.get('settle_steps') != 10):
                raise ValueError('Target server task/assets/horizon differ')
            self.metadata = metadata
        except BaseException:
            self.close()
            raise

    def reset(self, state, seed):
        return self.client.reset(state, seed)

    def step(self, actions):
        return self.client.step(actions)

    def close(self):
        if self.client is not None:
            try:
                self.client.close()
            except (OSError, EOFError):
                pass
            self.client = None
        if self.process is not None:
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)
            self.process = None
        if not self.log.closed:
            self.log.close()


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('output', 'checkpoint', 'openpi-root', 'dsrl-root', 'libero-root', 'simulator-python', 'scales'):
        ap.add_argument('--' + name, type=Path, required=True)
    for name in ('simulator-pythonpath', 'scales-sha256', 'source-commit', 'expected-hostname'):
        ap.add_argument('--' + name, required=True)
    ap.add_argument('--max-total-actions', type=int, default=MAX_DIAGNOSTIC_ACTIONS)
    ap.add_argument('--max-elapsed-seconds', type=float, required=True)
    ap.add_argument('--history-device', default='cuda:0')
    ap.add_argument('--port', type=int, default=8768)
    ap.add_argument('--resume', action='store_true')
    return ap


def main(argv=None):
    args = parser().parse_args(argv)
    if sys.platform != 'linux' or socket.gethostname() != args.expected_hostname:
        raise ValueError('Evaluate only on the explicitly identified allocated Linux host')
    if (not 0 < args.max_total_actions <= MAX_DIAGNOSTIC_ACTIONS or not np.isfinite(args.max_elapsed_seconds)
            or args.max_elapsed_seconds <= 0):
        raise ValueError('Subset target comparison exceeds the explicit remaining diagnostic reservation')
    from fluffy_bassoon.gpu_execution import validate_execution
    validate_execution(os.environ.get('FLUFFY_GPU_EXECUTION', 'baige'), os.environ)
    root = Path(__file__).resolve().parents[3]
    if (subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip() != args.source_commit
            or subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True).strip()):
        raise ValueError('Require the exact clean reviewed evaluation commit')
    deadline = time.monotonic() + args.max_elapsed_seconds
    scales = load_scales(args.scales, args.scales_sha256)
    plan = comparison_plan()
    contract = dict(format='source40_subset_target_comparison_v1', source_commit=args.source_commit,
        scales_sha256=args.scales_sha256, groups=plan, support_budgets=[0, 1],
        maximum_total_actions=args.max_total_actions, maximum_seconds_per_invocation=args.max_elapsed_seconds,
        source_population='One source goal; fixed25/100/500 optimizer-update scales on the same actual corpus',
        target_population='Predeclared LIBERO90 tasks0/1, state0, seed123; no semantic or outcome filtering',
        frozen_source_sac='Same-corpus source-trained pixel SAC; not ordinary target-online DSRL',
        dpt='Trained source candidates retained; DPT robotic comparison deferred in this slice',
        primary_episode_ceiling=51, primary_action_ceiling=20910,
        interpretation='Small development diagnostic, not full40 transfer or a negative test of meta-learning')
    args.output.mkdir(parents=True, exist_ok=args.resume)
    import fcntl
    lock = (args.output / 'run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.resume:
        if json.loads((args.output / 'contract.json').read_text()) != contract:
            raise ValueError('Subset evaluation resume contract differs')
    else:
        write_json(args.output / 'contract.json', contract)
    attempt = args.output / 'attempts' / str(time.time_ns())
    attempt.mkdir(parents=True)
    args.model_config, args.feature_interface = 'pi05_libero', 'vla-prefix'
    motor = FrozenOpenPI(SimpleNamespace(**dict(vars(args), output=attempt)))
    candidates, provenance = {}, {}
    for row in scales['scales']:
        candidates[row['updates']], provenance[row['updates']] = load_candidates(row['index_path'], row['index_sha256'],
            scales['assembly_path'], motor, seed=scales['seed'], device=args.history_device, allow_small_pilot=True)
    sac = load_sac(scales, args.dsrl_root, motor)
    if charged_actions(args.output) > args.max_total_actions:
        raise RuntimeError('Saved evaluation journal already exceeds its immutable physical cap')
    server, connected, completed = None, None, []
    stop_reason = None
    try:
        for ordinal, group in enumerate(plan):
            path = args.output / f'group-{ordinal:03d}'
            final = path / 'index.json'
            if final.exists():
                result = json.loads(final.read_text())
                if (result.get('completed') is not True or result.get('case') != group['case']
                        or result.get('weights_unchanged') is not True
                        or [(row['method'], row['support_budget']) for row in result.get('trials', [])]
                        != [(group['method'], 0), (group['method'], 1)]
                        or result.get('source_provenance', {}).get('optimizer_update_scale') != group['updates']):
                    raise ValueError('Saved group receipt differs from this paired comparison')
                completed.append(dict(group=group, result_path=str(final), result_sha256=file_sha256(final)))
                continue
            if time.monotonic() + 180 > deadline:
                stop_reason = 'invocation_time_reservation'
                break
            local_cost = 0
            if path.exists():
                costs = CostLedger(path / 'physical-journal', max_primitive_steps=MAX_DIAGNOSTIC_ACTIONS,
                    max_settling_steps=MAX_DIAGNOSTIC_ACTIONS).costs()
                local_cost = sum(row[name] for row in costs.values() for name in
                    ('primitive_steps', 'uncertain_primitive_steps_ceiling', 'settling_steps', 'uncertain_settling_steps_ceiling'))
            other_cost = charged_actions(args.output) - local_cost
            group_cap = args.max_total_actions - other_cost
            if not (path / 'contract.json').exists() and group_cap < 3 * 410:
                stop_reason = 'complete_group_physical_reservation'
                break
            case = TargetCase(**group['case'])
            if connected != case.task_id:
                if server is not None:
                    server.close()
                server = TargetServer(task_id=case.task_id, python=args.simulator_python,
                    pythonpath=args.simulator_pythonpath, output=attempt / f'server-{case.task_id}-{time.time_ns()}',
                    port=args.port, libero_root=args.libero_root)
                connected = case.task_id
            selected = {} if group['method'] in ('native', 'tied_gaussian') else (
                {'frozen_source_sac': dict(policy=sac, kind='source_sac', sha256=scales['frozen_source_sac']['checkpoint_sha256'])}
                if group['method'] == 'frozen_source_sac' else candidates[group['updates']])
            source = dict(provenance[25 if group['updates'] is None else group['updates']],
                group_maximum_total_actions=group_cap, optimizer_update_scale=group['updates'],
                frozen_source_sac=scales['frozen_source_sac'], dataset_scale='Same corpus at every optimizer scale',
                method_source_cost_role=('Native/tied controls require zero steering-source interactions; shared corpus costs here are experiment provenance'
                    if group['method'] in ('native', 'tied_gaussian') else
                    'Frozen SAC uses the source training teacher; its source validation and student optimization costs are not required by this baseline'
                    if group['method'] == 'frozen_source_sac' else
                    'AD uses the declared source training/validation corpus and its own cumulative controller optimizer budget'))
            def factory(spec):
                return spec['policy'] if spec.get('kind') == 'source_sac' else FrozenHistoryController(spec['policy'], mode=spec['mode'])
            def reset_motor_rng(seed):
                import jax
                motor.policy._rng = jax.random.key(seed)
            def qualify(observation):
                zero = np.zeros((10, 32), np.float32)
                a, b = motor.infer(observation, server.metadata['instruction'], zero), motor.infer(observation, server.metadata['instruction'], zero)
                if not np.allclose(a, b, rtol=0, atol=1e-5):
                    raise RuntimeError('Target same-noise motor qualification failed')
                write_json(path / 'qualification.json', dict(same_noise_max_delta=float(np.abs(a-b).max()),
                    features=motor.feature_encoder.qualify(observation, server.metadata['instruction'])))
            evaluate_case(server, motor, selected, case=case, output=path, provenance=source,
                max_primitive_steps=group_cap, max_settling_steps=group_cap, support_budgets=(0, 1),
                methods=(group['method'],), controller_factory=factory, reset_motor_rng=reset_motor_rng,
                motor_rng_state=lambda: save_motor_rng(motor.policy._rng), qualify=qualify,
                resume=(path / 'contract.json').exists(), max_elapsed_seconds=args.max_elapsed_seconds,
                external_deadline=deadline)
            completed.append(dict(group=group, result_path=str(final), result_sha256=file_sha256(final)))
            write_json(args.output / 'progress.json', dict(completed_groups=len(completed), next_group=ordinal + 1,
                charged_total_actions=charged_actions(args.output), maximum_total_actions=args.max_total_actions))
    finally:
        if server is not None:
            server.close()
    primary = sum(row['group']['primary'] for row in completed)
    write_json(args.output / 'index.json', dict(contract=contract, completed=primary == 17,
        completed_primary_groups=primary, completed_groups=completed, stop_reason=stop_reason,
        charged_total_actions=charged_actions(args.output),
        support_query_separate=True, deployment_optimizer_updates=0))


if __name__ == '__main__':
    main()
