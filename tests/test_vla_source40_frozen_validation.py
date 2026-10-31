"""Frozen source validation admission and accounting, on allocated Linux only."""
from copy import deepcopy
from dataclasses import asdict
import gzip
import json
import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import fluffy_bassoon.meta_rl.vla_source40_frozen_validation as module
from fluffy_bassoon.meta_rl.vla_source40_schedule import SOURCE_SUITES
from fluffy_bassoon.meta_rl.vla_rl2 import RL2Config


def tasks():
    return [dict(task_key=f'{suite}:{number:02d}', suite=suite, task_id=number,
        bddl_sha256=f'bddl-{suite}-{number}', initial_states_sha256=f'states-{suite}-{number}')
        for suite in SOURCE_SUITES for number in range(10)]


def source(mode='full'):
    keys = [row['task_key'] for row in tasks()]
    feature = dict(kind='qualified-prefix', observation_dim=4104)
    shared = dict(format='openpi_pi05_source40_shared_ppo_v1', source_only=True, source_data_from_targets=False,
        source_task_keys=keys, motor=dict(receipt_sha256=module.PI05_RECEIPT), feature_contract=feature,
        maximum_total_actions=172000, memory_mode=mode, source_commit='a' * 40, schedule_sha256='source-plan-sha',
        optimization=dict(learning_rate=1e-4, epochs=2))
    contract = dict(source=shared, config=asdict(RL2Config(observation_dim=4104)), mode=mode,
        attempts=4, train_states=list(range(40)), heldout_states=list(range(40, 50)),
        optimizer=dict(name='Adam', learning_rate=1e-4), ppo=dict(epochs=2),
        resource_budget=dict(max_total_actions=172000, max_primitive_steps=172000))
    payload = dict(version=1, boundary='between_trials', contract=contract, iteration=40,
        counters=dict(environment_steps=16000, settling_steps=1600, optimizer_steps=80),
        runtime=dict(motor_rng=dict(format='qualified-key'), task_cycle=dict(format='source40_ppo_task_cursor_v1',
            remaining=0, exposure={key: dict(episodes=4, primitive_steps=400, trials_started=1) for key in keys})),
        environment_rng=dict(position=10), rng=dict(torch='saved'),
        optimizer=dict(state={0: dict(step=torch.tensor(80.))}), policy=dict(weight=torch.ones(3)))
    return payload, feature, keys


def test_fixed_eight_goals_paired_disjoint_cases_and_joint_budget():
    selected = [f'{suite}:{number:02d}' for suite in SOURCE_SUITES for number in (0, 1)]
    slots = module.plan(dict(source_validation_task_keys=selected), dict(tasks=tasks()))
    assert len(slots) == 16 and sum(row['maximum_total_actions'] for row in slots) == 56000
    for full, reset in zip(slots[::2], slots[1::2], strict=True):
        assert full['mode'] == 'full' and reset['mode'] == 'episode'
        assert full['cases'] == reset['cases']
        assert [row['initial_state'] for row in full['cases']] == [40, 41]
        assert full['task'] == reset['task']
        assert 2 * 4 * 410 + 10 <= full['maximum_total_actions'] == 3500
    with pytest.raises(ValueError, match='fixed source goals'):
        module.plan(dict(source_validation_task_keys=selected[::-1]), dict(tasks=tasks()))


def test_invocation_stop_counts_loading_and_is_independent_of_resume_contract():
    assert module.invocation_deadline(14400, 1600, now_unix=1000, now_monotonic=50) == 650
    assert module.invocation_deadline(14400, 20000, now_unix=1000, now_monotonic=50) == 14450
    assert module.invocation_deadline(14400, 900, now_unix=1000, now_monotonic=50) == 50
    assert module.invocation_deadline(14400, now_unix=1000, now_monotonic=50) == 14450
    assert module.TRIAL_TIME_RESERVE == 360
    for elapsed, stop in ((14401, None), (0, None), (14400, float('nan'))):
        with pytest.raises(ValueError, match='four-hour ceiling'):
            module.invocation_deadline(elapsed, stop)


def test_final_source_requires_real_all40_coverage_mode_and_optimizer_advancement():
    payload, feature, keys = source()
    assert module.source_admission(payload, 'full', feature, keys, schedule_sha256='source-plan-sha').observation_dim == 4104
    for mutation in ('missing_task', 'no_updates', 'partial_trial', 'wrong_mode', 'wrong_features', 'larger_cap'):
        changed = deepcopy(payload)
        if mutation == 'missing_task':
            changed['runtime']['task_cycle']['exposure'][keys[-1]]['trials_started'] = 0
        elif mutation == 'no_updates':
            changed['optimizer']['state'][0]['step'] = torch.tensor(0.)
        elif mutation == 'partial_trial':
            changed['runtime']['task_cycle']['remaining'] = 1
        elif mutation == 'wrong_mode':
            changed['contract']['source']['memory_mode'] = 'episode'
        elif mutation == 'wrong_features':
            changed['contract']['source']['feature_contract'] = dict(old='RLinf130')
        else:
            changed['contract']['source']['maximum_total_actions'] = 200000
        with pytest.raises(ValueError):
            module.source_admission(changed, 'full', feature, keys, schedule_sha256='source-plan-sha')


def test_hash_bound_final_recoveries_export_immutable_frozen_models(tmp_path, monkeypatch):
    class TinyPolicy(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.weight = torch.nn.Parameter(torch.zeros(3))
    monkeypatch.setattr(module, 'RL2ActorCritic', TinyPolicy)
    models = {}
    for mode in module.MODES:
        payload, feature, keys = source(mode)
        path = tmp_path / f'{mode}-latest.pt'
        torch.save(payload, path)
        clock = tmp_path / f'{mode}-worker-time.json'
        clock.write_text(json.dumps(dict(worker_seconds=21600, attempts=[dict(charged_seconds=100, reserved_seconds=660)],
            contract=dict(source_commit='a' * 40, memory_mode=mode, maximum_total_actions=172000))))
        models[mode] = dict(recovery_path=str(path), recovery_sha256=module.file_sha256(path),
            worker_time_path=str(clock), worker_time_sha256=module.file_sha256(clock),
            terminal_root=str(tmp_path / mode), ledger_path=str(tmp_path / 'ledger.json'),
            cumulative_physical_cost=dict(total_action_charge=17600))
        terminal_index = tmp_path / f'{mode}-terminal-index.json'
        terminal_index.write_text(json.dumps(dict(summary=dict(updates=payload['iteration'], counters=payload['counters']),
            committed_task_exposure=payload['runtime']['task_cycle'], source_contract=payload['contract']['source'])))
        models[mode]['terminal_index_path'] = str(terminal_index)
    # This unit exercises model/recovery binding; full R1→R2→R3 terminal
    # metadata admission has its own model-free lineage checks.
    monkeypatch.setattr(module, 'terminal_source', lambda root, mode, ledger: models[mode])
    manifest = tmp_path / 'models.json'
    manifest.write_text(json.dumps(dict(format='source40_frozen_ppo_validation_models_v1', models=models)))
    output = tmp_path / 'validation'
    output.mkdir()
    motor = SimpleNamespace(feature_encoder=SimpleNamespace(contract=feature))
    policies, provenance = module.load_models(manifest, module.file_sha256(manifest), motor=motor, tasks=keys, schedule_sha256='source-plan-sha', output=output, device='cpu')
    assert all(not any(value.requires_grad for value in policy.parameters()) for policy in policies.values())
    _, again = module.load_models(manifest, module.file_sha256(manifest), motor=motor, tasks=keys, schedule_sha256='source-plan-sha', output=output, device='cpu')
    assert provenance == again
    for mode in module.MODES:
        assert provenance[mode]['source_updates'] == 40 and provenance[mode]['actual_source_tasks'] == 40
        assert (output / f'frozen-{mode}.pt').exists()
    (tmp_path / 'full-latest.pt').write_bytes(b'changed')
    with pytest.raises(ValueError, match='checksum'):
        module.load_models(manifest, module.file_sha256(manifest), motor=motor, tasks=keys, schedule_sha256='source-plan-sha', output=output, device='cpu')


def test_raw_validation_rpc_preserves_actual_inputs_outcomes_and_costs(tmp_path):
    class Simulator:
        metadata = dict(settle_steps=10)
        def reset(self, state, seed):
            return dict(observation=dict(image=np.ones((2, 2, 3), np.uint8)),
                initial_state_vector_sha256='state40', initial_observation_sha256='rgb')
        def step(self, actions):
            return dict(observation=dict(state=np.zeros(8, np.float32)), primitive_steps=len(actions),
                executed_actions=actions, rewards=[0.] * len(actions), done=True, success=False)
    sim = module.RawSimulator(Simulator(), tmp_path / 'execution-journal', 3500)
    sim.reset(40, 123)
    sim.noise = np.arange(32, dtype=np.float32) / 100
    sim.step(np.zeros((10, 7), np.float32))
    assert module.physical_charge(sim.journal)['total_action_charge'] == 20
    inputs = next(sim.journal.glob('raw-rpc-*.pkl.gz'))
    with gzip.open(inputs, 'rb') as stream:
        data = pickle.load(stream)
    np.testing.assert_array_equal(data['native_initial_noise'][0], sim.noise)
    assert data['observation']['image'].shape == (2, 2, 3)
    assert inputs.with_name(inputs.name + '.outcome').exists()
