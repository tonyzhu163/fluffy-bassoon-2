"""File-only immutable repair preservation; run inside allocated Linux preflight."""
from copy import deepcopy
import json
import random

import numpy as np
import pytest
import torch

from fluffy_bassoon.meta_rl.vla_source40_ppo_migration import prepare, validate, sha, fingerprint
from fluffy_bassoon.meta_rl.vla_source40_reset_probe import compare_resets


@pytest.mark.parametrize('mode,seconds', [('full', 1738), ('episode', 1739)])
def test_migration_preserves_all_numerical_state_journals_and_original_budget(tmp_path, mode, seconds):
    old, new = tmp_path / ('old-' + mode), tmp_path / ('new-' + mode)
    learner = old / 'experiment/learner'
    learner.mkdir(parents=True)
    (old / 'launcher.lock').touch()
    source = dict(format='openpi_pi05_source40_shared_ppo_v1', source_only=True, source_data_from_targets=False,
        source_commit='a' * 40, maximum_total_actions=172000, source_task_keys=['task-' + str(i) for i in range(40)])
    sampler = np.random.default_rng(2)
    saved = dict(version=1, boundary='between_trials', iteration=40,
        contract=dict(source=source, mode=mode, train_states=list(range(40))),
        counters=dict(environment_steps=100, optimizer_steps=80), policy=dict(weight=torch.arange(4.)),
        optimizer=dict(state={0: dict(step=torch.tensor(80.), exp_avg=torch.arange(4.))}),
        environment_rng=sampler.bit_generator.state,
        rng=dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state()),
        runtime=dict(motor_rng=dict(key_data=np.asarray([1, 2], np.uint32)),
            task_cycle=dict(cursor=40, remaining=0, active_task=39, exposure={'preserve': [1, 2, 3]})))
    torch.save(saved, learner / 'latest.pt')
    before = sha(learner / 'latest.pt')
    (old / 'experiment/source-contract.json').write_text(json.dumps(source))
    clock = dict(worker_seconds=21600, contract=dict(source_commit='a' * 40, run_id=old.name,
        maximum_total_actions=172000, memory_mode=mode),
        attempts=[dict(status='finished', charged_seconds=seconds, reserved_seconds=21540, path=str(old / 'attempt'))])
    (old / 'worker-time.json').write_text(json.dumps(clock))
    journal = old / 'experiment/execution-journal'
    journal.mkdir()
    for name, row in {'intent-1': dict(maximum_primitive_steps=10), 'step-1': dict(primitive_steps=10),
                      'intent-2': dict(maximum_primitive_steps=7),
                      'reset-a': dict(settling_steps=10, returned=True),
                      'reset-b': dict(settling_steps=10, returned=True)}.items():
        (journal / (name + '.json')).write_text(json.dumps(row))
    (old / 'experiment/task-resets').mkdir()
    (old / 'experiment/task-resets/reset.json').write_text('{}')
    for folder in ('raw', 'learner/traces'):
        path = old / 'experiment' / folder
        path.mkdir(exist_ok=True)
        (path / 'original-trial41-evidence').write_bytes(b'preserve failed and completed artifacts')
    ledger = tmp_path / 'ledger.json'
    record = dict(run_id=old.name, source_commit='a' * 40, maximum_total_actions=172000)
    ledger.write_text(json.dumps(dict(reservations={old.name: record})))
    result = prepare(old, new, 'b' * 40, ledger, old.name)
    assert result['carried_time_seconds'] == seconds and result['carried_physical_cost']['total_action_charge'] == 37
    assert result['state_sha256_before'] == result['state_sha256_after']
    assert sha(learner / 'latest.pt') == before
    actual = torch.load(new / 'experiment/learner/latest.pt', weights_only=False)
    assert actual['contract']['source']['source_commit'] == 'b' * 40
    assert fingerprint({k: v for k, v in saved.items() if k != 'contract'}) == fingerprint({k: v for k, v in actual.items() if k != 'contract'})
    carried_clock = json.loads((new / 'worker-time.json').read_text())
    assert carried_clock['attempts'] == clock['attempts'] and carried_clock['worker_seconds'] == 21600
    assert json.loads(ledger.read_text())['reservations'] == {old.name: record}
    assert (old / 'experiment/raw/original-trial41-evidence').exists()
    assert not (new / 'experiment/raw').exists()  # No writable link back into r1.
    validate(new / 'migration.json', output_root=new, source_commit='b' * 40,
        action_cap=172000, mode=mode, ledger_path=ledger)
    with pytest.raises(ValueError, match='fresh r2'):
        prepare(old, new, 'b' * 40, ledger, old.name)
    # The failed r2 probe is part of the SAME source/time/action lineage.
    (new / 'launcher.lock').touch()
    clock2 = json.loads((new / 'worker-time.json').read_text())
    clock2['attempts'].append(dict(status='finished', charged_seconds=187, reserved_seconds=19800, path=str(new / 'probe-attempt')))
    (new / 'worker-time.json').write_text(json.dumps(clock2))
    for number in range(4):
        (new / f'experiment/execution-journal/reset-probe-{number}.json').write_text(json.dumps(dict(settling_steps=10, returned=True)))
    probe = new / 'attempts/probe'
    probe.mkdir(parents=True)
    (probe / 'actual-reset-arrays.pkl.gz').write_bytes(b'preserved r2 arrays')
    third = tmp_path / ('r3-' + mode)
    chained = prepare(new, third, 'c' * 40, ledger, old.name, camera_max_lsb=1)
    assert chained['carried_time_seconds'] == seconds + 187
    assert chained['carried_physical_cost']['total_action_charge'] == 77
    assert chained['original_reservation_id'] == old.name
    assert chained['parent_source_commit'] == 'b' * 40 and chained['original_source_commit'] == 'a' * 40
    assert chained['state_sha256_before'] == result['state_sha256_before']
    assert chained['camera_max_uint8_lsb'] == 1
    assert json.loads(ledger.read_text())['reservations'] == {old.name: record}
    catalog = json.loads((third / 'prior-artifacts.json').read_text())['files']
    assert any(row['path'] == str(probe / 'actual-reset-arrays.pkl.gz') for row in catalog)
    assert any(row['path'] == str(old / 'experiment/raw/original-trial41-evidence') for row in catalog)
    validate(third / 'migration.json', output_root=third, source_commit='c' * 40,
        action_cap=172000, mode=mode, ledger_path=ledger, camera_max_lsb=1)



def test_reset_probe_requires_exact_physics_and_public_observations():
    result = dict(initial_state_vector_sha256='nominal', initial_simulator_state_sha256='actual',
        initial_observation_sha256='observation', initial_simulator_state=np.arange(3.),
        observation={key: np.zeros((2, 2), np.float32) for key in ('image', 'wrist', 'state')})
    values = [deepcopy(result) for _ in range(4)]
    assert compare_resets(values)['exact_actual_simulator_states']
    values[1]['observation']['image'][0, 0] = 1.
    with pytest.raises(RuntimeError, match='exact public observation'):
        compare_resets(values)
