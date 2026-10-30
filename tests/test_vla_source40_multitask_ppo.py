"""Task switching keeps one learner's retries and exposes actual source tasks."""
import copy
import json

import numpy as np
import pytest

from fluffy_bassoon.meta_rl.vla_source40_multitask_ppo import (
    TaskCycleSimulator, source_tasks, validate_task_metadata)
from fluffy_bassoon.meta_rl.vla_source40_schedule import LIBERO_COMMIT, SOURCE_SUITES


def task(suite, number):
    return dict(task_key=f'{suite}:{number:02d}', suite=suite, task_id=number,
        bddl_sha256=f'bddl-{suite}-{number}', initial_states_sha256=f'states-{suite}-{number}')


class Client:
    def __init__(self, selected):
        self.selected, self.closed, self.resets = selected, False, []
        self.metadata = dict(suite=selected['suite'], task_id=selected['task_id'],
            instruction='instruction-' + selected['task_key'], bddl_sha256=selected['bddl_sha256'],
            libero_commit=LIBERO_COMMIT, horizon=400, initial_states=50,
            fresh_environment_on_reset=True, settle_steps=10)

    def reset(self, state, seed):
        self.resets.append((state, seed))
        return dict(observation={}, initial_observation_sha256=f'{state}:{seed}',
            initial_state_vector_sha256=f'{self.selected["task_key"]}:{state}')

    def step(self, actions):
        return dict(primitive_steps=len(actions), done=True, success=True)

    def close(self):
        self.closed = True


def test_all40_source_order_is_fixed_and_rejects_missing_or_target_tasks():
    tasks = [task(suite, number) for suite in SOURCE_SUITES for number in range(10)]
    assert source_tasks(dict(tasks=list(reversed(tasks)))) == tasks
    with pytest.raises(ValueError, match='all40'):
        source_tasks(dict(tasks=tasks[:-1]))
    tasks[0] = task('libero_90', 0)
    with pytest.raises(ValueError, match='all40'):
        source_tasks(dict(tasks=tasks))


def test_qualification_and_four_retries_do_not_advance_to_another_task(tmp_path):
    tasks = [task('libero_spatial', 0), task('libero_object', 1)]
    clients = []
    def factory(selected):
        client = Client(selected)
        clients.append(client)
        return client
    sim = TaskCycleSimulator(tasks, 4, factory, receipts=tmp_path)
    metadata = sim.metadata
    sim.reset(0, 3)
    assert sim.cursor == 0 and sim.state_dict()['exposure'][tasks[0]['task_key']]['episodes'] == 0
    sim.begin_collection()
    for attempt in range(4):
        result = sim.reset(2, 17)
        sim.step(np.zeros((3, 7)))
        assert result['source_task_key'] == tasks[0]['task_key']
        assert sim.cursor == 1 and sim.remaining == 3 - attempt
    assert len(clients) == 5
    assert all(client.closed for client in clients[:-1])
    sim.reset(1, 18)
    assert clients[0].closed and len(clients) == 6
    assert sim.metadata is metadata and metadata['instruction'] == 'instruction-' + tasks[1]['task_key']
    receipts = [json.loads(path.read_text()) for path in tmp_path.glob('reset-*.json')]
    assert len(receipts) == 6 and all(row['returned'] for row in receipts)
    assert all('bddl_sha256' in row and 'initial_state_vector_sha256' in row for row in receipts)
    sim.close()
    assert clients[-1].closed


def test_task_cursor_restores_after_qualification_and_reuses_the_next_task():
    tasks = [task('libero_spatial', 0), task('libero_object', 1)]
    original = TaskCycleSimulator(tasks, 2, Client)
    original.begin_collection()
    original.reset(1, 9)
    original.reset(1, 9)
    saved = original.state_dict()
    original.close()
    resumed = TaskCycleSimulator(tasks, 2, Client)
    resumed.reset(0, 0)  # Requalification costs actions but never shifts the task cursor.
    resumed.load_state_dict(saved)
    assert resumed.state_dict() == saved
    resumed.reset(3, 8)
    assert resumed.metadata['task_id'] == 1 and resumed.metadata['suite'] == 'libero_object'
    assert resumed.cursor == 2
    resumed.close()


def test_changed_case_inside_trial_and_partial_cursor_recovery_are_rejected():
    sim = TaskCycleSimulator([task('libero_goal', 0)], 4, Client)
    sim.begin_collection()
    sim.reset(0, 4)
    with pytest.raises(ValueError, match='same state/seed'):
        sim.reset(1, 4)
    partial = copy.deepcopy(sim.state_dict())
    with pytest.raises(ValueError, match='complete-trial'):
        sim.load_state_dict(partial)
    sim.close()


def test_original_simulator_metadata_is_bound_to_the_expected_task():
    selected = task('libero_spatial', 0)
    metadata = Client(selected).metadata
    validate_task_metadata(selected, metadata)
    for changes in [dict(suite='libero_90'), dict(bddl_sha256='other'),
                    dict(fresh_environment_on_reset=False), dict(settle_steps=0)]:
        with pytest.raises(ValueError, match='declared original'):
            validate_task_metadata(selected, dict(metadata, **changes))


def test_retries_each_follow_first_reset_path_of_a_fresh_owned_process():
    class ProcessHistoryClient(Client):
        def reset(self, state, seed):
            result = super().reset(state, seed)
            result['initial_observation_sha256'] = 'first-process-reset' if len(self.resets) == 1 else 'different-later-reset'
            return result
    clients = []
    def factory(selected):
        client = ProcessHistoryClient(selected)
        clients.append(client)
        return client
    sim = TaskCycleSimulator([task('libero_spatial', 0)], 4, factory)
    sim.begin_collection()
    hashes = [sim.reset(19, 1916071045)['initial_observation_sha256'] for _ in range(4)]
    assert hashes == ['first-process-reset'] * 4
    assert sim.cursor == 1 and sim.remaining == 0
    assert len(clients) == 5 and all(client.closed for client in clients[:-1])
    sim.close()
