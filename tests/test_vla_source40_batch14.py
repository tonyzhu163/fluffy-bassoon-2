"""Native source-only planning and complete-trial cursor qualification."""
import copy

import numpy as np
import pytest

from fluffy_bassoon.meta_rl.vla_source40_batch14 import (
    NATIVE_HORIZONS, NativeTaskCycle, batch_plan, native_source_tasks, validate_native_metadata)
from fluffy_bassoon.meta_rl.vla_source40_schedule import LIBERO_COMMIT, SOURCE_SUITES


def task(suite, number):
    return dict(task_key=f'{suite}:{number:02d}', suite=suite, task_id=number,
        bddl_sha256=f'bddl-{suite}-{number}', initial_states_sha256=f'states-{suite}-{number}')


def tasks():
    return native_source_tasks(dict(tasks=[task(suite, number) for suite in SOURCE_SUITES for number in range(10)]))


class Client:
    def __init__(self, selected):
        self.selected, self.closed = selected, False
        self.metadata = dict(suite=selected['suite'], task_id=selected['task_id'],
            instruction=selected['task_key'], bddl_sha256=selected['bddl_sha256'], libero_commit=LIBERO_COMMIT,
            horizon=NATIVE_HORIZONS[selected['suite']], initial_states=50, fresh_environment_on_reset=True, settle_steps=10)

    def reset(self, state, seed):
        return dict(observation={}, initial_observation_sha256=f'{state}:{seed}',
            initial_state_vector_sha256=f'{self.selected["task_key"]}:{state}')

    def step(self, actions):
        return dict(primitive_steps=len(actions), done=True, success=True)

    def close(self):
        self.closed = True


def test_all40_interleaved_and_no_target_admission():
    selected = tasks()
    assert [row['suite'] for row in selected[:4]] == list(SOURCE_SUITES)
    assert [row['task_id'] for row in selected] == [number for number in range(10) for _ in SOURCE_SUITES]
    changed = copy.deepcopy(selected)
    changed[0] = task('libero_90', 0)
    with pytest.raises(ValueError, match='all40'):
        native_source_tasks(dict(tasks=changed))


def test_plans_are_source_only_outcome_independent_and_have_native_caps():
    selected = tasks()
    first = batch_plan(selected, 0, np.random.default_rng(7))
    assert first == batch_plan(selected, 0, np.random.default_rng(7))
    assert len(first) == 14 and len({row['suite'] for row in first}) == 4
    assert all(0 <= row['initial_state'] < 40 and row['maximum_actions'] == 4 * (row['horizon'] + 10) for row in first)
    assert sum(row['maximum_actions'] for row in first) + 10 <= 20_000
    assert NATIVE_HORIZONS == dict(libero_spatial=220, libero_object=280, libero_goal=300, libero_10=520)
    #100 updates is a target, never a promise under a1M physical ceiling.
    all_plans = [row for cursor in range(0, 1400, 14)
        for row in batch_plan(selected, cursor, np.random.default_rng(cursor))]
    assert sum(row['maximum_actions'] for row in all_plans) == 1_904_000
    assert all(sum(row['task_key'] == task['task_key'] for row in all_plans) == 35 for task in selected)
    assert all({row['initial_state'] for row in all_plans if row['task_key'] == task['task_key']} == set(range(35))
        for task in selected)


def test_native_metadata_rejects_shortened_long_tasks_and_wrong_source():
    selected = task('libero_10', 0)
    metadata = Client(selected).metadata
    validate_native_metadata(selected, metadata)
    for change in (dict(horizon=400), dict(suite='libero_90'), dict(initial_states=40),
            dict(fresh_environment_on_reset=False), dict(libero_commit='other')):
        with pytest.raises(ValueError, match='native-horizon'):
            validate_native_metadata(selected, dict(metadata, **change))


def test_task_cursor_only_advances_after_fixed_k4_and_survives_recovery():
    selected, clients = tasks(), []
    def factory(row):
        client = Client(row)
        clients.append(client)
        return client
    cycle = NativeTaskCycle(selected, 4, factory)
    cycle.reset(0, 7)  # Qualification never changes the committed task cycle.
    assert cycle.cursor == 0
    cycle.begin_collection()
    plan = batch_plan(selected, 0, np.random.default_rng(7))
    for descriptor in plan[:3]:
        for episode in range(4):
            result = cycle.reset(descriptor['initial_state'], descriptor['scene_seed'])
            assert result['source_task_key'] == descriptor['task_key']
            assert cycle.metadata['horizon'] == descriptor['horizon']
            cycle.step(np.zeros((2, 7)))
        assert cycle.remaining == 0
    saved = cycle.state_dict()
    assert saved['cursor'] == 3
    assert sum(row['episodes'] for row in saved['exposure'].values()) == 12
    cycle.close()
    resumed = NativeTaskCycle(selected, 4, factory)
    resumed.reset(0, 7)
    resumed.load_state_dict(saved)
    descriptor = plan[3]
    resumed.reset(descriptor['initial_state'], descriptor['scene_seed'])
    assert resumed.metadata['suite'] == 'libero_10' and resumed.metadata['horizon'] == 520
    assert all(client.closed for client in clients[:-1])
    resumed.close()
