"""Stage A headroom screen plan/selection/control flow; allocated Linux CPU fixtures only."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from fluffy_bassoon.meta_rl import vla_headroom_screen as screen


SOURCE, RUN = 'a' * 40, 'libero90-pi05-headroom-screen-fixture'


class Simulator:
    def __init__(self, task, succeed=lambda state, call: False):
        self.metadata = dict(instruction='put it there', task_id=task, suite='libero_90', horizon=400,
                             settle_steps=10, bddl_file='task.bddl')
        self.succeed, self.resets, self.steps = succeed, [], 0
    def observation(self):
        return dict(image=np.zeros((64, 64, 3), np.uint8), wrist=np.zeros((64, 64, 3), np.uint8),
                    state=np.zeros(8, np.float32))
    def reset(self, state, seed):
        self.resets.append((state, seed))
        self.state, self.elapsed = state, 0
        return dict(observation=self.observation(), initial_state_vector_sha256=f'state-{state}',
                    initial_observation_sha256=f'rgb-{state}')
    def step(self, actions):
        self.steps += 1
        self.elapsed += len(actions)
        success = self.elapsed >= 50 and self.succeed(self.state, len(self.resets))
        return dict(observation=self.observation(), primitive_steps=len(actions),
            executed_actions=np.asarray(actions), rewards=[float(success)] * len(actions),
            success=success, done=success or self.elapsed == 400)


class Motor:
    horizon, feature_encoder = 10, None
    def __init__(self):
        self.policy = SimpleNamespace(_rng=None)
        self.noises = []
    def infer(self, observation, instruction, noise):
        self.noises.append(np.asarray(noise).copy())
        return np.zeros((10, 7), np.float32)


def seed_motor(motor, seed):
    motor.policy._rng = seed


def grant(tmp_path, **changes):
    value = dict(format='pi05_libero90_headroom_screen_grant_v1', approved=True, source_commit=SOURCE,
        run_id=RUN, maximum_gpu_seconds=10800, maximum_total_actions=170000,
        maximum_output_bytes=20 * 1024 ** 3, tasks=list(screen.TASKS), episodes=384,
        motor_receipt_sha256=screen.PI05_RECEIPT, motor_parameter_sha256=screen.MOTOR_PARAMETER_SHA256,
        data_role='target-headroom-screen')
    value.update(changes)
    path = tmp_path / 'grant.json'
    path.write_text(json.dumps(value))
    return path, screen.onset.sha256(path)


def records(native, tied):
    """Synthetic completed records with given successes per task out of 16."""
    rows = []
    for entry in screen.episode_plan():
        counts = (native if entry['arm'] == 'native' else tied).get(entry['task_id'], 0)
        index = (entry['initial_state'] - 1) * 2 + entry['repeat']
        rows.append(dict(entry, success=index < counts))
    return rows


def test_plan_is_closed_task_major_and_deterministic():
    plan = screen.episode_plan()
    assert len(plan) == 384 == 12 * 8 * 2 * 2
    assert [entry['task_id'] for entry in plan[::32]] == list(screen.TASKS)
    assert {entry['initial_state'] for entry in plan} == set(range(1, 9))
    assert len({entry['identity'] for entry in plan}) == len({entry['episode_seed'] for entry in plan}) == 384
    assert plan == screen.episode_plan()
    assert [entry['arm'] for entry in plan[:4]] == ['native', 'tied', 'native', 'tied']
    assert 57 not in screen.TASKS and 40 not in screen.TASKS


@pytest.mark.parametrize('task,state', [(57, 1), (40, 1), (0, 0), (0, 40), (0, 49), (0, 9)])
def test_admission_refuses_unlisted_tasks_and_reserved_states(task, state):
    with pytest.raises(ValueError):
        screen.admit(task, state)


def test_contract_binds_frozen_motor_role_and_no_learner(tmp_path):
    path, digest = grant(tmp_path)
    screen.load_grant(path, digest, SOURCE, RUN)
    value = screen.scientific_contract(SOURCE, digest)
    assert value['episodes'] == 384 and value['optimizer_updates'] == 0 and value['learner'] is None
    assert value['data_role'] == 'target-headroom-screen' and not value['source_assembly_admission']
    assert value['entire_motor_frozen'] and value['native_horizon'] == value['execute_steps'] == 10
    assert value['forbidden_states'] == [0, *range(40, 50)]
    assert value['plan_sha256'] == screen.content_hash(screen.episode_plan())


@pytest.mark.parametrize('changes', [dict(approved=False), dict(maximum_gpu_seconds=10801),
    dict(maximum_total_actions=170001), dict(tasks=[57]), dict(episodes=400),
    dict(data_role='source-training'), dict(source_commit='b' * 40)])
def test_grant_cannot_expand_or_switch_scope(tmp_path, changes):
    path, digest = grant(tmp_path, **changes)
    with pytest.raises(ValueError, match='funding record'):
        screen.load_grant(path, digest, SOURCE, RUN)


def test_clock_charges_retries_against_one_ceiling(tmp_path):
    path, digest = grant(tmp_path)
    root = tmp_path / 'worker'
    first = screen.prepare_attempt(root, SOURCE, RUN, path, digest, started_unix=1000.)
    assert first['slice_seconds'] == 10740 and first['driver_stop_unix'] == first['stop_unix'] - 180
    screen.finish(root, first['attempt'], 1, now=1100.)
    second = screen.prepare_attempt(root, SOURCE, RUN, path, digest, resume=True, started_unix=2000.)
    assert second['prior_charged_seconds'] == 160 and second['slice_seconds'] == 10800 - 160 - 60
    with pytest.raises(ValueError, match='exact headroom screen'):
        screen.prepare_attempt(tmp_path / 'worker', SOURCE, RUN, path, digest, started_unix=3000.)


def test_selection_picks_closest_to_twenty_percent_with_tie_breaks():
    native = {0: 3, 29: 4, 33: 2, 38: 10, 44: 1, 53: 4}
    tied = {0: 2, 29: 5, 33: 9, 38: 8, 44: 8, 53: 6}
    result = screen.select(records(native, tied))
    # 3/16=.1875 is closest to .2; 4/16 ties 29 and 53, broken by more tied successes.
    assert result['status'] == 'selected' and result['selected_task_id'] == 0
    assert result['eligible_task_ids'] == [0, 53, 29, 33]
    assert result['runner_up_task_id'] == 53


def test_selection_requires_tied_reachability_and_reports_stop():
    result = screen.select(records({task: 4 for task in screen.TASKS}, {task: 1 for task in screen.TASKS}))
    assert result['status'] == 'stop_no_eligible_task' and result['selected_task_id'] is None
    assert len(result['rates']) == 12 and result['rates'][0]['native_successes'] == 4


def test_selection_band_edges_are_inclusive():
    result = screen.select(records({0: 2, 29: 9, 33: 10, 38: 1}, {0: 2, 29: 2, 33: 2, 38: 2}))
    assert result['eligible_task_ids'] == [0, 29]


def test_selection_rejects_incomplete_or_duplicate_records():
    rows = records({}, {})
    with pytest.raises(ValueError, match='exactly once'):
        screen.select(rows[:-1])
    with pytest.raises(ValueError, match='exactly once'):
        screen.select(rows + rows[:1])


def test_run_screen_arms_journal_resume_and_report(tmp_path):
    task = screen.TASKS[0]
    sim, motor = Simulator(task, succeed=lambda state, call: state == 1), Motor()
    contract = dict(format='fixture')
    plan = [e for e in screen.episode_plan() if e['task_id'] == task]
    assert screen.run_screen(sim, motor, task, root=tmp_path, contract=contract,
        seed_motor=seed_motor, plan=plan[:6]) == 'complete'
    rows = screen.read_records(tmp_path)
    assert [row['identity'] for row in rows] == [e['identity'] for e in plan[:6]]
    assert all(row['optimizer_updates'] == 0 and row['data_role'] == 'target-headroom-screen' for row in rows)
    assert [row['success'] for row in rows] == [True] * 4 + [False] * 2
    assert sim.resets == [(1, 123)] * 4 + [(2, 123)] * 2
    # Native chunks are independent H10 draws; tied chunks repeat one 32D draw.
    native, tied = motor.noises[0], motor.noises[5]
    assert not np.allclose(native[0], native[1]) and np.allclose(tied, tied[0])
    trace = tmp_path / rows[0]['trace']
    assert trace.exists() and screen.onset.sha256(trace) == rows[0]['trace_sha256']
    charged = screen.onset.joint_charge(tmp_path)
    assert charged == rows[-1]['cumulative_physical_actions'] == sum(r['primitive_steps'] + 10 for r in rows)
    # Resume skips completed identities and reproduces the same seeded noise.
    again = Motor()
    assert screen.run_screen(Simulator(task), again, task, root=tmp_path, contract=contract,
        seed_motor=seed_motor, plan=plan[:6]) == 'complete'
    assert again.noises == [] and len(screen.read_records(tmp_path)) == 6
    replay = Motor()
    screen.screen_episode(Simulator(task), replay, plan[0], seed_motor=seed_motor)
    assert np.array_equal(replay.noises[0], motor.noises[0])
    selection = screen.write_report(tmp_path)
    assert selection['status'] == 'incomplete' and (tmp_path / 'rates.csv').exists()
    assert selection['rates'][0]['native_successes'] == 2 and selection['rates'][0]['native_episodes'] == 3


def test_run_screen_stops_before_an_unfunded_episode(tmp_path, monkeypatch):
    task = screen.TASKS[0]
    monkeypatch.setattr(screen, 'MAX_ACTIONS', 409)
    status = screen.run_screen(Simulator(task), Motor(), task, root=tmp_path, contract={},
        seed_motor=seed_motor)
    assert status == 'action_reservation' and screen.read_records(tmp_path) == []


def test_changed_nominal_state_aborts(tmp_path):
    task = screen.TASKS[0]
    sim = Simulator(task)
    original = sim.reset
    def drifting(state, seed):
        value = original(state, seed)
        value['initial_state_vector_sha256'] = f'state-{state}-{len(sim.resets)}'
        return value
    sim.reset = drifting
    with pytest.raises(RuntimeError, match='nominal physical case'):
        screen.run_screen(sim, Motor(), task, root=tmp_path, contract={}, seed_motor=seed_motor)
