"""Source case isolation, actual warmup/update timing and complete-case recovery."""
import json

import numpy as np
import pytest

from fluffy_bassoon.meta_rl.vla_source40_schedule import case_plan, content_hash, load_schedule, FORMAT
from fluffy_bassoon.meta_rl.vla_source40_corpus import (OnlineSourceSimulator, has_pending_case,
    paired_arrays, physical_costs, restore_motor_rng, save_motor_rng, validate_simulator)


def test_many_seeded_cases_cover_each_bank_without_cross_partition_states():
    train = case_plan("libero_spatial:00", "train", 80, 4, 0)
    validation = case_plan("libero_spatial:00", "validation", 20, 4, 0)
    assert train == case_plan("libero_spatial:00", "train", 80, 4, 0)
    assert {row["initial_state"] for row in train} == set(range(40))
    assert {row["initial_state"] for row in validation} == set(range(40, 50))
    assert len({(row["initial_state"], row["scene_seed"]) for row in train}) == 80
    assert all(not row["learner_reset"] and row["episodes"] == 4 for row in train + validation)
    assert train[0]["scene_seed"] != train[40]["scene_seed"]


@pytest.mark.parametrize("task", ["libero_90:00", "LIBERO-Plus:00", "libero_pro:00", "libero_goal:10"])
def test_target_tasks_cannot_enter_source_plan(task):
    with pytest.raises(ValueError, match="source suites"):
        case_plan(task, "train", 2, 4, 0)


def test_even_rehashed_schedule_rejects_validation_state_leakage(tmp_path):
    case = case_plan("libero_goal:00", "validation", 1, 4, 0)[0]
    case["initial_state"] = 0
    plan = dict(format=FORMAT, source_only=True, tasks=[dict(suite="libero_goal", task_id=0,
        task_key="libero_goal:00", partitions=dict(train=case_plan("libero_goal:00", "train", 1, 4, 0), validation=[case]))])
    plan["schedule_sha256"] = content_hash(plan)
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="Case split"):
        load_schedule(path)


class Learner:
    def __init__(self):
        self.replay, self.updates, self.batches = [], 0, []
    def action(self, observation):
        return np.full(32, 4., np.float32)
    def update(self, batch, sampler):
        self.batches.append(batch)
        self.updates += 1
        return {"update": self.updates}


class Simulator:
    metadata = {}
    def reset(self, state, seed):
        return dict(observation=dict(state=state))
    def step(self, actions):
        return dict(observation=dict(state=1), success=False, done=True, primitive_steps=7)


def test_sac_persists_across_resets_and_updates_only_after_real_warmup():
    learner = Learner()
    collector = OnlineSourceSimulator(Simulator(), learner, np.random.default_rng(0),
        lambda observation: observation, warmup=2, batch=256, utd=20)
    noises = []
    for episode in range(4):
        collector.reset(episode, 100 + episode)
        noises.append(collector.choose(dict(state=episode)))
        collector.step(None)
        collector.finish_episode()
        assert len(learner.replay) == episode + 1
    assert learner.updates == 40 and learner.batches == [256] * 40
    assert np.abs(noises[0]).max() > 1  # Warmup is preserved, not silently clipped.
    np.testing.assert_array_equal(noises[3], np.ones(32))
    assert collector.provenance[0]["source_updates_before"] == 20
    assert collector.provenance[0]["source_updates_after"] == 20
    assert learner.replay[-1]["masks"] == 1  # Time-limit transitions bootstrap.
    assert learner.replay[-1]["discount"] == pytest.approx(.999 ** 7)


def test_crossing_warmup_keeps_gaussian_and_weights_fixed_until_episode_end():
    class FourDecisionSimulator(Simulator):
        decisions = 0

        def step(self, actions):
            self.decisions += 1
            return dict(super().step(actions), done=self.decisions == 4)

    learner = Learner()
    collector = OnlineSourceSimulator(FourDecisionSimulator(), learner, np.random.default_rng(0),
        lambda observation: observation, warmup=2, batch=256, utd=20)
    collector.reset(0, 0)
    for decision in range(4):
        collector.choose(dict(state=decision))
        collector.step(None)
        assert learner.updates == 0
    assert all(row["source_stage"] == "warmup_gaussian" for row in collector.provenance)
    assert all(row["source_updates_before"] == row["source_updates_after"] == 0
               for row in collector.provenance)
    collector.finish_episode()
    assert learner.updates == 80
    collector.reset(1, 1)
    np.testing.assert_array_equal(collector.choose(dict(state=1)), np.ones(32))


def test_resume_charges_unresolved_execution_intent_conservatively(tmp_path):
    (tmp_path / "intent-1.json").write_text(json.dumps(dict(maximum_primitive_steps=10)))
    (tmp_path / "decision-1.json").write_text(json.dumps(dict(primitive_steps=7)))
    (tmp_path / "intent-2.json").write_text(json.dumps(dict(maximum_primitive_steps=10)))
    assert physical_costs(tmp_path) == dict(known_primitive_steps=7,
        uncertain_primitive_steps_ceiling=10, primitive_step_budget_charge=17)
    # Receipt persistence resolves the charge even if its bulky trace write was interrupted.
    (tmp_path / "decision-2.json").write_text(json.dumps(dict(primitive_steps=2, trace_status="writing")))
    assert physical_costs(tmp_path) == dict(known_primitive_steps=9,
        uncertain_primitive_steps_ceiling=0, primitive_step_budget_charge=9)


@pytest.mark.parametrize("typed", [False, True])
def test_motor_rng_round_trip_preserves_next_random_draw(typed):
    import pickle
    jax = pytest.importorskip("jax")
    key = jax.random.key(17) if typed else jax.random.PRNGKey(17)
    saved = pickle.loads(pickle.dumps(save_motor_rng(key)))
    restored = restore_motor_rng(saved)
    np.testing.assert_array_equal(jax.random.key_data(jax.random.split(restored)),
                                  jax.random.key_data(jax.random.split(key)))


def test_recovery_reaching_target_mid_case_still_finishes_that_case():
    progress = dict(case_index=3, episode_in_case=2, chunks=601)
    assert has_pending_case(progress, 10, 600)
    progress.update(case_index=4, episode_in_case=0)
    assert not has_pending_case(progress, 10, 600)
    progress.update(chunks=599)
    assert has_pending_case(progress, 10, 600)


def test_paired_exports_keep_unbounded_executed_warmup_and_true_update_clock():
    rows = [dict(public_features=np.full(4104, i, np.float32), executed_noise=np.full(32, 2. - i, np.float32),
                 reward=-1., done=True, source_updates_before=i * 20) for i in (0, 1)]
    paired = paired_arrays(rows, np.zeros((2, 32), np.float32))
    np.testing.assert_array_equal(paired["ad"]["executed_noise"], paired["dpt"]["executed_noise"])
    np.testing.assert_array_equal(paired["ad"]["target_noise"], paired["ad"]["executed_noise"])
    assert paired["ad"]["executed_noise"][0, 0] == 2.
    assert list(paired["ad"]["teacher_steps"]) == [0, 20]
    assert list(paired["ad"]["episode_starts"]) == [True, True]


def test_collector_requires_fresh_original_source_task():
    task = dict(suite="libero_spatial", task_id=0, bddl_sha256="x", semantic_task="source")
    metadata = dict(suite="libero_spatial", task_id=0, bddl_sha256="x", bddl_file="source.bddl",
                    initial_states=50, horizon=400, fresh_environment_on_reset=True)
    validate_simulator(task, metadata)
    with pytest.raises(ValueError, match="source40"):
        validate_simulator(dict(task, suite="libero_90"), dict(metadata, suite="libero_90"))
    with pytest.raises(ValueError, match="reset"):
        validate_simulator(task, dict(metadata, fresh_environment_on_reset=False))
