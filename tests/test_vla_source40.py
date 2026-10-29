"""Source case isolation, actual warmup/update timing and complete-case recovery."""
import json

import numpy as np
import pytest

from fluffy_bassoon.meta_rl.vla_source40_schedule import (case_plan, content_hash, load_schedule,
    build_schedule, partition_budget, selected_partition_budget, FORMAT, LIBERO_COMMIT, SOURCE_SUITES)
from fluffy_bassoon.meta_rl.vla_source40_corpus import (OnlineSourceSimulator, has_pending_case,
    case_coverage, paired_arrays, physical_costs, publication_due,
    restore_motor_rng, save_motor_rng, validate_simulator)


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


def test_complete_case_mode_cannot_stop_after_chunk_target_or_on_recovery():
    progress = dict(case_index=3, episode_in_case=0, chunks=100000)
    assert has_pending_case(progress, 10, None, "complete_cases")
    progress.update(case_index=9, episode_in_case=4)
    assert has_pending_case(progress, 10, None, "complete_cases")  # Final publication still pending.
    progress.update(case_index=10, episode_in_case=0)
    assert not has_pending_case(progress, 10, None, "complete_cases")


def test_full_case_budgets_include_every_episode_and_partition_specific_ceiling():
    train = partition_budget(80, 4, collection_mode="complete_cases", publish_every_cases=16)
    validation = partition_budget(20, 4, collection_mode="complete_cases", publish_every_cases=16)
    assert (train["maximum_cases"], train["maximum_episodes"], train["maximum_chunks"], train["maximum_primitive_steps"]) == (80, 320, 12800, 128000)
    assert (validation["maximum_cases"], validation["maximum_episodes"], validation["maximum_chunks"], validation["maximum_primitive_steps"]) == (20, 80, 3200, 32000)
    assert train["maximum_settling_steps"] == 3210
    assert train["immutable_publication_case_numbers"] == [16, 32, 48, 64, 80]
    assert validation["immutable_publication_case_numbers"] == [16, 20]
    assert selected_partition_budget(dict(partition_budgets=dict(train=train, validation=validation)), "validation") == validation
    assert train["storage_allowance"]["total_bytes_with_ten_percent_reserve"] > train["storage_allowance"]["raw_trace_and_journal_bytes"]
    with pytest.raises(ValueError, match="cannot have a chunk"):
        partition_budget(80, 4, collection_mode="complete_cases", target_chunks=600)
    assert partition_budget(64, 4, collection_mode="chunk_target", target_chunks=600)["maximum_primitive_steps"] == 7600


def test_full_all40_schedule_does_not_advertise_unvisited_cases_as_covered(tmp_path, monkeypatch):
    mapping = {suite: [f"task_{i}" for i in range(10)] for suite in SOURCE_SUITES}
    task_map = tmp_path / "libero/libero/benchmark/libero_suite_task_map.py"
    task_map.parent.mkdir(parents=True)
    task_map.write_text("libero_task_map = " + repr(mapping))
    for suite, names in mapping.items():
        for name in names:
            for kind, suffix in (("bddl_files", ".bddl"), ("init_files", ".pruned_init")):
                path = tmp_path / f"libero/libero/{kind}/{suite}/{name}{suffix}"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"static fixture")
    monkeypatch.setattr("fluffy_bassoon.meta_rl.vla_source40_schedule.subprocess.check_output",
                        lambda *args, **kwargs: LIBERO_COMMIT + "\n")
    plan = build_schedule(tmp_path, train_cases=80, validation_cases=20, episodes_per_case=4,
        collection_mode="complete_cases", publish_every_cases=16)
    assert len(plan["tasks"]) == 40 and plan["target_chunks_per_task_partition"] is None
    assert plan["planned_requested_cases"] == 4000
    assert plan["planned_requested_episodes"] == 16000
    assert plan["planned_maximum_chunks"] == 640000
    assert plan["planned_maximum_primitive_steps"] == 6400000
    assert plan["planned_maximum_settling_steps"] == 160800
    assert plan["planned_storage_allowance_bytes"] == 1795711565880
    assert {task["suite"] for task in plan["tasks"]} == set(SOURCE_SUITES)
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    assert load_schedule(path)["schedule_sha256"] == plan["schedule_sha256"]
    # A rehashed file cannot silently delete requested cases while retaining
    # its all-task budget and coverage claim.
    plan["tasks"][0]["partitions"]["train"].pop()
    plan["schedule_sha256"] = content_hash({k: v for k, v in plan.items() if k != "schedule_sha256"})
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="full case count"):
        load_schedule(path)
    physical = build_schedule(tmp_path, train_cases=512, validation_cases=512, episodes_per_case=4,
        collection_mode="physical_budget", train_action_budget=10000, validation_action_budget=8750,
        publish_every_cases=16)
    assert physical["planned_maximum_total_actions"] == 750000
    assert physical["planned_requested_cases"] is None
    path.write_text(json.dumps(physical))
    assert load_schedule(path)["planned_maximum_total_actions"] == 750000
    physical["planned_maximum_total_actions"] += 1
    physical["schedule_sha256"] = content_hash({k: v for k, v in physical.items() if k != "schedule_sha256"})
    path.write_text(json.dumps(physical))
    with pytest.raises(ValueError, match="Aggregate physical-action budget"):
        load_schedule(path)


def test_publication_cadence_never_skips_final_requested_case():
    assert [i for i in range(1, 21) if publication_due(i, 20, 16)] == [16, 20]
    assert publication_due(1, 1, 16)


def test_actual_state_coverage_distinguishes_repeated_seeds_from_new_states():
    cases = case_plan("libero_goal:00", "validation", 20, 4, 0)
    receipts = [dict(case_id=case["case_id"], initial_state=case["initial_state"],
        scene_seed=case["scene_seed"], initial_state_vector_sha256=f"state{case['initial_state']}")
        for case in cases for _ in range(4)]
    measured = case_coverage(cases, receipts)
    assert measured["all_requested_cases_complete"]
    assert measured["completed_cases"] == 20 and measured["completed_episodes"] == 80
    assert measured["actual_distinct_initial_state_ids"] == 10
    assert measured["actual_distinct_nominal_state_vectors"] == 10
    assert measured["actual_distinct_state_seed_pairs"] == 20
    assert set(measured["actual_seed_counts_per_initial_state"].values()) == {2}
    assert measured["actual_repeat_episodes_beyond_first_per_seeded_case"] == 60
    partial = case_coverage(cases, receipts[:5])
    assert partial["completed_cases"] == 1 and partial["started_cases"] == 2
    assert not partial["all_requested_cases_complete"]
    with pytest.raises(ValueError, match="case identity"):
        case_coverage(cases, [dict(receipts[0], scene_seed=17)])


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


def test_physical_budget_reserves_joint_actions_and_does_not_request_whole_seed_bank():
    budget = partition_budget(512, 4, collection_mode="physical_budget",
        total_action_budget=10000, publish_every_cases=16)
    assert budget["maximum_total_actions"] == 10000
    assert budget["available_seeded_cases"] == 512
    assert budget["requested_cases"] is None and budget["requested_episodes"] is None
    assert budget["maximum_chunks"] == 999
    assert budget["maximum_episodes"] == 908
    assert not budget["warmup_clearing_guaranteed_by_episode_count"]
    assert budget["additional_terminal_chunk_target_export_allowance"] == 1
    with pytest.raises(ValueError, match="qualification plus one complete"):
        partition_budget(512, 4, collection_mode="physical_budget", total_action_budget=1649)
    with pytest.raises(ValueError, match="cannot have a chunk"):
        partition_budget(512, 4, collection_mode="physical_budget", total_action_budget=10000, target_chunks=600)
    with pytest.raises(ValueError, match="requires physical_budget"):
        partition_budget(10, 4, collection_mode="complete_cases", total_action_budget=10000)


def test_physical_budget_stops_before_case_and_retains_partial_recovery():
    from fluffy_bassoon.meta_rl.vla_source40_corpus import physical_case_allowed
    cursor = dict(case_index=2, episode_in_case=0)
    assert physical_case_allowed(cursor, 512, charged_actions=8360, total_budget=10000, episodes_per_case=4)
    assert not physical_case_allowed(cursor, 512, charged_actions=8361, total_budget=10000, episodes_per_case=4)
    # Final publication/recovery of an already started case cannot be skipped.
    assert physical_case_allowed(dict(cursor, episode_in_case=4), 512,
        charged_actions=9990, total_budget=10000, episodes_per_case=4)
    assert not physical_case_allowed(dict(case_index=512, episode_in_case=0), 512,
        charged_actions=100, total_budget=10000, episodes_per_case=4)
    with pytest.raises(RuntimeError, match="exceeds"):
        physical_case_allowed(cursor, 512, charged_actions=10001, total_budget=10000, episodes_per_case=4)


def test_physical_ledger_includes_failed_settling_and_unresolved_step_intents(tmp_path):
    from fluffy_bassoon.meta_rl.vla_source40_corpus import physical_action_charge
    (tmp_path / "intent-1.json").write_text(json.dumps(dict(maximum_primitive_steps=10)))
    (tmp_path / "decision-1.json").write_text(json.dumps(dict(primitive_steps=4)))
    (tmp_path / "intent-2.json").write_text(json.dumps(dict(maximum_primitive_steps=10)))
    (tmp_path / "reset-qualification-1.json").write_text(json.dumps(dict(settling_steps=10, returned=True)))
    (tmp_path / "reset-episode-1.json").write_text(json.dumps(dict(settling_steps=10, returned=False)))
    assert physical_action_charge(tmp_path) == 34


def test_physical_case_bank_coverage_reports_actual_complete_cases_without_false_request():
    from fluffy_bassoon.meta_rl.vla_source40_corpus import physical_case_coverage
    cases = case_plan("libero_goal:00", "validation", 512, 4, 0)
    receipts = [dict(case_id=case["case_id"], initial_state=case["initial_state"],
        scene_seed=case["scene_seed"], initial_state_vector_sha256=f"state{case['initial_state']}")
        for case in cases[:3] for _ in range(4)]
    coverage = physical_case_coverage(cases, receipts)
    assert coverage["available_cases"] == 512 and coverage["completed_cases"] == 3
    assert coverage["all_started_cases_complete"] and not coverage["candidate_bank_exhausted"]
    assert "requested_cases" not in coverage and "all_requested_cases_complete" not in coverage
    assert not physical_case_coverage(cases, receipts + [dict(receipts[0], case_id=cases[3]["case_id"],
        initial_state=cases[3]["initial_state"], scene_seed=cases[3]["scene_seed"])])["all_started_cases_complete"]
