"""First-wave coverage, support/query counts and shared per-case ownership."""
import ast

import pytest

from fluffy_bassoon.meta_rl.vla_source40_target_firstwave_plan import (
    ZERO_METHODS, ONE_METHODS, INTERIM_INPUTS, INTERIM_PROFILE, _literal, build_plan, content_hash)


def plan():
    return build_plan(worktree="/allocated/reviewed-worktree", source_commit="a" * 40)


def test_full90_plus_and_pro_are_predeclared_with_exact_physical_counts():
    value = plan()
    assert value["counts"] == dict(libero90_cases=90, plus_cases=4, pro_seeded_cases=4,
        total_cases=98, phase_a_groups=490, phase_b_groups=68, physical_episodes=626,
        nominal_actions=243920, worker_action_caps=249920, worker_gpu_seconds=21600)
    original = [row for row in value["cases"] if row["case"]["benchmark"] == "libero90"]
    assert [row["case"]["task_id"] for row in original] == list(range(90))
    assert all(row["case"]["initial_state"] == 0 and row["case"]["scene_seed"] == 123 for row in original)
    assert all(tuple(group["method"] for group in row["groups"][:5]) == ZERO_METHODS for row in value["cases"])
    assert value == plan()  # No target outcomes, wall time or random task selection.
    assert value["firstwave_sha256"] == content_hash({k: v for k, v in value.items() if k != "firstwave_sha256"})


def test_k1_phase_is_fixed_stride_and_all_shifted_cases_with_actual_support_query():
    value = plan()
    selected = [row for row in value["cases"] if any(group["phase"] == "B" for group in row["groups"])]
    assert [row["case"]["task_id"] for row in selected if row["case"]["benchmark"] == "libero90"] == list(range(0, 90, 10))
    assert len([row for row in selected if row["case"]["benchmark"] != "libero90"]) == 8
    for row in selected:
        groups = [group for group in row["groups"] if group["phase"] == "B"]
        assert tuple(group["method"] for group in groups) == ONE_METHODS
        assert all(group["support_budget"] == 1 and group["episodes"] == 2 for group in groups)
        assert all(group["deployment_optimizer_updates"] == 0 for group in groups)


def test_six_workers_own_entire_cases_and_caps_do_not_multiply_global_allowance():
    value = plan()
    case_ids = [row["case"]["case_id"] for row in value["cases"]]
    assigned = []
    for index, worker in enumerate(value["workers"]):
        assert worker["case_ids"] == case_ids[index::6]
        assert worker["action_cap"] == worker["nominal_actions"] + 1000
        assert worker["gpu_seconds"] == 3600
        assigned.extend(worker["case_ids"])
    assert len(assigned) == len(set(assigned)) == 98
    assert set(assigned) == set(case_ids)
    assert sum(worker["action_cap"] for worker in value["workers"]) <= value["contract"]["maximum_total_actions"]
    assert sum(worker["qualification_settling_allowance"] for worker in value["workers"]) == 0


def test_shifted_cases_describe_two_variants_and_pro_seed_repeats_honestly():
    value = plan()
    plus = [row for row in value["cases"] if row["case"]["benchmark"] == "plus"]
    pro = [row for row in value["cases"] if row["case"]["benchmark"] == "pro"]
    assert {(row["case"]["task_id"], row["case"]["initial_state"]) for row in plus} == {(667,0),(667,1),(290,0),(290,1)}
    assert len({row["declaration"]["family"] for row in pro}) == 2
    assert {row["case"]["initial_state"] for row in pro} == {0}
    assert {row["case"]["scene_seed"] for row in pro} == {123,124}
    assert all("seeded repeat" in row["physical_case_kind"] for row in pro)


def test_zero_shot_episode_reset_reuses_full_weights_and_not_static_controller():
    for row in plan()["cases"]:
        by_id = {group["group_id"]: group for group in row["groups"]}
        for objective in ("ad", "dpt"):
            group = by_id[row["zero_shot_reuse"][objective + "_episode_reset"]]
            assert group["method"] == objective + "_full" and group["support_budget"] == 0
        assert not any(group["phase"] == "A" and "episode_reset" in group["method"] for group in row["groups"])


def test_simulator_import_paths_follow_explicit_worktree_and_no_motor_runtime_mix():
    for row in plan()["cases"]:
        simulator = row["simulator"]
        assert simulator["pythonpath"].split(":")[0] == "/allocated/reviewed-worktree/src"
        assert "/.codex/" not in simulator["pythonpath"] and "rlinf" not in simulator["pythonpath"]
        assert "--port" not in simulator["arguments"] and "--output" not in simulator["arguments"]
        assert simulator["horizon"] == (400 if row["case"]["benchmark"] == "libero90" else 280)
        assert simulator["settle_steps"] == (5 if row["case"]["benchmark"] == "pro" else 10)


def test_declared_constant_reader_does_not_execute_arbitrary_expressions():
    with pytest.raises(ValueError, match="supported literal"):
        _literal(ast.parse("__import__('os').getcwd()", mode="eval").body, {})
    with pytest.raises(ValueError, match="six workers"):
        build_plan(worktree="/allocated/reviewed-worktree", source_commit="a" * 40, workers=5)


def test_fixed100_profile_retains_all90_and_exact_predeclared_support_grid():
    value = build_plan(worktree='/allocated/reviewed-worktree', source_commit='a'*40, profile=INTERIM_PROFILE)
    assert value['counts'] == dict(libero90_cases=90, plus_cases=0, pro_seeded_cases=0, total_cases=90,
        phase_a_groups=450, phase_b_groups=36, physical_episodes=522, nominal_actions=214020,
        worker_action_caps=219420, worker_gpu_seconds=16200)
    assert [c['case']['task_id'] for c in value['cases']] == list(range(90))
    assert {c['case']['benchmark'] for c in value['cases']} == {'libero90'}
    assert all(c['case']['initial_state'] == 0 and c['case']['scene_seed'] == 123 for c in value['cases'])
    assert [c['case']['task_id'] for c in value['cases'] if any(g['phase']=='B' for g in c['groups'])] == list(range(0,90,10))
    assert all(g['updates'] == (None if g['method']=='native' else 100) for g in value['groups'])
    assert all(g['episodes'] == g['support_budget']+1 for g in value['groups'])
    assert value['contract']['fixed_source_inputs'] == INTERIM_INPUTS
    assert value['contract']['maximum_total_actions'] == 220000
    assert value['contract']['maximum_gpu_seconds'] == 16200
    assert 'profile' not in plan()['contract']  # Legacy serialized defaults remain unchanged.


def test_fixed100_worker_ownership_and_reset_reuse_do_not_drop_support_cost():
    value = build_plan(worktree='/allocated/reviewed-worktree', source_commit='a'*40, profile=INTERIM_PROFILE)
    assert [w['nominal_actions'] for w in value['workers']] == [40590,30750,40590,30750,40590,30750]
    assert all(w['action_cap'] == w['nominal_actions']+900 and w['gpu_seconds'] == 2700 for w in value['workers'])
    assert len({k for w in value['workers'] for k in w['case_ids']}) == 90
    for c in value['cases']:
        by_id = {g['group_id']:g for g in c['groups']}
        for objective in ('ad','dpt'):
            full = by_id[c['zero_shot_reuse'][objective+'_episode_reset']]
            assert full['method'] == objective+'_full' and full['episodes'] == 1
        assert all(g['episodes'] == 2 and g['maximum_actions'] == 820 for g in c['groups'] if g['phase']=='B')
    with pytest.raises(ValueError, match='Unknown fixed'):
        build_plan(worktree='/allocated/reviewed-worktree', source_commit='a'*40, profile='pick-best-target')
