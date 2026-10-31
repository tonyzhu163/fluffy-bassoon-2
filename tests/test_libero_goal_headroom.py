"""CPU checks of the LIBERO-Goal headroom plan, noise pairing, traces and metrics."""
import math

import pytest

from fluffy_bassoon.rl import libero_goal as lg


def test_plan_counts_and_phase_order():
    plan = lg.build_plan()
    counts = {c: sum(r["condition"] == c for r in plan) for c in ("explicit", "vague", "swap")}
    assert counts == dict(explicit=120, vague=40, swap=180)
    assert [r["episode"] for r in plan] == list(range(340))
    assert [r["condition"] for r in plan[:20]] == ["explicit"] * 20
    assert {r["condition"] for r in plan[20:60]} == {"vague"}
    assert {r["condition"] for r in plan[60:240]} == {"swap"}
    assert {r["trial"] for r in plan[240:]} == set(range(2, 12))


def test_swap_covers_every_ordered_pair_on_paired_trials():
    plan = lg.build_plan()
    swaps = {(r["source_task"], r["instructed_goal"], r["trial"]) for r in plan if r["condition"] == "swap"}
    assert swaps == {(s, g, t) for s in range(10) for g in range(10) for t in range(2) if s != g}
    explicit = {(r["source_task"], r["trial"]) for r in plan if r["condition"] == "explicit"}
    vague = {(r["source_task"], r["trial"]) for r in plan if r["condition"] == "vague"}
    assert vague <= explicit
    assert all(r["source_task"] == r["instructed_goal"] for r in plan if r["condition"] == "explicit")
    assert all(r["instructed_goal"] is None for r in plan if r["condition"] == "vague")


def test_noise_is_paired_by_init_state_not_instruction_or_batch():
    torch = pytest.importorskip("torch")
    a = dict(source_task=3, trial=1, instructed_goal=3)
    b = dict(source_task=3, trial=1, instructed_goal=7)
    c = dict(source_task=4, trial=1, instructed_goal=7)
    first = lg.chunk_noise([a, c], 5, 2026)
    second = lg.chunk_noise([c, b], 5, 2026)
    assert first.shape == (2, lg.HORIZON, lg.MODEL_ACTION_DIM)
    assert torch.equal(first[0], second[1]) and torch.equal(first[1], second[0])
    assert not torch.equal(first[0], first[1])
    assert not torch.equal(lg.chunk_noise([a], 6, 2026), lg.chunk_noise([a], 5, 2026))


def test_noise_seed_rejects_collisions():
    with pytest.raises(ValueError):
        lg.noise_seed(1, 0, 100, 0)
    assert len({lg.noise_seed(1, s, t, c) for s in range(10) for t in range(50) for c in range(64)}) == 10 * 50 * 64


def test_instruction_text():
    langs = [f"task {i}" for i in range(10)]
    assert lg.instruction(dict(condition="swap", instructed_goal=4, vague_prompt=None), langs) == "task 4"
    assert lg.instruction(dict(condition="vague", instructed_goal=None, vague_prompt="do it"), langs) == "do it"


def test_goal_trace_first_steps_and_parity():
    trace = lg.GoalTrace(3, [False] * 3, source_task=1, native_initial=False)
    trace.update([False, False, True], native=False)
    trace.update([False, True, True], native=True)
    trace.update([False, True, False], native=False)  # native disagrees with goal 1
    row = trace.row()
    assert row["goals_first_step"] == [-1, 2, 1]
    assert row["achieved_goals"] == [1, 2] and row["first_goals"] == [2]
    assert row["goals_at_end"] == [False, True, False]
    assert row["native_parity_mismatches"] == 1 and row["steps"] == 3


def record(condition, source, trial, goal, achieved, prompt=None, first=None):
    steps = [-1] * 10
    for i, g in enumerate(achieved):
        steps[g] = 10 + i
    return dict(condition=condition, source_task=source, trial=trial, instructed_goal=goal,
                vague_prompt=prompt, goals_first_step=steps, achieved_goals=sorted(achieved),
                first_goals=[achieved[0]] if achieved else [], goals_initial=[False] * 10,
                native_parity_mismatches=0, steps=lg.STEPS)


def synthetic(follow=True, vague_goal=None):
    rows = [record("explicit", g, t, g, [g]) for g in range(10) for t in range(5)]
    rows += [record("swap", s, 0, g, [g] if follow else [s]) for s in range(10) for g in range(10) if g != s]
    rows += [record("vague", s, t, None, [] if vague_goal is None else [vague_goal], prompt=p)
             for s in range(10) for t in range(2) for p in lg.VAGUE_PROMPTS]
    return rows


def test_summary_ideal_steering_passes_every_criterion():
    s = lg.summarize(synthetic())
    assert s["explicit_overall"]["rate"] == 1 and s["swap"]["overall_follow_any"]["rate"] == 1
    assert s["swap"]["overall_follow_first"]["rate"] == 1
    assert all(v == 1 for v in s["gap"].values())
    c = s["criteria"]
    assert c["explicit_pass"] and c["swap_overall_pass"] and c["vague_pass"] and c["gap_pass"]
    assert c["goals_passing_all"] == list(range(10))
    assert s["vague"]["pooled"]["first_goal_counts"][10] == 40
    assert s["checks"] == dict(native_parity_mismatch_steps=0, episodes_with_goal_true_after_settle=0,
                               incomplete_episodes=0, init_states_with_unpaired_start=0)


def test_summary_detects_ignored_instruction_and_default_goal():
    s = lg.summarize(synthetic(follow=False, vague_goal=7))
    assert s["swap"]["overall_follow_any"]["rate"] == 0
    assert s["swap"]["by_instructed_goal"][3]["source_goal_instead"]["rate"] == 1
    assert s["swap"]["achieved_matrix"]["counts"][3][3] == 0
    assert s["vague"]["pooled"]["achieved_any"][7]["rate"] == 1
    assert math.isclose(s["gap"][7], 0) and math.isclose(s["gap"][2], 1)
    c = s["criteria"]
    assert not c["swap_overall_pass"] and not c["vague_per_goal"][7] and c["vague_per_goal"][2]
    assert 7 not in c["goals_passing_all"]
    assert s["uninstructed_goal_rate"][7]["k"] > 0


def test_missing_conditions_fail_closed():
    rows = [record("explicit", g, 0, g, [g]) for g in range(10)]
    c = lg.summarize(rows)["criteria"]
    assert c["explicit_pass"] and not c["swap_overall_pass"] and not c["vague_pass"] and not c["gap_pass"]


def test_wilson_interval_bounds():
    low, high = lg.wilson(8, 10)
    assert 0.4 < low < 0.8 < high < 1
    assert all(math.isnan(v) for v in lg.wilson(0, 0))


def test_unpaired_starts_counts_init_states_with_divergent_starts():
    rows = [dict(source_task=0, trial=0, model_xml_sha256="a", settled_state_sha256="x"),
            dict(source_task=0, trial=0, model_xml_sha256="a", settled_state_sha256="x"),
            dict(source_task=1, trial=0, model_xml_sha256="a", settled_state_sha256="x"),
            dict(source_task=1, trial=0, model_xml_sha256="a", settled_state_sha256="y")]
    assert lg.unpaired_starts(rows) == 1


def test_env_seed_is_shared_across_instructions():
    a = dict(source_task=3, trial=1, instructed_goal=3)
    b = dict(source_task=3, trial=1, instructed_goal=None)
    assert lg.env_seed(a) == lg.env_seed(b) != lg.env_seed(dict(source_task=3, trial=0))
