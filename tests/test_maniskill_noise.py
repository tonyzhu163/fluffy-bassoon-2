"""CPU checks of the noise-arm override, physics variants and Stage 0 gates (no simulator)."""
import json
import sys
import types

import numpy as np
import pytest
import torch

from fluffy_bassoon.meta_rl import noise_screen as ns
from fluffy_bassoon.rl import maniskill_pi05 as ms


def test_arm_noise_is_seeded_distinct_and_shaped():
    noise = ms.arm_noise([5, 6, 7])
    assert noise.shape == (3, *ms.NOISE_SHAPE) and noise.dtype == torch.float32
    torch.testing.assert_close(noise, ms.arm_noise([5, 6, 7]), rtol=0, atol=0)
    torch.testing.assert_close(noise[1], ms.arm_noise([6])[0], rtol=0, atol=0)
    assert not torch.equal(noise[0], noise[1])
    with pytest.raises(ValueError):
        ms.arm_noise([1.5])
    with pytest.raises(ValueError):
        ms.arm_noise([])


class FakeModel(torch.nn.Module):
    """Actions depend only on the initial noise, so outcomes expose what each env received."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.calls = []

    def predict_action_batch(self, obs, mode, rng, **kwargs):
        assert mode == "eval"
        self.calls.append(kwargs)
        noise = kwargs.get("noise")
        if noise is None:
            noise = torch.randn((obs["envs"], *ms.NOISE_SHAPE), generator=rng)
        return noise[:, :ms.CHUNK, :7].clone(), {}


class FakeEnv:
    device = torch.device("cpu")

    def __init__(self, objects):
        self.env = types.SimpleNamespace(unwrapped=types.SimpleNamespace(
            select_carrot_ids=torch.tensor(objects)))
        self.envs = len(objects)

    def reset(self, seed, options):
        return dict(task_descriptions=["put it"]*self.envs, envs=self.envs), {}

    def chunk_step(self, chunk):
        success = chunk[:, 0, 0] > 0
        info = dict(success=success, is_src_obj_grasped=success, consecutive_grasp=success,
                    carrot_plate_dist=torch.zeros(self.envs))
        return [dict(task_descriptions=[], envs=self.envs)], None, None, None, [info]*ms.CHUNK


@pytest.fixture
def prepare_actions(monkeypatch):
    utils = types.ModuleType("rlinf.envs.action_utils")
    utils.prepare_actions = lambda raw_chunk_actions, **kwargs: raw_chunk_actions
    for name in ("rlinf", "rlinf.envs"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "rlinf.envs.action_utils", utils)


def run(model, noise=None, fresh=None, objects=(0, 1, 2), seed=3):
    issued = []
    rows = ms.run_batch(model, FakeEnv(list(objects)), [dict(object=o, episode_id=o) for o in objects],
                        seed=seed, generator=torch.Generator(), initial_noise=noise, fresh_noise=fresh,
                        issued=issued)
    return rows, torch.stack(issued, 1)


def test_arm_actions_do_not_depend_on_slot_or_generator(prepare_actions):
    noise = ms.arm_noise([11, 12, 13])
    _, first = run(FakeModel(), noise, seed=3)
    _, swapped = run(FakeModel(), noise[[2, 0, 1]], seed=99)
    torch.testing.assert_close(swapped[1], first[0], rtol=0, atol=0)
    torch.testing.assert_close(swapped[0], first[2], rtol=0, atol=0)
    assert first.shape == (3, ms.STEPS // ms.CHUNK, ms.CHUNK, 7)
    # One fixed tensor serves every chunk of the attempt.
    assert all(torch.equal(first[:, i], first[:, 0]) for i in range(first.shape[1]))
    assert not torch.equal(first[0], first[1])


def test_fresh_rows_draw_new_seeded_noise_every_chunk(prepare_actions):
    noise, fresh = ms.arm_noise([1, 2]), torch.tensor([False, True])
    _, a = run(FakeModel(), noise, fresh, objects=(0, 1), seed=4)
    _, b = run(FakeModel(), noise, fresh, objects=(0, 1), seed=4)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.testing.assert_close(a[0, 0], noise[0, :ms.CHUNK, :7], rtol=0, atol=0)
    assert not torch.equal(a[1, 0], a[1, 1])
    _, c = run(FakeModel(), noise, fresh, objects=(0, 1), seed=5)
    assert torch.equal(a[0], c[0]) and not torch.equal(a[1], c[1])


def test_default_path_passes_no_noise_and_validates_override(prepare_actions):
    model = FakeModel()
    rows, _ = run(model)
    assert len(rows) == 3 and all(call == {} for call in model.calls)
    with pytest.raises(ValueError):
        run(FakeModel(), torch.zeros(2, *ms.NOISE_SHAPE))
    with pytest.raises(ValueError):
        run(FakeModel(), torch.full((3, *ms.NOISE_SHAPE), float("nan")))
    with pytest.raises(ValueError):
        run(FakeModel(), fresh=torch.tensor([True, False, False]))
    with pytest.raises(ValueError):
        run(FakeModel(), ms.arm_noise([1, 2, 3]), torch.tensor([1, 0, 0]))


class Task:
    obj_static_friction = 1.0
    obj_dynamic_friction = .8

    def __init__(self):
        self.model_db_carrot = {"carrot": {}}
        self.built = []

    def _build_actor_helper(self, name, path, density, scale, pose):
        self.built.append((name, density, self.obj_static_friction, self.obj_dynamic_friction))
        return name


def test_object_physics_scales_source_objects_only_and_restores():
    original = Task.__dict__["_build_actor_helper"]
    task = Task()
    with ms.object_physics(Task, friction_scale=.5, density_scale=4):
        task._build_actor_helper("carrot", None, 1000, 1, None)
        task._build_actor_helper("plate", None, 500, 1, None)
    task._build_actor_helper("carrot", None, 1000, 1, None)
    assert task.built == [("carrot", 4000, .5, .4), ("plate", 500, 1.0, .8), ("carrot", 1000, 1.0, .8)]
    assert Task.__dict__["_build_actor_helper"] is original and "obj_static_friction" not in vars(task)
    with pytest.raises(ValueError):
        with ms.object_physics(Task, friction_scale=0):
            pass
    with pytest.raises(ValueError):
        with ms.object_physics(type("Other", (Task,), {}), 1, 1):
            pass


def screen_rows(profiles, scenes=4, repeats=1):
    """profiles: {condition: {arm: success probability}}; deterministic rows by scene index."""
    rows = []
    for variant, arms in profiles.items():
        for scene in range(scenes):
            for arm, p in arms.items():
                for repeat in range(repeats):
                    rows.append(dict(variant=variant, object=0, scene_index=scene, arm=arm, repeat=repeat,
                                     row=len(rows), slot=repeat, success_once=bool(scene < p*scenes),
                                     issued_actions_sha256=f"{variant}-{scene}-{arm}"))
    return rows


def test_gates_detect_condition_specific_arms():
    # Each variant has its own working arm; native and other arms fail.
    profiles = {v: {-1: 0., **{a: (1. if a == v else 0.) for a in range(4)}} for v in range(4)}
    result = ns.gates(screen_rows(profiles), condition="variant", train=range(4), evaluate=range(4))
    assert result["summary"]["g1_headroom"]["mean"] == 1.
    assert result["summary"]["g2_heterogeneity"]["mean"] == 1.
    assert result["summary"]["g3_consistency"]["mean"] == pytest.approx(1)
    # No reference condition shares the working arm, so profiles do not transfer and G4 fails.
    assert result["summary"]["g4_structure"]["mean"] < 0 and not result["all_passed"]


def test_gates_reject_one_globally_best_arm():
    profiles = {v: {-1: .25, 0: 1., 1: .25, 2: .25} for v in range(3)}
    result = ns.gates(screen_rows(profiles), condition="variant", train=range(3), evaluate=range(3))
    assert result["summary"]["g1_headroom"]["passed"]
    assert result["summary"]["g2_heterogeneity"]["mean"] == 0. and not result["all_passed"]


def test_spearman_handles_ties_and_constants():
    assert ns.spearman([1, 2, 3], [3, 4, 9]) == pytest.approx(1)
    assert ns.spearman([1, 1, 2], [0, 0, 5]) == pytest.approx(1)
    assert np.isnan(ns.spearman([1, 1, 1], [0, 1, 2]))
    np.testing.assert_allclose(ns.average_ranks([3, 1, 3, 2]), [2.5, 0, 2.5, 1])


def test_profile_posterior_beats_independent_arms_when_profiles_transfer():
    references = [[.95, .05, .05], [.05, .95, .05]]
    observations = [(0, True), (1, False), (0, True), (2, False), (0, True), (1, False)]
    mixture, independent = ns.sequential_log_scores(observations, references)
    assert mixture > independent


def test_determinism_and_rows_validation(tmp_path):
    rows = screen_rows({0: {-1: .5, 0: .5, 1: .5}}, repeats=2)
    assert ns.determinism(rows)["deterministic"] and ns.determinism(rows)["groups_in_distinct_slots"] == 8
    assert (rows[3]["arm"], rows[3]["repeat"]) == (0, 1)
    rows[3]["issued_actions_sha256"] = "changed"
    assert not ns.determinism(rows)["deterministic"]
    path = tmp_path/"outcomes.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows + rows[:1]))
    with pytest.raises(ValueError):
        ns.load_rows(path)


def test_mode_effect_compares_first_chunks():
    rows = screen_rows({0: {-1: 0., 0: 0., 1: 0.}}, scenes=1, repeats=2)
    trajectories = {r["row"]: np.full((16, ms.CHUNK, 7), float(r["arm"] + 2*(r["arm"] == -1)*r["repeat"]))
                    for r in rows}
    effect = ns.mode_effect(rows, trajectories)
    assert effect["first_chunk_distance_between_arms"] == pytest.approx(np.sqrt(ms.CHUNK*7))
    assert effect["first_chunk_distance_native_repeats"] == pytest.approx(2*np.sqrt(ms.CHUNK*7))


def test_cli_writes_gate_file(tmp_path):
    profiles = {v: {-1: 0., **{a: (1. if a == v else 0.) for a in range(3)}} for v in range(3)}
    (tmp_path/"outcomes.jsonl").write_text("\n".join(json.dumps(r) for r in screen_rows(profiles)))
    ns.main(["--cache-dir", str(tmp_path), "--condition", "variant", "--train", "0,1,2", "--evaluate", "0,1,2",
             "--output", str(tmp_path/"gates.json")])
    result = json.loads((tmp_path/"gates.json").read_text())
    assert result["summary"]["g1_headroom"]["mean"] == 1. and "determinism" in result


def test_cli_checks_only_reports_determinism(tmp_path):
    (tmp_path/"outcomes.jsonl").write_text("\n".join(json.dumps(r) for r in screen_rows({0: {-1: 1., 0: 1.}}, repeats=2)))
    ns.main(["--cache-dir", str(tmp_path), "--output", str(tmp_path/"checks.json")])
    result = json.loads((tmp_path/"checks.json").read_text())
    assert result["summary"] is None and result["determinism"]["deterministic"]
    with pytest.raises(SystemExit):
        ns.main(["--cache-dir", str(tmp_path), "--condition", "variant", "--output", str(tmp_path/"x.json")])
