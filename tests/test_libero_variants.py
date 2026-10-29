"""CPU checks of the LIBERO-Plus/PRO noise-arm screen (fake env pool; real packages where installed)."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from fluffy_bassoon.rl import libero_variants as lv

MOKA = "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it"
BOOK = "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy"


def test_plus_names_map_to_perturbation_types():
    assert lv.plus_category(MOKA) == "clean"
    assert lv.plus_category(f"{MOKA}_table_3") == "background"
    assert lv.plus_category(f"{MOKA}_tb_3") == "background"
    assert lv.plus_category(f"{MOKA}_add_12") == "layout"
    assert lv.plus_category(f"{MOKA}_light_4") == "light"
    assert lv.plus_category(f"{MOKA}_language_2_view_0_0_100_0_0_initstate_0") == "language"
    assert lv.plus_category(f"{MOKA}_view_15_0_100_0_0_initstate_0") == "camera"
    assert lv.plus_category(f"{MOKA}_view_0_0_100_0_0_initstate_7") == "robot_init"
    assert lv.plus_category(f"{MOKA}_view_0_0_100_0_0_initstate_0_noise_30") == "sensor_noise"
    assert lv.plus_category(f"{BOOK}_level5_sample4") == "level"


def test_pro_directories_and_base_tasks():
    assert lv.pro_category("libero_10_lan") == "language"
    assert lv.pro_category("libero_10_swap") == "position"
    assert lv.pro_category("libero_10_with_red_stick") == "distractor"
    assert lv.pro_category("libero_10_task") == "task"
    assert lv.pro_category("libero_10_temp") is None
    bases = [MOKA, BOOK]
    assert lv.base_task(f"{MOKA}_light_4", bases) == MOKA
    assert lv.base_task("FLOOR_SCENE1_turn_on_the_stove_and_put_the_moka_pot_on_it", bases) == MOKA
    with pytest.raises(ValueError):
        lv.base_task("KITCHEN_SCENE9_open_the_door", bases)


def test_held_out_bases_are_the_split_test_tasks():
    bases = lv.held_out_bases()
    assert len(bases) == 10 and MOKA in bases and BOOK in bases


def test_select_spreads_over_base_tasks_and_is_seeded():
    variants = [dict(category=c, base_task=b, name=f"{b}_{c}_{i}")
                for c in ("light", "camera") for b in ("a", "b", "c") for i in range(3)]
    chosen = lv.select(variants, 3, seed=1)
    assert [v["variant"] for v in chosen] == list(range(6))
    for category in ("light", "camera"):
        assert {v["base_task"] for v in chosen if v["category"] == category} == {"a", "b", "c"}
    assert [v["name"] for v in lv.select(variants, 3, seed=1)] == [v["name"] for v in chosen]
    assert len(lv.select(variants, 50, seed=1)) == 18


class FakePool:
    """Each env succeeds once its summed first action passes 1; the reset seed is recorded."""

    def __init__(self):
        self.resets = []

    def reset(self, payloads):
        self.resets = payloads
        self.position = [0.0]*len(payloads)
        return [dict(obs=dict(state=np.zeros(8)), done=False, settled_state_sha256=f"s{p['seed']}", pid=1)
                for p in payloads]

    def step(self, chunks):
        out = []
        for i, chunk in enumerate(chunks):
            self.position[i] += float(np.sum(chunk[:, 0]))
            out.append(dict(obs=dict(state=np.zeros(8)), done=self.position[i] >= 1))
        return out


def test_run_batch_passes_arm_noise_and_stops_finished_envs():
    arms = lv.arm_noise([11, 12])
    seen = []

    def predict(states, prompts, noise):
        seen.append(noise.copy())
        return np.repeat(np.abs(noise[:, :lv.CHUNK, :1]).mean(axis=(1, 2), keepdims=True), lv.CHUNK, 1) \
            * np.ones((len(states), lv.CHUNK, 7)) * .2

    rows = [dict(bddl="x", init_state=np.zeros(3), scene_index=s, arm=a, episode_seed=7 + s, language="go",
                 variant=0, category="light") for s in (0, 1) for a in (lv.NATIVE, 0, 1)]
    pool = FakePool()
    out = lv.run_batch(predict, pool, rows, arms=arms, steps=60)
    assert [p["seed"] for p in pool.resets] == [lv.env_seed(r["scene_index"]) for r in rows]
    np.testing.assert_array_equal(seen[0][1], arms[0])
    np.testing.assert_array_equal(seen[0][0], lv.native_noise(7, 0))
    assert not np.array_equal(seen[0][0], seen[0][3])  # native differs across episode seeds
    records = [r for r, _ in out]
    assert all(r["success_once"] for r in records)
    assert all("init_state" not in r and "bddl" not in r for r in records)
    assert max(r["chunks"] for r in records) <= -(-60 // lv.CHUNK)
    assert len(seen[-1]) < len(rows)  # finished envs drop out of later batches
    assert out[1][1].shape == (1, lv.CHUNK, 7)


def load_script():
    spec = importlib.util.spec_from_file_location(
        "lv_cache", Path(__file__).resolve().parents[1]/"scripts/diagnostics/libero_variants_noise_cache.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_plan_and_analysis_per_perturbation_type(tmp_path):
    cache = load_script()
    variants = [dict(kind="plus", category=c, variant=i, base_task=b, name=f"{b}{c}", language="go", bddl="x")
                for i, (c, b) in enumerate([("light", "a"), ("light", "b"), ("camera", "a")])]
    rows = cache.plan_rows(variants, scenes=4, arms=3, repeats=1, base_seed=7,
                           init_states=lambda v: np.zeros((5, 3)))
    assert len(rows) == 3*4*4 and {r["arm"] for r in rows} == {-1, 0, 1, 2}
    with pytest.raises(ValueError):
        cache.plan_rows(variants, scenes=6, arms=3, repeats=1, base_seed=7, init_states=lambda v: np.zeros((5, 3)))
    (tmp_path/"trajectories").mkdir()
    with open(tmp_path/"outcomes.jsonl", "w") as stream:
        for index, row in enumerate(rows):
            best = {0: 0, 1: 1, 2: 2}[row["variant"]]
            record = {k: v for k, v in row.items() if k not in ("init_state", "bddl")}
            record.update(row=index, success_once=row["arm"] == best, issued_actions_sha256=str(index), slot=0)
            stream.write(json.dumps(record)+"\n")
    np.savez_compressed(tmp_path/"trajectories/first-chunks.npz", row=np.arange(len(rows)),
                        issued=np.zeros((len(rows), 1, lv.CHUNK, 7), np.float32))
    result = cache.analyse(tmp_path)
    light = result["categories"]["light"]
    assert light["native"] == 0 and light["gates"]["g1_headroom"] == pytest.approx(1.0)
    assert result["categories"]["camera"]["gates"] == "fewer than two variants"
    assert json.loads((tmp_path/"gates.json").read_text())["categories"].keys() == {"light", "camera"}


@pytest.mark.skipif(importlib.util.find_spec("liberoplus") is None or importlib.util.find_spec("liberopro") is None,
                    reason="LIBERO-Plus/PRO packages not installed")
def test_real_packages_enumerate_every_type_for_every_held_out_task():
    for kind, expected in (("plus", {"clean", "camera", "robot_init", "sensor_noise", "language", "light",
                                     "background", "layout", "level"}),
                           ("pro", {"clean", "language", "object", "position", "task", "distractor"})):
        variants = lv.enumerate_variants(kind)
        assert {v["category"] for v in variants} == expected
        for category in expected:
            assert len({v["base_task"] for v in variants if v["category"] == category}) == 10
        assert all(Path(v["bddl"].split("_view_")[0].removesuffix(".bddl") + ".bddl").exists() for v in variants)
