"""One registered 2 × 3 MuJoCo friction/continuation experiment."""
import hashlib
import importlib.util
import json
import os
import platform
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / "libero-temporal-recovery-20261119-r1/collect.py"
spec = importlib.util.spec_from_file_location("recovery", PRIOR)
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)
visual = recovery.visual
bottle = recovery.bottle
POSE = 24
SEED = 18180124
STEPS = 300
OBJECT = "wine_bottle_1"
MODES = ("default", "regrasp", "transport")
MATERIALS = ("ordinary", "reduced")
SUFFIXES = ("finger1_collision", "finger2_collision", "finger1_pad_collision", "finger2_pad_collision")


def dump(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot(sim, raw):
    fields = {key: visual.digest(value) for key, value in raw.items()}
    for camera in visual.CAMERAS:
        image = sim.render(width=256, height=256, camera_name=camera)
        assert image.shape == (256, 256, 3) and image.dtype == np.uint8
        fields[camera] = visual.digest(image)
    return fields


class FingerContacts:
    """Read actual solver contacts after each unmodified native integration step."""

    def __init__(self, world, reduced):
        self.sim = world.sim
        model = self.sim.model
        names = [model.geom_id2name(i) or "" for i in range(model.ngeom)]
        ids = []
        for suffix in SUFFIXES:
            matches = [i for i, name in enumerate(names) if name.endswith("_" + suffix)]
            assert len(matches) == 1, (suffix, matches)
            ids.extend(matches)
        self.fingers = np.zeros(model.ngeom, dtype=bool)
        self.fingers[ids] = True
        self.object_ids = {model.geom_name2id(n) for n in world.objects_dict[OBJECT].contact_geoms}
        before = model.geom_friction.copy()
        if reduced:
            model.geom_friction[ids, 0] = 0.05
        expected = before.copy()
        if reduced:
            expected[ids, 0] = 0.05
        np.testing.assert_array_equal(model.geom_friction, expected)
        self.settings = {names[i]: [before[i].tolist(), model.geom_friction[i].tolist()] for i in ids}
        self.names = names
        self.mu = {}
        self.first = None
        self.first_bottle = None
        self.substep = 0
        self.control = 0
        self.native_step = self.sim.step
        self.record()  # Check the settled state for pre-existing contacts.
        self.sim.step = self.step

    def record(self):
        contacts = self.sim.data._data.contact
        pairs = contacts.geom
        for idx in np.flatnonzero(self.fingers[pairs].any(axis=1)):
            g1, g2 = map(int, pairs[idx])
            contact = contacts[int(idx)]
            stamp = dict(control=self.control, substep=self.substep,
                         geoms=[self.names[g1], self.names[g2]])
            if self.first is None:
                self.first = stamp
            finger, other = (g1, g2) if self.fingers[g1] else (g2, g1)
            if other in self.object_ids and contact.efc_address >= 0:
                if self.first_bottle is None:
                    self.first_bottle = stamp
                pair = self.names[finger] + " / " + self.names[other]
                mu = float(contact.friction[0])
                if pair in self.mu:
                    assert self.mu[pair] == mu
                self.mu[pair] = mu

    def step(self):
        self.native_step()
        self.substep += 1
        self.record()


def execute(out, mode, material, path, bank, tape, source):
    env = visual.LiberoEnvironmentAdapter()
    env._env = visual.ControlEnv(str(path), use_camera_obs=False, has_offscreen_renderer=True, camera_names=[])
    env._init_states = bank
    env._max_steps = STEPS
    try:
        initial = env.reset(POSE, SEED)
        world = env._env.env
        sim = world.sim
        physics = initial["environment_reproducibility"]["initial_state_fingerprint"]
        assert physics == source["physics"] and not env._env.check_success()
        contacts = FingerContacts(world, material == "reduced")
        raw = initial["observation"]
        prefix = [snapshot(sim, raw)]
        renderer = visual.GL.glGetString(visual.GL.GL_RENDERER).decode()
        assert "llvmpipe" in renderer.lower() and os.environ["CUDA_VISIBLE_DEVICES"] == ""
        state = visual.robot_state(raw)
        obj = world.objects_dict[OBJECT]
        body = world.obj_body_id[OBJECT]
        objpos = [sim.data.body_xpos[body].copy()]
        states, actions, hits, rewards, ends, grasp = [], [], [], [], [], []
        close = None
        if mode == "regrasp":
            windows = np.lib.stride_tricks.sliding_window_view(tape["issued"][:, -1] > 0, 8)
            close = int(np.flatnonzero(windows.all(axis=1))[0])
        for idx in range(STEPS):
            a = recovery.correction(idx, tape, state, close) if mode == "regrasp" else tape["issued"][idx]
            contacts.control = idx + 1
            rec = env.step(a.copy())
            assert rec["worker_step_index"] == idx + 1
            raw = rec["observation"]
            if contacts.first is None or idx + 1 <= contacts.first["control"]:
                prefix.append(snapshot(sim, raw))
            state = visual.robot_state(raw)
            states.append(state)
            actions.append(a.copy())
            hits.append(rec["success"])
            rewards.append(rec["reward"])
            ends.append([rec["terminated"], rec["truncated"]])
            objpos.append(sim.data.body_xpos[body].copy())
            grasp.append(bool(world._check_grasp(world.robots[0].gripper, obj)))
        assert contacts.substep >= STEPS and world.perturbation_runtime.diagnostics()["case_id"] == "BASE"
    finally:
        env.close()
    fname = f"{mode}-{material}"
    arrays = dict(states=np.stack(states), issued=np.stack(actions), hits=np.asarray(hits),
                  rewards=np.asarray(rewards), ends=np.asarray(ends), bottle_xyz=np.stack(objpos),
                  bilateral=np.asarray(grasp))
    np.savez_compressed(out / (fname + ".npz"), **arrays)
    dump(out / (fname + "-observations.json"), prefix)
    first_hit = next((idx + 1 for idx, hit in enumerate(hits) if hit), None)
    row = dict(mode=mode, material=material, physics=physics, renderer=renderer,
               finger_settings=contacts.settings, solver_sliding_mu=contacts.mu, first_finger=contacts.first,
               first_finger_bottle=contacts.first_bottle, integration_steps=contacts.substep,
               first_hit=first_hit, ever=any(hits), final=hits[-1], bilateral_controls=sum(grasp),
               max_lift_m=float(np.max(arrays["bottle_xyz"][:, 2]) - objpos[0][2]),
               reward_at_first_hit=rewards[first_hit - 1] if first_hit else None,
               native_return_to_termination=float(sum(rewards[:first_hit])),
               native_reward_sum_300=float(sum(rewards)), final_reward=rewards[-1],
               commands_sha256=visual.digest(arrays["issued"]), regrasp_close_zero_based=close)
    if mode == "default" and material == "ordinary":
        row["archive_states_exact"] = bool(np.array_equal(arrays["states"], tape["states"]))
        row["archive_goal_hits_exact"] = bool(np.array_equal(hits, tape["hits"][:, 0]))
    if mode != "regrasp":
        np.testing.assert_array_equal(arrays["issued"], tape["issued"])
    dump(out / (fname + ".json"), row)
    print(mode, material, "ever", row["ever"], "final", row["final"], flush=True)
    return row, arrays, prefix


def paired(ordinary, reduced):
    a, x, px = ordinary
    b, y, py = reduced
    first = min(row["first_finger"]["control"] if row["first_finger"] else STEPS + 1 for row in (a, b))
    # Snapshots are indexed by completed controls, beginning at 0. Include the
    # settled observation even if a contact already existed at reset.
    limit = max(1, first)
    mismatches = {k: [i for i in range(limit) if px[i][k] != py[i][k]] for k in px[0]}
    common = a["solver_sliding_mu"].keys() & b["solver_sliding_mu"].keys()
    changed = {k: [a["solver_sliding_mu"][k], b["solver_sliding_mu"][k]]
               for k in sorted(common) if a["solver_sliding_mu"][k] != b["solver_sliding_mu"][k]}
    displacement = np.linalg.norm(x["bottle_xyz"] - y["bottle_xyz"], axis=1)
    prefix_equal = all(np.array_equal(x[k][:limit-1], y[k][:limit-1])
                       for k in ("issued", "rewards", "hits", "ends"))
    return dict(public_samples=limit, last_precontact_control=first-1,
                public_field_mismatch_indices=mismatches, reset_physics_equal=a["physics"] == b["physics"],
                precontact_commands_rewards_and_flags_equal=prefix_equal,
                public_precontact_equal=first > 0 and not any(mismatches.values()) and prefix_equal,
                full_commands_equal=np.array_equal(x["issued"], y["issued"]),
                changed_realized_mu_by_geom_pair=changed, max_bottle_gap_m=float(displacement.max()),
                first_bottle_gap_over_1mm=next((int(i) for i in np.flatnonzero(displacement > .001)), None))


def main(out):
    path, _, bank = bottle.archive.setup("live")
    tapes = {}
    tapes["default"] = bottle.read_tape(POSE, 0, "query")
    tapes["regrasp"] = tapes["default"]
    directory = recovery.REACTIVE / f"p{POSE}-g0-D18"
    rec = json.loads((directory / "result.json").read_text())
    assert (rec["pose"], rec["goal"], rec["mode"], rec["seed"]) == (POSE, 0, "D18", 19195024)
    with np.load(directory / "tape.npz") as z:
        tapes["transport"] = ({k: z[k] for k in ("issued", "states", "hits")}, rec, directory)
    prov = dict(pose=POSE, physics_seed=SEED, task="libero_goal/2/BASE", bddl=str(path),
                bddl_sha256=sha(Path(path)), python=sys.version, platform=platform.platform(),
                packages={k: version(k) for k in ("numpy", "mujoco", "robosuite", "scipy", "PyOpenGL")},
                correction_sha256=sha(PRIOR), runner_sha256=sha(Path(__file__)), sources={})
    for mode, (tape, rec, folder) in tapes.items():
        assert tape["issued"].shape == (STEPS, 7) and np.isfinite(tape["issued"]).all()
        prov["sources"][mode] = dict(folder=str(folder), tape_sha256=sha(folder / "tape.npz"),
                                     source_record=rec, actions_sha256=visual.digest(tape["issued"]))
    dump(out / "provenance.json", prov)
    rows, pairs = {}, {}
    for mode in MODES:
        tape, rec, _ = tapes[mode]
        results = [execute(out, mode, material, path, bank, tape, rec) for material in MATERIALS]
        rows[mode] = [r[0] for r in results]
        pairs[mode] = paired(*results)
    control = rows["default"][0]
    d = pairs["default"]
    effective = bool(d["changed_realized_mu_by_geom_pair"]) and d["max_bottle_gap_m"] > .001
    gate0 = (control["archive_states_exact"] and control["archive_goal_hits_exact"] and effective
             and all(p["public_precontact_equal"] and p["reset_physics_equal"] for p in pairs.values()))
    scores = {}
    for endpoint in ("ever", "final"):
        fixed = {mode: sum(r[endpoint] for r in rows[mode]) for mode in MODES}
        oracle = sum(any(rows[mode][i][endpoint] for mode in MODES) for i in range(2))
        scores[endpoint] = dict(fixed=fixed, oracle=oracle, best_fixed=max(fixed.values()),
                                oracle_gain=oracle-max(fixed.values()))
    dump(out / "result.json", dict(gate0=gate0, physically_effective_default=effective,
                                  gate1=gate0 and scores["ever"]["oracle_gain"] > 0, scores=scores,
                                  pairs=pairs, rows=rows, gpu_hours=0))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
