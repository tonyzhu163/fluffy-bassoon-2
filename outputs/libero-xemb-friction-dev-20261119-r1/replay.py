"""Native open-loop replay of the two preselected programs on eight development tasks."""
import hashlib
import json
import os
import random
import sys
from importlib.metadata import version
from pathlib import Path

import mujoco
import numpy as np
from OpenGL import GL
from libero.libero import benchmark
from libero.libero.envs.env_wrapper import ControlEnv
from libero_remote.environment.libero import LiberoEnvironmentAdapter
from libero_remote.environment.task_manifest import TaskManifest

HERE = Path(__file__).resolve().parent
ROOT = Path("/mowenjie/fluffy-bassoon-2")
SOURCE = ROOT / "outputs/libero-retry-headroom-20261119-r1/paired01/episodes"
RELEASE = ROOT / "sources/libero-pro2-e5500c69"
MODES = ("default", "paced")
MATERIALS = ("ordinary", "reduced")
CAMERAS = ("agentview", "robot0_eye_in_hand")
SUFFIXES = ("finger1_collision", "finger2_collision", "finger1_pad_collision", "finger2_pad_collision")
FIELDS = ("robot0_joint_pos", "robot0_joint_vel", "robot0_eef_pos", "robot0_eef_quat_xyzw", "robot0_gripper_qpos")


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(array):
    return hashlib.sha256(str((array.shape, array.dtype.str)).encode() + array.tobytes()).hexdigest()


def physical_state(sim):
    model, data = sim.model._model, sim.data._data
    fmt = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(model, fmt))
    mujoco.mj_getState(model, data, state, fmt)
    return {k: digest(v) for k, v in dict(integration=state, body_xyz=data.xpos,
                                           body_rot=data.xmat, camera_xyz=data.cam_xpos,
                                           camera_rot=data.cam_xmat).items()}


class Contacts:
    def __init__(self, world, material):
        self.sim = world.sim
        model = self.sim.model
        self.names = [model.geom_id2name(i) or f"<geom-{i}>" for i in range(model.ngeom)]
        ids = []
        for suffix in SUFFIXES:
            found = [i for i, n in enumerate(self.names) if n.endswith("_" + suffix)]
            assert len(found) == 1
            ids.extend(found)
        self.fingers = np.zeros(model.ngeom, dtype=bool)
        self.fingers[ids] = True
        self.owners = {}
        for kind, objects in (("object", world.objects_dict), ("fixture", world.fixtures_dict)):
            for name, obj in objects.items():
                for geom in obj.contact_geoms:
                    self.owners[geom] = (kind, name)
        before = model.geom_friction.copy()
        if material == "reduced":
            model.geom_friction[ids, 0] = 0.05
        want = before.copy()
        if material == "reduced":
            want[ids, 0] = 0.05
        np.testing.assert_array_equal(want, model.geom_friction)
        self.settings = {self.names[i]: [before[i].tolist(), model.geom_friction[i].tolist()] for i in ids}
        self.control = self.integration = 0
        self.first = None
        self.pairs = {}
        self.native_step = self.sim.step
        self.sample()
        self.sim.step = self.step

    def sample(self):
        contact = self.sim.data._data.contact
        pairs = contact.geom
        active = np.flatnonzero(self.fingers[pairs].any(axis=1))
        if self.first is None and len(active):
            self.first = dict(control=self.control, substep=self.integration,
                              geoms=[self.names[g] for g in pairs[active[0]]])
        for i in active:
            c = contact[int(i)]
            if c.efc_address < 0:
                continue
            a, b = pairs[i]
            finger, other = (a, b) if self.fingers[a] else (b, a)
            finger, other = self.names[finger], self.names[other]
            key = finger + " / " + other
            mu = float(c.friction[0])
            if key not in self.pairs:
                kind, owner = self.owners.get(other, ("other", None))
                self.pairs[key] = dict(kind=kind, owner=owner, first_control=self.control,
                                       min_mu=mu, max_mu=mu, count=0)
            row = self.pairs[key]
            row["count"] += 1
            row["min_mu"] = min(row["min_mu"], mu)
            row["max_mu"] = max(row["max_mu"], mu)

    def step(self):
        self.native_step()
        self.integration += 1
        self.sample()


def replay(out, label, bank, path, rec, tape, material):
    mode, case, ident = rec["mode"], rec["case"], rec["identity"]
    horizon = ident["horizon"]
    np.random.seed(case["seed"])
    random.seed(case["seed"])
    env = LiberoEnvironmentAdapter()
    env._init_states = bank
    env._max_steps = horizon
    env._env = ControlEnv(str(path), use_camera_obs=False, camera_names=[], has_offscreen_renderer=True)
    try:
        first = env.reset(case["init_state_id"], case["seed"])
        physics = first["environment_reproducibility"]["initial_state_fingerprint"]
        raw = first["observation"]
        assert set(raw) == set(FIELDS) and not env._env.check_success()
        world = env._env.env
        assert world.perturbation_runtime.diagnostics()["case_id"] == "BASE"
        sim = world.sim
        contacts = Contacts(world, material)
        frame_state = physical_state(sim)
        views = {}
        for camera in CAMERAS:
            sensors, names = world._create_camera_sensors(camera, 256, 256, False, None)
            image = sensors[0]({})
            assert names[0] == camera + "_image" and image.shape == (256, 256, 3) and image.dtype == np.uint8
            views[names[0]] = digest(image)
        assert frame_state == physical_state(sim) and env._step_index == 0
        renderer = GL.glGetString(GL.GL_RENDERER).decode()
        assert "llvmpipe" in renderer.lower() and os.environ["CUDA_VISIBLE_DEVICES"] == ""
        states = {k: [raw[k].copy()] for k in FIELDS}
        start_public = {k: digest(v) for k, v in raw.items()}
        rewards, flags = [], []
        for idx, action in enumerate(tape["issued"], 1):
            contacts.control = idx
            step = env.step(action.copy())
            assert step["worker_step_index"] == idx
            for k in FIELDS:
                states[k].append(step["observation"][k].copy())
            rewards.append(step["reward"])
            flags.append([step[k] for k in ("success", "terminated", "truncated")])
            if step["terminated"] or step["truncated"]:
                break
        assert len(rewards) <= horizon and contacts.integration >= len(rewards)
    finally:
        env.close()
    n = len(rewards)
    states = {k: np.stack(v) for k, v in states.items()}
    arrays = dict(**states, rewards=np.asarray(rewards), flags=np.asarray(flags), issued=tape["issued"][:n])
    np.savez_compressed(out / f"{label}-{mode}-{material}.npz", **arrays)
    success, terminated, truncated = flags[-1]
    err = np.linalg.norm(states["robot0_eef_pos"][1:].astype(float) - tape["positions"][:n], axis=1)
    repro = dict(success_match=rec["success"] == success, length_match=rec["controls"] == n,
                 reward_mismatches=int(np.count_nonzero(arrays["rewards"] != tape["rewards"][:n])),
                 wrist_exact_steps=int(np.count_nonzero(err == 0)), wrist_max_error_m=float(max(err)))
    row = dict(task=label, mode=mode, material=material, seed=case["seed"], reset=case["init_state_id"],
               physics=physics, frame_state=frame_state, public=start_public, cameras=views, renderer=renderer,
               archive_physics_match=physics == ident["physics"],
               archive_public_match={k: v == ident["public"][k] for k, v in {**start_public, **views}.items()},
               settings=contacts.settings, solver_pairs=contacts.pairs, first_finger=contacts.first,
               integration_steps=contacts.integration, tape_length=rec["controls"], horizon=horizon,
               controls=n, success=success, terminated=terminated, truncated=truncated,
               stop="terminated" if terminated else "truncated" if truncated else "tape_end",
               native_return=sum(rewards), final_reward=rewards[-1], native_reward_equals_success=bool(
                   np.array_equal(arrays["rewards"], arrays["flags"][:, 0])),
               archive_success=rec["success"], archive_return=float(tape["rewards"].sum()),
               ordinary_reproduction=repro if material == "ordinary" else None)
    with (out / "branches.jsonl").open("a") as f:
        print(json.dumps(row, allow_nan=False), file=f)
    print(label, mode, material, n, row["stop"], row["native_return"], flush=True)
    return row, arrays


def pair_metrics(a, b):
    x, xs = a
    y, ys = b
    common_length = min(x["controls"], y["controls"])
    first = min((r["first_finger"]["control"] if r["first_finger"] else common_length + 1) for r in (x, y))
    last = min(common_length, first - 1)
    fields_equal = {k: np.array_equal(xs[k][:last + 1], ys[k][:last + 1]) for k in FIELDS}
    actions_equal = np.array_equal(xs["issued"][:common_length], ys["issued"][:common_length])
    flags_equal = all(np.array_equal(xs[k][:max(0,last)], ys[k][:max(0,last)]) for k in ("rewards","flags"))
    common = x["solver_pairs"].keys() & y["solver_pairs"].keys()
    changed = {}
    for name in sorted(common):
        ax, by = x["solver_pairs"][name], y["solver_pairs"][name]
        if (ax["min_mu"], ax["max_mu"]) != (by["min_mu"], by["max_mu"]):
            changed[name] = [ax["kind"], ax["owner"], ax["min_mu"], ax["max_mu"], by["min_mu"], by["max_mu"]]
    return dict(physics_equal=x["physics"] == y["physics"], frame_state_equal=x["frame_state"] == y["frame_state"],
                camera_equal={k: v == y["cameras"][k] for k, v in x["cameras"].items()},
                last_complete_precontact_control=last, precontact_equal=fields_equal,
                precontact_rewards_flags_equal=flags_equal, matched_actions_in_overlap=actions_equal,
                common_solver_pairs=sorted(common), changed_mu=changed)


def main(out):
    splits = read(HERE.parent / "libero-meta-learning-20261117-r1/SPLITS.json")
    design = read(SOURCE / "design.json")
    assert design["splits"] == splits["development"] and design["modes"] == {"default": [5, 0], "paced": [10, 14]}
    assert Path(design["checkpoint"]).name == "pi05-libero-openpi-20261027-r1"
    manifest = TaskManifest.load(str(RELEASE / "assets/task_manifest.json"))
    assert manifest.sha256 == splits["manifest_sha256"]
    schedule = [(s, t) for s, tasks in sorted(splits["development"].items()) for t in tasks]
    assert len(schedule) == len(set(schedule)) == 8
    prov = dict(archive=str(SOURCE), design=design, packages={k: version(k) for k in
                ("mujoco", "robosuite", "numpy", "PyOpenGL")}, sources={})
    panel, pairs = {}, {}
    for suite, task in schedule:
        task_label = f"{suite}-{task}"
        path = manifest.resolve_bddl(suite, task, "BASE")
        bank = benchmark.get_benchmark_dict()[suite]().get_task_init_states(task)
        panel[task_label] = {}
        pairs[task_label] = {}
        source_ids = []
        for mode in MODES:
            folder = SOURCE / f"{task_label}-r0-BASE" / mode
            record = read(folder / "result.json")
            seed = 19110001 + sorted(splits["development"]).index(suite)*10000 + task*100
            assert record["case"] == dict(suite=suite, task_id=task, init_state_id=1, seed=seed, variant="BASE")
            ident = record["identity"]
            assert record["mode"] == mode and ident["horizon"] == manifest.require(suite, task, "BASE")["max_steps"]
            with np.load(folder / "public.npz", allow_pickle=False) as data:
                tape = {k: data[k] for k in ("issued", "positions", "rewards", "waits", "opening")}
            n = record["controls"]
            assert 0 < n <= ident["horizon"] and tape["issued"].shape == (n, 7)
            assert tape["positions"].shape == (n, 3) and tape["rewards"].shape == (n,)
            assert all(np.isfinite(v).all() for v in tape.values())
            assert record["waits"] == sum(tape["waits"]) and digest(tape["opening"]) == record["opening_sha256"]
            source_ids.append((ident, record["opening_sha256"]))
            prov["sources"][f"{task_label}-{mode}"] = dict(folder=str(folder), record_sha256=sha(folder / "result.json"),
                npz_sha256=sha(folder / "public.npz"), actions_sha256=digest(tape["issued"]), bddl=str(path),
                bddl_sha256=sha(Path(path)), record=record)
            results = [replay(out, task_label, bank, path, record, tape, mat) for mat in MATERIALS]
            panel[task_label][mode] = [r["success"] for r, _ in results]
            pairs[task_label][mode] = pair_metrics(*results)
        assert source_ids[0] == source_ids[1]
        write(out / "sources.json", prov)
        write(out / "pairs.json", pairs)
    maps = {m0 + "/" + m1: sum(v[m0][0] + v[m1][1] for v in panel.values()) for m0 in MODES for m1 in MODES}
    task_fixed = sum(max(sum(v[m]) for m in MODES) for v in panel.values())
    full = sum(any(v[m][i] for m in MODES) for v in panel.values() for i in range(2))
    write(out / "score.json", dict(panel=panel, material_maps=maps, best_fixed=max(maps[m+"/"+m] for m in MODES),
                                   material_only=max(maps.values()), task_fixed=task_fixed, task_and_material=full))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
