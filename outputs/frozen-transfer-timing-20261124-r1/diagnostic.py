"""Matched, reward-blind fitting and native intervention on archived frozen actions."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import itertools
import json
import os
from pathlib import Path
import time

import numpy as np

DATA = Path("/mowenjie/fluffy-bassoon-2/outputs")
FIT = DATA / "libero-retry-headroom-20261119-r1/paired01"
TEST = DATA / "frozen-vla-native-transfer-20261123-r1/gpu01"
TASKS = [("libero_object", 0), ("libero_object", 7), ("libero_spatial", 1), ("libero_spatial", 7)]
QUERIES = [("libero_10", 8), ("libero_goal", 3), ("libero_goal", 6)]
MODES = ("identity", "static", "timing", "shifted")
TARGETS = ("IIWA", "UR5e")
FIT_PIN = "c45aa965db73ad0644382da578ff03a30b2083d7"
TEST_PIN = "a8a997e7d06bb131d3f9710204c58e106f0f0904"


def read(path):
    return json.loads(path.read_bytes())


def write(path, obj):
    with path.open("x") as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write("\n")


def digest(a):
    return sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def inputs(role, suite, task, layout):
    if role == "fit":
        suffix = (1, 4).index(layout)
        base = FIT / f"episodes/{suite}-{task}-r{suffix}-BASE/default"
        meta, tape = base / "result.json", base / "public.npz"
        d = read(meta)
        assert d["mode"] == "default" and FIT_PIN in (FIT / "launch.txt").read_text()
        with np.load(tape, allow_pickle=False) as f:
            u = f["issued"]
            xyz = f["positions"]
            np.testing.assert_array_equal(f["decision_offsets"], np.arange(0, len(u), 5))
            assert not f["waits"].any()
        case, reset = d["case"], d["identity"]["physics"]
        files = [meta, tape]
    else:
        tape = TEST / f"{suite}-{task}-{layout}-Panda.json"
        d = read(tape)
        assert TEST_PIN in (TEST / "launch.txt").read_text()
        assert d["case"]["arm"] == d["model"]["arm"] == "Panda"
        assert d["record"]["inferred"] == dict(yaw_deg=0., xy_scale=1., gripper_sign=1)
        chunks = d["chunks"]
        assert all(len(c["cmd"]) == 5 for c in chunks[:-1]) and 0 < len(chunks[-1]["cmd"]) <= 5
        u = np.concatenate([c["cmd"] for c in chunks])
        xyz = np.concatenate([c["xyz"] for c in chunks])
        case, reset = d["case"], d["full_reset"]["initial_state_fingerprint"]
        files = [tape]
    assert (case["suite"], case["task_id"], case["init_state_id"], case["variant"]) == (suite, task, layout, "BASE")
    u = np.clip(u, -1, 1).astype(np.float32)
    assert u.shape == (len(xyz), 7) and xyz.shape == (len(u), 3) and np.isfinite(u).all()
    event = np.flatnonzero(u[:, 6] > 0)
    start = (int(event[0]) // 5 + 1) * 5 if event.size else None
    return u, xyz, case, reset, start, {str(p): sha256(p.read_bytes()).hexdigest() for p in files}


def adapt(block, last_z, mode, model):
    new = block.copy()
    if mode == "identity":
        return new, np.zeros(len(block)), 0
    z = block[:, 2].astype(float)
    gain = model["g0"] if mode == "static" else model["g"]
    d = np.zeros(len(block))
    if mode in ("timing", "shifted"):
        d = model["b"] * np.diff(z, prepend=last_z)
        if mode == "shifted":
            d = np.roll(d, 2)
    updated = gain * z + d
    new[:, 2] = np.clip(updated, -1, 1)
    return new, d, int(np.sum(np.abs(updated) > 1))


class Contacts:
    """Log external, force-bearing finger contacts during the registered window."""
    def __init__(self, env):
        import mujoco
        robot = env.robots[0]
        sim = self.sim = env.sim
        m = self.model = sim.model._model
        self.fingers = {m.geom(f"gripper0_finger{i}{pad}_collision").id
                        for i in (1, 2) for pad in ("", "_pad")}
        own_bodies = set(robot.robot_model.bodies) | set(robot.gripper.bodies)
        if robot.robot_model.mount:
            own_bodies.update(robot.robot_model.mount.bodies)
        self.external = {i: (m.geom(i).name or m.body(m.geom_bodyid[i]).name)
                         for i in range(m.ngeom) if m.body(m.geom_bodyid[i]).name not in own_bodies}
        self.names = None
        force = np.zeros(6)
        advance = sim.step

        def measure():
            advance()
            if self.names is None:
                return
            data = sim.data._data
            for k, c in enumerate(data.contact):
                if c.efc_address < 0:
                    continue
                for a, b in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                    if a in self.fingers and b in self.external:
                        mujoco.mj_contactForce(m, data, k, force)
                        if force[0] > 0:
                            self.names.add(self.external[b])
        sim.step = measure


def simulate(work):
    from arm_env import ArmEnv
    from OpenGL import GL
    role, suite, task, layout, arm, mode, model, directory = work
    out = Path(directory)
    assert len(os.sched_getaffinity(0)) <= 2 and os.getenv("CUDA_VISIBLE_DEVICES") == ""
    u, ref, case, orig_reset, start, files = inputs(role, suite, task, layout)
    name = f"{role}-{suite}-{task}-{layout}-{arm}-{mode}"
    t0 = time.monotonic()
    with ArmEnv(arm, suite, task) as session:
        first = session.reset(layout, case["seed"])
        env = session._env
        renderer = GL.glGetString(GL.GL_RENDERER).decode()
        assert "llvmpipe" in renderer.lower()
        model_id = first["model_identity"]
        assert (model_id["control_freq"], model_id["controller"], model_id["gripper"]) == (20, "OSC_POSE", "PandaGripper")
        for key, sensor in env.env._observables.items():
            if key.endswith("_image"):
                sensor.set_enabled(False)
                sensor.set_active(False)
        contacts = Contacts(env)
        xyz = [first["observation"]["robot0_eef_pos"].tolist()]
        rewards, issued, hits, corrections, clip_count = [], [], [], [], 0
        horizon = first["max_steps"]
        stop = min(len(u), horizon)
        if role == "fit" and start is not None:
            stop = min(stop, start + 10)
        prefix = None
        terminal = False
        for i in range(0, stop, 5):
            active = start is not None and start <= i < start + 10
            block = u[i:min(i+5, stop)]
            if i == start:
                prefix = digest(env.get_sim_state())
            if active:
                block, corr, nclip = adapt(block, u[i-1, 2], mode, model)
                corrections.extend(corr.tolist())
                clip_count += nclip
            for action in block:
                contacts.names = set() if active else None
                raw, reward, done, _ = env.step(action.copy())
                success = bool(env.check_success())
                assert reward in (0., 1.) and bool(reward) == success
                xyz.append(np.asarray(raw["robot0_eef_pos"], dtype=np.float32).tolist())
                rewards.append(float(reward))
                issued.append(action)
                hits.append(sorted(contacts.names) if active else [])
                terminal = bool(done or success)
                if terminal:
                    break
            if terminal:
                break
        xyz, issued = np.asarray(xyz), np.array(issued)
        n = len(rewards)
        if start is None:
            np.testing.assert_array_equal(issued, u[:n])
        else:
            np.testing.assert_array_equal(issued[:start], u[:min(start, n)])
            np.testing.assert_array_equal(issued[:, [0, 1, 3, 4, 5, 6]], u[:n, [0, 1, 3, 4, 5, 6]])
            np.testing.assert_array_equal(issued[start+10:], u[start+10:n])
        if mode == "identity":
            np.testing.assert_array_equal(issued, u[:n])
        reset = first["environment_reproducibility"]
        rec = dict(role=role, case=case, arm=arm, mode=mode, files=files, renderer=renderer,
                   sim_model=model_id, scene=reset["scene_fingerprint"], reset=reset["initial_state_fingerprint"],
                   archive_reset=orig_reset, before_window=prefix, trigger_start=start, horizon=horizon,
                   n=n, tape_length=len(u), terminal=terminal, exhausted_horizon=n==horizon,
                   native_return=sum(rewards), tape_censored=bool(role == "test" and not terminal and n<horizon),
                   prefix_xyz_hash=digest(xyz[:start+1] if start is not None else xyz),
                   prefix_actions_hash=digest(issued[:start]), commands_hash=digest(issued),
                   clip_count=clip_count, derivative=corrections,
                   archive_panda_max_abs_mm=(float(1000*np.abs(xyz[1:] - ref[:n]).max()) if arm=="Panda" else None),
                   elapsed_s=round(time.monotonic()-t0, 3))
        payload = dict(**rec, xyz_m=xyz.tolist(), issued=issued.tolist(), rewards=rewards, finger_contacts=hits)
        write(out / f"{name}.json", payload)
        print(name, "controls", n, "native" if role=="test" else "fit", rec["native_return"] if role=="test" else "", flush=True)
    return name


def fit_model(directory):
    pairs = []
    cases = []
    for suite, task in TASKS:
        for layout in (1, 4):
            p, s = [read(directory / f"fit-{suite}-{task}-{layout}-{a}-identity.json") for a in ("Panda","Sawyer")]
            assert p["scene"] == s["scene"] and p["files"] == s["files"] and p["trigger_start"] == s["trigger_start"]
            start = s["trigger_start"]
            ok = start is not None and min(p["n"],s["n"]) >= start+10
            cases.append(dict(suite=suite,task=task,layout=layout,start=start,complete=ok))
            if ok:
                pv, sv = [np.diff(np.array(r["xyz_m"])[:,2]) for r in (p,s)]
                y, x = pv[start:start+10], sv[start:start+10]
                pairs.append((y, x, x-sv[start-1:start+9]))
    y, x, dx = np.concatenate(pairs, axis=1)
    design = np.column_stack((x, dx))
    coef, _, rank, sv = np.linalg.lstsq(design, y, rcond=None)
    g0 = float(x @ y / (x @ x))
    model = dict(g0=g0, g=float(coef[0]), b=float(coef[1]), rank=int(rank), singular_values=sv.tolist(),
                 fit_cases=cases, n_controls=len(y), plan_sha256=sha256(Path(__file__).with_name("PLAN.md").read_bytes()).hexdigest())
    assert rank == 2 and g0>0 and model["g"]>0 and np.isfinite(coef).all()
    model["effective_lead_ticks"] = model["b"]/model["g"]
    for label, v in (("identity",x),("static",g0*x),("timing",design@coef)):
        model[f"fit_{label}_mm_per_tick"] = float(1000*np.sqrt(np.mean((y-v)**2)))
    return model


def run(role, out):
    start = time.monotonic()
    if role == "fit":
        jobs = [("fit",suite,t,i,a,"identity",{},str(out)) for suite,t in TASKS for i in (1,4) for a in ("Panda","Sawyer")]
    else:
        model = read(out.parent / "fit/model.json")
        jobs = [("test",suite,t,i,a,m,model,str(out)) for suite,t in QUERIES for i in (14,15)
                for a in ("Panda",*TARGETS) for m in (("identity",) if a=="Panda" else MODES)]
    with ProcessPoolExecutor(max_workers=2) as pool:
        list(pool.map(simulate, jobs, chunksize=1))
    if role == "fit":
        write(out/"model.json", fit_model(out))
    write(out/"completed.json", dict(count=len(jobs), elapsed_s=time.monotonic()-start, cores=sorted(os.sched_getaffinity(0))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("role", choices=["fit","test"])
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    run(args.role, args.out)
