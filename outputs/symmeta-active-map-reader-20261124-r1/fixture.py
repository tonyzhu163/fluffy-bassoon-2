"""Collect numeric physical banks. No target scoring or model fitting here."""
import hashlib
import io
import json
import os
import socket
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "symmeta-query-direction-probe-20261124-r1"),
                str(HERE.parents[1] / "src")]
from arm_env import ArmEnv
from collect import Recorder, restore, snapshot
from design import SCHEDULES, checked, inputs, save, schedule, source
from fluffy_bassoon.xemb.wider_family import interface_matrix, sample_interfaces

SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")
TARGET = (8, 6, 8, 8)
SOURCE_FAMILIES = ("native", "family", "swap", "aniso", "shear", "general+", "general-", "general")
TARGET_FAMILIES = ("native", "family", "swap", "general+", "general-")


def matrix(rng, name):
    if name == "native":
        return np.eye(2)
    family = name.rstrip("+-")
    for _ in range(200):
        candidates = interface_matrix(sample_interfaces(rng, family, 64))
        for a in candidates:
            if np.linalg.norm(a, np.inf) > 1:
                continue
            if name[-1] in "+-" and (name[-1] == "+") != (np.linalg.det(a) > 0):
                continue
            assert abs(np.linalg.det(a)) > 0
            return a
    raise ValueError(name)


def original(row):
    with np.load(io.BytesIO(checked(row["path"], row["sha256"]))) as z:
        u = z["issued_mean"][z["valid"]].astype(float)
        d = z["delta_eef"][z["valid"]].astype(float)
    assert u.shape == (3, 7) and d.shape == (3, 3)
    return u, d


def compact_inputs(arm, suite_i, task, fit, refs):
    suite = SUITES[suite_i]
    if arm in ("UR5e", "IIWA"):
        support, prop, _, obj, query, xyz, qr = inputs(refs, arm, suite, task)
        pos = 5 * max(0, len(query) // 10 - 2)
        return dict(proposals=prop, query=query, archived_xyz=xyz, prefix=pos,
                    support_layout=17, support_seed=24112600 + suite_i,
                    query_layout=14, query_seed=obj["case"]["seed"],
                    archived_reset=obj["full_reset"], source_record=support, query_record=qr)
    rows = sorted((r for r in fit["sources"] if r["arm"] == arm and r["task"] == f"{suite}/{task}"),
                  key=lambda r: r["start"])
    assert [r["start"] for r in rows] == list(range(4))
    # Historical nominal commands only. Each real command is the five-control mean repeated five times.
    basestart = (suite_i + task) % 4
    indices = [(basestart + i) % 4 for i in (1, 2)]
    prop = original(rows[basestart])[0]
    query = np.repeat(np.concatenate([original(rows[i])[0] for i in indices]), 5, axis=0)
    return dict(proposals=prop, query=query, archived_xyz=np.zeros((0,3)), prefix=0,
                support_layout=4 + (suite_i + task) % 4, support_seed=24112401 + 100*suite_i + task,
                query_layout=8 + (suite_i + task) % 3, query_seed=24112700 + 100*suite_i + task,
                source_record=rows[basestart], query_record=[rows[i] for i in indices])


class Numeric:
    """Use the exact reset, delta OSC and raw public encoders. Omit camera sampling after reset."""
    def __init__(self, env, initial):
        self.env = env._env
        for name, observable in self.env.env._observables.items():
            if "image" in name or "depth" in name:
                observable.set_enabled(False)
        self.rec = Recorder(env)
        self.initial = initial
        self.state = snapshot(self.env)

    def observe(self, obs):
        r = self.rec
        q = np.asarray(obs["robot0_joint_pos"]).copy()
        r.scratch.qpos[r.idx] = q
        import mujoco
        mujoco.mj_kinematics(r.sim.model._model, r.scratch)
        return dict(q=q, fk=r.scratch.site_xpos[r.site].copy(),
                    public_xyz=np.asarray(obs["robot0_eef_pos"]).copy(), time=float(r.sim.data.time))

    def run(self, actions, a, filename, initial=None, snap=None):
        restore(self.env, self.state if snap is None else snap)
        obs = self.initial if initial is None else initial
        rows = [self.observe(obs)]
        issued, physical, goals, terminal = [], [], [], []
        controller = self.rec.controller
        set_goal = controller.set_goal
        def goal(*args, **kwargs):
            result = set_goal(*args, **kwargs)
            goals.append((controller.goal_pos - controller.ee_pos).copy())
            return result
        with patch.object(controller, "set_goal", side_effect=goal):
            for u in actions:
                u = np.asarray(u, np.float32)
                ctrl = u.copy()
                ctrl[:2] = a @ u[:2]
                assert abs(u).max() <= 1 and abs(ctrl).max() <= 1
                obs, _, done, _ = self.env.step(ctrl.copy())
                issued.append(u.copy())
                physical.append(ctrl)
                term = bool(done or self.env.check_success())
                terminal.append(term)
                rows.append(self.observe(obs))
                if term:
                    break
        data = {k:np.asarray([r[k] for r in rows]) for k in rows[0]}
        data.update(issued=np.asarray(issued, np.float32).reshape(-1,7),
                    physical=np.asarray(physical, np.float32).reshape(-1,7),
                    goal_m=np.asarray(goals).reshape(-1,3), terminal=np.asarray(terminal, bool))
        np.testing.assert_allclose(data["goal_m"], .05*data["physical"][:,:3], atol=1e-8, rtol=0)
        assert len(data["goal_m"]) == len(data["issued"])
        np.savez_compressed(filename, **data)
        return obs, data


def run_case(out, refs, fit, amp, arm, suite_i, task, split):
    suite = SUITES[suite_i]
    folder = out / f"{arm}-{suite}-{task}"
    folder.mkdir()
    info = compact_inputs(arm, suite_i, task, fit, refs)
    p, query, xyz = (info.pop(k) for k in ("proposals", "query", "archived_xyz"))
    np.savez_compressed(folder/"archive-controls.npz", support_chunk_means=p, query=query, archived_query_xyz=xyz)
    target = split == "target"
    names = TARGET_FAMILIES if target else SOURCE_FAMILIES
    seed = (24112501 + suite_i*100 if target else
            24112401 + 10000 * (arm == "Sawyer") + suite_i*1000 + task*10)
    matrices = {name:matrix(np.random.default_rng(seed+i), name) for i, name in enumerate(names)}
    meta = dict(arm=arm, suite=suite, task=task, split=split, source_amplitude=amp,
                maps={k:dict(A=v.tolist(), seed=seed+i) for i,(k,v) in enumerate(matrices.items())}, **info)
    save(folder/"case.json", meta)
    with ArmEnv(arm, suite, task) as env:
        start = env.reset(info["support_layout"], info["support_seed"])
        meta.update(support_reset=start["environment_reproducibility"], model=start["model_identity"])
        sim = Numeric(env, start["observation"])
        for family, a in matrices.items():
            first = None
            for mode in SCHEDULES:
                u = np.repeat(np.asarray(list(schedule(p, mode, amp))), 5, axis=0)
                _, data = sim.run(u, a, folder/f"{family}-{mode}.npz")
                assert len(data["issued"]) == 15
                if first is None:
                    first = data
                else:
                    for key in ("fk", "q", "time"):
                        np.testing.assert_allclose(data[key][:6], first[key][:6], atol=1e-7, rtol=0)
        held = env.reset(info["query_layout"], info["query_seed"])
        meta["held_reset"] = held["environment_reproducibility"]
        assert meta["support_reset"]["released_row_fingerprint"] != meta["held_reset"]["released_row_fingerprint"]
        if target:
            for key in ("released_row_fingerprint", "bddl_sha256", "scene_fingerprint", "perturbation_plan_fingerprint"):
                assert meta["held_reset"][key] == info["archived_reset"][key]
        sim = Numeric(env, held["observation"])
        n = info["prefix"]
        obs, prefix = sim.run(query[:n], np.eye(2), folder/"native-prefix.npz")
        if len(prefix["issued"]) == n and not prefix["terminal"].any():
            middle = snapshot(sim.env)
            for family, a in matrices.items():
                sim.run(query[n:n+20], a, folder/f"{family}-held.npz", obs, middle)
    meta["complete"] = True
    save(folder/"case.json", meta)
    print(arm, suite, task, split, "complete", flush=True)


def main():
    out, cpu, arm = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    assert os.sched_getaffinity(0) == {cpu} and os.environ["CUDA_VISIBLE_DEVICES"] == ""
    refs, fit, amp = source()
    source_arm = arm in ("Panda", "Sawyer")
    assert source_arm or arm in ("UR5e", "IIWA")
    if source_arm:
        # Sufficient raw inputs to refit source-only laws for the cross-body diagnostics.
        rows = [r for r in fit["sources"] if r["arm"] == arm]
        arrays = [original(r) for r in rows]
        np.savez_compressed(out/f"law-{arm}.npz", task=[r["task"] for r in rows],
                            u=[u[:,:3] for u,_ in arrays], d=[d for _,d in arrays])
    save(out/f"runtime-{arm}.json", dict(arm=arm, host=socket.gethostname(), cpu=cpu,
         prior=fit["prior"], residual=fit["residual_power"], source_hash=refs["source_fit_sha256"], amplitude=amp))
    for i, suite in enumerate(SUITES):
        ids = sorted({int(r["task"].split("/")[1]) for r in fit["sources"] if r["task"].split("/")[0]==suite})
        assert len(ids) == 6
        for j, task in enumerate(ids if source_arm else [TARGET[i]]):
            split = ("train" if j < 4 else "val" if j == 4 else "test") if source_arm else "target"
            run_case(out, refs, fit, amp, arm, i, task, split)


if __name__ == "__main__":
    main()
