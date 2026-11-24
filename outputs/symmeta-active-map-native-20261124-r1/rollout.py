"""Use the released fixed motor loop and observer. Apply the full XY correction at its issued-command port."""
import argparse
import gzip
import os
from pathlib import Path
import socket
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from cohort import (ALONG, ESSENTIAL, HERE, MANIFEST_SHA, RELEASE_PIN, ROBOTS, array_sha, assignments,
                    atomic, bounded_issue, checked, digest, encode, official_banks, order, parse, query, require)


class FixturePolicy:
    _model = SimpleNamespace(action_horizon=10, action_dim=32)

    def infer(self, item, *, noise):
        return {"actions":noise[:, :7].copy()}  # exercises nominal clipping, including non-XY channels


class Observe:
    def __init__(self, sim, policy, row, estimate):
        self.sim, self.policy, self.row, self.estimate = sim, policy, row, estimate
        self._model = policy._model

    def call(self, op, value):
        if op == "create":
            assert value == dict(query(self.row), embodiment="nom")
            spec, self.first = self.sim.call(op, dict(value, forward_map=self.row["true_map"]))
            assert spec == dict(value, max_steps=self.row["reference"]["horizon"])
            self.chunks = []
            self.latest = self.first
            return spec, self.first
        assert op == "chunk"
        b = self.chunks[-1]
        raw = np.asarray(b["raw"], np.float32)
        np.testing.assert_array_equal(value, np.clip(raw[:len(value)], -1, 1))
        issued = bounded_issue(value, self.estimate)
        self.latest = self.sim.call(op, issued)
        steps = self.latest["native_steps"]
        np.testing.assert_array_equal(issued[:len(steps)], np.asarray([s["issued"] for s in steps], np.float32))
        b["steps"] = steps
        return self.latest

    def infer(self, item, *, noise):
        from ur5e_motor import to_openpi_observation
        expected = to_openpi_observation(self.latest["observation"], self.first["language_instruction"])
        assert set(expected) == set(item) == {"prompt","observation/image","observation/wrist_image","observation/state"}
        assert all(np.array_equal(v, item[k]) for k,v in expected.items())
        pred = self.policy.infer(item, noise=noise)
        self.chunks.append(dict(raw=np.asarray(pred["actions"], np.float32).tolist(), noise=array_sha(noise),
                                motor_inputs={k:array_sha(v) for k,v in item.items() if k!="prompt"}))
        return pred


def episode(sim, motor, row, name, anchor, cpu):
    from ur5e_motor import run_episode
    from fluffy_bassoon.xemb.embodiment import get_embodiment_spec
    from audit import inspect_episode, opening

    estim = row["estimates"][name]
    a = np.asarray(estim["A"], dtype=estim["dtype"]) if estim else None
    obs = Observe(sim, motor, row, a)

    class CreateMismatch(ValueError):
        pass

    def check_reset(reset):
        if anchor and anchor["reset"] != reset:
            raise CreateMismatch("pre-inference reset differs from registered pair")
        require(reset["physics"] != row.get("support_physics"), "query and support share physics")

    for attempt in range(1,4):
        try:
            native = run_episode(obs, obs, query(row), get_embodiment_spec("nom"), check_reset=check_reset,
                                 max_controls=5 if cpu else None)
            break
        except CreateMismatch:
            if attempt == 3:
                raise
    obj = dict(native=native, env=obs.first["environment_reproducibility"], body=obs.first["model_identity"],
               chunks=obs.chunks, pre_action_create_attempt=attempt)
    inspect_episode(row, name, obj, cpu)
    if anchor:
        require(anchor == opening(obj), "paired reset, physics, frames, noise or opening proposal mismatch")
    return obj


def cpu_cases(refs):
    rows = []
    # Both bodies, orientations and official layouts. Real resets, synthetic source-independent corrections.
    for r in assignments():
        if r["index"] not in (0, 7, 10, 13, 17, 22, 27, 28):
            continue
        truth = [[.20,.60],[.45,-.35]] if r["family"]=="general-" else [[.20,-.60],[.45,.35]]
        est = {"A":[[.25,.5],[.125,.25]],"dtype":"float32"}
        rows.append(dict(r, true_map=truth, estimates={"raw":None,"true":dict(A=truth,dtype="float64"),
                         "fixture_singular":est}, reference=refs[r["suite"],r["task"],r["layout"]]))
    return rows


def run(a):
    import ur5e_sim_worker as worker
    from ur5e_motor import OPENPI_CKPT, load_frozen_pi05
    from audit import opening

    cpu = a.mode == "cpu"
    if cpu:
        require(os.environ["CUDA_VISIBLE_DEVICES"] == "" and len(os.sched_getaffinity(0)) == 1, "one CPU, no CUDA")
        cases = cpu_cases(official_banks())
        methods_by_phase = [("raw","true","fixture_singular")]
        policy = FixturePolicy()
        panel_sha = digest(encode(cases))
        atomic(a.out/"cpu-cases.json", encode(cases))
    else:
        panel = parse(checked(a.panel, a.sha256))
        require(panel["assignments"] == assignments() and panel["registration"] == digest((HERE/"PLAN.md").read_bytes()),
                "native registration changed")
        cases = [r for r in panel["cases"] if r["arm"] == a.arm]
        require(len(cases)==16 and [r["index"] for r in cases]==[r["index"] for r in assignments() if r["arm"]==a.arm],
                "wrong arm shard")
        checked(OPENPI_CKPT/"transport-manifest.json","28310343bab088490f4e84110ce13861ccdeb3eda1bb4dd7c4bc26df84b8428c")
        ready = parse(checked(OPENPI_CKPT/"READY.json","ec064cce32f5003e2e3979c5712d7d6385018f6bfcc59cb2161105881e70d170"))
        require(ready["status"]=="verified" and ready["checkpoint_path"]==str(OPENPI_CKPT), "stock pi05")
        probe = parse((a.out/"gpu.json").read_bytes())
        require(probe["hostname"]==socket.gethostname() and probe["status"]=="passed" and probe["purpose"]=="evaluation"
                and probe["allocation_type"]=="baige_container" and len(probe["devices"])==1
                and "RTX 4090" in probe["devices"][0]["cuda_name"].upper(), "GPU allocation")
        policy = load_frozen_pi05()
        panel_sha = a.sha256
        methods_by_phase = [ESSENTIAL, ALONG]
    require((policy._model.action_horizon, policy._model.action_dim)==(10,32), "stock motor horizon")
    with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES":""}), patch.object(worker,"__file__",str(HERE/"map_worker.py")):
        sim = worker.UR5eSimulator(allow_mesa=cpu)
    anchors = {}
    with sim:
        for methods in methods_by_phase:
            previous = None
            for row in cases:
                if (row["arm"],row["task_index"]) != previous:
                    previous = row["arm"],row["task_index"]
                    sim.call("create", dict(query(row), embodiment="nom", forward_map=row["true_map"]))
                i = row["index"]
                for name in order(row, methods):
                    obj = episode(sim,policy,row,name,anchors.get(i),cpu)
                    anchors[i] = opening(obj)
                    receipt = dict(panel_sha256=panel_sha,index=i,method=name,cpu_fixture=cpu,episode=obj)
                    atomic(a.out/"episodes"/f"{i:02d}-{name}.json.gz",gzip.compress(encode(receipt),compresslevel=2,mtime=0))
                group = dict(index=i,panel_sha256=panel_sha,methods=list(order(row,methods)),opening=anchors[i])
                atomic(a.out/"groups"/f"{i:02d}-{methods[0]}.json", encode(group))
                print("verified group",i,methods[0],flush=True)
    atomic(a.out/"completed.json",encode(dict(cpu_fixture=cpu,groups=len(anchors),panel_sha256=panel_sha,
           episodes=sum(len(m) for m in methods_by_phase)*len(cases),scored_native_retries=0 if cpu else 128)))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("cpu","gpu"))
    p.add_argument("out", type=Path)
    p.add_argument("--arm",choices=ROBOTS)
    p.add_argument("--panel",type=Path)
    p.add_argument("--sha256")
    run(p.parse_args())
