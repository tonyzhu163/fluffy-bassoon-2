"""Native frozen motor retry with a separate precomputed calibration yaw."""
import argparse
import gzip
import os
from pathlib import Path
import socket
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from protocol import HERE, atomic, digest, encode, historical, parse, require
from score import fp, inspect
from paired_inputs import native
from fluffy_bassoon.xemb.embodiment import get_embodiment_spec, retarget_observation_for_motor
import ur5e_sim_worker


class CPUActionFixture:
    """Exercise the production interface with reproducible nontrivial commands, without loading a motor."""
    _model = SimpleNamespace(action_horizon=10, action_dim=32)

    def infer(self, item, *, noise):
        return {"actions": np.tanh(noise[:, :7]).astype("f4")}


class ObservedQuery:
    def __init__(self, sim, policy, ev, row, correction):
        self.sim, self.policy, self.ev = sim, policy, ev
        self.row, self.correction = row, correction
        self._model = policy._model

    def call(self, op, value):
        result = self.sim.call(op, value)
        if op == "create":
            spec, self.first = result
            require(spec == dict(self.row["query"], embodiment=self.row["embodiment"],
                                 max_steps=self.row["reference"]["horizon"]), "native query")
            self.latest = self.first
            self.chunks = []
        else:
            require(op == "chunk" and len(result["native_steps"]) == result["steps"], "unexpected motor wire call")
            self.chunks[-1].update(issued=value[:result["steps"]].tolist(), steps=result["native_steps"])
            self.latest = result
        return result

    def infer(self, item, *, noise):
        observation = self.latest["observation"]
        if self.chunks and self.correction["yaw_deg"]:
            observation = retarget_observation_for_motor(observation, self.first["anchor_eef_pos"],
                                                         self.correction["yaw_deg"])
        expected = self.ev.to_openpi_observation(observation, self.row["prompt"])
        for k in expected:
            require(np.array_equal(expected[k], item[k]), f"motor input {k}")
        result = self.policy.infer(item, noise=noise)
        proposal = np.asarray(result["actions"], dtype=np.float32)
        self.chunks.append(dict(noise=fp(noise), motor_inputs={k:fp(v) for k,v in item.items() if k!="prompt"},
                                raw=proposal.tolist()))
        return result


def episode(ev, sim, base, row, name, resets, cpu):
    c = row["corrections"][name]
    interface = ObservedQuery(sim, base, ev, row, c)
    spec = get_embodiment_spec(row["embodiment"])
    rec = ev.run_paired_query(ev.run_episode, interface, interface, row["query"], spec, arm_name=name,
                              key=row["index"], resets=resets, expected_prompt=row["prompt"],
                              support_physics=row.get("calibration_physics", row["support_physics"]),
                              create_attempts=3, yaw_hat=c["yaw_deg"], scale_hat=c["xy_scale"],
                              grip_hat=c["gripper_sign"], max_controls=10 if cpu else None)
    obj = dict(native=rec, chunks=interface.chunks, env=interface.first["environment_reproducibility"],
               body=interface.first["model_identity"])
    inspect(row, name, obj, cpu)
    return obj


def fixtures(rows):
    out = []
    for r in rows:
        t, m = r["task_shard"], r["slot"] - 4
        if m not in (t, (t+2)%8):
            continue
        # Known historical predictions exercise yaw-only changes on each native release layout.
        c = dict(r["nominal"])
        out.append(dict(r, study="cpu_fixture", angle_deg=0, corrections={
            "A":dict(c, yaw_deg=r["historical_symmeta"]["yaw_deg"]), "B":c}))
    require(len(out) == len({(r["task_shard"],r["query_layout"]) for r in out}) == 16 and
            len({r["embodiment"] for r in out}) == 8, "two official layouts per task, all eight actual command maps")
    return out


def main(out, pin, mode, release, sha):
    cpu = mode == "cpu"
    ev = native()
    rows, runtime = historical()
    if cpu:
        require(os.environ["CUDA_VISIBLE_DEVICES"] == "" and len(os.sched_getaffinity(0)) == 2, "CPU-only fixture")
        chosen = fixtures(rows)
    else:
        from export_panel import assemble
        chosen = assemble(rows, release, sha, out)
    panel = dict(pin=pin, registration=digest((HERE/"PLAN.md").read_bytes()), cases_sha256=digest((HERE/"cases.csv").read_bytes()),
                 cpu_fixture=cpu, runtime=runtime, release_sha256=sha, rows=chosen)
    raw = encode(panel)
    atomic(out/"panel.json", raw)
    if mode == "gpu":
        prepared = parse(Path(os.environ["CAMNATIVE_PREPARED_PANEL"]).read_bytes())
        require(prepared == panel, "CPU-prepared panel differs from the GPU input panel")
    if mode == "prepare":
        print(f"Verified {len(chosen)} real post-view prediction pairs", flush=True)
        return
    if cpu:
        policy = CPUActionFixture()
    else:
        probe = parse((out/"gpu.json").read_bytes())
        require(probe["hostname"] == socket.gethostname() and probe["status"] == "passed" and
                probe["allocation_type"] == "baige_container" and probe["purpose"] == "evaluation" and
                len(probe["devices"]) == 1 and "RTX 4090" in probe["devices"][0]["cuda_name"].upper(), "GPU")
        policy = ev.load_frozen_policy("pi05")
    require((policy._model.action_horizon, policy._model.action_dim) == (10, 32), "pi05 action bank")
    with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES":""}), \
         patch.object(ur5e_sim_worker, "__file__", str(HERE/"audited_worker.py")):
        sim = ev.UR5eSimulator(allow_mesa=cpu)
    with sim:
        previous = None
        for row in chosen:
            if row["task_shard"] != previous:
                sim.call("create", dict(row["query"], embodiment=row["embodiment"]))
                previous = row["task_shard"]
            names = list(row["corrections"])
            if (row["task_shard"] + row["slot"]) % 2:
                names[:2] = names[1::-1]
            arms, resets = {}, {}
            for name in names:
                arms[name] = episode(ev, sim, policy, row, name, resets, cpu)
                base = arms[names[0]]
                require(arms[name]["native"]["reset"] == base["native"]["reset"] and
                        arms[name]["chunks"][0]["raw"] == base["chunks"][0]["raw"] and
                        arms[name]["env"] == base["env"], "same-process opening")
            pair = dict(panel_sha256=digest(raw), index=row["index"], execution_order=names, arms=arms)
            atomic(out/"pairs"/f"{row['index']:02d}.json.gz", gzip.compress(encode(pair), compresslevel=4, mtime=0))
            print(f"Completed {row['study']} pair {row['index']} order={names}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    p.add_argument("pin")
    p.add_argument("mode", choices=("cpu", "prepare", "gpu"))
    p.add_argument("--release", type=Path)
    p.add_argument("--sha256")
    a = p.parse_args()
    main(a.out, a.pin, a.mode, a.release, a.sha256)
