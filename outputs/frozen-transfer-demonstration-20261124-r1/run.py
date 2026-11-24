"""Closed-loop four-mode Kinova3 drawer study using the pinned native run_episode."""
import argparse
import json
import os
from pathlib import Path
import runpy
import socket
import subprocess
import sys
from unittest.mock import patch

import numpy as np

from study import BANK_SHA, CUT, HERE, IDENTITY, MODES, ROWS, TASK, WEIGHT_SHA, cases, fp, require, save
from study import sources, summarize, verify
from contract import check_weights, checked
from fluffy_bassoon.xemb.embodiment import get_embodiment_spec, retarget_observation_for_motor
from ur5e_motor import OPENPI_CKPT, load_frozen_pi05, public_digest, run_episode, to_openpi_observation
import ur5e_sim_worker


def fingerprints():
    import torch
    release = Path(os.environ["CROSSARM_RELEASE"])
    hasher = runpy.run_path(str(release / "libero_remote/reproducibility.py"))["numerical_state_fingerprint"]
    bank = release / ("simulator-source/libero/libero/init_files/libero_goal/"
                      "open_the_top_drawer_and_put_the_bowl_inside.pruned_init")
    checked(bank.read_bytes(), BANK_SHA)
    states = torch.load(bank, map_location="cpu", weights_only=False)
    require(states.shape == (50,79), "release bank")
    return {r:hasher(states[r]) for r in ROWS}


class Interface:
    def __init__(self, sim, base, fixture, mode, tapes, correction):
        self.sim = sim
        self.base = base
        self._model = base._model
        self.fixture = fixture
        self.mode = mode
        self.tape = tapes.get(mode)
        self.cut = CUT.get(mode, 0)
        self.correction = correction
        self.calls = []
        self.t = 0
        self.pending = None

    def call(self, op, value):
        if op == "create":
            spec, self.first = self.sim.call(op, value)
            self.pending = self.first
            return spec, self.first
        require(op == "chunk" and self.pending is None, "missing proposal")
        r = self.calls[-1]
        require(r["t"] == self.t and "native_steps" not in r, "duplicate chunk")
        chosen = self.tape[self.t:self.t+len(value)] if r["prefix"] else value
        obs = self.sim.call(op, chosen)
        steps = obs["native_steps"]
        sent = np.array([s["action"] for s in steps], np.float32)
        require(sent.tobytes() == chosen[:len(steps)].tobytes(), "action changed before native step")
        r.update(native_steps=steps, after_public=obs["after_public"])
        self.t += len(steps)
        self.pending = obs
        return obs

    def infer(self, item, *, noise):
        require(self.pending is not None and self.pending["language_instruction"] == item["prompt"] == TASK,
                "fresh current observation and original instruction")
        public = self.pending["observation"]
        raw = public
        yaw = self.correction["yaw_deg"]
        if self.t and yaw:
            raw = retarget_observation_for_motor(public, self.first["anchor_eef_pos"], yaw)
        expected = to_openpi_observation(raw, TASK)
        inputs = {k:fp(v) for k,v in item.items() if k != "prompt"}
        expected_inputs = {k:fp(v) for k,v in expected.items() if k != "prompt"}
        require(inputs == expected_inputs, "motor did not receive current target state/images")
        use_fixture = self.fixture is not None and (self.mode not in CUT or self.t != self.cut)
        policy = self.fixture if use_fixture else self.base
        a = np.asarray(policy.infer(item, noise=noise)["actions"], np.float32)
        self.calls.append(dict(t=self.t, public=public_digest(public), inputs=inputs, expected_inputs=expected_inputs,
                               noise=fp(noise), prompt=item["prompt"], proposal=fp(a), first_five=a[:5].tolist(),
                               prefix=self.t < self.cut, motor="fixture" if use_fixture else "pi05_libero"))
        self.pending = None
        return {"actions": a}


class ResetMismatch(ValueError):
    pass


def episode(sim, base, fixture, q, support, mode, tapes, row_fp, ref, pin):
    c = IDENTITY if mode == "raw" else support["tuned"]
    for attempt in range(1, 4):
        wire = Interface(sim, base, fixture, mode, tapes, c)

        def check(reset):
            env = wire.first["environment_reproducibility"]
            require(env["released_row_fingerprint"] == row_fp and reset["prompt"] == TASK and
                    reset["arm"] == "Kinova3" and reset["worker_step_index"] == 0, "native reset")
            if ref and (reset != ref["native"]["reset"] or env != ref["env"]):
                raise ResetMismatch("opening differs before inference")

        try:
            rec = run_episode(wire, wire, q, get_embodiment_spec("nom"), yaw_hat=c["yaw_deg"],
                              scale_hat=c["xy_scale"], grip_hat=c["gripper_sign"], check_reset=check,
                              canonicalize_quat=False, max_controls=wire.cut+5 if fixture is not None else None)
        except ResetMismatch:
            if attempt == 3:
                raise
            continue
        obj = dict(pin=pin, mode=mode, query=q, support=support, release_row=row_fp, fixture=fixture is not None,
                   weight_manifest=WEIGHT_SHA, shape=[int(base._model.action_horizon), int(base._model.action_dim)],
                   accepted_reset=attempt, native=rec, env=wire.first["environment_reproducibility"],
                   body=wire.first["model_identity"], calls=wire.calls)
        return obj


def main(out, pin, cpu):
    require(subprocess.check_output(["git","-C",str(HERE),"rev-parse","HEAD"], text=True).strip() == pin, "pin")
    tapes, info = sources()
    bank = fingerprints()
    case, = (r for r in cases("Kinova3", verify_tapes=True)
             if (r["query"]["suite"], r["query"]["task_id"]) == ("libero_goal",3))
    check_weights(OPENPI_CKPT)
    fixture = None
    if cpu:
        import jax
        from openpi.policies import policy_config
        from openpi.training import config
        sys.path.insert(0, str(HERE.parent / "frozen-transfer-cue-20261124-r1"))
        from fixture import Fixture
        require(os.sched_getaffinity(0) == {38,39} and os.environ["CUDA_VISIBLE_DEVICES"] == ""
                and all(d.platform == "cpu" for d in jax.devices()), "CPU affinity or device")
        fixture = Fixture()
        print("Loading frozen pi05_libero for the two CPU handoff queries", flush=True)
        base = policy_config.create_trained_policy(config.get_config("pi05_libero"), OPENPI_CKPT)
    else:
        probe = json.loads((out/"gpu.json").read_bytes())
        require(socket.gethostname().startswith("job-") and probe["hostname"] == socket.gethostname() and
                probe["status"] == "passed" and len(probe["devices"]) == 1 and
                "RTX 4090" in probe["devices"][0]["cuda_name"].upper(), "scheduler 4090")
        base = load_frozen_pi05()
    save(out/"origin.json", dict(pin=pin, host=socket.gethostname(), fixture=cpu, sources=info,
         release_rows=bank, release_bank_sha256=BANK_SHA, cpu_affinity=sorted(os.sched_getaffinity(0)),
         policy=str(OPENPI_CKPT), support=case["support"], shape=[int(base._model.action_horizon),int(base._model.action_dim)]))
    (out/"episodes").mkdir()
    # The pinned client selects its child executable from its own __file__. Point only that
    # construction at the run-local audit entry. The child imports the unmodified native serve loop.
    with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES":""}), \
         patch.object(ur5e_sim_worker, "__file__", str(HERE/"audited_worker.py")):
        real = ur5e_sim_worker.UR5eSimulator(allow_mesa=cpu)
    with real as sim:
        for row in (ROWS[:1] if cpu else ROWS):
            q = dict(case["query"], init_state_id=row, seed=42710300+row)
            sim.call("create", dict(q, embodiment="nom"))
            group = {}
            for m in MODES:
                rec = episode(sim, base, fixture, q, case["support"], m, tapes, bank[row], group.get("raw"), pin)
                save(out/"episodes"/f"{row}-{m}.json", rec)
                verify(rec, tapes)
                group[m] = rec
                print(row, m, rec["native"]["controls"], flush=True)
    save(out/("qualified.json" if cpu else "result.json"), summarize(out, pin, cpu))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    p.add_argument("pin")
    p.add_argument("--cpu", action="store_true")
    a = p.parse_args()
    main(a.out, a.pin, a.cpu)
