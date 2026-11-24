"""Use the predecessor's native simulator client and motor loop with scheduled input and command routing."""
import argparse
import json
import os
from pathlib import Path
import runpy
import socket
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from protocol import HERE, LOCK, MODES, OLD, ROWS, TASKS, WEIGHT_SHA, assets, fp, query, require, save, source_tapes, supports
from audit import inspect_episode, summarize
from contract import check_weights
from fluffy_bassoon.xemb.embodiment import get_embodiment_spec, retarget_action_from_motor, retarget_observation_for_motor
from ur5e_motor import OPENPI_CKPT, load_frozen_pi05, public_digest, run_episode, to_openpi_observation
import ur5e_sim_worker


predecessor = runpy.run_path(str(OLD / "run.py"))
OriginalInterface = predecessor["Interface"]
ResetMismatch = predecessor["ResetMismatch"]


class Fixture:
    _model = SimpleNamespace(action_horizon=10, action_dim=32)

    def infer(self, item, *, noise):
        require(noise.shape == (10, 32) and item["observation/image"].shape == (224, 224, 3)
                and item["observation/wrist_image"].shape == (224, 224, 3)
                and item["observation/state"].shape == (8,), "fixture inputs")
        a = .02 * noise[:, :7].copy()
        a[:, 0] += item["observation/image"].mean() / 1000
        a[:, 1] += item["observation/wrist_image"].mean() / 1000
        a[:, 2] += .1 * np.tanh(item["observation/state"].sum())
        return {"actions": a}


class Interface(OriginalInterface):
    def __init__(self, sim, base, cpu, mode, source, task, correction):
        super().__init__(sim, base, None, mode, {}, correction)
        self.cpu = cpu
        self.task = task
        self.cut = len(source)
        self.tape = source
        if mode == "corrected":
            # Use exactly the production inverse on each complete five-command block.
            self.tape = np.concatenate([
                retarget_action_from_motor(a, correction["yaw_deg"], correction["xy_scale"],
                                           correction["gripper_sign"], -np.ones(7, "f4"), np.ones(7, "f4"))
                for a in np.split(source, self.cut // 5)])

    def call(self, op, value):
        if op == "chunk" and self.mode == "switch" and self.t < self.cut:
            # The default loop has computed the tuned candidate. Issue the actual raw motor proposal.
            value = np.clip(np.asarray(self.calls[-1]["first_five"], np.float32), -1, 1)
        return super().call(op, value)

    def infer(self, item, *, noise):
        require(self.pending is not None and self.pending["language_instruction"] ==
                item["prompt"] == self.task["prompt"], "current frame and released prompt")
        public = self.pending["observation"]
        prompt = item["prompt"]
        anchor = self.first["anchor_eef_pos"]
        raw_input = to_openpi_observation(public, prompt)
        tuned_obs = retarget_observation_for_motor(public, anchor, self.correction["yaw_deg"]) if self.t else public
        expected = to_openpi_observation(tuned_obs, prompt)
        require(all(np.array_equal(item[k], expected[k]) for k in expected), "upstream tuned frame")
        raw_stage = self.mode == "switch" and self.t < self.cut
        model_input = raw_input if raw_stage else item
        a = np.asarray(self.base.infer(model_input, noise=noise)["actions"], np.float32)
        r = dict(t=self.t, prefix=self.mode in ("source", "corrected") and self.t < self.cut,
                 raw_stage=raw_stage, public=public_digest(public), inputs=public_digest(
                     {k: v for k, v in model_input.items() if k != "prompt"}),
                 raw_state=raw_input["observation/state"].tolist(), tuned_state=expected["observation/state"].tolist(),
                 state_dtype=raw_input["observation/state"].dtype.str, prompt=prompt,
                 noise=fp(noise), proposal=fp(a), first_five=a[:5].tolist(),
                 motor="fixture" if self.cpu else "pi05_libero")
        self.calls.append(r)
        self.pending = None
        return {"actions": a}


def episode(sim, base, cpu, key, row, mode, source, support, row_fp, ref, pin):
    task = TASKS[key]
    q = query(task, row)
    c = support["tuned"]
    rejections = []
    for attempt in range(1, 4):
        wire = Interface(sim, base, cpu, mode, source, task, c)

        def check(reset):
            env = wire.first["environment_reproducibility"]
            anchor = wire.first["anchor_eef_pos"]
            require(env["released_row_fingerprint"] == row_fp and reset["worker_step_index"] == 0 and
                    reset["horizon"] == task["bank"]["native_horizon"] and reset["prompt"] == task["prompt"] and
                    np.array_equal(anchor, np.float32(wire.first["observation"]["robot0_eef_pos"])), "reset anchor")
            if ref and (reset != ref["native"]["reset"] or env != ref["env"] or anchor.tolist() != ref["anchor"]):
                rejections.append(dict(attempt=attempt, physics=reset["physics"], public=reset["public"]))
                raise ResetMismatch("pre-inference reset mismatch")
        try:
            rec = run_episode(wire, wire, q, get_embodiment_spec("nom"), prompt=task["prompt"],
                              yaw_hat=c["yaw_deg"], scale_hat=c["xy_scale"], grip_hat=c["gripper_sign"],
                              check_reset=check, max_controls=wire.cut+5 if cpu else None)
        except ResetMismatch:
            if attempt == 3:
                raise
            continue
        return dict(pin=pin, task=key, query=q, mode=mode, cut=wire.cut, fixture=cpu,
                    support=support, weight_manifest=WEIGHT_SHA, shape=[int(base._model.action_horizon),
                    int(base._model.action_dim)], body=wire.first["model_identity"], release_row=row_fp,
                    native=rec, env=wire.first["environment_reproducibility"], anchor=wire.first["anchor_eef_pos"].tolist(),
                    accepted_reset=attempt, rejected_before_inference=rejections, calls=wire.calls)


def main(out, pin, cpu):
    require(subprocess.check_output(["git", "-C", str(HERE), "rev-parse", "HEAD"], text=True).strip() == pin, "pin")
    rows = assets()
    tapes = source_tapes()
    support = supports(check=True)
    check_weights(OPENPI_CKPT)
    if cpu:
        require(os.sched_getaffinity(0) == {30, 31} and os.environ["CUDA_VISIBLE_DEVICES"] == "" and
                os.environ["JAX_PLATFORMS"] == "cpu", "CPU qualification")
        base = Fixture()
    else:
        probe = json.loads((out / "gpu.json").read_bytes())
        require(probe["hostname"] == socket.gethostname() and socket.gethostname().startswith("job-") and
                probe["status"] == "passed" and len(probe["devices"]) == 1 and
                "RTX 4090" in probe["devices"][0]["cuda_name"].upper(), "scheduler 4090")
        base = load_frozen_pi05()
    require((base._model.action_horizon, base._model.action_dim) == (10, 32), "qualified model noise shape")
    save(out / "origin.json", dict(pin=pin, host=socket.gethostname(), fixture=cpu, lock=LOCK, release_rows=rows,
         support=support, policy=str(OPENPI_CKPT), shape=[10, 32], cpu_affinity=sorted(os.sched_getaffinity(0)),
         egl_device=os.environ["MUJOCO_EGL_DEVICE_ID"]))
    (out / "episodes").mkdir()
    with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}), \
         patch.object(ur5e_sim_worker, "__file__", str(OLD / "audited_worker.py")):
        worker = ur5e_sim_worker.UR5eSimulator(allow_mesa=cpu)
    with worker as sim:
        for key, task in TASKS.items():
            for row in ROWS[:1] if cpu else ROWS:
                # Predecessor's scene warm reset, zero controls and zero policy queries.
                sim.call("create", dict(query(task, row), embodiment="nom"))
                ref = None
                for mode in MODES:
                    obj = episode(sim, base, cpu, key, row, mode, tapes[key], support[key], rows[key][str(row)], ref, pin)
                    save(out / "episodes" / f"{key}-{row}-{mode}.json", obj)
                    inspect_episode(obj, tapes[key], support[key], rows[key][str(row)], pin, cpu)
                    if ref is None:
                        ref = obj
                    print(key, row, mode, obj["native"]["controls"], flush=True)
    save(out / ("qualified.json" if cpu else "result.json"), summarize(out, pin, cpu))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("pin")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    main(args.out, args.pin, args.cpu)
