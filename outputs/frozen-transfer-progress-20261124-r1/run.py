"""Single-arm matched native benchmark or real-simulator CPU wiring pass."""
import argparse
import json
import os
from pathlib import Path
import socket
from unittest.mock import patch

import numpy as np

from fluffy_bassoon.xemb.embodiment import get_embodiment_spec
from ur5e_motor import OPENPI_CKPT, load_frozen_pi05, run_episode
from ur5e_sim_worker import UR5eSimulator

from contract import ARMS, IDENTITY, MODES, SOURCE_SHA, WEIGHT_SHA, cases, check_weights, require, save, source_law
from intervention import Phase, Policy, Simulator
from readback import analyze, pairing, verify


class ResetMismatch(ValueError):
    pass


def episode(sim, base, mode, row, origin, law, reference, cpu):
    q, sup = row["query"], row["support"]
    c = IDENTITY if mode == "raw" else sup["tuned"]
    for attempt in range(1, 4):
        if cpu:
            from qualify import Fixture
            base = Fixture()
        policy = Policy(base, Phase(mode, c, law))
        wrapped = Simulator(sim, policy)

        def check_reset(reset):
            renderer_ok = ("llvmpipe" in reset["renderer"].lower() if cpu
                           else reset["renderer_vendor"] == "NVIDIA Corporation")
            require(reset["physics"] != sup["physics"] and reset["arm"] == q["arm"] and renderer_ok,
                    "invalid physical reset")
            if reference is not None and (reference["native"]["reset"] != reset or
                    reference["environment"] != wrapped.first["environment_reproducibility"]):
                raise ResetMismatch("pre-inference physical reset mismatch")

        try:
            rec = run_episode(wrapped, policy, q, get_embodiment_spec("nom"), yaw_hat=c["yaw_deg"],
                              scale_hat=c["xy_scale"], grip_hat=c["gripper_sign"],
                              max_controls=11 if cpu else None, check_reset=check_reset, canonicalize_quat=False)
        except ResetMismatch:
            if attempt == 3:
                raise
            continue
        out = dict(case=row, mode=mode, fixture=cpu, origin=origin, accepted_create=attempt,
                   native=rec, model=wrapped.first["model_identity"],
                   environment=wrapped.first["environment_reproducibility"], chunks=policy.chunks,
                   noise_shape=[int(policy._model.action_horizon), int(policy._model.action_dim)],
                   source_sha256=SOURCE_SHA, weights_sha256=None if cpu else WEIGHT_SHA)
        verify(out, row, mode, cpu=cpu)
        return out


def main(arm, out, cpu, pin):
    law = source_law()
    rows = cases(arm, verify_tapes=True)
    base = None
    if cpu:
        from qualify import numerical
        require(os.environ["CUDA_VISIBLE_DEVICES"] == "" and os.environ["JAX_PLATFORMS"] == "cpu" and
                len(os.sched_getaffinity(0)) <= 2, "software-only CPU qualification")
        numerical()
        device = "CPU"
    else:
        probe = json.loads((out/"gpu.json").read_bytes())
        require(socket.gethostname().startswith("job-") and probe["status"] == "passed" and
                len(probe["devices"]) == 1 and probe["hostname"] == socket.gethostname(), "scheduler admission")
        device = probe["devices"][0]["cuda_name"]
        check_weights(OPENPI_CKPT)
        base = load_frozen_pi05()
    origin = dict(pin=pin, host=socket.gethostname(), arm=arm, fixture=cpu, device=device,
                  egl_index=os.environ["MUJOCO_EGL_DEVICE_ID"], attempt=out.name)
    save(out/"origin.json", origin)
    (out/"episodes").mkdir()
    # The already-resolved EGL ordinal identifies the allocated GPU. Robosuite's renderer child
    # validates CUDA mask strings as EGL indices; it performs no CUDA inference.
    with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}):
        real = UR5eSimulator(allow_mesa=cpu)
    with real as sim:
        for row in rows:
            q = row["query"]
            sim.call("create", dict(q, embodiment="nom"))
            quartet = {}
            for mode in MODES:
                obj = episode(sim, base, mode, row, origin, law, quartet.get("raw"), cpu)
                quartet[mode] = obj
                pairing(quartet)
                save(out/"episodes"/f"{q['suite']}-{q['task_id']}-{mode}.json", obj)
                print(arm, q["suite"], q["task_id"], mode, "controls", obj["native"]["controls"], flush=True)
    if cpu:
        _, details = analyze(out, arm, cpu=True)
        require(all(len(v["diagnostics"]) == 4 for v in details), "missing qualification mode")
        save(out/"qualified.json", dict(status="passed", origin=origin, complete_cases=8, episodes=32,
             numerical="six indices, ties, clipping, gain, yaw, no history, horizon and terminal prefix",
             controls=sum(json.loads(p.read_bytes())["native"]["controls"] for p in (out/"episodes").glob("*.json")),
             released_rows=[dict(suite=r["query"]["suite"], task=r["query"]["task_id"], layout=r["query"]["init_state_id"],
                                 fingerprint=json.loads((out/"episodes"/f"{r['query']['suite']}-{r['query']['task_id']}-raw.json").read_bytes())["environment"]["released_row_fingerprint"])
                            for r in rows], renderer=obj["native"]["reset"]["renderer"]))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("arm", choices=ARMS)
    p.add_argument("out", type=Path)
    p.add_argument("--pin", required=True)
    p.add_argument("--cpu", action="store_true")
    a = p.parse_args()
    main(a.arm, a.out, a.cpu, a.pin)
