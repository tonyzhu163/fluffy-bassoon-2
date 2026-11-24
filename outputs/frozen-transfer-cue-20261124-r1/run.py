"""Native drawer cue study on two physical arms, with one shared π0.5 load."""
import argparse
import json
import os
from pathlib import Path
import socket
from unittest.mock import patch

import numpy as np

from cue import ARMS, BANK_SHA, HERE, IDENTITY, LAYOUTS, MODEL_SHA, MODES, QueryInterface, checked, frozen_model, require, save
from contract import MANIFEST_SHA, RELEASE_SHA, WEIGHT_SHA, cases, check_weights
from fluffy_bassoon.xemb.embodiment import get_embodiment_spec
from score import paired, summarize, verify
from ur5e_motor import OPENPI_CKPT, load_frozen_pi05, run_episode
from ur5e_sim_worker import UR5eSimulator


def release_rows():
    from libero_remote.reproducibility import numerical_state_fingerprint
    import torch
    bank = Path(os.environ["CROSSARM_RELEASE"]) / ("simulator-source/libero/libero/init_files/libero_goal/"
                "open_the_top_drawer_and_put_the_bowl_inside.pruned_init")
    checked(bank.read_bytes(), BANK_SHA)
    states = torch.load(bank, map_location="cpu", weights_only=False)
    require(states.shape == (50, 79), "wrong released goal/3 bank")
    return {i: numerical_state_fingerprint(states[i]) for i in LAYOUTS}


class ResetMismatch(ValueError):
    pass


def episode(sim, base, model, q, support, mode, reference, row_hash, cpu, pin):
    c = IDENTITY if mode == "raw" else support["tuned"]
    for attempt in range(1, 4):
        wire = QueryInterface(sim, base, mode, model)

        def check(reset):
            full = wire.first["environment_reproducibility"]
            body = wire.first["model_identity"]
            require(body["arm"] == reset["arm"] == q["arm"] and body["model_class"] == "Mounted"+q["arm"]
                    and body["controller"] == "OSC_POSE" and body["control_freq"] == 20
                    and body["gripper"] == "PandaGripper" and body["joint_count"] == 7, "physical robot")
            require(full["source_pin"] == RELEASE_SHA and full["manifest_sha256"] == MANIFEST_SHA and
                    full["released_row_fingerprint"] == row_hash and full["perturbation_plan"]["case_id"] == "BASE"
                    and all(full[k] == q[k] for k in ("suite", "task_id", "init_state_id", "seed"))
                    and reset["physics"] != support["physics"] and reset["horizon"] == 300
                    and reset["worker_step_index"] == 0, "physical start or official horizon")
            require("llvmpipe" in reset["renderer"].lower() if cpu else reset["renderer_vendor"] == "NVIDIA Corporation",
                    "unexpected renderer")
            if reference and (reference["native"]["reset"] != reset or reference["env"] != full):
                raise ResetMismatch("unmatched reset before inference")
        try:
            rec = run_episode(wire, wire, q, get_embodiment_spec("nom"), yaw_hat=c["yaw_deg"],
                              scale_hat=c["xy_scale"], grip_hat=c["gripper_sign"], check_reset=check,
                              max_controls=10 if cpu else None, canonicalize_quat=False)
        except ResetMismatch:
            if attempt == 3:
                raise
            continue
        obj = dict(pin=pin, fixture=cpu, query=q, support=support, mode=mode, accepted_reset=attempt,
                   env=wire.first["environment_reproducibility"], body=wire.first["model_identity"],
                   shape=[int(base._model.action_horizon), int(base._model.action_dim)],
                   model_sha=MODEL_SHA, weight_manifest_sha=WEIGHT_SHA, native=rec, calls=wire.queries)
        verify(obj)
        return obj


def main(out, pin, cpu):
    from fixture import Fixture, check_routing
    import subprocess
    require(subprocess.check_output(["git","-C",str(HERE),"rev-parse","HEAD"], text=True).strip() == pin,
            "executable pin")
    model = frozen_model()
    distinct = release_rows()
    require(len(set(distinct.values())) == 4, "duplicate released starts")
    check_weights(OPENPI_CKPT)
    if cpu:
        require(os.sched_getaffinity(0) == {38,39} and os.environ["CUDA_VISIBLE_DEVICES"] == "" and
                os.environ["JAX_PLATFORMS"] == "cpu", "two-core CPU qualification")
        base = Fixture()
    else:
        probe = json.loads((out/"gpu.json").read_bytes())
        require(socket.gethostname().startswith("job-") and probe["hostname"] == socket.gethostname() and
                probe["status"] == "passed" and len(probe["devices"]) == 1 and
                "RTX 4090" in probe["devices"][0]["cuda_name"].upper(), "one scheduler 4090")
        base = load_frozen_pi05()
    (out/"episodes").mkdir()
    with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}):
        real = UR5eSimulator(allow_mesa=cpu)
    with real as sim:
        for arm in ARMS:
            trained, = (r for r in cases(arm, verify_tapes=True)
                        if (r["query"]["suite"], r["query"]["task_id"]) == ("libero_goal", 3))
            for layout, row_hash in distinct.items():
                q = dict(trained["query"], init_state_id=layout, seed=42710300+layout)
                _, warm = sim.call("create", dict(q, embodiment="nom"))
                if cpu and layout == LAYOUTS[0]:
                    save(out/f"routing-{arm}.json", check_routing(warm, q, model))
                quartet = {}
                for mode in MODES:
                    record = episode(sim, base, model, q, trained["support"], mode, quartet.get("raw"),
                                     row_hash, cpu, pin)
                    save(out/"episodes"/f"{arm}-{layout}-{mode}.json", record)
                    quartet[mode] = record
                    print(arm, layout, mode, record["native"]["controls"], flush=True)
                paired(quartet)
    result = summarize(out, cpu, pin)
    save(out/("qualified.json" if cpu else "result.json"), result)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    p.add_argument("pin")
    p.add_argument("--cpu", action="store_true")
    a = p.parse_args()
    main(a.out, a.pin, a.cpu)
