"""Replay immutable raw actions. Write independent public features and privileged scoring labels."""
import argparse
import hashlib
import json
import os
import socket
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from arm_env import ArmEnv
from ur5e_sim_worker import _probe_egl
from public import features, observation

ROOT = Path("/mowenjie/fluffy-bassoon-2/outputs")
NATIVE = ROOT/"frozen-vla-native-transfer-20261123-r1/gpu01"
MULTI = ROOT/"frozen-vla-multiarm-confirmation-20261123-r1"
FEEDBACK = ROOT/"frozen-vla-tracking-feedback-20261123-r1/gpu01"


def cases(split):
    if split == "source":
        donors = sorted(NATIVE.glob("*-15-Panda.json"))
        assert len(donors) == 3
    else:
        donors = [p for arm in ("IIWA", "UR5e") for p in sorted((MULTI/f"gpu01-{arm}").glob("*-14-raw.json"))]
        donors += sorted(FEEDBACK.glob("*-raw.json"))
        assert len(donors) == 22
    for p in donors:
        spec = json.loads(p.read_bytes())["case"]
        for layout in ([spec["init_state_id"], 18, 19] if split == "source" else [spec["init_state_id"], 20]):
            yield p, spec, layout


class Contacts:
    """Extract contact-bearing finger bodies, rigid movable roots and external object supports."""
    def __init__(self, model):
        names = [model.body(i).name or "world" for i in range(model.nbody)]
        self.parts = []
        self.objects = []
        self.bodies = []
        for body in model.geom_bodyid:
            body = int(body)
            name = names[body]
            self.bodies.append(name)
            side = None
            ancestor = body
            while ancestor:
                parent_name = names[ancestor]
                if parent_name.endswith("leftfinger"):
                    side = "left"
                    break
                if parent_name.endswith("rightfinger"):
                    side = "right"
                    break
                if parent_name.startswith("gripper0_"):
                    side = "palm"
                ancestor = int(model.body_parentid[ancestor])
            self.parts.append(side)
            root = int(model.body_rootid[body])
            addr = int(model.body_jntadr[root])
            free = addr >= 0 and model.jnt_type[addr] == 0
            self.objects.append(names[root] if free and not names[root].startswith(("robot0_", "gripper0_","mount0_")) else None)
        assert {"left", "right", "palm"} <= set(self.parts) and any(self.objects)

    def endpoint(self, data):
        touches = {"left": set(), "right": set(), "palm": set()}
        supported = set()
        for contact in data.contact[:data.ncon]:
            if contact.efc_address < 0 or contact.exclude != 0:
                continue
            for a, b in ((contact.geom1, contact.geom2), (contact.geom2, contact.geom1)):
                obj = self.objects[a]
                if obj and obj != self.objects[b] and self.parts[b] is None:
                    supported.add(obj)
                part = self.parts[a]
                if part and self.parts[b] is None:
                    touches[part].add(self.objects[b] or self.bodies[b])
        both = touches["left"] & touches["right"]
        carried = (both & set(self.objects)) - supported
        return {"carry": sorted(carried), "empty": not any(touches.values()),
                "left": sorted(touches["left"]), "right": sorted(touches["right"]), "palm": sorted(touches["palm"]),
                "supported": sorted(supported)}


def write(path, data):
    path.write_text(json.dumps(data, allow_nan=False, separators=(",", ":"))+"\n")


def record(out, path, spec, layout, split):
    blob = path.read_bytes()
    tape = json.loads(blob)
    donor = spec["init_state_id"]
    seed = spec["seed"] + layout - donor
    arm_name, suite, task = spec["arm"], spec["suite"], spec["task_id"]
    case = f"{arm_name}-{suite}-{task}-tape{donor}-layout{layout}"
    out.mkdir(exist_ok=True, parents=True)
    if (out/f"{case}.lineage.json").exists():
        return
    pub, truth, rewards, tcp_err, jaw_err = [], [], [], [], []
    with ArmEnv(arm_name, suite, task) as env:
        init = env.reset(layout, seed)
        reset = init["environment_reproducibility"]
        if layout == donor:
            for key in ("released_row_fingerprint", "scene_fingerprint", "bddl_sha256", "perturbation_plan_fingerprint"):
                assert reset[key] == tape["full_reset"][key], (case,key)
        sim = env._env.sim
        physical = Contacts(sim.model._model)
        prev = observation(init["observation"])
        blank = np.zeros((256,256,3), np.uint8)
        render_rgb = sim.render
        def render(*args, **kwargs):
            if t % 5 == 0 and kwargs.get("camera_name") == "robot0_eye_in_hand":
                return render_rgb(*args, **kwargs)
            return blank
        age = 0
        endpoints = []
        commands = [(u, p, j) for chunk in tape["chunks"] for u,p,j in zip(chunk["cmd"],chunk["xyz"],chunk["jaw"], strict=True)]
        assert len(commands) == tape["record"]["controls"] and spec["variant"] == "BASE"
        with patch.object(sim, "render", side_effect=render):
            for t,(action,xyz,jaw) in enumerate(commands, 1):
                step = env.step(np.asarray(action, np.float32))
                assert step["worker_step_index"] == t
                obs = step["observation"]
                if layout == donor:
                    tcp_err.append(float(np.linalg.norm(obs["robot0_eef_pos"] - xyz)))
                    jaw_err.append(float(np.max(np.abs(obs["robot0_gripper_qpos"] - jaw))))
                age = age + 1 if action[6] > 0 else 0
                rewards.append(step["reward"])
                endpoints.append(physical.endpoint(sim.data._data))
                terminal = bool(step["terminated"] or step["truncated"])
                if t % 5 == 0:
                    current = observation(obs)
                    if age >= 5 and not terminal:
                        pub.append(features(prev,current,t,age))
                        common = set.intersection(*(set(e["carry"]) for e in endpoints))
                        y = 1 if common else 0 if all(e["empty"] for e in endpoints) else None
                        truth.append(dict(control=t,y=y,endpoints=endpoints))
                    prev = current
                    endpoints = []
                if terminal:
                    break
    source_reward = tape["record"]["native_rewards"]
    lineage = dict(split=split,case=case,spec=spec,layout=layout,seed=seed,original=layout==donor,
                   input_path=str(path),input_sha256=hashlib.sha256(blob).hexdigest(),reset=reset,
                   model=init["model_identity"], source_controls=len(commands),cpu_controls=t,
                   cpu_rewards=rewards,original_rewards=source_reward,cpu_success=bool(step["success"]),
                   original_success=tape["record"]["success"],cpu_terminated=bool(step["terminated"]),
                   cpu_truncated=bool(step["truncated"]),cpu_terminal=terminal,censored=not terminal,
                   tcp_delta_mm=(1000*max(tcp_err) if tcp_err else None),jaw_delta_mm=(1000*max(jaw_err) if jaw_err else None))
    write(out/f"{case}.public.json", pub)
    write(out/f"{case}.truth.json", truth)
    write(out/f"{case}.lineage.json", lineage)
    print(case, "complete", t, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("out",type=Path)
    p.add_argument("split", choices=["source","target"])
    args = p.parse_args()
    cv2.setNumThreads(1)
    renderer = _probe_egl(True)
    cpus = sorted(os.sched_getaffinity(0))
    assert len(cpus) <= 2 and not os.environ["CUDA_VISIBLE_DEVICES"] and "llvmpipe" in renderer[1].lower()
    write(args.out/f"runtime-{args.split}.json",dict(renderer=renderer,cpus=cpus,host=socket.gethostname()))
    for donor_path,spec,layout in cases(args.split):
        record(args.out/args.split,donor_path,spec,layout,args.split)
