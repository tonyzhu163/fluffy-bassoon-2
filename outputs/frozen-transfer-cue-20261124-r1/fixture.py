"""Exercise the production prompt path using a deterministic motor and labelled camera changes."""
from types import SimpleNamespace

import numpy as np

from cue import CUE, MODULE, TASK, QueryInterface, common, require
from fluffy_bassoon.xemb.embodiment import get_embodiment_spec
from ur5e_motor import run_episode


class Fixture:
    _model = SimpleNamespace(action_horizon=10, action_dim=32)

    def infer(self, item, *, noise):
        require(item["prompt"] in (TASK, CUE) and noise.shape == (10, 32), "fixture input")
        rgb, wrist, state = (item[k] for k in ("observation/image", "observation/wrist_image", "observation/state"))
        require(rgb.shape == wrist.shape == (224,224,3) and state.shape == (8,), "native OpenPI input")
        a = .01*noise[:,:7].copy()
        a[:,0] += rgb.mean()/1000.
        a[:,1] += wrist.mean()/1000.
        a[:,2] += float(np.tanh(state.sum()))*.1
        a[:,3] += .5*(item["prompt"] == CUE)
        return {"actions": a}


class CameraFixture:
    def __init__(self, opening, switch):
        self.opening = opening
        self.switch = switch
        self.tiles = []

    def call(self, op, arg):
        self.t = 0 if op == "create" else self.t + len(arg)
        public = self.opening["observation"].copy()
        image = np.full_like(public["agentview_image"], 200 if self.t < self.switch else 40)
        public["agentview_image"] = image
        self.tiles.append(common.tiles(image))
        obs = dict(self.opening, observation=public)
        if op == "create":
            return dict(arm=arg["arm"], max_steps=300), obs
        return dict(obs, steps=5, rewards=[0.]*5, terminated=False, truncated=False, success=False,
                    positions=[public["robot0_eef_pos"]]*5, joint_positions=[public["robot0_joint_pos"]]*5,
                    gripper_positions=[public["robot0_gripper_qpos"]]*5)


def check_routing(opening, q, model):
    from ur5e_motor import to_openpi_observation
    base = Fixture()
    item = to_openpi_observation(opening["observation"], TASK)
    noise = np.zeros((10,32), np.float32)
    a = base.infer(item, noise=noise)["actions"]
    text_diff = base.infer(dict(item, prompt=CUE), noise=noise)["actions"] - a
    image_diff = base.infer(dict(item, **{"observation/image":np.zeros_like(item["observation/image"])}),
                            noise=noise)["actions"] - a
    require(np.all(text_diff[:,3] == .5) and np.all(image_diff[:,0] != 0), "fixture sensitivity")
    checks = []
    for switch in (10, 60, 65):
        by_mode = {}
        for mode in ("tuned", "clock", "visual"):
            sim = CameraFixture(opening, switch)
            wire = QueryInterface(sim, base, mode, model)
            run_episode(wire, wire, q, get_embodiment_spec("nom"), max_controls=75)
            xs = np.array(sim.tiles[:len(wire.queries)])
            times = 5*np.arange(len(xs))
            pred = MODULE.predict(times, xs-xs[0], model)
            for i, row in enumerate(wire.queries):
                np.testing.assert_allclose([row["visual"],row["clock"]],
                                           [pred["instant"][i],pred["clock"][i]], rtol=0, atol=1e-12)
            fires = [v["t"] for v in wire.queries if v["prompt"] == CUE]
            require(fires == ({"clock":[60],"visual":[switch]}.get(mode, [])), "cue count/time")
            by_mode[mode] = wire.queries
        for mode in ("clock", "visual"):
            for x, y in zip(by_mode[mode], by_mode["tuned"], strict=True):
                require(x["inputs"] == y["inputs"] and x["noise"] == y["noise"] and
                        (x["proposal"] != y["proposal"]) == (x["prompt"] != y["prompt"]) and
                        (x["sent"] != y["sent"]) == (x["prompt"] != y["prompt"]), "cue failed to reach motor")
        checks.append(dict(visual_at=switch, clock_at=60, actual_prompts={
            m:[v["prompt"] for v in calls] for m,calls in by_mode.items()},
            matching_observations=15, restored_on_next_query=True))
    sim = CameraFixture(opening, 5)
    wire = QueryInterface(sim, base, "visual", model)
    class StaleText:
        _model = base._model
        def infer(self, item, *, noise):
            return wire.infer(dict(item, prompt=TASK), noise=noise)
    try:
        run_episode(wire, StaleText(), q, get_embodiment_spec("nom"), max_controls=10)
    except ValueError as e:
        require("policy prompt differs" in str(e) and len(wire.queries) == 1, e)
    else:
        raise ValueError("per-query text overwrite went unnoticed")
    return dict(arm=q["arm"], exact_reader=str(MODULE.__file__), sensitivity_at_fixed_noise=True,
                stale_environment_text_rejected_before_second_inference=True, checks=checks)
