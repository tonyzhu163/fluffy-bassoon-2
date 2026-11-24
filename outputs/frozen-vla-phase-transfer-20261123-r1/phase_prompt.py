"""Prepare an isolated one-replan prompt intervention at the native run_episode simulator interface."""
import numpy as np
from scipy.special import expit

from common import tiles


class PhasePrompt:
    """Wrap one simulator for the registered raw Kinova3 drawer task, preserving its native reward."""
    def __init__(self, simulator, model, mode):
        if mode not in {"raw", "visual", "clock"}:
            raise ValueError(mode)
        self.simulator = simulator
        self.model = model
        self.mode = mode
        self.events = []
        self.base = None

    def call(self, op, arg):
        if op == "create":
            if (arg["suite"],arg["task_id"],arg["arm"],arg["variant"]) != ("libero_goal",3,"Kinova3","BASE"):
                raise ValueError("Unregistered task")
            spec, obs = self.simulator.call(op,arg)
            self.base = tiles(obs["observation"]["agentview_image"])
            self.released = obs["language_instruction"]
            self.elapsed = 0
            self.fired = False
            self.events = []
            return spec,self._read(obs)
        if op != "chunk" or self.base is None:
            raise ValueError("Expected a reset followed by native five-control replans")
        obs = self.simulator.call(op,arg)
        self.elapsed += obs["steps"]
        if obs["terminated"] or obs["truncated"]:
            return obs
        if obs["steps"] != 5:
            raise ValueError("Wrong replanning cadence")
        return self._read(obs)

    def _read(self, obs):
        if obs["language_instruction"] != self.released:
            raise ValueError("Native task prompt changed")
        m = self.model
        sample = tiles(obs["observation"]["agentview_image"]) - self.base
        x = (sample[m["signed_luminance_tile_ids"]] - m["mean"]) / m["scale"]
        visual = float(expit(m["beta"][0] + x @ m["beta"][1:]))
        clock = m["clock"][min(self.elapsed//5,len(m["clock"])-1)]
        evidence = visual if self.mode == "visual" else clock
        fire = self.mode != "raw" and not self.fired and evidence >= 0.5
        self.fired |= fire
        self.events.append(dict(t=self.elapsed,visual=visual,clock=clock,fire=fire))
        if fire:
            return dict(obs, language_instruction="put the bowl in the top drawer")
        return obs
