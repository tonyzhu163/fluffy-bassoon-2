"""Run the archived camera/clock gate on the actual observation at each native policy call."""
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ARCHIVE = HERE.parent / "frozen-vla-phase-transfer-20261123-r1"
PROGRESS = HERE.parent / "frozen-transfer-progress-20261124-r1"
sys.path[:0] = [str(ARCHIVE), str(PROGRESS)]

import common
import phase_prompt
import reader as MODULE
from contract import IDENTITY, checked, require, save
from ur5e_motor import array_digest, public_digest

MODEL_SHA = "ca8906c86514d8917c1743726929c1e70aaf1f45e07de8892574c58a9ddf7008"
BANK_SHA = "1cd5354ade9ab9f323a518515f961df96123b7daeca017df1f62d3e9ff18794f"
TASK = "open the top drawer and put the bowl inside"
CUE = "put the bowl in the top drawer"
ARMS = ("Kinova3", "IIWA")
MODES = ("raw", "tuned", "clock", "visual")
LAYOUTS = (34, 35, 36, 37)


def frozen_model():
    for module, digest in (
        (common, "b8adee68069369be9f1499561499f9f678f896d89b7673592c180de701265591"),
        (phase_prompt, "e33c64449c6b57c489806c2e3241948062df9d2076fbb570a475f9462304a014"),
        (MODULE, "51cdb6e395dbc767e289bc102f4839009b338937f241371e4faad01d46b2ac52"),
    ):
        path = Path(module.__file__).resolve()
        require(path == ARCHIVE / (module.__name__ + ".py"), "wrong archived module")
        checked(path.read_bytes(), digest)
    return json.loads(checked((ARCHIVE/"evidence/model.json").read_bytes(), MODEL_SHA))


class Gate(phase_prompt.PhasePrompt):
    def call(self, op, arg):
        if op == "create":
            require((arg["suite"], arg["task_id"], arg["variant"], arg["arm"]) in
                    {("libero_goal", 3, "BASE", a) for a in ARMS}, "unregistered task or arm")
            if arg["arm"] == "IIWA":
                # The archived create guard admits only Kinova3. Forward the genuine IIWA
                # reset, then initialize the same archive state. All scored reads are inherited.
                spec, obs = self.simulator.call(op, arg)
                require(spec["arm"] == obs["model_identity"]["arm"] == "IIWA", "body substitution")
                self.base = common.tiles(obs["observation"]["agentview_image"])
                self.released = obs["language_instruction"]
                self.elapsed, self.fired, self.events = 0, False, []
                return spec, self._read(obs)
        return super().call(op, arg)


class QueryInterface:
    """One native episode with a single gate and a single policy query per executed block."""
    def __init__(self, sim, base, mode, model):
        self.gate = Gate(sim, model, mode if mode in ("clock", "visual") else "raw")
        self.base = base
        self._model = base._model
        self.queries = []
        self.first = None
        self.pending = None

    def call(self, op, arg):
        if op == "create":
            require(self.first is None, "gate reused across episodes")
            spec, self.first = self.gate.call(op, arg)
            require(self.gate.released == self.first["language_instruction"] == TASK, "unpaired opening prompt")
            self.pending = self.first
            return spec, self.first
        require(op == "chunk" and self.pending is None and self.queries, "control without one inference")
        query = self.queries[-1]
        require("sent" not in query, "multiple chunks per query")
        obs = self.gate.call(op, arg)
        n = obs["steps"]
        query.update(sent=np.asarray(arg[:n]).tolist(), rewards=list(obs["rewards"]),
                     terminated=obs["terminated"], truncated=obs["truncated"])
        if not (obs["terminated"] or obs["truncated"]):
            self.pending = obs
        return obs

    def infer(self, item, *, noise):
        require(self.pending is not None and len(self.gate.events) == len(self.queries)+1,
                "policy called without a fresh observed query")
        event = self.gate.events[-1]
        expected = CUE if event["fire"] else TASK
        # Never reconstruct the item with the environment's original task here. Fail before
        # the model call if any part of the per-query native runner restored stale text.
        require(item["prompt"] == self.pending["language_instruction"] == expected,
                "policy prompt differs from the registered cue at inference")
        inputs = {k: array_digest(v) for k, v in item.items() if k != "prompt"}
        proposal = np.asarray(self.base.infer(item, noise=noise)["actions"], dtype=np.float32)
        require(proposal.shape == (self._model.action_horizon, 7) and np.isfinite(proposal).all(),
                "invalid decoded proposal")
        self.queries.append(dict(**event, prompt=item["prompt"], inputs=inputs,
                                 public=public_digest(self.pending["observation"]),
                                 noise=array_digest(noise), proposal=array_digest(proposal),
                                 first_five=proposal[:5].tolist()))
        self.pending = None
        return {"actions": proposal}
