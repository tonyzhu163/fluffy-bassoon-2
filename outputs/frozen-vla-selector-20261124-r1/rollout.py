"""Attach a three-way decision to the normal five-control OpenPI replan."""
import hashlib

import numpy as np

from fluffy_bassoon.xemb.embodiment import ACTIVE_PROBE_BURSTS, get_embodiment_spec
from ur5e_motor import array_digest, run_episode

from learner import choose
from signals import Signals


RAW = "continue"
MENU = (RAW, "correct", "cue")
TASK = "open the top layer of the drawer and put the bowl inside"


def collect_support(sim, case):
    """Measure three prescribed bursts on an independent layout of this body."""
    _, first = sim.call("create", dict(case, embodiment="nom"))
    raw = first["observation"]
    inputs, responses, joints = [], [], []
    previous = np.zeros(3)
    for burst in ACTIVE_PROBE_BURSTS:
        step = sim.call("chunk", burst)
        if step["steps"] != 5 or step["terminated"] or step["truncated"]:
            raise ValueError("Incomplete 15-control calibration")
        following = step["observation"]
        mean = burst[:, :3].mean(axis=0)
        inputs.append(np.r_[mean, previous].tolist())
        responses.append((following["robot0_eef_pos"] - raw["robot0_eef_pos"]).tolist())
        joints.append(raw["robot0_joint_pos"].tolist())
        raw, previous = following, mean
    return dict(inputs=inputs, response=responses, joints=joints, case=case,
                physics=first["environment_reproducibility"]["initial_state_fingerprint"])


def actions(result, horizon):
    value = np.asarray(result["actions"], dtype=np.float32)
    if value.shape != (horizon, 7) or not np.isfinite(value).all():
        raise ValueError("Invalid frozen motor proposal")
    return value


class SelectorEpisode:
    """Select from public inputs and audit the resulting native command window.

    Candidate windows use the reset action bounds. The executed delta is the
    largest absolute difference from nominal over simulator-consumed controls.
    """

    def __init__(self, sim, policy, support, boundary, mode, model=None, repeated=False):
        if mode not in (*MENU, "factorized", "generic"):
            raise ValueError(f"Unknown branch {mode!r}")
        if boundary < 5 or boundary % 5:
            raise ValueError("The boundary must be a positive five-control replan")
        if mode not in MENU and (model is None or model["kind"] != mode):
            raise ValueError("Provide a fitted selector of the requested kind")
        self.sim, self.policy, self.support = sim, policy, support
        self._model = getattr(policy, "_model", None)  # native motor owns the noise bank
        self.horizon = int(getattr(self._model, "action_horizon", 10))
        self.boundary, self.mode, self.model, self.repeated = boundary, mode, model, repeated
        self.elapsed = 0
        self.prefix = hashlib.sha256()
        self.boundary_record = None
        self.events = []

    def call(self, op, value):
        if op == "create":
            if self.elapsed or self.events:
                raise ValueError("Create a fresh selector for every rollout")
            support_case = self.support["case"]
            if (value["suite"], value["task_id"], value["variant"], value["embodiment"]) != (
                "libero_goal", 3, "BASE", "nom"
            ):
                raise ValueError("Only the native drawer-and-bowl contract is registered")
            if support_case["arm"] != value["arm"] or support_case["init_state_id"] == value["init_state_id"]:
                raise ValueError("Expected independent calibration on the query arm")
            spec, first = self.sim.call(op, value)
            if first["language_instruction"] != TASK or first["model_identity"]["arm"] != value["arm"]:
                raise ValueError("Unexpected native prompt or robot")
            if first["environment_reproducibility"]["initial_state_fingerprint"] == self.support["physics"]:
                raise ValueError("Calibration reused the query reset")
            bounds = first["action_schema"]
            self.low = np.asarray(bounds["low"], dtype=np.float32)
            self.high = np.asarray(bounds["high"], dtype=np.float32)
            self.raw = first["observation"]
            self.signals = Signals(self.support, self.raw)
            return spec, first
        if op != "chunk":
            raise ValueError(op)
        step = self.sim.call(op, value)
        steps = step["steps"]
        if not 0 < steps <= len(value):
            raise ValueError("Simulator step count is outside the submitted command window")
        issued = value[:steps]
        # Match this simulator receipt to the decision at the start of this chunk.
        if self.events and self.events[-1]["control"] == self.elapsed:
            event = self.events[-1]
            nominal_prefix = event["nominal_bounded"][:steps]
            event.update(
                submitted_steps=len(value), executed_steps=steps, executed_commands=issued.tolist(),
                executed_delta_linf=float(np.max(np.abs(issued - nominal_prefix))),
            )
        following = step["observation"]
        if self.elapsed < self.boundary:
            self.prefix.update(array_digest(issued).encode())
        self.signals.observe(self.raw, issued, following)
        self.raw = following
        self.elapsed += steps
        return step

    def infer(self, item, *, noise):
        result = self.policy.infer(item, noise=noise)
        at_fork = self.elapsed == self.boundary
        if not at_fork and not (self.repeated and self.elapsed > self.boundary):
            return result
        nominal = actions(result, self.horizon)
        canonical = np.clip(nominal[:5], self.low, self.high)
        proposal_digest = array_digest(nominal)
        noise_digest = array_digest(noise)
        motion, progress = self.signals.features(self.raw, canonical)
        if at_fork:
            self.boundary_record = dict(
                physics=self.sim.call("fork_digest", None), prefix=self.prefix.hexdigest(),
                proposal=proposal_digest, noise=noise_digest,
                motion=np.asarray(motion).tolist(), progress=np.asarray(progress).tolist(),
            )
        branch = self.mode if self.mode in MENU else choose(self.model, motion, progress)
        # Snapshot the nominal decision before a cue can reuse the policy's output buffer.
        event = dict(control=self.elapsed, branch=branch, nominal_prompt=item["prompt"], noise_sha256=noise_digest,
                     nominal_proposal_sha256=proposal_digest, nominal_bounded=canonical.tolist())
        selected = nominal
        if branch == "correct":
            selected = nominal.copy()
            selected[:5] = self.signals.correct(self.raw, canonical)
            result = dict(result, actions=selected)
        elif branch == "cue":
            # One re-query on identical observations and noise. Later replans use the released task.
            event["prompt"] = self.signals.cue(self.raw)
            result = self.policy.infer(dict(item, prompt=event["prompt"]), noise=noise)
            selected = actions(result, self.horizon)
        elif branch != RAW:
            raise ValueError(f"Unexpected selector output {branch!r}")
        event.update(selected_proposal_sha256=array_digest(selected),
                     selected_bounded=np.clip(selected[:5], self.low, self.high).tolist())
        self.events.append(event)
        return result


def rollout(sim, policy, support, case, boundary, mode, model=None, repeated=False):
    episode = SelectorEpisode(sim, policy, support, boundary, mode, model, repeated)
    record = run_episode(episode, episode, case, get_embodiment_spec("nom"))
    return dict(record=record, prefix=episode.prefix.hexdigest(), fork=episode.boundary_record,
                events=episode.events)
