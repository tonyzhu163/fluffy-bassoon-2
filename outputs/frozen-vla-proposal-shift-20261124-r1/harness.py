"""Run the frozen five-control motor loop with an adapter on its conditioning or sampling."""
import copy
import operator
from pathlib import Path
import sys

import numpy as np

_MOTOR = str(Path(__file__).resolve().parents[1] / "libero-xemb-ur5e-sysid-20261120-r1")
if _MOTOR not in sys.path:
    sys.path.append(_MOTOR)

import ur5e_motor
from ur5e_motor import NOISE_SEED_OFFSET, array_digest
from fluffy_bassoon.xemb.embodiment import ACTIVE_PROBE_BURSTS, get_embodiment_spec


TASK = "open the top layer of the drawer and put the bowl inside"
PROPRIO = ("robot0_joint_pos", "robot0_eef_pos", "robot0_eef_quat", "robot0_eef_quat_xyzw",
           "robot0_gripper_qpos")
CANDIDATE_STREAM = 90  # second word of the per-episode seed for Adapter.noises


class Adapter:
    """Every hook defaults to the untreated policy loop.

    `raw` is a private copy of the public observation: camera images and PROPRIO.
    """

    name = "raw"

    def reset(self, support, first):
        """Start an episode from the calibration record and the public reset reply."""

    def observation(self, raw):
        """Return the observation the frozen policy input is packed from."""
        return raw

    def prompt(self, raw, original):
        return original

    def noises(self, base, rng):
        """Return the candidate policy-noise arrays for this replan, each shaped like `base`."""
        return [base]

    def choose(self, raw, proposals):
        """Return the index of the (horizon, 7) proposal to execute."""
        return 0

    def command(self, raw, proposal):
        """Return the window to send. Its first five rows are clipped to the action bounds."""
        return proposal

    def observe(self, before, issued, after):
        """Receive the consumed commands and the public observations around them."""


def public(raw):
    return {key: np.array(value) for key, value in raw.items() if key in PROPRIO or key.endswith("_image")}


def collect_support(sim, case):
    """Measure three prescribed bursts on an independent layout of this body.

    A superset of the selector record: `frames` holds the four public proprioceptive
    readings around the bursts in `commands`.
    """
    _, first = sim.call("create", dict(case, embodiment="nom"))
    raw = first["observation"]
    inputs, responses, joints = [], [], []
    frames = [{key: np.asarray(raw[key]).tolist() for key in PROPRIO if key in raw}]
    previous = np.zeros(3)
    for burst in ACTIVE_PROBE_BURSTS:
        step = sim.call("chunk", burst)
        if step["steps"] != 5 or step["terminated"] or step["truncated"]:
            raise ValueError("Incomplete 15-control calibration")
        following = step["observation"]
        mean = burst[:, :3].mean(axis=0)
        inputs.append(np.r_[mean, previous].tolist())
        responses.append((following["robot0_eef_pos"] - raw["robot0_eef_pos"]).tolist())
        joints.append(np.asarray(raw["robot0_joint_pos"]).tolist())
        frames.append({key: np.asarray(following[key]).tolist() for key in PROPRIO if key in following})
        raw, previous = following, mean
    return dict(inputs=inputs, response=responses, joints=joints, case=case,
                physics=first["environment_reproducibility"]["initial_state_fingerprint"],
                commands=ACTIVE_PROBE_BURSTS.tolist(), frames=frames)


def input_digests(item):
    return {key: array_digest(value) for key, value in item.items() if key != "prompt"}


class Episode:
    """Stand between `ur5e_motor.run_episode`, the simulator and the frozen policy.

    Receipts are taken at the simulator handoff. Reward, termination, fingerprints
    and non-public observation keys never reach the adapter.
    """

    def __init__(self, sim, policy, support, adapter):
        self.sim, self.policy, self.support, self.adapter = sim, policy, support, adapter
        self._model = getattr(policy, "_model", None)  # the motor loop sizes its noise bank from this
        self.horizon = int(getattr(self._model, "action_horizon", 10))
        self.elapsed = 0
        self.policy_calls = 0
        self.opening = None
        self.replans = []

    def fail(self, hook, message):
        raise ValueError(f"Adapter {self.adapter.name!r} {hook}: {message}")

    def call(self, op, value):
        if op == "create":
            if self.elapsed or self.replans:
                raise ValueError("Create a fresh episode for every rollout")
            if (value["suite"], value["task_id"], value["variant"], value["embodiment"]) != (
                "libero_goal", 3, "BASE", "nom"
            ):
                raise ValueError("Only the native drawer-and-bowl contract is registered")
            calibration = self.support["case"]
            if calibration["arm"] != value["arm"] or calibration["init_state_id"] == value["init_state_id"]:
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
            # One candidate stream per start, shared by every branch of that start.
            self.rng = np.random.default_rng([int(value["seed"]) + NOISE_SEED_OFFSET, CANDIDATE_STREAM])
            self.adapter.reset(
                copy.deepcopy({key: item for key, item in self.support.items() if key != "physics"}),
                dict(observation=public(self.raw), language_instruction=first["language_instruction"],
                     action_schema=copy.deepcopy(bounds), arm=first["model_identity"]["arm"],
                     joint_count=int(first["model_identity"]["joint_count"]), max_steps=int(spec["max_steps"])),
            )
            return spec, first
        if op != "chunk":
            raise ValueError(op)
        step = self.sim.call(op, value)
        steps = int(step["steps"])
        if not 0 < steps <= len(value):
            raise ValueError("Simulator step count is outside the submitted command window")
        if not self.replans or self.replans[-1]["control"] != self.elapsed:
            raise ValueError("Command window without a recorded replan")
        issued = np.array(value[:steps], dtype=np.float32)
        self.replans[-1].update(submitted_steps=len(value), executed_steps=steps,
                                executed_commands=issued.tolist())
        following = step["observation"]
        self.adapter.observe(public(self.raw), issued.copy(), public(following))
        self.raw = following
        self.elapsed += steps
        return step

    def infer(self, item, *, noise):
        original = item["prompt"]
        base = np.array(noise)
        answers = {}

        def query(entry, sample):
            key = (entry["prompt"], *sorted(input_digests(entry).items()), array_digest(sample))
            if key not in answers:
                self.policy_calls += 1
                value = np.array(self.policy.infer(entry, noise=sample)["actions"], dtype=np.float32)
                if value.shape != (self.horizon, 7) or not np.isfinite(value).all():
                    raise ValueError("Invalid frozen motor proposal")
                answers[key] = value
            return answers[key]

        if self.elapsed == 0:
            # Branches of a start are matched on this even when the adapter changes the first replan.
            self.opening = array_digest(query(item, base))
        seen = self.adapter.observation(public(self.raw))
        if not isinstance(seen, dict):
            self.fail("observation", "expected a dictionary")
        prompt = self.adapter.prompt(public(self.raw), original)
        if not isinstance(prompt, str):
            self.fail("prompt", "expected a string")
        entry = ur5e_motor.to_openpi_observation(seen, prompt)
        samples = self.adapter.noises(base.copy(), self.rng)
        if not isinstance(samples, (list, tuple)) or not samples:
            self.fail("noises", "expected a nonempty list")
        samples = [np.asarray(sample, dtype=np.float32) for sample in samples]
        if any(sample.shape != base.shape or not np.isfinite(sample).all() for sample in samples):
            self.fail("noises", f"expected finite arrays of shape {base.shape}")
        proposals = [query(entry, sample) for sample in samples]
        try:
            index = operator.index(self.adapter.choose(public(self.raw), [p.copy() for p in proposals]))
        except TypeError:
            self.fail("choose", "expected an integer index")
        if not 0 <= index < len(proposals):
            self.fail("choose", f"index {index} is outside {len(proposals)} proposals")
        selected = proposals[index]
        window = np.asarray(self.adapter.command(public(self.raw), selected.copy()), dtype=np.float32)
        if window.ndim != 2 or window.shape[1] != 7 or len(window) < 5 or not np.isfinite(window).all():
            self.fail("command", "expected a finite window of at least five seven-dimensional controls")
        bounded = np.clip(window[:5], self.low, self.high)
        sent = selected.copy()
        sent[:5] = bounded
        inputs, nominal = input_digests(entry), input_digests(item)
        event = dict(
            control=self.elapsed, original_prompt=original, prompt=prompt, policy_inputs=inputs,
            inputs_changed=sorted(key for key in inputs.keys() | nominal.keys() if inputs.get(key) != nominal.get(key)),
            base_noise_sha256=array_digest(base), noise_sha256=[array_digest(sample) for sample in samples],
            proposal_sha256=[array_digest(proposal) for proposal in proposals],
            candidates=len(proposals), chosen=index, selected_proposal_sha256=array_digest(selected),
            command_delta_linf=float(np.max(np.abs(bounded - np.clip(selected[:5], self.low, self.high)))),
        )
        event["untreated"] = bool(
            not event["inputs_changed"] and prompt == original and event["command_delta_linf"] == 0
            and event["noise_sha256"] == [event["base_noise_sha256"]]
        )
        self.replans.append(event)
        return {"actions": sent}


def rollout(sim, policy, support, case, adapter):
    """One native episode. Adapters act from its first replan."""
    episode = Episode(sim, policy, support, adapter)
    record = ur5e_motor.run_episode(episode, episode, case, get_embodiment_spec("nom"))
    replans = episode.replans
    consumed = np.asarray([row for event in replans for row in event["executed_commands"]], dtype=np.float32)
    return dict(
        record, adapter=adapter.name, case=case, untreated_opening_proposal_sha256=episode.opening,
        policy_calls=episode.policy_calls, candidates=max(event["candidates"] for event in replans),
        changed_replans=sum(not event["untreated"] for event in replans),
        commands_sha256=array_digest(consumed), replans=replans,
    )
