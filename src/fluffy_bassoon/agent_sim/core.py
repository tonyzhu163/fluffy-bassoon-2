"""Persistent, single-owner episodes with control-rate training records.

The agent supplies a short public decision summary before acting. Native
outcomes are separate from agent claims. No provider's private reasoning is
required. A failed dispatch poisons the episode: never silently replay it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Protocol

import numpy as np


FORMAT = "agent_sim_episode_v1"


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported record value: {type(value).__name__}")


def encode(value):
    return json.dumps(value, default=json_value, allow_nan=False, sort_keys=True)


def write_json(path, value):
    with Path(path).open("x") as stream:
        stream.write(encode(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value):
        raise ValueError("Require a short alphanumeric identifier, without path separators")
    return value


@dataclass
class Observation:
    images: dict[str, np.ndarray]
    state: dict[str, np.ndarray]
    instruction: str
    sim_time: float


@dataclass
class Transition:
    observation: Observation
    action: np.ndarray  # canonical action actually passed to the native adapter
    reward: float
    terminated: bool
    truncated: bool
    success: bool
    private: dict = field(default_factory=dict)


class Backend(Protocol):
    def describe(self) -> dict: ...
    def reset(self, task: str, seed: int) -> Observation: ...
    def canonicalize(self, actions: np.ndarray) -> np.ndarray: ...
    def step(self, action: np.ndarray) -> Transition: ...
    def snapshot(self) -> dict: ...
    def close(self) -> None: ...


class Session:
    def __init__(self, backend: Backend, output, *, client, model, source_commit,
                 max_controls=1000, max_chunk=50, feedback="hidden"):
        if feedback not in ("hidden", "native"):
            raise ValueError("feedback must be hidden or native")
        if any(type(x) is not int or x < 1 for x in (max_controls, max_chunk)):
            raise ValueError("Control and chunk limits must be positive integers")
        if not all(isinstance(x, str) and x.strip() for x in (client, model, source_commit)):
            raise ValueError("Declare the client, model and source commit")
        self.backend = backend
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=False)
        self.max_controls, self.max_chunk, self.feedback = max_controls, max_chunk, feedback
        self.manifest = dict(format=FORMAT, client=client, model=model,
            model_identity_source="launcher_declaration", source_commit=source_commit,
            max_controls=max_controls, max_chunk=max_chunk, feedback=feedback,
            physics_between_calls="paused", usage=None,
            usage_note="MCP does not receive provider token usage; join client telemetry separately",
            backend=backend.describe())
        write_json(self.output / "session.json", self.manifest)
        self.episode = None
        self.active = self.failed = self.closed = False
        self.controls = self.decisions = 0
        self.current = None
        self.journal = None

    def _event(self, kind, **data):
        self.journal.write(encode(dict(kind=kind, wall_unix=time.time(), **data)) + "\n")
        self.journal.flush()
        os.fsync(self.journal.fileno())

    def _check(self):
        if self.closed or self.failed:
            raise RuntimeError("Session closed or failed; preserve records and start a fresh session")

    def _observation(self, obs):
        from PIL import Image

        if (not obs.images or not obs.state or not isinstance(obs.instruction, str)
                or not math.isfinite(obs.sim_time)):
            raise ValueError("Require RGB views, public state, instruction and simulation clock")
        folder = self.episode / "observations" / f"{self.controls:06d}"
        folder.mkdir(parents=True)
        frames = {}
        for name, value in obs.images.items():
            identifier(name)
            value = np.asarray(value)
            if value.dtype != np.uint8 or value.ndim != 3 or value.shape[-1] != 3:
                raise ValueError("Require unmodified uint8 HWC RGB camera observations")
            path = folder / f"{name}.png"
            Image.fromarray(value).save(path)
            frames[name] = dict(path=path.relative_to(self.output).as_posix(),
                               sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        state = {}
        for name, value in obs.state.items():
            value = np.asarray(value)
            if value.ndim != 1 or not np.isfinite(value).all():
                raise ValueError("Require finite public state vectors")
            state[name] = value.tolist()
        result = dict(control=self.controls, images=frames, state=state,
                      instruction=obs.instruction, sim_time=float(obs.sim_time))
        write_json(folder / "public.json", result)
        return result

    def _private(self, value):
        # Kept away from MCP and from causal training inputs, including XML,
        # object poses and simulator state used for offline replay/analysis.
        write_json(self.episode / "private" / f"{self.controls:06d}.json", value)

    def reset(self, task, seed, episode_id):
        self._check()
        if self.active:
            raise RuntimeError("Finish the current episode before resetting")
        identifier(task); identifier(episode_id)
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("Require an explicit uint32 scene seed")
        self.episode = self.output / episode_id
        self.episode.mkdir()
        (self.episode / "private").mkdir()
        if self.journal is not None:
            self.journal.close()
        self.journal = (self.episode / "events.jsonl").open("x")
        self.controls = self.decisions = 0
        self.started = time.monotonic()
        self.return_ = 0.
        self.success = self.terminated = self.truncated = False
        self.episode_id = episode_id
        write_json(self.episode / "episode.json", dict(format=FORMAT, task=task, seed=seed,
            episode_id=episode_id, session="../session.json"))
        self._event("reset_intent", task=task, seed=seed)
        try:
            obs = self.backend.reset(task, seed)
            self.current = self._observation(obs)
            self._private(self.backend.snapshot())
            self.active = True
            self._event("reset", observation=self.current, interface=self.backend.describe())
        except BaseException as error:
            self.failed = True
            self._event("error", operation="reset", error=type(error).__name__)
            raise
        return self.observe()

    def observe(self):
        self._check()
        if self.current is None:
            raise RuntimeError("Reset an episode first")
        return dict(episode_id=self.episode_id, observation=self.current,
                    status=self.status(), interface=self.backend.describe())

    def status(self):
        self._check()
        result = dict(active=self.active, controls=self.controls, decisions=self.decisions,
                      max_controls=self.max_controls, feedback=self.feedback)
        if self.current is not None:
            result.update(terminated=self.terminated, truncated=self.truncated)
            if self.feedback == "native":
                result.update(native_return=self.return_, success_once=self.success)
        return result

    def execute(self, actions, *, observation_control, subgoal, summary, expected_outcome):
        self._check()
        if not self.active or self.terminated or self.truncated:
            raise RuntimeError("No active episode with remaining controls")
        if type(observation_control) is not int or observation_control != self.controls:
            raise ValueError("Stale observation: observe again before proposing actions")
        context = dict(subgoal=subgoal, summary=summary, expected_outcome=expected_outcome)
        if any(not isinstance(v, str) or not v.strip() for v in context.values()):
            raise ValueError("Supply a public subgoal, brief decision summary and expected outcome")
        requested = np.asarray(actions, dtype=np.float64)
        if (requested.ndim != 2 or not 1 <= len(requested) <= self.max_chunk
                or len(requested) + self.controls > self.max_controls
                or not np.isfinite(requested).all()):
            raise ValueError("Require a finite nonempty action chunk within the remaining allowance")
        commands = np.asarray(self.backend.canonicalize(requested.copy()), dtype=np.float64)
        if commands.shape != requested.shape or not np.isfinite(commands).all():
            raise ValueError("Backend returned malformed canonical actions")
        decision = self.decisions
        before = self.current
        self._event("decision", decision=decision, observation=before, context=context,
                    requested_actions=requested, commands=commands)
        executed = []
        started = time.monotonic()
        try:
            for command in commands:
                self._event("dispatch", decision=decision, control=self.controls, action=command)
                transition = self.backend.step(command.copy())
                actual = np.asarray(transition.action, dtype=np.float64)
                if (actual.shape != command.shape or not np.isfinite(actual).all()
                        or not math.isfinite(transition.reward)
                        or transition.observation.sim_time <= self.current["sim_time"]):
                    raise ValueError("Invalid native transition or non-advancing simulation clock")
                old = self.current
                self.controls += 1
                self.current = self._observation(transition.observation)
                self._private(transition.private)
                self.return_ += float(transition.reward)
                self.success |= bool(transition.success)
                self.terminated = bool(transition.terminated)
                self.truncated = bool(transition.truncated) or self.controls == self.max_controls
                self._event("transition", decision=decision, observation_control=old["control"],
                    action=actual, next_observation=self.current, reward=float(transition.reward),
                    success=bool(transition.success), terminated=self.terminated, truncated=self.truncated)
                executed.append(actual.tolist())
                if self.terminated or self.truncated:
                    break
            self.decisions += 1
            result = self.observe()
            result.update(executed_actions=executed, executed_controls=len(executed),
                          sim_seconds=self.current["sim_time"] - before["sim_time"])
            if self.feedback == "native":
                result["last_reward"] = float(transition.reward)
            self._event("tool_result", decision=decision, result=result,
                        execution_wall_seconds=time.monotonic() - started)
            return result
        except BaseException as error:
            self.failed = True
            self._event("error", operation="execute", decision=decision,
                        error=type(error).__name__, completed_controls=len(executed))
            raise

    def finish(self, *, summary, success_claim):
        self._check()
        if not self.active or not isinstance(summary, str) or not summary.strip() or type(success_claim) is not bool:
            raise ValueError("Finish an active episode with a summary and an explicit agent success claim")
        self._event("finish", summary=summary, success_claim=success_claim, control=self.controls)
        write_json(self.episode / "result.json", dict(completed=True, controls=self.controls,
            decisions=self.decisions, native_return=self.return_, native_success_once=self.success,
            agent_success_claim=success_claim, terminated=self.terminated, truncated=self.truncated,
            ended_by="native_boundary" if self.terminated or self.truncated else "agent",
            wall_seconds=time.monotonic() - self.started))
        self.active = False
        self.backend.close()
        return self.status()

    def close(self):
        if self.closed:
            return
        try:
            if self.active and self.journal is not None:
                self._event("interrupted", control=self.controls)
            self.backend.close()
        finally:
            if self.journal is not None:
                self.journal.close()
            self.closed = True
