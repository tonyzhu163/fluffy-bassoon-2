"""Policy and simulator wrappers for one native run_episode, with no cross-reset state."""
import numpy as np
from fluffy_bassoon.xemb.embodiment import retarget_action_from_motor, rot2
from ur5e_motor import array_digest

from contract import MODES, require

LOW = np.full(7, -1., dtype=np.float32)
HIGH = -LOW


def issued(actions, correction):
    canonical = np.clip(actions, LOW, HIGH)
    return retarget_action_from_motor(canonical, correction["yaw_deg"], correction["xy_scale"],
                                      correction["gripper_sign"], LOW, HIGH)


def estimated_source(commands, correction):
    xyz = np.array(commands[..., :3], dtype=np.float32, copy=True)
    xyz *= np.float32(correction["xy_scale"])
    xyz[..., :2] = xyz[..., :2] @ rot2(correction["yaw_deg"]).T
    return np.clip(xyz, -1., 1.).astype(float)


def alignment(donor, before, after, carry, correction, law):
    hypothetical = issued(donor[:5], correction)
    commands = estimated_source(hypothetical, correction)
    previous = estimated_source(carry, correction).mean(axis=0)
    local = np.asarray(after, dtype=float) - np.asarray(before, dtype=float)
    measured = local.copy()
    measured[:2] = local[:2] @ rot2(correction["yaw_deg"]).T
    prefix = np.vstack((np.zeros(3), np.cumsum(commands, axis=0)))
    waypoints = (prefix @ law[:3] + np.arange(6)[:, None] * (previous @ law[3:])) / 5
    distance = np.linalg.norm(waypoints - measured, axis=1)
    index = 5 - int(np.argmin(distance[::-1]))
    return index, dict(delta_local_m=local.tolist(), delta_source_m=measured.tolist(),
                       prefix_m=waypoints.tolist(), distances_m=distance.tolist(),
                       donor_issued_xyz=hypothetical[:, :3].tolist())


class Phase:
    """Sees a correction, metric endpoints, issued commands and policy outputs only."""
    def __init__(self, mode, correction, law):
        require(mode in MODES, mode)
        self.mode = mode
        self.correction = correction
        self.law = law
        self.previous = None
        self.carry = np.zeros((5, 7), np.float32)
        self.completed = None

    def propose(self, raw):
        out = raw.copy()
        j, info = None, None
        if self.previous is not None:
            require(self.completed is not None, "policy requery before native controls completed")
            before, after, sent = self.completed
            require(len(sent) == 5, "requery after a partial block")
            if self.mode == "progress":
                j, info = alignment(self.previous, before, after, self.carry, self.correction, self.law)
            elif self.mode == "fixed":
                j = 5
            if j is not None:
                old = np.clip(self.previous[j:j+5, :3], -1., 1.)
                new = np.clip(raw[:5, :3], -1., 1.)
                out[:5, :3] = (new + old) * np.float32(.5)
            self.carry = sent
        self.previous = raw.copy()
        self.completed = None
        return out, j, info

    def observe(self, before, after, sent):
        require(self.previous is not None and self.completed is None, "unsolicited simulator block")
        self.completed = (before.copy(), after.copy(), sent.copy())


class Policy:
    def __init__(self, base, phase):
        self.base, self.phase = base, phase
        self._model = base._model
        require(int(self._model.action_horizon) >= 10 and int(self._model.action_dim) > 0,
                "model lacks two five-control windows")
        self.chunks = []

    def infer(self, item, *, noise):
        h, width = int(self._model.action_horizon), int(self._model.action_dim)
        require(noise.shape == (h, width), "noise axes differ from frozen model")
        inputs = {k: array_digest(v) for k, v in item.items() if k != "prompt"}
        raw = np.asarray(self.base.infer(item, noise=noise)["actions"], dtype=np.float32)
        require(raw.shape == (h, 7) and np.isfinite(raw).all(), "invalid original decoded chunk")
        output, j, info = self.phase.propose(raw)
        self.chunks.append(dict(original=raw.tolist(), output=output.tolist(), j=j, alignment=info,
                                policy_input=inputs, prompt=item["prompt"], noise_sha256=array_digest(noise)))
        return {"actions": output}


class Simulator:
    def __init__(self, real, policy):
        self.real, self.policy = real, policy
        self.first = None
        self.tcp = None

    def call(self, op, value):
        if op == "create":
            require(self.first is None, "episode wrappers must be fresh")
            spec, self.first = self.real.call(op, value)
            self.tcp = np.array(self.first["observation"]["robot0_eef_pos"], copy=True)
            require(self.first["action_schema"]["low"] == LOW.tolist() and
                    self.first["action_schema"]["high"] == HIGH.tolist(), "unexpected action bounds")
            return spec, self.first
        require(op == "chunk" and self.first is not None, "unexpected simulator operation")
        chunk = self.policy.chunks[-1]
        require("sent" not in chunk and 0 < len(value) <= 5, "wrong cadence")
        reply = self.real.call(op, value)
        n = int(reply["steps"])
        pos = np.asarray(reply["positions"])
        require(0 < n <= len(value) and pos.shape == (n, 3) and np.isfinite(pos).all() and
                np.array_equal(pos[-1], reply["observation"]["robot0_eef_pos"]),
                "invalid public metric position or executed count")
        sent = np.asarray(value, dtype=np.float32)[:n]
        self.policy.phase.observe(self.tcp, pos[-1], sent)
        chunk.update(before=self.tcp.tolist(), tcp=pos.tolist(), sent=sent.tolist())
        self.tcp = pos[-1].copy()
        return reply
