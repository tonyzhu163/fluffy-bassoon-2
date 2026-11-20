"""Native closed-loop execution of a configurable prefix of a frozen five-control proposal."""

import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / "libero-paligemma-crossarm-20261120-r1"
sys.path.insert(1, str(PRIOR))

import numpy as np

import adapt_eval as ev
from motor_bank import verify_reset

STRIDES = (1, 2, 5)
CONDITIONS = ("raw", "correct", "empty", "wrong", "calibration")


def observed(obs):
    return {k: obs[k] if k == "prompt" else ev.digest_array(obs[k]) for k in ev.PUBLIC}


def plan_input(obs_hashes, seed):
    data = json.dumps([obs_hashes, seed], sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


class PairedRollout:
    """One frozen motor, one execution stride. Each episode starts with a fresh native reset."""

    def __init__(self, motor, stride, checkpoint_sha256, fit_id, prompt):
        if type(stride) is not int or stride not in STRIDES:
            raise ValueError(f"Expected a replan stride in {STRIDES}")
        if motor.training or any(p.requires_grad for p in motor.parameters()):
            raise ValueError("Motor must be frozen and in evaluation mode")
        self.motor = motor
        self.stride = stride
        self.checkpoint = checkpoint_sha256
        self.fit_id = fit_id
        self.prompt = prompt

    @staticmethod
    def transfer_transforms(reader, correct_history, wrong_history, panda_response, target_response):
        """Construct the existing R1 residual and calibration callables under R2 condition names."""
        return {"raw": None, "correct": ev.learned(reader, correct_history),
                "empty": ev.learned(reader, None), "wrong": ev.learned(reader, wrong_history),
                "calibration": ev.privileged(panda_response, target_response)}

    def episode(self, out, body, pose, condition="raw", transform=None, expected=None):
        """Pass the same R1 transform and raw opening as R2 transfer.evaluate."""
        if condition not in CONDITIONS or (condition == "raw") != (transform is None):
            raise ValueError("Expected raw controls or the specified condition's transform")
        if condition != "raw" and expected is None:
            raise ValueError("A transformed query requires its raw opening")
        path = out / f"query-{body}-{pose}-{condition}.npz"
        if path.exists():
            raise FileExistsError(path)
        chunks, plans, actions, raw, motion, rewards, dones, successes = [], [], [], [], [], [], [], []
        with ev.session(body) as env:
            obs, physics, horizon = ev.reset(env, body, pose)
            verify_reset(env.reset_info, body, pose, ev.seed(pose))
            if horizon != ev.HORIZON:
                raise ValueError("Unexpected native horizon")
            while len(actions) < horizon:
                t = len(actions)
                if obs["prompt"] != self.prompt:
                    raise ValueError("Unexpected public instruction")
                hashes = observed(obs)
                seed = ev.plan_seed(pose, t)
                chunk = ev.real_array(ev.proposal(self.motor, obs, seed), (5, 7), "proposal", bounded=True)
                chunk_hash = ev.digest_array(chunk)
                if t == 0:
                    opening = dict(observation=hashes, physics=physics, horizon=horizon,
                                   chunk=chunk.tolist(), chunk_sha256=chunk_hash)
                    if expected is not None and opening != expected:
                        raise ValueError(f"Unpaired opening on {body}/{pose}/{condition}")
                prefix = chunk[:min(self.stride, horizon - t)]
                issued = prefix.copy() if transform is None else transform(prefix.copy())
                issued = ev.real_array(issued, prefix.shape, "issued prefix", bounded=True)
                chunks.append(chunk)
                plans.append(dict(control=t, noise_seed=seed, observation=hashes,
                                  input_sha256=plan_input(hashes, seed), chunk_sha256=chunk_hash))
                for proposal, command in zip(prefix, issued, strict=True):
                    nxt, delta, reward, done, success = ev.step(env, command)
                    raw.append(proposal.copy())
                    actions.append(command.copy())
                    motion.append(delta)
                    rewards.append(reward)
                    dones.append(done)
                    successes.append(success)
                    if done or success:
                        break
                if done or success or len(actions) == horizon:
                    break
                obs = ev.public_obs(nxt)
        n = len(actions)
        if not dones[-1] or (n < horizon and not successes[-1]) or any(rewards[:-1]):
            raise ValueError("Incomplete native episode")
        arrays = {k: np.asarray(v) for k, v in dict(chunks=chunks, proposals=raw, actions=actions,
                  motion=motion, rewards=rewards, done=dones, success=successes).items()}
        row = dict(body=body, pose=pose, condition=condition, seed=ev.seed(pose), stride=self.stride,
                   fit_id=self.fit_id, motor_checkpoint_sha256=self.checkpoint, opening=opening,
                   steps=n, model_calls=len(plans), first_action=actions[0].tolist(),
                   native_return=sum(rewards), terminal_reward=rewards[-1], success=successes[-1],
                   done=dones[-1], array_sha256={k: ev.digest_array(v) for k, v in arrays.items()})
        tmp = path.with_suffix(".npz.tmp")
        with tmp.open("xb") as stream:
            np.savez_compressed(stream, receipt=json.dumps(row, allow_nan=False, sort_keys=True),
                                plans=json.dumps(plans, allow_nan=False, sort_keys=True), **arrays)
        tmp.replace(path)
        digest = ev.digest_file(path)
        verified = check_tape(path, stride=self.stride, body=body, pose=pose, condition=condition,
                              checkpoint=self.checkpoint, fit_id=self.fit_id, prompt=self.prompt, tape_sha256=digest)
        return {**verified, "tape": path.name, "tape_sha256": digest}


def check_tape(path, *, stride, body, pose, condition, checkpoint, fit_id, prompt, tape_sha256):
    """Read back a stride tape and verify the exact issued prefix, plan schedule and native result."""
    from pretrain_reader import NATIVE_RESET

    if ev.digest_file(path) != tape_sha256:
        raise ValueError(f"Stride tape digest differs: {path}")
    with np.load(path, allow_pickle=False) as tape:
        row = json.loads(tape["receipt"].item())
        plans = json.loads(tape["plans"].item())
        fields = ("chunks", "proposals", "actions", "motion", "rewards", "done", "success")
        data = {k: tape[k] for k in fields}
    n = row["steps"]
    if (stride not in STRIDES or condition not in CONDITIONS or not 1 <= n <= ev.HORIZON
            or row["array_sha256"] != {k: ev.digest_array(v) for k, v in data.items()}):
        raise ValueError(f"Bad stride, control count or stream digest: {path}")
    starts = range(0, n, stride)
    chunks = ev.real_array(data["chunks"], (len(starts), 5, 7), "all proposals", bounded=True)
    actions = ev.real_array(data["actions"], (n, 7), "issued controls", bounded=True)
    raw = ev.real_array(data["proposals"], (n, 7), "executed raw prefix", bounded=True)
    ev.real_array(data["motion"], (n, 7), "measured motion")
    rewards = ev.real_array(data["rewards"], (n,), "native rewards")
    dones, successes = data["done"], data["success"]
    for t, plan, chunk in zip(starts, plans, chunks, strict=True):
        hashes = plan["observation"]
        seed = ev.plan_seed(pose, t)
        if (set(hashes) != set(ev.PUBLIC) or hashes["prompt"] != prompt or plan["control"] != t
                or any(len(hashes[k]) != 64 for k in ("state", "rgb", "wrist"))
                or plan["noise_seed"] != seed or plan["input_sha256"] != plan_input(hashes, seed)
                or plan["chunk_sha256"] != ev.digest_array(chunk)
                or not np.array_equal(raw[t:t + stride], chunk[:min(stride, n - t)])):
            raise ValueError(f"Bad observation, noise, output or prefix at control {t}: {path}")
    first = row["opening"]
    labels = dict(body=body, pose=pose, seed=ev.seed(pose), stride=stride, condition=condition,
                  fit_id=fit_id, motor_checkpoint_sha256=checkpoint, model_calls=len(starts))
    if (any(row[k] != v for k, v in labels.items()) or first["horizon"] != ev.HORIZON
            or any(first["physics"].get(k) != v for k, v in NATIVE_RESET.items())
            or first["observation"] != plans[0]["observation"] or first["chunk_sha256"] != plans[0]["chunk_sha256"]
            or first["chunk"] != chunks[0].tolist() or row["first_action"] != actions[0].tolist()
            or (condition == "raw" and not np.array_equal(raw, actions))
            or any(v.shape != (n,) or v.dtype != np.bool_ for v in (dones, successes))
            or not np.array_equal(successes, rewards > 0) or any(rewards[:-1]) or any(dones[:-1])
            or not dones[-1] or (n < ev.HORIZON and not successes[-1])
            or row["native_return"] != float(rewards.sum()) or row["terminal_reward"] != float(rewards[-1])
            or row["success"] != bool(successes[-1]) or row["done"] != bool(dones[-1])):
        raise ValueError(f"Stride receipt disagrees with native tape: {path}")
    return row


def compare_plans(reference_path, other_path):
    """Pair on elapsed control. Require the same public input while the action histories agree."""
    with np.load(reference_path, allow_pickle=False) as a, np.load(other_path, allow_pickle=False) as b:
        left = {p["control"]: p for p in json.loads(a["plans"].item())}
        right = {p["control"]: p for p in json.loads(b["plans"].item())}
        actions_a, actions_b = a["actions"], b["actions"]
    length = min(len(actions_a), len(actions_b))
    difference = np.flatnonzero(np.any(actions_a[:length] != actions_b[:length], axis=1))
    first_change = int(difference[0]) if len(difference) else None
    shared = sorted(left.keys() & right.keys())
    matched = 0
    for t in shared:
        lplan, rplan = left[t], right[t]
        same_input = lplan["observation"] == rplan["observation"]
        if lplan["noise_seed"] != rplan["noise_seed"] or (not same_input and (first_change is None or t <= first_change)):
            raise ValueError(f"Unpaired noise or observation at common control {t}")
        if same_input:
            if lplan["chunk_sha256"] != rplan["chunk_sha256"]:
                raise ValueError(f"Identical public input and noise yielded different proposals at {t}")
            matched += 1
    return dict(common_plan_controls=len(shared), same_input_and_proposal=matched,
                first_different_issued_control=first_change)
