"""Native Goal 5 prefix capture, exact replay and teacher-only bank publication."""

import sys
from multiprocessing.connection import Connection
from pathlib import Path
from tempfile import TemporaryDirectory

import mujoco
import numpy as np
from recover import (
    BODIES,
    PROMPT,
    action,
    array_hash,
    checked_prefixes,
    code_ids,
    digest,
    public_hashes,
    read,
    reset_identity,
    sha256,
    write,
)

# isort: split
# recover adds the pinned sibling run to the import path.
from demos import save_tape
from expert import session
from recover_teacher import (
    ARCHIVE_SHA,
    bank_choices,
    official_reference,
    paddle_handoff,
    pick,
    selected_source,
    tracker_handoff,
)
from sim import _pack, _public


class ReplayMismatch(ValueError):
    pass


def require_equal(label, observed, expected):
    if observed != expected:
        raise ReplayMismatch(f"{label}: expected {expected}, observed {observed}")


def fingerprint(env, obs):
    wrapper = env.native.arm._env
    model, data = wrapper.sim.model._model, wrapper.sim.data._data
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    integration = np.empty(mujoco.mj_stateSize(model, spec), dtype=np.float64)
    mujoco.mj_getState(model, data, integration, spec)
    if obs["prompt"] != PROMPT:
        raise ValueError("Goal 5 instruction changed during native rollout")
    return dict(
        native=array_hash(wrapper.get_sim_state()), integration=array_hash(integration), **public_hashes(obs)
    )


def reset(env, source, approved, states):
    body, pose, seed = (source[k] for k in ("body", "pose", "seed"))
    # Validate the destination before any reset, including during replay.
    donor = reset_identity(approved["reset"], body, pose, seed)
    obs = env.reset(pose, seed)
    live = reset_identity(env.reset_info, body, pose, seed)
    for key in ("released_row_fingerprint", "initial_state_fingerprint"):
        require_equal("approved " + key, live[key], donor[key])
    opening = fingerprint(env, obs)
    require_equal("approved opening hand state", opening["state"], array_hash(states[0]))
    return obs, opening


def issue(env, requested):
    # Session.step omits issued_action. Read it from the same ArmEnv.step packet
    # as the native reward and next observation so labels are actuator receipts.
    requested = action(requested)
    packet = env.native.arm.step(requested)
    issued = packet["issued_action"]
    require_equal("issued float32 control", array_hash(issued), array_hash(requested))
    obs = _public(packet)
    motion = np.asarray(packet["sensor_motion"], dtype=np.float32)
    reward, success = float(packet["reward"]), bool(packet["success"])
    done = bool(packet["terminated"] or packet["truncated"])
    if (
        motion.shape != (7,)
        or not np.isfinite(motion).all()
        or not np.isfinite(reward)
        or reward < 0
        or (reward > 0) != success
        or (success and not done)
    ):
        raise ValueError("Invalid native response")
    step = {
        "t": int(packet["worker_step_index"]),
        "action": issued.tolist(),
        "action_sha256": array_hash(issued),
        "motion": motion.tolist(),
        "reward": reward,
        "done": done,
        "success": success,
        "post": fingerprint(env, obs),
    }
    return obs, step


def prefix_chain(opening, steps):
    value = digest(opening)
    for step in steps:
        value = digest([value, step])
    return value


def capture_roll(channel, config, source, approved, states):
    steps, proposals = [], []
    limit = max(config["prefixes"])
    with session(source["body"], True) as env:
        obs, opening = reset(env, source, approved, states)
        while len(steps) < limit:
            channel.send(_pack(obs))
            input_hashes, proposed = channel.recv()
            require_equal("motor input and current public frame", input_hashes, public_hashes(obs))
            chunk = np.asarray(proposed, dtype=np.float32)
            if chunk.shape != (5, 7):
                raise ValueError("Wrong motor action horizon")
            proposals.append(
                {
                    "start": len(steps),
                    "input": input_hashes,
                    "actions": chunk.tolist(),
                    "sha256": array_hash(chunk),
                }
            )
            for a in chunk[: limit - len(steps)]:
                obs, step = issue(env, a)
                require_equal("native control index", step["t"], len(steps) + 1)
                steps.append(step)
                if step["done"]:
                    break
            if steps[-1]["done"]:
                break
        return {
            "source": source,
            "reset": env.reset_info,
            "opening": opening,
            "steps": steps,
            "proposals": proposals,
            "motor_sha256": config["motor"]["sha256"],
            "prefix_sha256": {
                str(n): prefix_chain(opening, steps[:n]) for n in config["prefixes"] if n <= len(steps)
            },
        }


def capture_native(fd, config_path):
    config = read(config_path)
    available = bank_choices(config)
    keys = pick(config, available)
    index = {"version": 1, "config_sha256": sha256(config_path), "rolls": []}
    starts = set()
    with Connection(fd) as channel:
        for body, pose in keys:
            source, row, states, _ = selected_source(available[(body, pose)][0])
            start = (body, row["reset"]["environment_reproducibility"]["released_row_fingerprint"])
            if start in starts:
                raise ValueError("Two requested rolls use the same body and released row")
            starts.add(start)
            roll = capture_roll(channel, config, source, row, states)
            path = config_path.parent / f"learner-{body}-{pose}.json"
            write(path, roll)
            index["rolls"].append({"file": path.name, "sha256": sha256(path)})
            print("captured", body, pose, len(roll["steps"]), flush=True)
    write(config_path.parent / "capture.json", index)


def verify_roll(roll, config):
    steps = roll["steps"]
    if not 0 < len(steps) <= max(config["prefixes"]):
        raise ValueError("Invalid captured prefix length")
    if len(steps) < max(config["prefixes"]) and not steps[-1]["done"]:
        raise ValueError("Incomplete frozen motor roll")
    proposals = roll["proposals"]
    if [p["start"] for p in proposals] != list(range(0, len(steps), 5)):
        raise ValueError("Learner proposals are not aligned with native rollout chunks")
    for p in proposals:
        chunk = np.asarray(p["actions"], dtype=np.float32)
        start = p["start"]
        observation = steps[start - 1]["post"] if start else roll["opening"]
        expected = {k: observation[k] for k in ("state", "rgb", "wrist", "prompt")}
        if p["input"] != expected or chunk.shape != (5, 7) or array_hash(chunk) != p["sha256"]:
            raise ValueError("Frozen proposal digest mismatch")
    for i, step in enumerate(steps):
        issued = action(step["action"])
        proposed = action(proposals[i // 5]["actions"][i % 5])
        if (
            step["t"] != i + 1
            or step["action_sha256"] != array_hash(issued)
            or array_hash(proposed) != array_hash(issued)
            or (step["done"] and i < len(steps) - 1)
        ):
            raise ValueError("Learner prefix contains a substituted command or premature termination")
    expected = {
        str(n): prefix_chain(roll["opening"], steps[:n]) for n in config["prefixes"] if n <= len(steps)
    }
    if roll["prefix_sha256"] != expected or roll["motor_sha256"] != config["motor"]["sha256"]:
        raise ValueError("Frozen motor or learner prefix hash mismatch")


def check_labels(path, handoff, trace):
    """Verify the training tensors against native evidence before publishing the row."""
    pre = [handoff] + [step["post"] for step in trace[:-1]]
    with np.load(path, allow_pickle=False) as tape:
        for key in ("state", "rgb", "wrist"):
            values = tape[key]
            require_equal("suffix " + key, [array_hash(v) for v in values], [p[key] for p in pre])
            del values
        require_equal(
            "suffix actions", [array_hash(a) for a in tape["actions"]], [s["action_sha256"] for s in trace]
        )
        require_equal("suffix motion", tape["motion"].tolist(), [s["motion"] for s in trace])
        require_equal("suffix reward", tape["rewards"].tolist(), [s["reward"] for s in trace])
        require_equal("terminal state", array_hash(tape["terminal_state"]), trace[-1]["post"]["state"])


def attempt(out, roll, approved, states, bank_actions, ref, n, trace):
    source = roll["source"]
    body, pose, seed = (source[k] for k in ("body", "pose", "seed"))
    row = {
        "body": body,
        "pose": pose,
        "seed": seed,
        "prefix_steps_requested": n,
        "steps": 0,
        "route": "recovery_" + source["teacher"],
        "accepted": False,
        "native_success": False,
        "done": False,
        "terminal_reward": None,
        "source": source,
        "motor_checkpoint_sha256": roll["motor_sha256"],
    }
    if len(roll["steps"]) < n:
        row.update(
            outcome="learner_terminated_before_handoff",
            learner_steps=len(roll["steps"]),
            learner_terminal=roll["steps"][-1],
        )
        return row
    with session(body, True) as env:
        try:
            obs, opening = reset(env, source, approved, states)
            for field in ("model_identity", "environment_reproducibility", "max_steps"):
                require_equal("reset " + field, digest(env.reset_info[field]), digest(roll["reset"][field]))
            require_equal("opening", opening, roll["opening"])
            initial_scene = env.expert_scene()
            replay = []
            for expected in roll["steps"][:n]:
                obs, actual = issue(env, expected["action"])
                require_equal(f"native prefix step {actual['t']}", actual, expected)
                replay.append(actual)
            chain = prefix_chain(opening, replay)
            require_equal("replayed prefix digest", chain, roll["prefix_sha256"][str(n)])
        except ReplayMismatch as exc:
            row.update(outcome="prefix_mismatch", error=str(exc))
            return row
        row.update(reset=env.reset_info, learner_steps=n, prefix_sha256=chain, handoff=replay[-1]["post"])
        if replay[-1]["done"]:
            row.update(outcome="learner_terminal_at_handoff", learner_terminal=replay[-1])
            return row
        if source["teacher"] == "official_tracker" and ref is None:
            row.update(outcome="official_reference_unavailable")
            return row
        scene = env.expert_scene()
        geom = env.native.geometry
        if source["teacher"] == "paddle":
            command, alignment = paddle_handoff(approved["params"], scene, geom)
            row["params"] = approved["params"]
        else:
            command, alignment = tracker_handoff(ref, initial_scene, obs, scene)
            evidence = ref["evidence"]
            row.update(
                reference_demo=evidence["demo"], reference_pose=evidence["pose"], reference_evidence=evidence
            )
        row["phase_at_handoff"] = alignment
        # Compare to the successful tape on the same reset at this control count.
        if n < len(states):
            row["bank_timed_xyz_gap_m"] = float(np.linalg.norm(obs["state"][:3] - states[n, :3]))
        overlap = min(n, len(bank_actions))
        issued_prefix = np.asarray([s["action"] for s in replay[:overlap]], np.float32)
        row["bank_prefix_compared_steps"] = overlap
        row["bank_prefix_action_mae"] = float(np.abs(issued_prefix - bank_actions[:overlap]).mean())
        frames, actions, motions, rewards = [], [], [], []
        for _ in range(env.reset_info["max_steps"] - n):
            requested, decision = command(obs, scene, geom)
            frames.append(obs)
            obs, actual = issue(env, requested)
            require_equal("teacher native control index", actual["t"], n + len(actions) + 1)
            trace.append(dict(**actual, decision=decision))
            actions.append(action(actual["action"]))
            motions.append(np.array(actual["motion"], dtype=np.float32))
            rewards.append(actual["reward"])
            if actual["done"]:
                break
            scene = env.expert_scene()
        if not trace or not trace[-1]["done"]:
            raise ValueError("Missing native endpoint within the remaining episode horizon")
        accepted = bool(actual["success"] and actual["reward"] > 0)
        row.update(
            steps=len(trace),
            total_steps=n + len(trace),
            done=actual["done"],
            native_success=actual["success"],
            terminal_reward=actual["reward"],
            native_return=float(sum(rewards)),
            accepted=accepted,
            outcome="recovered" if accepted else "teacher_failed",
        )
        if accepted:
            file = Path(body) / f"layout-{pose}-seed-{seed}-prefix-{n}.npz"
            with TemporaryDirectory(dir=out) as scratch:
                pending = Path(scratch) / file.name
                save_tape(pending, frames, actions, motions, rewards, obs["state"], PROMPT, body, pose, seed)
                frames.clear()
                check_labels(pending, row["handoff"], trace)
                row.update(file=str(file), sha256=sha256(pending))
                (out / body).mkdir(exist_ok=True)
                pending.replace(out / file)
        return row


def collect(args):
    capture_file = args.capture.resolve(strict=True)
    if sha256(capture_file) != args.capture_sha256:
        raise ValueError("Pinned capture manifest SHA-256 mismatch")
    capture = read(capture_file)
    root = capture_file.parent
    if capture["version"] != 1 or sha256(root / "config.json") != capture["config_sha256"]:
        raise ValueError("Pinned capture config mismatch")
    config = read(root / "config.json")
    if config["code"] != code_ids():
        raise ValueError("Capture and recovery implementation differ")
    prefixes = checked_prefixes(config["prefixes"])
    available = bank_choices(config)
    keys = pick(config, available)
    if len(capture["rolls"]) != len(keys):
        raise ValueError("Incomplete capture manifest")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "receipts").mkdir()
    index = {
        "suite": "libero_goal",
        "task_id": 5,
        "case": "BASE",
        "bodies": [b for b in BODIES if any(k[0] == b for k in keys)],
        "poses": sorted({k[1] for k in keys}),
        "status": "running",
        "attempts": [],
        "capture": str(capture_file),
        "capture_sha256": args.capture_sha256,
        "revision": config["revision"],
        "motor": config["motor"],
        "argv": sys.argv,
    }
    write(out / "index.json", index)
    archive_verified = False
    for key, item in zip(keys, capture["rolls"], strict=True):
        body, pose = key
        path = root / f"learner-{body}-{pose}.json"
        if item["file"] != path.name or sha256(path) != item["sha256"]:
            raise ValueError("Captured source tape mismatch")
        roll = read(path)
        verify_roll(roll, config)
        source, approved, states, commands = selected_source(available[key][0])
        if source != roll["source"]:
            raise ValueError("Approved teacher receipt or source bank tape changed")
        ref, reference_error = None, None
        if source["teacher"] == "official_tracker":
            if not archive_verified:
                if not args.archive or sha256(args.archive) != ARCHIVE_SHA:
                    raise ValueError("Provide the pinned released Goal 5 HDF5 for Panda tracking")
                archive_verified = True
            try:
                ref = official_reference(args.archive, approved)
            except RuntimeError as exc:
                if not str(exc).startswith("Released Panda replay ended at"):
                    raise
                reference_error = str(exc)
        for n in prefixes:
            trace = []
            row = attempt(out, roll, approved, states, commands, ref, n, trace)
            row.update(learner_file=path.name, learner_file_sha256=item["sha256"])
            if reference_error:
                row["reference_error"] = reference_error
            receipt = Path("receipts") / f"{body}-{pose}-prefix-{n}.json"
            write(out / receipt, {"result": row, "teacher_steps": trace})
            row.update(receipt=str(receipt), receipt_sha256=sha256(out / receipt))
            index["attempts"].append(row)
            write(out / "index.json", index)
            print(body, pose, n, row["outcome"], "teacher_steps", row["steps"], flush=True)
    index.update(
        status="complete",
        accepted=sum(r["accepted"] for r in index["attempts"]),
        labels=sum(r["steps"] for r in index["attempts"] if r["accepted"]),
    )
    write(out / "index.json", index)
    print("bank_sha256", sha256(out / "index.json"), "accepted", index["accepted"], "labels", index["labels"])
    if any(r["outcome"] == "prefix_mismatch" for r in index["attempts"]):
        raise SystemExit(2)


if __name__ == "__main__":
    capture_native(int(sys.argv[1]), Path(sys.argv[2]))
