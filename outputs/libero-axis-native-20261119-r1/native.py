"""Fresh BASE physics probes and open-loop playback of menu02 development tapes."""

import argparse
from contextlib import contextmanager
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

import numpy as np


HERE = Path(__file__).resolve().parent
SPLITS = HERE.parent / "libero-meta-learning-20261117-r1" / "SPLITS.json"
NAMES = ("nom", "x", "y", "xy")
SIGNS = dict(zip(NAMES, ((1, 1), (-1, 1), (1, -1), (-1, -1))))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(array):
    return hashlib.sha256(str((array.shape, array.dtype.str)).encode() + array.tobytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def panel(episodes, development):
    rows = {}
    for row in read_json(episodes / "development" / "progress.json"):
        support = row["support"]["case"]
        if support["variant"] != "BASE":
            continue
        key = (support["suite"], support["task_id"], support["init_state_id"])
        require(key not in rows, "Duplicate BASE pair: %s" % (key,))
        rows[key] = row

    order = sorted((suite, task, rep * 3) for suite, tasks in development.items()
                   for task in tasks for rep in (0, 1))
    require(len(order) == 16 and set(rows) == set(order), "Incomplete BASE development panel")
    schedule = []
    for i, key in enumerate(order):
        suite, task, reset_id = key
        row = rows[key]
        rep = reset_id // 3
        trial = "%s-%s-r%s-BASE" % (suite, task, rep)
        source = row["support"]["case"]
        query = row["queries"]["default"]
        require(row["trial"] == trial, "Trial key mismatch: " + trial)
        require(query["case"] == dict(source, init_state_id=reset_id + 1, seed=source["seed"] + 1),
                "Unexpected query reset: " + trial)
        require(row["support"]["identity"]["physics"] != query["identity"]["physics"],
                "Archived support/query reset collision: " + trial)
        # Each repeat has two of each direction pair. Repeats within a task are opposites.
        direction = [v * (1 - 2 * rep) for v in SIGNS[NAMES[(i // 2) % 4]]]
        schedule.append(dict(trial=trial, repeat=rep, direction=direction, source=source,
                             probe=dict(source, init_state_id=reset_id + 2, seed=source["seed"] + 20),
                             query=query["case"]))
    return schedule, {row["trial"]: row for row in rows.values()}


def load_tape(folder, record):
    require(read_json(folder / "result.json") == record, "Source result disagrees with progress.json")
    with np.load(str(folder / "public.npz"), allow_pickle=False) as src:
        tape = {k: src[k] for k in ("issued", "positions", "initial_position", "rewards", "waits", "opening")}
    n = record["controls"]
    require(0 < n <= record["identity"]["horizon"], "Invalid source tape length")
    for name, shape in (("issued", (n, 7)), ("positions", (n, 3)), ("initial_position", (3,)),
                        ("rewards", (n,)), ("waits", (n,)), ("opening", (10, 7))):
        require(tape[name].shape == shape and np.isfinite(tape[name]).all(), "Invalid source " + name)
    require(not tape["waits"].any() and record["waits"] == 0, "Expected default controls")
    require(digest(tape["initial_position"]) == record["identity"]["public"]["robot0_eef_pos"],
            "Archive reset position mismatch")
    require(digest(tape["opening"]) == record["opening_sha256"], "Archive opening mismatch")
    burst = min(5, n)
    require(np.array_equal(tape["opening"][:burst], tape["issued"][:burst]), "Opening/actions misaligned")
    return tape


@contextmanager
def episode(bddl, case, horizon):
    from libero.libero import envs
    from libero.libero.envs.env_wrapper import ControlEnv
    from libero_remote.environment.libero import LiberoEnvironmentAdapter

    env = LiberoEnvironmentAdapter()
    headless = partial(ControlEnv, use_camera_obs=False, has_offscreen_renderer=False)
    try:
        spec = dict(case, max_steps=horizon, control_mode="relative", camera_names=[],
                    camera_height=256, camera_width=256, _bddl_path=str(bddl))
        with patch.object(envs, "OffScreenRenderEnv", headless):
            reset = env.create(spec)
        world = env._env
        raw = reset["observation"]
        require(reset["worker_step_index"] == 0 and all(not k.endswith("_image") for k in raw),
                "Invalid headless reset")
        start = dict(physics=reset["environment_reproducibility"]["initial_state_fingerprint"],
                     public={k: digest(v) for k, v in raw.items()}, prompt=str(world.language_instruction),
                     horizon=horizon, initial_position=raw["robot0_eef_pos"].tolist(),
                     sim_time_s=float(world.sim.data.time), control_hz=int(world.env.control_freq))
        yield env, start
    finally:
        env.close()


class WireAdapter:
    """Only this actuator converts a published command to the hidden physical convention."""

    def __init__(self, env, convention):
        self.env = env
        self.signs = SIGNS[convention]
        self.controls = 0

    def step(self, wire):
        physical = wire.copy()
        physical[:2] *= self.signs
        result = self.env.step(physical)
        self.controls += 1
        require(result["worker_step_index"] == self.controls, "Native control timestamp misaligned")
        return result


def probe(adapter, start, directions):
    before = np.asarray(start["initial_position"])
    tokens, samples = [], []
    for axis, direction in enumerate(directions):
        wire = np.zeros(7, dtype=np.float32)
        wire[axis] = 0.45 * direction
        for _ in range(5):
            prev = before
            result = adapter.step(wire)
            position = result["observation"]["robot0_eef_pos"].astype(float)
            before = position
            samples.append(dict(control=adapter.controls, wire_xy=wire[:2].tolist(),
                                sim_time_s=float(adapter.env._env.sim.data.time), position=position.tolist(),
                                reward=float(result["reward"]), success=bool(result["success"])))
            if (result["terminated"] or result["truncated"]) and adapter.controls < 10:
                raise ValueError("Native episode ended before the second probe endpoint")
        tokens.append(wire[:2].tolist() + (position - prev)[:2].tolist())
    return dict(reset=start, tokens=tokens, samples=samples)


def playback(adapter, start, tape, record, nominal):
    rewards, timestamps, errors = [], [], []
    initial = np.asarray(start["initial_position"])
    for index, wire in enumerate(tape["issued"]):
        result = adapter.step(wire)
        position = result["observation"]["robot0_eef_pos"].astype(float)
        rewards.append(float(result["reward"]))
        timestamps.append(float(adapter.env._env.sim.data.time))
        if nominal:
            # Source positions[i] is post-action, not the start of action i.
            errors.append(float(np.linalg.norm(position - tape["positions"][index])))
        if result["terminated"] or result["truncated"]:
            break
    n = adapter.controls
    require(np.all(np.diff([start["sim_time_s"]] + timestamps) > 0), "Non-advancing native clock")
    terminated, truncated = bool(result["terminated"]), bool(result["truncated"])
    stop = "terminated" if terminated else "truncated" if truncated else "tape_exhausted"
    outcome = dict(reset=start, success=bool(result["success"]), rewards=rewards, native_return=sum(rewards),
                   terminated=terminated, truncated=truncated, stop=stop, tape_exhausted=n == record["controls"],
                   controls=n, budget_controls=record["controls"], first_control=1, last_control=n,
                   synthetic_controls=0, post_control_sim_time_s=timestamps, sim_end_s=timestamps[-1],
                   first_reward_control=next((i for i, v in enumerate(rewards, 1) if v > 0), None),
                   final_position=position.tolist(), displacement_m=(position - initial).tolist())
    if nominal:
        outcome["reproduction"] = dict(
            source_success=record["success"], success_matches=outcome["success"] == record["success"],
            controls_match=n == record["controls"], compared_controls=n,
            reward_mismatches=int(np.count_nonzero(np.asarray(rewards) != tape["rewards"][:n])),
            exact_position_controls=errors.count(0.0), first_position_drift_m=errors[0],
            last_aligned_position_drift_m=errors[-1], max_position_drift_m=max(errors),
            source_final_position=tape["positions"][-1].tolist(),
            endpoint_drift_m=float(np.linalg.norm(position - tape["positions"][-1])))
    return outcome


def branch_lookup():
    # Evaluator-only lookup. With a fixed estimate, the two sign maps commute.
    names = {sign: name for name, sign in SIGNS.items()}
    return {true: {est: names[tuple(a * b for a, b in zip(SIGNS[true], SIGNS[est]))] for est in NAMES}
            for true in NAMES}


def run(args, cpu_start):
    from libero_remote.environment.task_manifest import TaskManifest

    design = read_json(args.episodes / "design.json")
    require(Path(design["motor"]).name == "pi05-libero-openpi-20261027-r1", "Unexpected menu02 motor")
    split = read_json(SPLITS)
    schedule, rows = panel(args.episodes, split["development"])
    manifest = TaskManifest.load(str(args.release / "assets" / "task_manifest.json"))
    require(manifest.sha256 == split["manifest_sha256"], "Unexpected release manifest")

    target = args.output if args.output.name == "summary.json" else args.output / "summary.json"
    target.parent.mkdir(exist_ok=True, parents=True)
    report = dict(status="running", started_unix_s=time.time(), schedule=schedule,
                  episodes=str(args.episodes.resolve()), motor=design["motor"], release=str(args.release.resolve()),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  manifest_sha256=manifest.sha256, source_revision="b90ee5a0c049d301529d0bf661d0a682b858ed78",
                  convention_order=NAMES, signs=SIGNS, branch_by_true_and_estimate=branch_lookup(),
                  pair_order=["suite ascending", "numeric task_id ascending", "repeat ascending"],
                  token_columns=["wire_x", "wire_y", "control5_delta_x_m", "control5_delta_y_m"],
                  scoring="Open-loop issued tape; stop at native termination, truncation or tape length.",
                  timebase="Reset at control 0; rewards[i] and source positions[i] follow control i+1.",
                  probe_episodes=0, query_episodes=0, pairs={})

    def save():
        report.update(updated_unix_s=time.time(), process_cpu_s=time.process_time() - cpu_start)
        tmp = target.with_suffix(".tmp")
        tmp.write_text(json.dumps(report, allow_nan=False, separators=(",", ":")) + "\n")
        os.replace(str(tmp), str(target))

    # Reserve a fresh result path. Publish the complete schedule before the first reset.
    with target.open("x"):
        pass
    save()
    try:
        for item in schedule:
            trial = item["trial"]
            row = rows[trial]
            record = row["queries"]["default"]
            source = record["identity"]
            query = item["query"]
            key = query["suite"], query["task_id"], "BASE"
            horizon = manifest.require(*key)["max_steps"]
            require(horizon == source["horizon"], "Source horizon mismatch: " + trial)
            bddl = manifest.resolve_bddl(*key)
            tape = load_tape(args.episodes / "development" / trial / "default", record)
            expected = {k: v for k, v in source["public"].items() if not k.endswith("_image")}
            pair = dict(query_first_xy=tape["issued"][0, :2].tolist(), source_controls=record["controls"],
                        source_success=record["success"], issued_sha256=digest(tape["issued"]),
                        source_query_reset=dict(source, public=expected),
                        source_support_physics=row["support"]["identity"]["physics"],
                        opening_sha256=record["opening_sha256"], probes={}, branches={})
            report["pairs"][trial] = pair
            archived_starts = (source["physics"], row["support"]["identity"]["physics"])
            for name in NAMES:
                with episode(bddl, item["probe"], horizon) as (env, start):
                    require(start["physics"] not in archived_starts, "Probe shares archived reset: " + trial)
                    if pair["probes"]:
                        require(start == pair["probes"]["nom"]["reset"], "Unpaired probe start: " + trial)
                    pair["probes"][name] = probe(WireAdapter(env, name), start, item["direction"])
                report["probe_episodes"] += 1
                save()

            for name in NAMES:
                with episode(bddl, query, horizon) as (env, start):
                    require(start["physics"] == source["physics"] and start["public"] == expected
                            and start["prompt"] == source["prompt"], "Source query reset mismatch: " + trial)
                    if pair["branches"]:
                        require(start == pair["branches"]["nom"]["reset"], "Unpaired query start: " + trial)
                    pair["branches"][name] = playback(WireAdapter(env, name), start, tape, record,
                                                      nominal=name == "nom")
                report["query_episodes"] += 1
                save()
            print(trial, {name: pair["branches"][name]["success"] for name in NAMES}, flush=True)
        require(report["probe_episodes"] == report["query_episodes"] == 64, "Incomplete panel")
        report["status"] = "complete"
    except Exception as exc:
        report.update(status="partial", error="%s: %s" % (type(exc).__name__, exc))
        raise
    finally:
        save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=Path, required=True, help="menu02/episodes directory")
    parser.add_argument("--release", type=Path, required=True, help="Pinned LIBERO-Pro2 release root")
    parser.add_argument("--output", type=Path, required=True, help="Output directory or fresh summary.json")
    args = parser.parse_args()
    # The qualified old simulator environment supplies robosuite and the configured assets.
    sys.path.insert(0, str(args.release.resolve()))
    sys.path.insert(0, os.environ.get("LIBERO_PRO_ROOT") or str(args.release.resolve() / "simulator-source"))
    run(args, time.process_time())


if __name__ == "__main__":
    main()
