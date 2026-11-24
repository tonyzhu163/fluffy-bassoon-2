"""Read back every actual native step and recompute the fixed paired contrasts."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from protocol import (LOCK, MANIFEST_SHA, MODES, RELEASE_SHA, ROWS, TASKS, WEIGHT_SHA,
                      fp, native_commands, query, require, source_tapes, supports)
from readback import rotation


def inspect_episode(obj, tape, support, row_fp, pin, cpu):
    key, mode = obj["task"], obj["mode"]
    task = TASKS[key]
    q, env, body = obj["query"], obj["env"], obj["body"]
    rec, calls = obj["native"], obj["calls"]
    reset = rec["reset"]
    row = q["init_state_id"]
    cut, n = len(tape), rec["controls"]
    c = support["tuned"]
    horizon = task["bank"]["native_horizon"]
    require(obj["pin"] == pin and obj["fixture"] is cpu and obj["cut"] == cut == task["cut"] and
            obj["support"] == support and rec["inferred"] == c and obj["weight_manifest"] == WEIGHT_SHA and
            row in ROWS and mode in MODES and q == query(task, row), "study identity")
    require(obj["release_row"] == env["released_row_fingerprint"] == row_fp and
            reset["physics"] == env["initial_state_fingerprint"] != support["physics"] and
            reset["scene_fingerprint"] == env["scene_fingerprint"] and env["source_pin"] == RELEASE_SHA and
            env["manifest_sha256"] == MANIFEST_SHA and env["bddl_sha256"] == task["bank"]["bddl_sha256"] and
            env["perturbation_plan"]["case_id"] == "BASE" and
            all(env[k] == q[k] for k in ("suite", "task_id", "init_state_id", "seed")), "release")
    require(body["arm"] == reset["arm"] == "Kinova3" and body["model_class"] == reset["model_class"] == "MountedKinova3"
            and body["gripper"] == "PandaGripper" and body["joint_count"] == 7 and body["controller"] == "OSC_POSE"
            and body["control_freq"] == 20 and reset["action_schema"]["control_mode"] == "relative"
            and reset["worker_step_index"] == 0 and reset["horizon"] == horizon and
            reset["prompt"] == rec["released_prompt"] == task["prompt"], "target")
    require("llvmpipe" in reset["renderer"].lower() if cpu else reset["renderer_vendor"] == "NVIDIA Corporation",
            "renderer")
    seed = 817117 + q["seed"]
    noise = np.random.default_rng(seed).standard_normal(((horizon+4)//5, 10, 32)).astype("f4")
    require(obj["shape"] == [10, 32] and reset["noise_seed"] == seed and reset["noise_sha256"] == fp(noise)
            and 0 < n <= horizon and len(calls) == (n+4)//5, "noise bank and cadence")
    public = reset["public"]
    anchor = np.asarray(obj["anchor"], np.float32)
    require(anchor.shape == (3,) and np.isfinite(anchor).all(), "reset anchor")
    rewards = []
    switched_observations = []
    handback = None
    for i, r in enumerate(calls):
        t, k = 5*i, min(5, n-5*i)
        raw = mode == "switch" and t < cut
        donor = mode in ("source", "corrected") and t < cut
        require(r["t"] == t and r["prefix"] == donor and r["raw_stage"] == raw and r["public"] == public and
                r["noise"] == fp(noise[i]) and r["prompt"] == task["prompt"] and
                r["motor"] == ("fixture" if cpu else "pi05_libero"), "policy call")
        raw_state = np.asarray(r["raw_state"], dtype=r["state_dtype"])
        tuned_state = np.asarray(r["tuned_state"], dtype=r["state_dtype"])
        require(raw_state.shape == tuned_state.shape == (8,) and np.array_equal(tuned_state[3:], raw_state[3:])
                and r["inputs"]["observation/state"] == fp(raw_state if raw else tuned_state), "proprioception routed")
        eef = np.asarray(raw_state[:3], np.float32)
        d = eef - anchor
        xy = d[:2] @ rotation(c["yaw_deg"]).T
        expected = anchor + np.array([*xy, d[2]], np.float32) if t and c["yaw_deg"] else raw_state[:3]
        require(np.array_equal(tuned_state[:3], expected) and (t or np.array_equal(eef, anchor)), "anchor transform")
        if not np.array_equal(tuned_state, raw_state):
            switched_observations.append(t)
        if t == 0:
            require(r["inputs"] == reset["motor_frame0"] and r["proposal"] == rec["opening_proposal_sha256"], "first query")
        plan = np.asarray(r["first_five"], np.float32)
        require(plan.shape == (5, 7) and np.isfinite(plan).all(), "policy actions")
        if donor:
            block = tape[t:t+5]
            expected_actions = native_commands(block, c) if mode == "corrected" else block
        else:
            expected_actions = np.clip(plan, -1, 1) if raw else native_commands(plan, c)
        steps = r["native_steps"]
        a = np.asarray([s["action"] for s in steps], np.float32)
        require(a.tobytes() == expected_actions[:k].tobytes() and
                [s["index"] for s in steps] == list(range(t+1, t+k+1)), "issued native controls")
        for s in steps:
            require(s["reward"] == float(s["success"]) and s["success"] == s["terminated"] and
                    s["truncated"] == (s["index"] == horizon and not s["success"]) and
                    (s["index"] == n or not (s["terminated"] or s["truncated"])), "native predicate")
            rewards.append(s["reward"])
        if t == cut and mode != "tuned":
            handback = dict(t=t, query=i, public=public, motor=r["motor"], inputs=r["inputs"], noise=r["noise"],
                            proposal=r["proposal"], commands=k, raw_vs_tuned_state=not np.array_equal(raw_state, tuned_state),
                            raw_vs_tuned_action_components=int(np.count_nonzero(a != np.clip(plan[:k], -1, 1))))
        public = r["after_public"]
    last = calls[-1]["native_steps"][-1]
    require(all(last[k] == rec[k] for k in ("success", "terminated", "truncated")) and
            rewards == rec["native_rewards"] and sum(rewards) == rec["native_return"] == int(rec["success"]),
            "endpoint")
    if cpu:
        require(n == cut+5 and not last["terminated"] and not last["truncated"] and
                any(t < cut for t in switched_observations), "fixture qualification")
        if handback:
            require(handback["raw_vs_tuned_state"] and handback["raw_vs_tuned_action_components"] > 0,
                    "fixture did not discriminate the handback")
    else:
        require(last["terminated"] != last["truncated"], "incomplete evaluation")
    return dict(controls=n, success=int(last["success"]), calls=len(calls), handback=handback,
                prefix_controls=min(n, cut) if mode != "tuned" else 0,
                ends_before_cut=n < cut, states_distinguishing_raw_and_tuned=len(switched_observations),
                accepted_reset=obj["accepted_reset"])


def prefix_differences(tape, c, n):
    # Count identical commands and components after clipping, including changes exactly erased by clipping.
    corrected = np.concatenate([native_commands(tape[t:t+5], c) for t in range(0, len(tape), 5)])[:n]
    source = tape[:n]
    before_clip = source.copy()
    inv = np.float32(1 / np.clip(c["xy_scale"], .25, 4))
    for t in range(0, len(tape), 5):
        block = tape[t:t+5]
        before_clip[t:t+5, :2] = ((block[:, :2] @ rotation(-c["yaw_deg"]).T) * inv)[:len(before_clip[t:t+5])]
    before_clip[:, 2] *= inv
    before_clip[:, 6] *= c["gripper_sign"]
    equal = source == corrected
    return dict(compared_controls=n, identical_controls=int(equal.all(axis=1).sum()),
                equal_components_by_channel=equal.sum(axis=0).tolist(),
                erased_by_clipping=int(((before_clip != source) & equal).sum()),
                max_abs_change=float(np.max(np.abs(source-corrected))))


def exact_interval(strata):
    """Finite empirical paired resampling, retaining each task's sample size."""
    hist = Counter({0: 1})
    for d in strata:
        counts = Counter(d)
        for _ in d:
            nxt = Counter()
            for total, ways in hist.items():
                for change, mult in counts.items():
                    nxt[total+change] += ways*mult
            hist = nxt
    denom = sum(hist.values())
    n = sum(map(len, strata))
    bounds = []
    cum = 0
    for total, count in sorted(hist.items()):
        cum += count
        while len(bounds) < 2 and 40*cum >= (1, 39)[len(bounds)]*denom:
            bounds.append(100*total/n)
    return bounds


def contrasts(groups):
    result = {}
    pairs = [("source", "tuned"), ("source", "switch"), ("source", "corrected"),
             ("switch", "tuned"), ("corrected", "tuned"), ("corrected", "switch")]
    for mode, base in pairs:
        strata, wins, losses = [], [], []
        for task, rows in groups.items():
            d = []
            for row, r in rows.items():
                delta = r["modes"][mode]["success"] - r["modes"][base]["success"]
                d.append(delta)
                if delta == 1:
                    wins.append(f"{task}/{row}")
                if delta == -1:
                    losses.append(f"{task}/{row}")
            strata.append(d)
        n = sum(map(len, strata))
        result[f"{mode}-{base}"] = dict(rescues=wins, losses=losses, ties=n-len(wins)-len(losses),
            effect_pp=100*(len(wins)-len(losses))/n, exact_paired_bootstrap_95_pp=exact_interval(strata))
    return result


def summarize(out, pin, cpu):
    origin = json.loads((out / "origin.json").read_bytes())
    require(origin["pin"] == pin and origin["fixture"] is cpu and origin["lock"] == LOCK, "origin")
    tapes, support = source_tapes(), supports()
    result = dict(pin=pin, fixture=cpu, status="passed", tasks={}, episodes=0)
    all_rows, seeds, hashes = {}, set(), set()
    rows = ROWS[:1] if cpu else ROWS
    files = {f"{key}-{row}-{mode}.json" for key in TASKS for row in rows for mode in MODES}
    require({p.name for p in (out / "episodes").glob("*.json")} == files, "episode inventory")
    for key in TASKS:
        group = {}
        for row in rows:
            modes = {}
            ref = None
            for mode in MODES:
                obj = json.loads((out / "episodes" / f"{key}-{row}-{mode}.json").read_bytes())
                require((obj["task"], obj["query"]["init_state_id"], obj["mode"]) == (key, row, mode), "record key")
                modes[mode] = inspect_episode(obj, tapes[key], support[key], origin["release_rows"][key][str(row)], pin, cpu)
                if ref is None:
                    ref = obj
                require(all(obj[k] == ref[k] for k in ("body", "env", "anchor", "support", "shape", "query")) and
                        obj["native"]["reset"] == ref["native"]["reset"] and
                        all(obj["calls"][0][k] == ref["calls"][0][k] for k in ("public", "inputs", "noise", "proposal",
                                                                                "first_five")), "paired opening")
            reset = ref["native"]["reset"]
            require(reset["noise_seed"] not in seeds and reset["noise_sha256"] not in hashes, "nonindependent bank")
            seeds.add(reset["noise_seed"])
            hashes.add(reset["noise_sha256"])
            common_n = min(modes["source"]["prefix_controls"], modes["corrected"]["prefix_controls"])
            group[str(row)] = dict(release_row=ref["release_row"], physics=reset["physics"], noise_seed=reset["noise_seed"],
                noise_sha256=reset["noise_sha256"], modes=modes,
                observed_prefix_difference=prefix_differences(tapes[key], support[key]["tuned"], common_n))
        all_rows[key] = group
        result["tasks"][key] = dict(rows=group, cut=len(tapes[key]), full_prefix_difference=prefix_differences(
            tapes[key], support[key]["tuned"], len(tapes[key])))
        if not cpu:
            result["tasks"][key].update(successes={m: sum(r["modes"][m]["success"] for r in group.values()) for m in MODES},
                                        contrasts=contrasts({key: group}))
        result["episodes"] += len(group)*len(MODES)
    if not cpu:
        result["secondary_pooled"] = contrasts(all_rows)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    p.add_argument("pin")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    result = summarize(args.out, args.pin, args.cpu)
    original = "qualified.json" if args.cpu else "result.json"
    require(result == json.loads((args.out / original).read_bytes()), "saved result disagrees")
    print(json.dumps(dict(status="reproduced", episodes=result["episodes"], pin=args.pin)))
