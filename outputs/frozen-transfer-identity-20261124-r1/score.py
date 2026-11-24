"""Read the registered BASE role map and fixed archive. No model fitting or rollout."""
import argparse
import csv
import hashlib
import json
import os
import random
import socket
import subprocess
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARCHIVE = HERE.parent / "frozen-transfer-object-20261124-r1/evidence/trajectories.tar.gz"
ARCHIVE_SHA = "56bed093a4f48f5f293f342541878dd18929fd1c9f64093d7a2005d5ffa52f44"
HANDS = ("left", "right", "palm")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def role(body, task):
    for name in task["objects"]:
        if body in (name, name + "_main"):
            if name in task["pickups"]:
                return "allowed"
            if name in task["receivers"]:
                return "loose_receiver"
            return "same_class_distractor" if any(
                task["objects"][p] == task["objects"][name] for p in task["pickups"]
            ) else "other_distractor"
    for name in task["fixtures"]:
        if body == name or body.startswith(name + "_"):
            return "fixed_receiver" if name in task["receivers"] else "fixture"
    return "unresolved"


WRONG = {"same_class_distractor", "other_distractor", "loose_receiver"}


def window(endpoints, existing, task):
    assert len(endpoints) == 5
    held, touches = [], []
    known_free = {b for name in task["objects"] for b in (name, name + "_main")}
    for e in endpoints:
        hands = [set(e[k]) for k in HANDS]
        touched = set.union(*hands)
        carry = set(e["carry"])
        both = hands[0] & hands[1]
        unsupported = both - set(e["supported"])
        assert carry <= unsupported and (unsupported & known_free) == (carry & known_free)
        assert e["empty"] == (not touched)
        held.append(carry)
        touches.append(touched)
    common = set.intersection(*held)
    all_held = set.union(*held)
    assert existing == (1 if common else 0 if not any(touches) else None)
    obj, label = "", "no_strict_load"
    if common and len(all_held) == 1:
        obj = next(iter(common))
        identity = role(obj, task)
        label = "correct" if identity == "allowed" else "wrong" if identity in WRONG else "ambiguous_load"
    elif len(all_held) > 1:
        label = "ambiguous_load"
    elif not any(touches):
        label = "empty"
    events = set()
    for e, hold, touch in zip(endpoints, held, touches):
        roles = {role(body, task) for body in touch}
        events.update("touch_" + key for key in roles)
        if len(touch) > 1:
            events.add("concurrent_touches")
        if "allowed" in roles and roles & WRONG:
            events.add("target_and_nontarget_free_contact")
        if hold and touch - hold:
            events.add("carry_and_other_contact")
        if any(role(b, task) in WRONG for b in hold):
            events.add("wrong_at_endpoint")
        if any(role(b, task) in WRONG for b in set(e["left"]) & set(e["right"]) & set(e["supported"])):
            events.add("supported_bilateral_nontarget")
        if any(role(b, task) in WRONG for b in touch - hold):
            events.add("nontarget_touch_without_carry")
        if len(hold) > 1:
            events.add("simultaneous_carries")
    if "wrong_at_endpoint" in events and label != "wrong":
        events.add("wrong_endpoint_without_strict_wrong_window")
    return label, obj, events, set.union(*touches), all_held


def group_name(split, layout, donor_row):
    if split == "arm_swap":
        return "swapped_arm"
    return split + ("_original" if layout == donor_row else "_alternate")


def episode(tar, m, rules):
    def read(field):
        return json.load(tar.extractfile(m.name.replace(".lineage.", "." + field + ".")))
    row = read("lineage")
    split = row["split"]
    assert m.name.split("/")[0] == split
    spec = row["spec"]
    task_id = f"{spec['suite']}/{spec['task_id']}"
    roles = rules["tasks"][task_id]
    reset = row["reset"]
    assert spec["variant"] == "BASE" and reset["bddl_sha256"] == roles["bddl_sha256"]
    assert reset["source_pin"] == rules["release"] and reset["manifest_sha256"] == rules["manifest_sha256"]
    assert row["model"]["arm"] == row["actuated_arm"] and row["donor_arm"] == spec["arm"]
    group = group_name(split, row["layout"], spec["init_state_id"])
    assert (row["actuated_arm"] == row["donor_arm"]) == (group != "swapped_arm")
    assert bool(max(row["original_rewards"])) == bool(row["original_success"])
    assert bool(max(row["cpu_rewards"])) == bool(row["cpu_success"])
    assert len(row["cpu_rewards"]) == row["cpu_controls"] and len(row["original_rewards"]) == row["source_controls"]
    counts, touch_names, carry_names = Counter(), Counter(), Counter()
    observations = {}
    strict_ids = defaultdict(list)
    for truth, pub in zip(read("truth"), read("public"), strict=True):
        t = truth["control"]
        assert t == pub["control"] and t not in observations and t % 5 == 0 and pub["close_age"] >= 5
        assert 5 <= t <= row["cpu_controls"] - int(row["cpu_terminal"])
        label, obj, events, touched, held = window(truth["endpoints"], truth["y"], roles)
        counts[label] += 1
        counts.update(events)
        if label in ("correct", "wrong"):
            strict_ids[obj].append(t)
        observations[t] = (label, obj, pub["close_age"], pub["aperture_mm"])
        touch_names.update(touched)
        carry_names.update(held)
    record = dict(group=group, case=row["case"], task=task_id, arm=row["actuated_arm"], donor_arm=spec["arm"],
                  donor_row=spec["init_state_id"], layout=row["layout"], seed=row["seed"],
                  donor_hash=row["input_sha256"], scene=reset["scene_fingerprint"], released_row=reset["released_row_fingerprint"],
                  perturbation=reset["perturbation_plan_fingerprint"], bddl=reset["bddl_sha256"],
                  cpu_controls=row["cpu_controls"], original_controls=row["source_controls"],
                  native_success=int(row["original_success"]), replay_success=int(row["cpu_success"]),
                  replay_censored=int(row["censored"]), replay_truncated=int(row["cpu_truncated"]),
                  rewards_match=int(row["cpu_rewards"] == row["original_rewards"]),
                  max_tcp_delta_mm=row["tcp_delta_mm"], max_jaw_delta_mm=row["jaw_delta_mm"],
                  eligible=len(observations), observed_endpoints=5 * len(observations),
                  first_saved_endpoint=(min(observations) - 4 if observations else None),
                  possible_nontarget=int(bool(set(roles["objects"]) - set(roles["pickups"]))),
                  window_counts=dict(counts), touch_names_windows=dict(touch_names), carried_names_any_endpoint_windows=dict(carry_names),
                  strict_controls_by_root=dict(strict_ids))
    return record, observations


def summary(rows):
    failures = [r for r in rows if not r["native_success"]]
    counts = Counter()
    for row in rows:
        counts.update(row["window_counts"])
    def any_event(event, records):
        return sum(r["window_counts"].get(event, 0) > 0 for r in records)
    return dict(episodes=len(rows), replay_success=sum(r["replay_success"] for r in rows),
                replay_censored=sum(r["replay_censored"] for r in rows), donor_failures=len(failures),
                matched_outcomes=sum(r["replay_success"] == r["native_success"] for r in rows),
                exact_reward_tapes=sum(r["rewards_match"] for r in rows), eligible_windows=sum(r["eligible"] for r in rows),
                zero_eligible=sum(not r["eligible"] for r in rows), controls=sum(r["cpu_controls"] for r in rows),
                episodes_correct=any_event("correct", rows), episodes_wrong=any_event("wrong", rows),
                failed_with_correct=any_event("correct", failures), failed_with_wrong=any_event("wrong", failures),
                failed_with_wrong_at_endpoint=any_event("wrong_at_endpoint", failures),
                failed_with_distractor_touch=sum(any(r["window_counts"].get("touch_" + k) for k in
                    ("other_distractor", "same_class_distractor")) for r in failures),
                failed_with_loose_receiver_touch=any_event("touch_loose_receiver", failures),
                failed_with_fixed_receiver_touch=any_event("touch_fixed_receiver", failures),
                failed_with_concurrent_target_and_other=any_event("target_and_nontarget_free_contact", failures),
                failed_with_empty=any_event("empty", failures),
                failed_zero_eligible=sum(not r["eligible"] for r in failures),
                failed_with_documented_load=sum(any(r["window_counts"].get(k) for k in ("correct","wrong")) for r in failures),
                windows=dict(counts))


def matched(rows, all_obs):
    original = {r["donor_hash"]: r for r in rows if r["group"].endswith("_original")}
    assert len(original) == 25
    result = {}
    for group in ("target_alternate", "swapped_arm", "source_alternate"):
        cases = []
        for b in [r for r in rows if r["group"] == group]:
            a = original[b["donor_hash"]]
            assert a["task"] == b["task"] and a["donor_row"] == b["donor_row"] and a["bddl"] == b["bddl"]
            if group == "swapped_arm":
                assert a["arm"] == "IIWA" and b["arm"] == "UR5e"
                assert all(a[f] == b[f] for f in ("seed", "layout", "scene", "released_row", "perturbation"))
            else:
                assert a["arm"] == b["arm"] and a["layout"] != b["layout"]
            x = all_obs[a["group"], a["case"]]
            y = all_obs[group, b["case"]]
            shared = sorted(x.keys() & y.keys())
            pairs = []
            both_loaded = 0
            for t in shared:
                assert x[t][2] == y[t][2]
                labels = {x[t][0], y[t][0]}
                both_loaded += labels <= {"correct", "wrong"}
                if labels == {"correct", "wrong"}:
                    pairs.append(dict(control=t, original=x[t], alternate=y[t]))
            cases.append(dict(task=a["task"], donor=a["case"], alternative=b["case"], common_windows=len(shared),
                              both_strictly_loaded=both_loaded, opposed=pairs))
        result[group] = cases
    return result


def uncertainty(rows):
    by_task = defaultdict(list)
    for r in rows:
        by_task[r["task"]].append(r)
    groups = []
    for records in by_task.values():
        failures = [r for r in records if not r["native_success"]]
        groups.append((sum(bool(r["window_counts"].get("wrong")) for r in failures), len(failures)))
    supported = sum(n > 0 for _, n in groups)
    affected = sum(x > 0 for x, _ in groups)
    d = dict(tasks_with_failures=supported, tasks_with_wrong_on_failure=affected)
    if not affected:
        d["conditional_task_incidence_one_sided_95_upper_iid"] = 1 - 0.05 ** (1 / supported) if supported else None
    else:
        rng = random.Random(1124)
        rates = []
        for _ in range(20000):
            sample = rng.choices(groups, k=len(groups))
            n = sum(y for _, y in sample)
            if n:
                rates.append(sum(x for x, _ in sample) / n)
        rates.sort()
        d["whole_task_bootstrap_95"] = [rates[round(q*(len(rates)-1))] for q in (0.025, 0.975)]
        d["defined_bootstrap_draws"] = len(rates)
    return d


def main(out):
    assert set(os.sched_getaffinity(0)) == {32, 33} and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert digest(ARCHIVE) == ARCHIVE_SHA
    rules = json.loads((HERE / "tasks.json").read_bytes())
    rows, all_obs = [], {}
    with tarfile.open(ARCHIVE) as tar:
        for m in sorted(tar.getmembers(), key=lambda x: x.name):
            if m.name.endswith(".lineage.json"):
                row, obs = episode(tar, m, rules)
                key = (row["group"], row["case"])
                assert key not in all_obs
                all_obs[key] = obs
                rows.append(row)
    groups = Counter(r["group"] for r in rows)
    assert groups == dict(target_original=22, target_alternate=22, swapped_arm=8, source_original=3, source_alternate=6)
    primary = [r for r in rows if r["group"] == "target_original"]
    primary_fail = [r for r in primary if not r["native_success"]]
    assert len({r["donor_hash"] for r in primary}) == 22
    pair_details = matched(rows, all_obs)
    result = dict(group={g: summary([r for r in rows if r["group"] == g]) for g in groups},
                  primary_by_task={k: summary([r for r in primary if r["task"] == k]) for k in rules["tasks"]},
                  primary_by_arm={k: summary([r for r in primary if r["arm"] == k]) for k in sorted({r["arm"] for r in primary})},
                  primary_failures=summary(primary_fail), primary_successes=summary([r for r in primary if r["native_success"]]),
                  primary_with_nontarget_available=summary([r for r in primary if r["possible_nontarget"]]),
                  primary_concordant=summary([r for r in primary if r["replay_success"] == r["native_success"]]),
                  uncertainty=uncertainty(primary),
                  matched={k: dict(trajectory_pairs=len(v), common_windows=sum(p["common_windows"] for p in v),
                           both_loaded=sum(p["both_strictly_loaded"] for p in v), correct_wrong=sum(len(p["opposed"]) for p in v))
                           for k,v in pair_details.items()})
    result["decision"] = ("stop_no_observed_strict_wrong_pickup_on_original_native_failure"
                          if not result["primary_failures"]["episodes_wrong"] else "assess_independent_source_and_matched_contrasts")
    out.mkdir(exist_ok=True, parents=True)
    def save(name, data):
        (out / name).write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")
    save("result.json", result)
    save("matched.json", pair_details)
    with (out / "episodes.jsonl").open("w") as f:
        for r in rows:
            print(json.dumps(r, sort_keys=True, separators=(",", ":")), file=f)
    save("runtime.json", dict(cpu_affinity=sorted(os.sched_getaffinity(0)), gpu_hours=0, host=socket.gethostname(),
         pin=subprocess.check_output(["git", "-C", str(HERE), "rev-parse", "HEAD"], text=True).strip(),
         archive_sha256=ARCHIVE_SHA, tasks_sha256=digest(HERE/"tasks.json"), plan_sha256=digest(HERE/"PLAN.md"),
         scorer_sha256=digest(Path(__file__)), python=os.sys.version))
    print(result["decision"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    main(parser.parse_args().out)
