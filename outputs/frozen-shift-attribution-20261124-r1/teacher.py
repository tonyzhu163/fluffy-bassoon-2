"""Teacher-forced attribution: replay a donor tape on Kinova3 and query the frozen policy under input swaps.

teacher.py run <out> [--pairs 34:35,35:35,37:35,14:34] [--views panda|none] [--adapters a,b]
teacher.py score <out>
Rewards of the replay are audit data. No condition reads them.
"""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
import tarfile
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE.parent
sys.path.insert(0, str(OUTPUTS / "frozen-vla-proposal-shift-20261124-r1"))
import harness
import ur5e_motor
from ur5e_motor import NOISE_SEED_OFFSET, array_digest, write_json

DONOR = OUTPUTS / "frozen-transfer-donor-20261124-r1/inputs"
# Kinova3 layout, IIWA tape, Kinova3 seed of the replay that earned reward on Baige CPU.
PAIRS = {(34, 35): 42710334, (35, 35): 42710335, (37, 35): 42710337, (14, 34): 41810314}
FRESH_STREAM = 91
PERMUTATION = (3, 6, 0, 7, 1, 2, 4, 5)  # fixed derangement of the eight state coordinates
OPEN, PUT = "open the top drawer", "put the bowl in the top drawer"
CAMERAS = ("agentview_image", "robot0_eye_in_hand_image")
GROUPS = dict(all=slice(0, 7), xyz=slice(0, 3), rotation=slice(3, 6), gripper=slice(6, 7))
FRESH = ("noise_a", "noise_b")


def condition(name, *, view=(), state=None, prompt=None, noise=None, first_only=False, adapter=None):
    return dict(name=name, view=view, state=state, prompt=prompt, noise=noise, first_only=first_only, adapter=adapter)


def conditions(views=True, adapters=()):
    rows = [condition("raw"), condition("raw_repeat", first_only=True),
            *(condition(name, noise=name) for name in FRESH)]
    if views:
        rows += [condition("panda_external", view=CAMERAS[:1]), condition("panda_both", view=CAMERAS)]
    rows += [condition("state_mapped", state="mapped"), condition("state_zero", state="zero"),
             condition("state_shuffled", state="shuffled"),
             condition("prompt_open", prompt=OPEN), condition("prompt_put", prompt=PUT)]
    return rows + [condition("adapter:" + adapter.name, adapter=adapter) for adapter in adapters]


def load_tape(row, root=DONOR):
    """The IIWA raw tape of `row`: issued commands, the policy's own first five, and its noise bank."""
    manifest = json.loads((root / "manifest.json").read_bytes())
    archive = root / "raw-tapes.tar.gz"
    if hashlib.sha256(archive.read_bytes()).hexdigest() != manifest["raw_archive_sha256"]:
        raise ValueError("Donor archive differs from its manifest")
    name = f"gpu01/episodes/IIWA-{row}-raw.json"
    with tarfile.open(archive) as members:
        blob = members.extractfile(name).read()
    if hashlib.sha256(blob).hexdigest() != manifest["members"][name]:
        raise ValueError(f"{name} differs from its manifest")
    tape = json.loads(blob)
    seed = tape["query"]["seed"]
    if tape["mode"] != "raw" or not tape["native"]["success"] or seed != 42710300 + row:
        raise ValueError("Expected a successful raw IIWA tape")
    bank = np.random.default_rng(seed + NOISE_SEED_OFFSET).standard_normal((60, *tape["shape"])).astype(np.float32)
    calls = tape["calls"]
    for index, call in enumerate(calls):
        if call["t"] != 5 * index or call["noise"] != array_digest(bank[index]):
            raise ValueError("Donor noise or boundary does not match the tape")
    return dict(row=row, seed=seed, prompt=tape["native"]["released_prompt"], noise=bank[:len(calls)],
                sent=[np.asarray(call["sent"], dtype=np.float32) for call in calls],
                first_five=[np.asarray(call["first_five"], dtype=np.float32) for call in calls],
                commands_sha256=array_digest(np.concatenate([np.asarray(c["sent"], dtype=np.float32) for c in calls])))


def target_view(raw):
    """Public Kinova3 observation without the audit-side Panda renders."""
    return {key: value for key, value in harness.public(raw).items() if not key.startswith("panda_")}


def build(raw, prompt, rule, pack):
    """Policy input for one condition from the public observation (plus Panda renders of the same state)."""
    seen = target_view(raw)
    for key in rule["view"]:
        seen[key] = np.array(raw["panda_" + key])
    if rule["adapter"] is not None:
        prompt = rule["adapter"].prompt(harness.public(seen), prompt)
        seen = rule["adapter"].observation(harness.public(seen))
    entry = dict(pack(seen, rule["prompt"] or prompt, canonicalize_quat=rule["state"] == "mapped"))
    state = np.array(entry["observation/state"])
    if rule["state"] == "zero":
        entry["observation/state"] = np.zeros_like(state)
    elif rule["state"] == "shuffled":
        entry["observation/state"] = state[list(PERMUTATION)]
    return entry


def teacher(sim, policy, tape, case, rules, pack, support=None):
    """Replay `tape` on `case` and query every condition before each five-control window."""
    spec, first = sim.call("create", dict(case, embodiment="nom"))
    if first["language_instruction"] != tape["prompt"] or first["model_identity"]["arm"] != case["arm"]:
        raise ValueError("Unexpected native prompt or robot")
    low, high = (np.asarray(first["action_schema"][key], dtype=np.float32) for key in ("low", "high"))
    raw, prompt = first["observation"], first["language_instruction"]
    for rule in rules:
        if rule["adapter"] is not None:
            rule["adapter"].reset(
                {key: value for key, value in (support or {}).items() if key != "physics"},
                dict(observation=harness.public(raw), language_instruction=prompt,
                     action_schema=first["action_schema"], arm=case["arm"],
                     joint_count=int(first["model_identity"]["joint_count"]), max_steps=int(spec["max_steps"])))
    fresh = np.random.default_rng([int(case["seed"]) + NOISE_SEED_OFFSET, FRESH_STREAM])
    boundaries, rewards, control, step = [], [], 0, None
    for index, sent in enumerate(tape["sent"]):
        base = tape["noise"][index]
        draws = {name: fresh.standard_normal(base.shape).astype(np.float32) for name in FRESH}
        proposals, missing = {}, []
        for rule in rules:
            if rule["first_only"] and index:
                continue
            if any("panda_" + key not in raw for key in rule["view"]):
                missing.append(rule["name"])  # no admissible Panda pose or render check at this boundary
                continue
            entry = build(raw, prompt, rule, pack)
            noise = draws.get(rule["noise"], base)
            actions = np.array(policy.infer(entry, noise=noise)["actions"], dtype=np.float32)
            if actions.ndim != 2 or actions.shape[1] != 7 or len(actions) < 5 or not np.isfinite(actions).all():
                raise ValueError("Invalid frozen motor proposal")
            proposals[rule["name"]] = dict(
                prompt=entry["prompt"], inputs=harness.input_digests(entry), noise_sha256=array_digest(noise),
                proposal_sha256=array_digest(actions), first_five=np.clip(actions[:5], low, high).tolist())
        before = raw
        step = sim.call("chunk", sent)
        if int(step["steps"]) != len(sent):
            raise ValueError("Simulator did not consume the donor window")
        rewards.extend(float(value) for value in step["rewards"])
        raw = step["observation"]
        for rule in rules:
            if rule["adapter"] is not None:
                rule["adapter"].observe(target_view(before), sent.copy(), target_view(raw))
        boundaries.append(dict(
            boundary=index, control=control, donor_sent=sent.tolist(), donor_first_five=tape["first_five"][index].tolist(),
            donor_noise_sha256=array_digest(base), eef_pos=np.asarray(before["robot0_eef_pos"]).tolist(),
            proposals=proposals, missing=missing, rewards=[float(value) for value in step["rewards"]]))
        control += len(sent)
        if step["terminated"] or step["truncated"]:
            break
    reproducibility = first["environment_reproducibility"]
    rewarded = [index for index, value in enumerate(rewards) if value > 0]
    return dict(
        case=case, layout=case["init_state_id"], donor=dict(arm="IIWA", layout=tape["row"], seed=tape["seed"],
                                                           commands_sha256=tape["commands_sha256"]),
        physics=reproducibility["initial_state_fingerprint"], renderer=first.get("renderer"),
        conditions=[rule["name"] for rule in rules], controls=control, tape_controls=sum(map(len, tape["sent"])),
        native_rewards=rewards, replay_success=bool(step["success"]),
        first_reward_control=rewarded[0] + 1 if rewarded else None, boundaries=boundaries)


def rms(a, b, columns=GROUPS["all"]):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    count = min(len(a), len(b), 5)
    return float(np.sqrt(np.mean((a[:count, columns] - b[:count, columns]) ** 2)))


def layout_interval(by_layout):
    """Mean of layout means and the 95% range over resamples of whole layouts (all ordered resamples when few)."""
    means = np.array([np.mean(values) for values in by_layout.values()], dtype=np.float64)
    count = len(means)
    if count ** count <= 50000:
        draws = np.array(list(itertools.product(range(count), repeat=count)))
    else:
        draws = np.random.default_rng(42710300).integers(count, size=(10000, count))
    low, high = np.quantile(means[draws].mean(axis=1), (.025, .975))
    return dict(mean=float(means.mean()), layout_interval95=[float(low), float(high)], layouts=count,
                boundaries=int(sum(len(values) for values in by_layout.values())),
                by_layout={str(key): float(np.mean(values)) for key, values in by_layout.items()})


def distances(record):
    """Per boundary and condition: RMS to the donor's next controls, to the donor arm's proposal, and to raw."""
    rows = []
    for event in record["boundaries"]:
        raw = event["proposals"]["raw"]["first_five"]
        row = dict(boundary=event["boundary"], control=event["control"], conditions={})
        for name, proposal in event["proposals"].items():
            five = proposal["first_five"]
            row["conditions"][name] = dict(
                donor={group: rms(five, event["donor_sent"], columns) for group, columns in GROUPS.items()},
                source={group: rms(five, event["donor_first_five"], columns) for group, columns in GROUPS.items()},
                raw={group: rms(five, raw, columns) for group, columns in GROUPS.items()})
        rows.append(row)
    return rows


def score(records):
    """Paired contrasts per boundary, averaged within layout, with intervals over layouts."""
    report = {}
    scopes = dict(all=records, replay_rewarded=[record for record in records if record["replay_success"]])
    for scope, members in scopes.items():
        tables = {record["layout"]: distances(record) for record in members}
        names = list(dict.fromkeys(name for rows in tables.values() for row in rows for name in row["conditions"]))
        section = report[scope] = dict(layouts=sorted(tables), conditions={})
        if not tables:
            continue
        for name in names:
            entry = section["conditions"][name] = {}
            for target in ("donor", "source", "raw"):
                for group in GROUPS:
                    absolute, minus_raw, minus_fresh = {}, {}, {}
                    for layout, rows in tables.items():
                        for row in rows:
                            cells = row["conditions"]
                            if name not in cells:
                                continue
                            value = cells[name][target][group]
                            absolute.setdefault(layout, []).append(value)
                            minus_raw.setdefault(layout, []).append(value - cells["raw"][target][group])
                            if all(key in cells for key in FRESH):
                                control = np.mean([cells[key][target][group] for key in FRESH])
                                minus_fresh.setdefault(layout, []).append(value - control)
                    cell = dict(distance=layout_interval(absolute))
                    if target != "raw":
                        cell["minus_raw"] = layout_interval(minus_raw)
                    if minus_fresh:
                        cell["minus_fresh_noise"] = layout_interval(minus_fresh)
                    entry[f"{target}/{group}"] = cell
    report["replays"] = [dict(layout=r["layout"], donor=r["donor"]["layout"], controls=r["controls"],
                              missing_views=sum(bool(event["missing"]) for event in r["boundaries"]),
                              replay_success=r["replay_success"], first_reward_control=r["first_reward_control"])
                         for r in records]
    return report


def parse_pairs(text):
    pairs = [tuple(map(int, item.split(":"))) for item in text.split(",")]
    unknown = [pair for pair in pairs if pair not in PAIRS]
    if unknown:
        raise ValueError(f"Unregistered target:donor pairs {unknown}")
    return pairs


def plan(pairs, rules):
    boundaries = {f"{layout}<-{row}": len(load_tape(row)["sent"]) for layout, row in pairs}
    each = sum(not rule["first_only"] for rule in rules)
    extra = sum(rule["first_only"] for rule in rules)
    return dict(pairs=boundaries, conditions=[rule["name"] for rule in rules],
                policy_calls=sum(boundaries.values()) * each + len(pairs) * extra)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("run", "score", "list"))
    p.add_argument("out", type=Path)
    p.add_argument("--pairs", default=",".join(f"{a}:{b}" for a, b in PAIRS))
    p.add_argument("--views", choices=("panda", "none"), default="panda")
    p.add_argument("--adapters", default="", help="comma-separated lane adapters added as conditions")
    args = p.parse_args(argv)
    if args.mode == "score":
        records = [json.loads(path.read_bytes()) for path in sorted(args.out.glob("teacher-*.json"))]
        write_json(args.out / "summary.json", score(records))
        return
    pairs = parse_pairs(args.pairs)
    names = [name for name in args.adapters.split(",") if name]
    adapters = []
    if names:
        import panel
        factories, _ = panel.discover(names=names)
        adapters = [factories[name]() for name in names]
    rules = conditions(views=args.views == "panda", adapters=adapters)
    print(json.dumps(plan(pairs, rules), indent=2))
    if args.mode == "list":
        return
    from ur5e_motor import load_frozen_pi05
    from ur5e_sim_worker import UR5eSimulator

    policy = load_frozen_pi05()
    with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}):
        if args.views == "panda":
            import views
            sim = views.Simulator(args.out / "views.jsonl", [layout for layout, _ in pairs])
        else:
            sim = UR5eSimulator()
    with sim:
        support = harness.collect_support(sim, dict(arm="Kinova3", suite="libero_goal", task_id=3, variant="BASE",
                                                    init_state_id=11, seed=42710311)) if adapters else None
        records = []
        for layout, row in pairs:
            case = dict(arm="Kinova3", suite="libero_goal", task_id=3, variant="BASE",
                        init_state_id=layout, seed=PAIRS[layout, row])
            record = teacher(sim, policy, load_tape(row), case, rules, ur5e_motor.to_openpi_observation, support)
            write_json(args.out / f"teacher-{layout}-from-{row}.json", record)
            records.append(record)
            write_json(args.out / "summary.json", score(records))
            print(layout, row, record["controls"], record["replay_success"], flush=True)


if __name__ == "__main__":
    main()
