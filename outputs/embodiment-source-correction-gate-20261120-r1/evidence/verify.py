"""Independent Linux readback of per-action NPZ data, training identities and paired outcomes."""
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(condition, msg):
    if not condition:
        raise ValueError(msg)


def save_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def audit_case(path, expected_row, modes, split, controls=None, archived=None):
    rec = read(path)
    tape = path.with_suffix(".npz")
    seed = expected_row["layout_seed"]
    arm = path.parent.name
    check((rec["layout_seed"], rec["layout_id"], rec["arm"], rec["donor_sha256"], rec["scene_hash"]) ==
          (seed, expected_row["layout_id"], arm, expected_row["initial_state_sha256"], expected_row["scene_hash"]),
          f"unmatched donor {path}")
    check(rec["tape_sha256"] == sha(tape), f"changed tape {path}")
    results = rec["branches"]
    check(set(modes) == set(results), f"wrong branches {path}")
    row = dict(arm=arm, layout_seed=seed, layout_id=rec["layout_id"], donor_sha256=rec["donor_sha256"], tape_sha256=rec["tape_sha256"])
    fields = {}
    with np.load(tape, allow_pickle=False) as arrays:
        for mode in modes:
            a = {k: arrays[f"trace_{mode}_{k}"] for k in (
                "actions", "base", "native_steps", "rewards", "success", "terminated", "truncated")}
            a["observations"] = json.loads(arrays[f"trace_{mode}_observations"].item())
            reset = json.loads(arrays[f"trace_{mode}_reset"].item())
            phy = reset["environment_reproducibility"]
            a["opening"] = reset["admission"]["raw_state_sha256"]
            check(reset["model_identity"]["arm"] == arm and phy["layout_seed"] == seed and
                  phy["generated_row_fingerprint"] == row["donor_sha256"] and
                  "NVIDIA" in reset["renderer"][0].upper(), f"wrong reset or renderer {path} {mode}")
            if f"trace_{mode}_integration_sha256" in arrays:
                a["integration"] = arrays[f"trace_{mode}_integration_sha256"]
                a["snapshots"] = json.loads(arrays[f"trace_{mode}_physical_snapshots"].item())
                a["rendered"] = json.loads(arrays[f"trace_{mode}_live_observations"].item())
                check(len(a["snapshots"]) == len(a["base"])+1 and len(a["integration"]) == len(a["actions"]) and
                      len(a["rendered"]) == len(a["observations"]), f"missing physical samples {path} {mode}")
                for camera in ("rgb","wrist"):
                    differences = [5*j for j,(public,live) in enumerate(zip(a["observations"],a["rendered"],strict=True))
                                   if public[camera] != live[camera]]
                    check(differences == rec["camera_replay_differences"][mode][camera], f"wrong camera difference count {path} {mode}")
                    row[f"{mode}_{camera}_rerecorded_different"] = len(differences)
                check(all(all(public[k]==live[k] for k in ("prompt","state")) for public,live in
                          zip(a["observations"],a["rendered"],strict=True)), f"live proprioception differs {path} {mode}")
            fields[mode] = a
            acts, base, rewards = a["actions"], a["base"], a["rewards"]
            n = len(acts)
            terminal = a["terminated"] | a["truncated"]
            win = bool(a["success"][-1])
            check(0 < n <= 300 and acts.shape == (n, 7) and np.isfinite(acts).all() and
                  np.max(abs(acts)) <= 1 and base.shape == ((n+4)//5, 5, 7) and len(a["observations"]) == len(base),
                  f"invalid actions {path} {mode}")
            check(np.array_equal(a["native_steps"], np.arange(1, n+1)) and
                  all(a[k].shape == (n,) for k in ("success", "rewards", "terminated", "truncated")) and
                  np.all((rewards == 0) | (rewards == 1)) and np.array_equal(rewards == 1, a["success"]) and
                  not np.any(a["success"][:-1]) and not np.any(terminal[:-1]) and terminal[-1] and
                  not np.any(a["terminated"] & a["truncated"]) and (a["terminated"][-1] if win else n == 300),
                  f"invalid native indexing, terminal or reward {path} {mode}")
            check((n, int(rewards.sum()), win, n > 60) ==
                  (results[mode]["steps"], results[mode]["native_return"], results[mode]["success"],
                   results[mode]["decision_reached"]), f"receipt differs from arrays {path} {mode}")
            off_controls = n if mode == "off" else min(60, n)
            check(np.array_equal(acts[:off_controls], base.reshape(-1,7)[:off_controls]),
                  f"raw action mismatch {path} {mode}")
            if controls is not None:
                payload = np.empty(n, dtype=[("native_step","<i4"),("action","<f4",(7,)),("raw_proposal","<f4",(7,)),
                                             ("reward","<f4"),("success","?"),("terminated","?"),("truncated","?")])
                for key, value in (("native_step",a["native_steps"]),("action",acts),
                                   ("raw_proposal",base.reshape(-1,7)[:n]),("reward",rewards)):
                    payload[key] = value
                for k in ("success","terminated","truncated"):
                    payload[k] = a[k]
                controls[f"{arm}__{seed}__{mode}"] = payload
            if mode in ("always","gated"):
                delta = np.abs(acts.astype(np.float64)-base.reshape(-1,7)[:n].astype(np.float64))
                caps = np.array([0.5,0.5,0.5,0.25,0.25,0.25,2.0])
                check(np.all(delta <= caps + np.finfo(np.float32).eps), f"residual exceeded saved caps {path} {mode}")
                row[f"{mode}_controls_changed_from_proposal"] = int(np.any(delta>0,axis=1).sum())
            row[f"{mode}_success"] = int(win)
            row[f"{mode}_steps"] = n
            row[f"{mode}_return"] = int(rewards.sum())
            source_key = f"trace_{mode}_proposal_source"
            if source_key in arrays:
                sources = arrays[source_key].tolist()
                check(len(sources) == len(base), f"source history missing {path} {mode}")
                for j, source in enumerate(sources):
                    if source == "live":
                        continue
                    check(source in modes[:modes.index(mode)], f"unexpected source {source}")
                    prior = fields[source]
                    check(j < len(prior["observations"]) and prior["observations"][j] == a["observations"][j]
                          and np.array_equal(prior["base"][j], base[j]), f"invalid cached proposal {path} {mode}/{j}")
                row[f"{mode}_live_replans"] = sources.count("live")
                row[f"{mode}_reused_replans"] = len(sources)-sources.count("live")
        raw = fields["off"]
        prefix = min(60, len(raw["actions"]))
        plans = (prefix+4)//5 + int(len(raw["actions"]) > 60)
        for mode in modes[1:]:
            other = fields[mode]
            if "integration" in raw:
                check(raw["snapshots"][:(prefix+4)//5+1] == other["snapshots"][:(prefix+4)//5+1] and
                      np.array_equal(raw["integration"][:prefix],other["integration"][:prefix]),
                      f"physical prefix differs {path} {mode}")
            check(raw["opening"] == other["opening"] and raw["observations"][:plans] == other["observations"][:plans]
                  and np.array_equal(raw["base"][:plans], other["base"][:plans])
                  and all(np.array_equal(raw[k][:prefix], other[k][:prefix]) for k in (
                      "actions", "native_steps", "rewards", "success", "terminated", "truncated")),
                  f"opening or prefix differs {path} {mode}")
        row["prefix_controls"] = prefix
        row["fork_reached"] = int(len(raw["actions"]) > 60)
        if "gated" in modes:
            decision = results["gated"]
            threshold = 0.10
            switch = bool(decision["decision_reached"] and decision["gate_score"] > threshold)
            chosen = "always" if switch else "off"
            check(decision["enabled"] == switch and rec["gated_reproduces"] == chosen, f"wrong gate decision {path}")
            a, b = fields["gated"], fields[chosen]
            if "integration" in a:
                check(a["snapshots"] == b["snapshots"] and np.array_equal(a["integration"],b["integration"]),
                      f"selected continuation physics differs {path}")
                if row["fork_reached"]:
                    pre = read(path.parent.parent/"decisions"/arm/path.name)
                    check(rec["precommitted_gate_score"] == decision["gate_score"] == pre["score"] and
                          pre["enabled"] == switch and pre["public"] == raw["observations"][12] and
                          np.array_equal(pre["raw_proposal"], raw["base"][12]) and pre["elapsed_controls"] == 60,
                          f"invalid gate precommitment {path}")
            check(a["observations"] == b["observations"] and all(np.array_equal(a[k], b[k]) for k in (
                "actions", "native_steps", "rewards", "success", "truncated", "terminated", "base")),
                f"selected trajectory differs {path}")
            row.update(switch=int(switch), gate_score=decision["gate_score"], selected_controller=chosen)
            if split == "dev":
                arc = rec["archived_raw"]
                if archived is not None:
                    original, old_report = archived
                    old = old_report[arm,seed]
                    tape_path = original / old["tape"]
                    check(sha(tape_path) == old["tape_sha256"] == arc["tape_sha256"] and
                          old["layout_id"] == row["layout_id"] and old["scene_hash"] == rec["scene_hash"] and
                          old["movable_xy_hash"] == expected_row["movable_xy_hash"],
                          f"wrong archived trajectory {path}")
                    with np.load(tape_path, allow_pickle=False) as reference:
                        ok = dict(issued_actions_match=np.array_equal(reference["actions"],raw["actions"]),
                                  rewards_match=np.array_equal(reference["rewards"],raw["rewards"]),
                                  steps_match=len(reference["actions"]) == len(raw["actions"]),
                                  outcome_match=bool(reference["success"][-1]) == bool(row["off_success"]),
                                  opening_observation_match=json.loads(reference["observations"].item())[0] ==
                                      raw["observations"][0])
                    check(all(arc[k]==v for k,v in ok.items()) and old["success"] == arc["success"],
                          f"wrong archived comparison {path}")
                for k in ("success", "steps", "layout_match", "issued_actions_match", "rewards_match",
                          "outcome_match", "steps_match", "opening_observation_match"):
                    row[f"archived_{k}"] = arc.get(k)
        else:
            check(arrays["raw_success"].item() == row["off_success"] and
                  arrays["teacher_success"].item() == row["teacher_success"], f"bad training labels {path}")
            valid = arrays["valid_lengths"]
            suffix = fields["teacher"]["actions"][60:]
            targets = arrays["teacher_actions"]
            check(valid.sum() == len(suffix) and (not len(suffix) or np.array_equal(
                suffix, np.concatenate([v[:n] for n,v in zip(valid,targets,strict=True)]))),
                f"teacher labels differ from issued actions {path}")
    return row


def main(run, out):
    out.mkdir(exist_ok=True)
    archive = run.parent.parent / "libero-paligemma-dualarm-20261120-r2/large-panel-v3/dev/manifest.json"
    train = read(run/"layouts/manifest.json")["layouts"]
    dev = read(archive)["layouts"]
    check(len(train) == 64 and len({r["layout_seed"] for r in train}) == 64 and
          set(range(740000,740050)) == {r["layout_seed"] for r in dev} and len(dev) == 50,
          "invalid training or development starts")
    for k in ["layout_seed", "layout_id", "initial_state_sha256", "scene_hash", "movable_xy_hash", "mobile_joint_xy_hash"]:
        check(set(r[k] for r in train).isdisjoint(r[k] for r in dev), f"train/dev overlap {k}")
    pilot_size = len(read(run / "pilot/train/contract.json")["seeds"])
    check(pilot_size in (1,8), "unexpected number of pilot training layouts")
    groups = [("pilot/train", train[:pilot_size], ("off","teacher"), "train"),
              ("pilot/eval", train[pilot_size:2*pilot_size], ("off","always","gated"), "train"),
              ("train", train, ("off","teacher"), "train"),
              ("eval", dev, ("off","always","gated"), "dev")]
    old_dir = archive.parent / "scores/77d18c386b2cc176252cfb68449e4415748382f554046ca50367ed4592676d9f/stride-5"
    original_report = read(old_dir / "report.json")
    old_rows = {(arm,r["layout_seed"]):r for arm in ("Panda","Sawyer") for r in original_report["arms"][arm]["episodes"]}
    summary = {}
    controls = {}
    for sub, cohort, modes, phase in groups:
        cases = []
        root = run / sub
        contract = read(root / "contract.json")
        check(contract["seeds"] == [r["layout_seed"] for r in cohort] and contract["split"] == phase and
              contract["checkpoint_sha256"].startswith("77d18c386b2c") and contract["inference_batch"] == 8 and
              contract["fork"] == 60 and contract["horizon"] == 300 and contract["cadence"] == 5,
              f"wrong registered stage {sub}")
        if sub == "eval":
            check(contract["model_sha256"] == sha(run / "fitted/model.npz"), "development model differs")
        for arm in ("Panda","Sawyer"):
            check(len(list((root/arm).glob("*.json"))) == len(cohort), f"extra/missing cases {sub} {arm}")
            cases.extend(audit_case(root/arm/f"{r['layout_seed']}.json", r, modes, phase,
                                    controls if sub == "eval" else None, (old_dir,old_rows) if sub == "eval" else None)
                         for r in cohort)
        save_csv(out / (sub.replace("/","-")+"-cases.csv"), cases)
        by_arm = {}
        for arm in ("Panda","Sawyer"):
            data = [r for r in cases if r["arm"] == arm]
            info = dict(paired_starts=len(data), fork_reached=sum(r["fork_reached"] for r in data),
                        native_episodes=len(data)*len(modes),
                        issued_controls=sum(r[f"{m}_steps"] for r in data for m in modes),
                        successes={m:sum(r[f"{m}_success"] for r in data) for m in modes})
            if f"{modes[0]}_rgb_rerecorded_different" in data[0]:
                info["rerecorded_camera_hash_mismatches"] = {
                    m:{camera:sum(r[f"{m}_{camera}_rerecorded_different"] for r in data)
                       for camera in ("rgb","wrist")} for m in modes}
                info["cases_with_any_rerecorded_camera_difference"] = sum(any(
                    r[f"{m}_{camera}_rerecorded_different"] for camera in ("rgb","wrist") for m in modes) for r in data)
            info["both_outcomes"] = dict(Counter("".join(str(r[f"{m}_success"]) for m in modes) for r in data))
            if "gated" in modes:
                info["switch_count"] = sum(r["switch"] for r in data)
                info["switch_rate_all"] = info["switch_count"]/len(data)
                info["switch_rate_actionable"] = (info["switch_count"]/info["fork_reached"] if info["fork_reached"] else None)
                info["hindsight_successes"] = sum(max(r["off_success"],r["always_success"]) for r in data)
                for name,raw,always in (("available_rescues",0,1),("avoidable_losses",1,0),
                                          ("both_succeeded",1,1),("both_failed",0,0)):
                    subset = [r for r in data if (r["off_success"],r["always_success"]) == (raw,always)]
                    info[name] = {"count":len(subset), "switched":sum(r["switch"] for r in subset)}
                if phase == "dev":
                    info["archive_matches"] = {k:sum(r[f"archived_{k}"] is True for r in data) for k in
                        ("layout_match","issued_actions_match","rewards_match","outcome_match","steps_match","opening_observation_match")}
            by_arm[arm] = info
        summary[sub] = by_arm
    check(len(controls) == 300, "missing evaluation control records")
    np.savez_compressed(out/"eval-controls.npz", **controls)
    (out/"readback.json").write_text(json.dumps(summary, indent=2)+"\n")


if __name__ == "__main__":
    main(*map(Path, sys.argv[1:]))
