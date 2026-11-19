"""Run the three paired first-POC queries using the joint reader's predictions.json.

Supply --predictions, the completed --pilot live-eval-summary.json, and a fresh
--output JSON file. A partial output retains completed cases. Use the pilot's
native OpenPI and simulator environment. No support replay or fitting is needed.
"""

import argparse
import hashlib
import json
from pathlib import Path

from score_poc import PILOT_ARMS, VISUAL, check_reset, digest, episode, finite, require


HERE = Path(__file__).resolve().parent
ARCHIVE = HERE / "evidence/gpu02col/archive/archive.json"
ARCHIVE_SHA = "832029f23f298c4edb0a5caf6c0da634660230acc6cb380d3a44689910a97767"
ARM = "analytic_public_joint_sysid"
CASES = {28: "fam_nom", 29: "comp_180_lo_inv", 30: "cont_45_nom"}
QUERY = dict(suite="libero_goal", task_id=1, variant="BASE", init_state_id=13, seed=20410113)
FACTORS = ("yaw_deg", "xy_scale", "gripper_sign")
DIAGNOSTICS = ("wrist_yaw_deg", "wrist_referenced_yaw_deg", "joint_raw_yaw_deg", "predicted_world_delta_m")


def unique_object(pairs):
    result = dict(pairs)
    require(len(result) == len(pairs), "Duplicate JSON field")
    return result


def read_source(path):
    raw = path.read_bytes()
    return json.loads(raw, object_pairs_hook=unique_object), hashlib.sha256(raw).hexdigest()


def index_cases(rows):
    by_name = {row["embodiment"]: row for row in rows}
    require(len(rows) == 3 and by_name.keys() == set(CASES.values()), "Unexpected or duplicate first-POC keys")
    return by_name


def read_estimates(doc, archive):
    require(set(doc) == {"arm", "archive_sha256", "rows"} and doc["arm"] == ARM
            and doc["archive_sha256"] == ARCHIVE_SHA, "Expected joint worker predictions.json")
    rows = index_cases(doc["rows"])
    cases = {}
    for index, name in CASES.items():
        support = archive["trials"][index]
        identity = dict(suite=QUERY["suite"], task_id=1, variant="BASE", embodiment=name,
                        support_init_state_id=11, support_seed=20210111, support_archive_index=index,
                        support_controls=15, query_init_state_id=13)
        context = f"support {index}/{name}"
        require((support["index"], support["split"], support["suite"], support["task_id"], support["embodiment"],
                 support["init_state_id"], support["seed"], support["active_controls"]) ==
                (index, "dev", QUERY["suite"], 1, name, 11, 20210111, 15), f"{context}: archive mismatch")
        row = rows[name]
        require(set(row) == identity.keys() | set(FACTORS + DIAGNOSTICS), f"{context}: unexpected prediction schema")
        require(all(type(row[k]) is type(v) and row[k] == v for k, v in identity.items()), f"{context}: identity mismatch")
        require(all(finite(row[k]) for k in FACTORS + DIAGNOSTICS[:-1]) and -180 <= row["yaw_deg"] < 180
                and row["xy_scale"] > 0 and type(row["gripper_sign"]) is int and row["gripper_sign"] in (-1, 1),
                f"{context}: invalid joint reader estimate")
        xyz = row["predicted_world_delta_m"]
        require(isinstance(xyz, list) and len(xyz) == 3
                and all(isinstance(v, list) and len(v) == 3 and all(map(finite, v)) for v in xyz),
                f"{context}: expected three XYZ estimates")
        cases[name] = dict(**identity, query_seed=QUERY["seed"], support_physics=support["physics"],
                           inferred={k: row[k] for k in FACTORS}, diagnostics={k: row[k] for k in DIAGNOSTICS})
    return cases


def pilot_reference(doc, cases):
    require(doc["panel"] == "poc" and doc["archive_sha256"] == ARCHIVE_SHA and doc["trials"] == 3
            and doc["seed_ckpt"] == 41 and doc["matched_support_cap"] == 15, "Wrong pilot panel or source")
    arms = set(PILOT_ARMS)
    if doc["visual_baseline_available"] is False:
        arms.remove(VISUAL)
    require(set(doc["arms"]) == arms and len(doc["arms"]) == len(arms), "Incomplete pilot arm inventory")
    rows = index_cases(doc["rows"])
    reference = rows["fam_nom"]["arms"]["nominal"]
    reset, opening = reference["reset"], reference["opening_proposal_sha256"]
    require(digest(opening), "Missing pilot opening proposal")
    for name, row in rows.items():
        require(row.get("variant", "BASE") == "BASE"
                and all(type(row[k]) is type(v) and row[k] == v for k, v in QUERY.items() if k != "variant")
                and set(row["arms"]) == arms and reset["physics"] != cases[name]["support_physics"],
                f"{name}: wrong pilot query identity, seed, or arms")
        results = {}
        for arm, rec in row["arms"].items():
            context = f"{name}/{arm}"
            check_reset(rec["reset"], row, rec, context)
            require(rec["reset"] == reset, f"{context}: pilot query resets differ")
            if arm != "ours_online_prefix":
                require(rec["opening_proposal_sha256"] == opening, f"{context}: pilot opening proposals differ")
            results[arm] = dict(**episode(rec, context, legacy=False), inferred={k: rec["inferred"][k] for k in FACTORS})
        cases[name]["pilot_arms"] = results
    return reset, opening


class PairedOpening:
    """Check the first unmodified policy response before the motor issues a control."""

    def __init__(self, policy, expected):
        self.policy = policy
        self.expected = expected

    def infer(self, item, *, noise):
        result = self.policy.infer(item, noise=noise)
        if self.expected is not None:
            import numpy as np
            from motor_iface import array_digest

            actual = array_digest(np.asarray(result["actions"], dtype=np.float32))
            require(actual == self.expected, f"Opening frozen proposal {actual} differs from pilot {self.expected}")
            self.expected = None
        return result


def run(args):
    predictions, predictions_sha = read_source(args.predictions)
    pilot, pilot_sha = read_source(args.pilot)
    archive, archive_sha = read_source(ARCHIVE)
    require(archive_sha == ARCHIVE_SHA, "Wrong pinned first-POC support archive")
    cases = read_estimates(predictions, archive)
    reset, opening = pilot_reference(pilot, cases)

    from fluffy_bassoon.xemb.embodiment import get_embodiment_spec
    from motor_iface import load_frozen_pi05, run_episode, write_json
    from sim_worker import XEmbSimulator

    def match_reset(observed):
        require(observed == reset, "Fresh query reset differs from pilot on " +
                str([k for k in observed.keys() | reset.keys() if observed.get(k) != reset.get(k)]))

    sources = dict(predictions_sha256=predictions_sha, pilot_sha256=pilot_sha, archive_sha256=archive_sha)
    for file in ("joint_queries.py", "motor_iface.py", "sim_worker.py", "score_poc.py"):
        sources[file] = hashlib.sha256((HERE / file).read_bytes()).hexdigest()
    summary = dict(arm=ARM, complete=False, sources=sources, query_reset=reset,
                   opening_proposal_sha256=opening, rows=list(cases.values()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.touch(exist_ok=False)
    write_json(args.output, summary)
    policy = load_frozen_pi05()
    with XEmbSimulator() as simulator:
        for name, case in cases.items():
            est = case["inferred"]
            rec = run_episode(simulator, PairedOpening(policy, opening), QUERY, get_embodiment_spec(name),
                              prompt=reset["prompt"], check_reset=match_reset, yaw_hat=est["yaw_deg"],
                              scale_hat=est["xy_scale"], grip_hat=est["gripper_sign"])
            require(rec["opening_proposal_sha256"] == opening and rec["inferred"] == est,
                    f"{name}: fresh opening proposal or applied estimate differs")
            case["joint_query"] = dict(**episode(rec, name, legacy=False), native_rewards=rec["native_rewards"],
                                       opening_proposal_sha256=rec["opening_proposal_sha256"])
            write_json(args.output, summary)
    summary["complete"] = True
    write_json(args.output, summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("--predictions", "--pilot", "--output"):
        parser.add_argument(flag, type=Path, required=True)
    run(parser.parse_args())
