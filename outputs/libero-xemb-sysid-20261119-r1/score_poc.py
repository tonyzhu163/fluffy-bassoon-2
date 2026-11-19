#!/usr/bin/env python3
"""Read final Baige Gate 1 and/or first-pilot JSON. Write the independent score to stdout."""

import argparse
from collections import Counter
from hashlib import sha256
import json
import math
from pathlib import Path
import re


NOM, ORACLE = "nominal", "privileged_oracle"
ACTIVE, PASSIVE = "ours_retained_active", "ours_retained_passive"
CLEAR, SWAP, ONLINE = "ours_cleared", "ours_swapped_donor", "ours_online_prefix"
PROPRIO, VISUAL = "analytic_proprio_sysid", "analytic_visual_sysid"
JOINT, NO_EQUIV = "joint_transformer_active", "factored_no_equiv_active"
PILOT_ARMS = (NOM, ORACLE, ACTIVE, PASSIVE, CLEAR, SWAP, PROPRIO, VISUAL, JOINT, NO_EQUIV, ONLINE)
TASKS = {"libero_spatial": (1, 7), "libero_object": (0, 7), "libero_goal": (0, 1), "libero_10": (0, 3)}
HORIZONS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520}
BASE, COMP, OBLIQUE = "fam_nom", "comp_180_lo_inv", "cont_45_nom"
LATENTS = {BASE: (0, 1, 1), "fam_yaw90": (90, 1, 1), COMP: (180, 0.65, -1),
           "comp_m90_hi_inv": (-90, 1.55, -1), OBLIQUE: (45, 1, 1), "cont_135_075_inv": (135, 0.75, -1)}
PILOT = {("libero_goal", 1, e) for e in (BASE, COMP, OBLIQUE)}
GATE = {(suite, t, e) for suite, tasks in TASKS.items() for t in tasks for e in LATENTS}
REGIMES = {"fam": "familiar", "comp": "compositional_ood", "cont": "continuous_angle_ood"}
RESET_FIELDS = ("physics", "perturbation_plan", "public", "motor_frame0", "reader_frame0", "prompt",
                "observation_schema", "observation_schema_version", "action_schema", "renderer", "renderer_vendor")
FACTORS = ("yaw_deg", "xy_scale", "gripper_sign")
FRAME_FIELDS = {
    "public": ("agentview_image", "robot0_eye_in_hand_image", "robot0_joint_pos", "robot0_joint_vel",
               "robot0_eef_pos", "robot0_eef_quat_xyzw", "robot0_gripper_qpos"),
    "motor_frame0": ("observation/image", "observation/wrist_image", "observation/state"),
    "reader_frame0": ("image", "state"),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    raw = path.read_bytes()
    return json.loads(raw), {"path": str(path), "sha256": sha256(raw).hexdigest()}


def digest(value):
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None


def finite(value):
    return type(value) in (float, int) and math.isfinite(value)


def episode(rec, context, legacy):
    reset, n, total = rec["reset"], rec["controls"], rec["native_return"]
    require(type(n) is int and 0 < n <= reset["horizon"], f"{context}: invalid controls {n}")
    for key in ("success", "terminated", "truncated"):
        require(type(rec[key]) is bool, f"{context}: invalid {key}")
    require(rec["terminated"] or rec["truncated"], f"{context}: incomplete native endpoint")
    require(finite(total), f"{context}: invalid archived return")
    if "native_rewards" in rec:
        rewards = rec["native_rewards"]
        require(isinstance(rewards, list) and len(rewards) == n and all(map(finite, rewards)),
                f"{context}: expected {n} finite native rewards")
        raw_sum = sum(rewards, 0.0)
        require(raw_sum == total, f"{context}: per-control sum {raw_sum} differs from archive {total}")
        total = raw_sum
        source = "per_control"
    else:
        require(legacy, f"{context}: missing native_rewards")
        source = "archived_sum_only"
    return dict(success=rec["success"], native_return=total, return_source=source, controls=n,
                terminated=rec["terminated"], truncated=rec["truncated"])


def check_reset(reset, row, rec, context):
    require(all(reset[k] for k in RESET_FIELDS) and reset["worker_step_index"] == 0
            and reset["horizon"] == HORIZONS[row["suite"]] and reset["prompt"] == rec["released_prompt"],
            f"{context}: incomplete or inconsistent reset")
    for view, fields in FRAME_FIELDS.items():
        require(all(digest(reset[view][key]) for key in fields), f"{context}: incomplete {view} frame zero")
    require(digest(reset["noise_sha256"]) and reset["noise_seed"] == row["seed"] + 817117,
            f"{context}: invalid frozen noise identity")


def check_budget(arm, rec, context):
    budget, n = rec["control_budget"], rec["controls"]
    acquired = budget["support_acquired"]
    prefix = min(15, n) if arm == ONLINE else 0
    external = arm not in (NOM, ORACLE, ONLINE)
    require(type(acquired) is int and (0 < acquired <= 15 if external else acquired == 0),
            f"{context}: invalid acquired controls {acquired}")
    expected = dict(support_observed=0 if arm == CLEAR else acquired, query_prefix=prefix, query_motor=n - prefix,
                    query_limit=rec["reset"]["horizon"], total_executed=acquired + n)
    require(all(type(budget[k]) is int and budget[k] == v for k, v in expected.items()),
            f"{context}: control accounting differs from {expected}")
    return budget


def group_score(cases, readers):
    arms = sorted(cases[0]["arms"])
    n = len(cases)
    totals, paired = {}, {}
    for arm in arms:
        records = [case["arms"][arm] for case in cases]
        successes = sum(r["success"] for r in records)
        total = sum(r["native_return"] for r in records)
        totals[arm] = dict(n=n, successes=successes, success_rate=successes / n, return_sum=total,
                           return_mean=total / n, controls=sum(r["controls"] for r in records),
                           return_sources=dict(Counter(r["return_source"] for r in records)))
    for reader in readers:
        if reader not in arms:
            continue
        paired[reader] = {}
        for arm in arms:
            if arm == reader:
                continue
            pairs = [(c["arms"][reader], c["arms"][arm]) for c in cases]
            diffs = [a["success"] - b["success"] for a, b in pairs]
            paired[reader][arm] = dict(n=n, wins=diffs.count(1), losses=diffs.count(-1), ties=diffs.count(0),
                                       success_gain_pp=100 * sum(diffs) / n,
                                       mean_return_gain=sum(a["native_return"] - b["native_return"] for a, b in pairs) / n)
    return dict(cases=n, arms=totals, paired=paired)


def pilot_outcomes(cases, arms):
    by_emb = {c["embodiment"]: c for c in cases}
    require({c["donor_embodiment"] for c in cases} == by_emb.keys(), "donors are not a permutation")
    rescues = {}
    core = {NOM, ORACLE, CLEAR, SWAP}
    readers = [a for a in (ACTIVE, PASSIVE) if a in arms]
    for case in cases:
        emb, records = case["embodiment"], case["arms"]
        if ACTIVE in arms and SWAP in arms:
            donor = by_emb[case["donor_embodiment"]]["arms"][ACTIVE]
            require(records[SWAP]["inferred"] == donor["inferred"], f"{emb}: swapped estimate differs from donor")
        case["active_passive_both_15"] = (all(records[a]["control_budget"]["support_observed"] == 15 for a in readers)
                                          if len(readers) == 2 else None)
        if emb == BASE or not core <= records.keys():
            continue
        if records[ORACLE]["success"] and not any(records[a]["success"] for a in (NOM, CLEAR, SWAP)):
            winners = [a for a in readers if records[a]["success"]]
            if winners:
                rescues[emb] = winners
    complete = (core | {ACTIVE, PASSIVE}) <= set(arms)
    decision = bool(rescues) if rescues or complete else None
    return dict(first_poc_outcome_rule=decision, first_poc_witnesses=rescues,
                missing_registered_arms=sorted(set(PILOT_ARMS) - set(arms)))


def score(doc, pilot):
    expected = PILOT if pilot else GATE
    panel = "poc" if pilot else "development"
    require(isinstance(doc, dict) and doc.get("panel") == panel, f"expected final {panel} summary object")
    rows = doc["rows"]
    keys = [(r["suite"], r["task_id"], r["embodiment"]) for r in rows]
    require(len(keys) == len(expected) and set(keys) == expected, f"incomplete, duplicate or unexpected {panel} cases")
    arms = doc["arms"] if pilot else [NOM, ORACLE]
    require(len(arms) == len(set(arms)) and {NOM, ORACLE} <= set(arms), "invalid summary arm names")
    if pilot:
        require(doc["seed_ckpt"] == 41 and doc["matched_support_cap"] == 15 and doc["trials"] == len(keys),
                "pilot seed, support cap or trial count differs from registration")
        require(doc["visual_baseline_available"] is (VISUAL in arms), "visual availability differs from arms")
        require(digest(doc["archive_sha256"]), "missing pilot support archive identity")
    # 95388e6e omits both arrays and opening proposals on every Gate 1 episode.
    legacy = not pilot and all("native_rewards" not in r[a] and "opening_proposal_sha256" not in r[a]
                               for r in rows for a in arms)
    resets, openings, cases = {}, {}, []
    for key, row in zip(keys, rows):
        suite, task, emb = key
        task_key = (suite, task)
        require(row["init_state_id"] == 13, f"{key}: wrong query reset")
        latent = row["true_latent"] if pilot else row
        regime = REGIMES[emb.split("_", 1)[0]]
        require(row["regime"] == regime, f"{key}: incorrect regime")
        require(tuple(latent[k] for k in FACTORS) == LATENTS[emb], f"{key}: wrong factor labels")
        recs = row["arms"] if pilot else {a: row[a] for a in arms}
        require(set(recs) == set(arms), f"{key}: arm inventory differs from summary")
        case = dict(suite=suite, task_id=task, embodiment=emb, true_latent=dict(zip(FACTORS, LATENTS[emb])),
                    regime=regime, seed=row["seed"], arms={})
        for arm, rec in recs.items():
            context = f"{key}/{arm}"
            reset = rec["reset"]
            check_reset(reset, row, rec, context)
            ref = resets.setdefault(task_key, reset)
            if reset != ref:
                fields = sorted(k for k in ref.keys() | reset.keys() if ref.get(k) != reset.get(k))
                raise ValueError(f"{context}: mismatched reset fields {fields}")
            if not legacy and arm != ONLINE:
                opening = rec["opening_proposal_sha256"]
                require(digest(opening) and openings.setdefault(task_key, opening) == opening,
                        f"{context}: opening frozen proposal missing or unequal")
            result = episode(rec, context, legacy)
            inferred = tuple(rec["inferred"][k] for k in FACTORS)
            require(all(map(finite, inferred)) and inferred[1] > 0 and inferred[2] in (1, -1),
                    f"{context}: invalid inferred interface")
            if arm in (NOM, ORACLE):
                require(inferred == LATENTS[BASE if arm == NOM else emb], f"{context}: incorrect control interface")
            if pilot:
                result["control_budget"] = check_budget(arm, rec, context)
                result["inferred"] = rec["inferred"]
            case["arms"][arm] = result
        if pilot:
            donor = row["donor_embodiment"]
            require((suite, task, donor) in expected and donor != emb, f"{key}: wrong donor identity")
            case["donor_embodiment"] = donor
        cases.append(case)

    readers = (ACTIVE, PASSIVE) if pilot else (ORACLE,)
    groups = {"overall": cases, "shifted": [c for c in cases if c["embodiment"] != BASE]}
    for field in ("regime", "embodiment"):
        for value in sorted({c[field] for c in cases}):
            groups[value] = [c for c in cases if c[field] == value]
    scores = {name: group_score(group, readers) for name, group in groups.items()}
    result = dict(panel=panel, counts=dict(cases=len(cases), episodes=sum(len(c["arms"]) for c in cases)),
                  parity=[dict(suite=s, task_id=t, init_state_id=13, physics=r["physics"], noise_seed=r["noise_seed"],
                               noise_sha256=r["noise_sha256"], identical_query_resets=True,
                               opening_proposal_sha256=openings.get((s, t))) for (s, t), r in sorted(resets.items())],
                  groups=scores, cases=cases)
    discrepancies = []
    archived_groups = dict(doc["by_regime"])
    if not pilot:
        archived_groups.update(doc["by_embodiment"])
    for name in (("overall", "shifted", *REGIMES.values()) if pilot else groups):
        archived = archived_groups[name]
        counts = {a: s["successes"] for a, s in scores[name]["arms"].items()}
        recorded = archived["successes"] if pilot else {NOM: archived["nominal_successes"],
                                                        ORACLE: archived["oracle_successes"]}
        if archived["trials"] != len(groups[name]) or recorded != counts:
            discrepancies.append(name)
    result["archived_count_disagreements"] = discrepancies
    if pilot:
        result.update(pilot_outcomes(cases, arms))
        if "pilot_rescues" in doc:
            result["archived_decision_agrees"] = doc["pilot_rescues"] == result["first_poc_witnesses"]
    else:
        all_arms, base = scores["overall"]["arms"], scores[BASE]["arms"]
        predicates = dict(overall_oracle_rate=all_arms[ORACLE]["successes"] >= 36,
                          overall_gain=all_arms[ORACLE]["successes"] - all_arms[NOM]["successes"] >= 24,
                          no_base_regression=base[ORACLE]["successes"] >= base[NOM]["successes"])
        passed = all(predicates.values())
        result.update(predicates=predicates, passed=passed, archived_passed=doc["passed"],
                      archived_decision_agrees=doc["passed"] is passed)
    return result


def check_support(path, archive_sha, pilot):
    archive, source = read(path)
    require(source["sha256"] == archive_sha and archive["panel"] == "poc", "support archive identity differs")
    require(set(archive["modes"]) == {"active_probe", "passive_attempt1"}
            and archive["passive_max_controls"] == archive["matched_reader_controls"] == 15, "wrong support modes or cap")
    train_embs = ("nom", "yaw90", "yaw180", "yawm90", "scale_lo", "scale_hi", "grip_inv")
    expected = {("train", "libero_goal", t, i, e) for t in (5, 7) for i in (0, 1) for e in train_embs}
    expected.update(("dev", suite, task, 11, emb) for suite, task, emb in PILOT)
    dev, starts = {}, {}
    for row in archive["trials"]:
        key = tuple(row[k] for k in ("split", "suite", "task_id", "init_state_id", "embodiment"))
        require(key in expected, f"unexpected or duplicate support trial {key}")
        expected.remove(key)
        identity = tuple(row[k] for k in ("seed", "physics", "frame0_img_sha256", "frame0_state_sha256",
                                          "passive_noise_sha256", "passive_first_proposal_sha256"))
        require(identity[1] and all(map(digest, identity[2:])) and starts.setdefault(key[:-1], identity) == identity,
                f"{key}: missing or unequal support reset, passive noise or opening")
        for mode in ("active", "passive"):
            n = row[f"{mode}_controls"]
            require(type(n) is int and 0 < n <= 15, f"{key}: invalid {mode} support length")
        if row["split"] == "dev":
            dev[row["embodiment"]] = row
    require(not expected, "incomplete support archive")
    for case in pilot["cases"]:
        support, donor = dev[case["embodiment"]], dev[case["donor_embodiment"]]
        require(support["physics"] != pilot["parity"][0]["physics"], "query and support physical starts coincide")
        for arm, rec in case["arms"].items():
            acquired = support["active_controls"]
            if arm in (NOM, ORACLE, ONLINE):
                acquired = 0
            elif arm == PASSIVE:
                acquired = support["passive_controls"]
            elif arm == SWAP:
                acquired = donor["active_controls"]
            require(rec["control_budget"]["support_acquired"] == acquired,
                    f"{case['embodiment']}/{arm}: acquired controls differ from support archive")
    return dict(source=source, train_trials=28, dev_trials=len(dev), support_reset=11, query_reset=13,
                paired_support_reset_groups=len(starts), support_query_physics_distinct=True)


def check_fit(path, archive_sha, visual_available):
    fit, source = read(path)
    require(fit["archive_sha256"] == archive_sha and fit["complete"] is True and fit["panel"] == "poc",
            "reader fit identity differs from pilot")
    require(fit["seeds"] == [41] and fit["steps"] == 1500 and fit["matched_support_cap"] == 15,
            "unexpected reader fit seed, step or support cap")
    for mode in ("active_probe", "passive_attempt1"):
        require((fit["modes"][mode]["train_trials"], fit["modes"][mode]["dev_trials"]) == (28, 3),
                f"{mode}: wrong reader fit counts")
    visual = fit["modes"]["active_probe"][VISUAL]
    require(visual["available"] is visual_available, "live analytic visual availability differs from fit")
    return dict(source=source, steps=fit["steps"], seeds=fit["seeds"], visual_available=visual_available,
                visual_unavailable_reason=visual.get("reason"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate1", type=Path, help="development gate1-summary.json")
    parser.add_argument("--pilot", type=Path, help="poc live-eval-summary.json")
    parser.add_argument("--archive", type=Path, help="pilot support archive.json for a cross-check")
    parser.add_argument("--fit", type=Path, help="pilot offline-audit.json for a cross-check")
    args = parser.parse_args()
    if not (args.gate1 or args.pilot) or ((args.archive or args.fit) and not args.pilot):
        parser.error("supply --gate1 and/or --pilot; --archive and --fit require --pilot")
    reports = {}
    bad_archive = False
    for stage, path in (("gate1", args.gate1), ("pilot", args.pilot)):
        if path is None:
            continue
        try:
            doc, source = read(path)
            report = score(doc, stage == "pilot")
            report["source"] = source
            if stage == "pilot":
                archive_sha = doc["archive_sha256"]
                report["support_archive_sha256"] = archive_sha
                report["support_verification"] = check_support(args.archive, archive_sha, report) if args.archive else None
                report["fit_verification"] = check_fit(args.fit, archive_sha, doc["visual_baseline_available"]) if args.fit else None
            reports[stage] = report
            if report.get("archived_decision_agrees") is False or report["archived_count_disagreements"]:
                bad_archive = True
        except (OSError, ValueError, KeyError, TypeError, IndexError) as err:
            reports[stage] = dict(path=str(path), error=str(err))
            bad_archive = True
    print(json.dumps(reports, indent=2, allow_nan=False))
    return 2 if bad_archive else 0


if __name__ == "__main__":
    raise SystemExit(main())
