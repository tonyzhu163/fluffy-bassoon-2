"""Read Baige native01 receipts on one Linux CPU. Writes JSON only to stdout."""
import hashlib
import json
import os
import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np


ROOT = Path("/mowenjie/fluffy-bassoon-2/outputs/frozen-vla-selector-20261124-r1/native01")
JOB = "job-umsgs123dma7"
PIN = "85696bd8231b8115cc0d7959a1662dad46bb71b2"
ARMS = {"Panda": range(10), "UR5e": range(10), "Kinova3": range(36, 40)}
MENU = ("continue", "correct", "cue")
SOURCE = ("continue", "repeat", "correct", "cue")
HELD = SOURCE + ("correct_online", "factorized", "factorized_online", "generic", "generic_online")
SEED = 42710300
TASK = "open the top layer of the drawer and put the bowl inside"


def read(path):
    return json.loads(path.read_text())


def check(ok, reason):
    if not ok:
        raise ValueError(reason)


def identity(arm, layout):
    return dict(arm=arm, suite="libero_goal", task_id=3, variant="BASE",
                init_state_id=layout, seed=SEED + layout)


def digest(array):
    header = repr((array.shape, array.dtype.str)).encode()
    return hashlib.sha256(header + array.tobytes()).hexdigest()


def decision(event, name, start, controls, bounds):
    action = event["branch"]
    fixed = {"repeat": "continue", "correct_online": "correct"}.get(name, name)
    check(action in (MENU if fixed.startswith(("factorized", "generic")) else (fixed,)),
          f"unexpected {action} in {name}")
    check(event["control"] == start and event["nominal_prompt"] == TASK, "decision schedule/prompt")
    select, nominal, issued = (np.asarray(event[k], dtype=float) for k in
                               ("selected_bounded", "nominal_bounded", "executed_commands"))
    consumed = min(5, controls - start)
    check(select.shape == nominal.shape == (5, 7) and issued.shape == (consumed, 7),
          "decision command dimensions")
    for proposal in (select, nominal):
        check(np.all((bounds["low"] <= proposal) & (proposal <= bounds["high"])), "action bounds")
    check(event["submitted_steps"] == 5 and event["executed_steps"] == consumed
          and np.array_equal(issued, select[:consumed]), "consumed commands vs selected prefix")
    delta = np.abs(issued - nominal[:consumed])
    check(delta.max() == event["executed_delta_linf"], "executed delta vs nominal")
    if action == "continue":
        check(np.array_equal(select, nominal)
              and event["nominal_proposal_sha256"] == event["selected_proposal_sha256"], "continue changed")
    elif action == "correct":
        check(np.array_equal(select[:, 3:], nominal[:, 3:]), "correction outside XYZ")
    else:
        check(event["prompt"] in ("open the top drawer", "put the bowl in the top drawer"), "cue prompt")
    return consumed, int(np.count_nonzero(np.any(delta != 0, axis=1)))


def audit_case(cell, arm, layout, boundary, names, support):
    branches = cell["branches"]
    check(cell["case"] == identity(arm, layout) and cell["boundary"] == boundary, "case identity/seed/boundary")
    check(set(branches) == set(names), "incomplete or unexpected branch set")
    base = branches["continue"]
    ref, fork = base["record"], base["fork"]
    reset = ref["reset"]
    check(reset["arm"] == arm and reset["noise_seed"] == SEED + layout + 817117
          and reset["physics"] != support["physics"] and reset["worker_step_index"] == 0, "reset/seed/support")
    shape = ((reset["horizon"] + 4) // 5, 10, 32)
    bank = np.random.default_rng(SEED + layout + 817117).standard_normal(shape).astype(np.float32)
    check(digest(bank) == reset["noise_sha256"], "noise bank vs registered seed")
    noise_by_replan = [digest(draw) for draw in bank]
    digests = [reset["noise_sha256"], reset["physics"], ref["opening_proposal_sha256"], base["prefix"]]
    if fork is not None:
        digests.extend(fork[k] for k in ("physics", "prefix", "proposal", "noise"))
        check(fork["prefix"] == base["prefix"] and len(fork["motion"]) == 19
              and len(fork["progress"]) == 13, "fork prefix/features")
    check(all(re.fullmatch("[0-9a-f]{64}", s) for s in digests), "missing/invalid digest")
    rows = {}
    for name in names:
        b = branches[name]
        r, events = b["record"], b["events"]
        controls, rewards = r["controls"], r["native_rewards"]
        check(r["reset"] == reset and r["released_prompt"] == reset["prompt"] == TASK
              and r["opening_proposal_sha256"] == ref["opening_proposal_sha256"]
              and b["prefix"] == base["prefix"] and b["fork"] == fork
              and rewards[:boundary] == ref["native_rewards"][:boundary], f"{name} unmatched fork/reset")
        check(0 < controls == len(rewards) <= reset["horizon"] and all(x in (0, 1) for x in rewards)
              and (r["terminated"] or r["truncated"]) and sum(rewards) == r["native_return"]
              and r["success"] is any(rewards), f"{name} native tape/return/termination")
        starts = range(boundary, controls if name.endswith("_online") else min(controls, boundary + 1), 5)
        check((fork is not None) == (controls > boundary) and len(events) == len(starts), f"{name} decision coverage")
        if events:
            e = events[0]
            check(e["noise_sha256"] == fork["noise"] and e["nominal_proposal_sha256"] == fork["proposal"]
                  and e["nominal_bounded"] == base["events"][0]["nominal_bounded"], f"{name} decision opening")
        consumed = changed = 0
        for event, start in zip(events, starts):
            check(noise_by_replan[start // 5] == event["noise_sha256"], f"{name} seeded replan noise at {start}")
            n, c = decision(event, name, start, controls, reset["action_schema"])
            consumed += n
            changed += c
        rows[name] = dict(success=int(any(rewards)), controls=controls, decisions=len(events),
                          decision_commands=consumed, changed_commands=changed,
                          short_windows=sum(e["executed_steps"] < e["submitted_steps"] for e in events),
                          first_choice=events[0]["branch"] if events else None)
        if name.endswith("_online"):
            rows[name]["choice_counts"] = dict(Counter(e["branch"] for e in events))
        if name in ("factorized", "generic") and events:
            chosen = branches[events[0]["branch"]]
            rows[name]["chosen_fixed_commands_match"] = events[0]["executed_commands"] == chosen["events"][0]["executed_commands"]
            rows[name]["chosen_fixed_outcome_matches"] = r["success"] == chosen["record"]["success"]
    repeat = branches["repeat"]
    return dict(layout=layout, seed=SEED + layout, boundary=boundary, reached_fork=fork is not None,
                file=f"{arm}-{layout}.json", branches=rows,
                untreated_reward_tapes_equal=repeat["record"]["native_rewards"] == ref["native_rewards"],
                untreated_decision_events_equal=repeat["events"] == base["events"])


def comparisons(cases, names):
    values = {k: np.array([c["branches"][k]["success"] for c in cases]) for k in names}
    values["one_choice_hindsight"] = np.maximum.reduce([values[k] for k in MENU])
    n = len(cases)
    draws = np.random.default_rng(SEED).integers(n, size=(10000, n))

    def paired(a, b):
        d = values[a] - values[b]
        return dict(successes=int(values[a].sum()), total=n, rescues=int((d > 0).sum()),
                    losses=int((d < 0).sum()), delta=float(d.mean()),
                    layout_interval95=np.percentile(d[draws].mean(axis=1), [2.5, 97.5]).tolist())

    summary = {k: paired(k, "continue") for k in values}
    extra = {}
    if "factorized" in names:
        for suffix in ("", "_online"):
            name = "factorized_minus_generic" + suffix
            difference = paired("factorized" + suffix, "generic" + suffix)
            summary[name] = {k: difference[k] for k in ("delta", "layout_interval95")}
            extra[name] = difference
        for head in ("factorized", "generic"):
            for control in ("repeat", "correct", "cue"):
                extra[f"{head}_minus_{control}"] = paired(head, control)
    return summary, extra


def disposition(held):
    scores = held["paired_against_continue"]
    raw, repeat, oracle = (scores[k] for k in ("continue", "repeat", "one_choice_hindsight"))
    fac, gen = (scores[k] for k in ("factorized", "generic"))
    n = raw["total"]
    headroom = (f"One-choice hindsight rescued {oracle['rescues']} of {n - raw['successes']} nominal failures. "
                if n > raw["successes"] else "Continuing succeeded on every layout, exhausting one-choice headroom. ")
    if oracle["rescues"] == 0:
        finding = "This panel shows no additional success available to a one-choice selector."
    else:
        finding = "The fixed menu contains a rescue. The four paired layouts give only pilot evidence for choosing it."
    head_pair = held["other_held_pairs"]["factorized_minus_generic"]
    low, high = head_pair["layout_interval95"]
    uncertainty = (f"With zero head-to-head discordances in {n} pairs, the one-sided exact 95% upper bound "
                   f"on discordance would still be {1 - 0.05 ** (1 / n):.1%} under independent layout sampling."
                   if not (head_pair["rescues"] or head_pair["losses"]) else
                   "The interval reflects only variation among these four layouts.")
    return (f"On four held-out Kinova3 layouts of the same drawer-and-bowl task, continue and repeat scored "
            f"{raw['successes']}/{n} and {repeat['successes']}/{n}, with {held['untreated_outcome_disagreements']} "
            f"discordant outcomes. {headroom}Factorized and pooled one-choice selectors scored {fac['successes']}/{n} "
            f"({fac['rescues']} rescues, {fac['losses']} losses) and {gen['successes']}/{n} "
            f"({gen['rescues']} rescues, {gen['losses']} losses) against continue. {finding} "
            f"The paired 95% layout-bootstrap interval for factorized minus pooled is "
            f"[{100 * low:g}, {100 * high:g}] percentage points. Each layout contributes 25 points. {uncertainty}")


def audit():
    now = datetime.now(timezone(timedelta(hours=8)))
    launch = (ROOT / "launch.txt").read_text().strip()
    terminal = ROOT / "exit.txt"
    exit_fields = terminal.read_text().split() if terminal.is_file() else []
    status = int(exit_fields[0]) if exit_fields else None
    published = read(ROOT / "summary.json") if (ROOT / "summary.json").is_file() else {}
    errors = []
    if launch != f"host={JOB}-master-0 pin={PIN}":
        errors.append("launch host or pin mismatch")
    if status not in (None, 0):
        errors.append(f"launcher exit {status}")
    report = dict(job=JOB, pin=PIN, source_directory=str(ROOT), checked_shanghai=now.isoformat(timespec="seconds"),
                  launch_receipt=launch, exit_receipt=" ".join(exit_fields) or None, errors=errors,
                  audit_runtime=dict(system=os.uname().sysname, host=os.uname().nodename,
                                     cpu_affinity=sorted(os.sched_getaffinity(0)),
                                     cuda_visible_devices=os.getenv("CUDA_VISIBLE_DEVICES")), arms={})
    missing = []
    for arm, layouts in ARMS.items():
        names = HELD if arm == "Kinova3" else SOURCE
        support_file = ROOT / f"{arm}-support.json"
        cases = []
        arm_report = dict(cases=cases, support_file=None)
        report["arms"][arm] = arm_report
        support = {}
        if support_file.is_file():
            try:
                support = read(support_file)
                check(support["case"] == identity(arm, 11) and
                      all(len(support[k]) == 3 for k in ("inputs", "joints", "response")), "calibration case/bursts")
                arm_report["support_file"] = support_file.name
            except (ValueError, KeyError, TypeError) as exc:
                errors.append(f"{support_file.name}: {exc}")
        for idx, layout in enumerate(layouts):
            path = ROOT / f"{arm}-{layout}.json"
            if not path.is_file():
                missing.append(path.name)
                continue
            try:
                cases.append(audit_case(read(path), arm, layout, (45, 60, 75, 90)[idx % 4], names, support))
            except (ValueError, KeyError, IndexError, TypeError) as exc:
                errors.append(f"{path.name}: {exc}")
        if cases:
            summary, extra = comparisons(cases, names)
            arm_report["paired_against_continue"] = {k: v for k, v in summary.items() if "minus" not in k}
            if extra:
                arm_report["other_held_pairs"] = extra
            arm_report["count_totals"] = {k: {key: sum(c["branches"][k][key] for c in cases)
                                              for key in ("controls", "decisions", "decision_commands",
                                                          "changed_commands", "short_windows")}
                                          for k in names}
            repeat = summary["repeat"]
            arm_report["untreated_outcome_disagreements"] = repeat["rescues"] + repeat["losses"]
            arm_report["untreated_tape_disagreements"] = sum(not c["untreated_reward_tapes_equal"] for c in cases)
            arm_report["untreated_decision_disagreements"] = sum(not c["untreated_decision_events_equal"] for c in cases)
        if arm in published:
            match = len(cases) == len(layouts) and published[arm] == summary
            arm_report["published_summary_matches"] = match
            if not match:
                errors.append(f"summary.json: {arm} disagrees with case readback")
        else:
            arm_report["published_summary_matches"] = None
    expected = {f"{a}-{i}.json" for a, ls in ARMS.items() for i in ls}
    extras = sorted(p.name for p in ROOT.glob("*-[0-9]*.json") if p.name not in expected)
    if extras or set(published) - set(ARMS):
        errors.append(f"unexpected cases {extras} or arms {sorted(set(published) - set(ARMS))}")
    rows = [r for a in report["arms"].values() for c in a["cases"] for r in c["branches"].values()]
    report.update(planned_queries=116, verified_queries=len(rows), missing_case_files=missing,
                  verified_matched_cases=sum(len(a["cases"]) for a in report["arms"].values()),
                  calibrations=sum(bool(a["support_file"]) for a in report["arms"].values()),
                  decisions=sum(r["decisions"] for r in rows), short_decision_windows=sum(r["short_windows"] for r in rows),
                  query_controls=sum(r["controls"] for r in rows), decision_commands=sum(r["decision_commands"] for r in rows),
                  changed_decision_commands=sum(r["changed_commands"] for r in rows))
    if status is not None and (len(rows) != 116 or set(published) != set(ARMS)):
        errors.append("launcher exited before all planned cases and summaries verified")
    complete = len(rows) == 116 and set(published) == set(ARMS)
    report["status"] = ("failed_verification" if errors else "verified" if complete else
                        "deadline_without_terminal_receipt" if (now.hour, now.minute) >= (13, 28) else "in_progress")
    if report["status"] == "verified":
        report["held_arm_disposition"] = disposition(report["arms"]["Kinova3"])
    report["checks_per_verified_case"] = [
        "Registered arm, suite, task, variant, layout, query seed, support identity, boundary and branch set",
        "Noise bank regenerated from the registered seed and every logged decision draw checked by digest",
        "Equal reset, opening proposal, prefix commands and rewards, fork state, features and nominal decision",
        "Binary native tape, consumed control count, summed return, terminal flag and success flag",
        "Decision cadence, bounded proposals, commands consumed through termination and changed components",
    ]
    report["count_definitions"] = ("Controls count all simulator steps on verified query reward tapes. Decision "
        "commands count consumed steps at logged decisions, including unchanged steps. Changed commands differ "
        "in at least one component from that event's bounded nominal action. Calibrations are separate.")
    report["pairing"] = ("Rescues and losses count layouts switching from failure to success and success to failure. "
        "Deltas are success-rate differences. Intervals resample paired layouts 10,000 times with seed 42710300. "
        "One-choice hindsight takes the per-layout maximum of continue, correct and cue. Online branches can "
        "intervene at every later replan and have a different attainable success rate.")
    report["arms"] = report.pop("arms")
    return report


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2, allow_nan=False))
