"""Reduce CPU-only evidence and check that common pairing and scoring mistakes are detectable."""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
from panel import MODES, cases, key, require, save
from check_fit import sha
from verify import statistics, summarize, validate


def check_scoring():
    # One rescue and one loss cancel on one arm. The same rescued row is a tie vs policy-only.
    rows = []
    for i, case in enumerate(c for c in cases() if c["arm"] == "Kinova3"):
        modes = {m: dict(success=False) for m in MODES}
        modes["joint"]["success"] = i == 0
        modes["policy_only"]["success"] = i == 0
        modes["original"]["success"] = i == 1
        rows.append(dict(case=case, modes=modes))
    score = statistics(rows)
    vs_base, vs_policy = (score["joint_vs"][m] for m in ("original", "policy_only"))
    require((vs_base["rescues"], vs_base["losses"], vs_base["joint_minus"], vs_base["pairs"]) == (1, 1, 0, 12)
            and (vs_policy["rescues"], vs_policy["losses"], score["interaction"]) == (0, 0, 1/12),
            "paired contrasts/interaction")
    require(vs_policy["paired_bootstrap_95"] == [0, 0] and vs_base["paired_bootstrap_95"][0] < 0 <
            vs_base["paired_bootstrap_95"][1], "paired bootstrap")
    return True


def check_tamper(row):
    def corrupt(record, field):
        if field == "issued":
            record["calls"][1]["issued"][1][0] += .25
        elif field == "history":
            record["calls"][1]["history"][-1] += 1
        elif field == "noise":
            record["calls"][1]["noise"] = record["calls"][0]["noise"]
        else:
            record["native"]["native_rewards"][0] += 1

    rejected = []
    for field in ("issued", "history", "noise", "reward"):
        modified = deepcopy(row)
        corrupt(modified, field)
        try:
            validate(modified, True)
        except (AssertionError, ValueError):
            rejected.append(field)
    require(len(rejected) == 4, "invalid tape was admitted")
    return rejected


def readback(out):
    result = summarize(out, True)
    require(result == json.loads((out/"cpu-summary.json").read_text()), "CPU readback differs from writer")
    rows = result["rows"]
    p = out/"episodes"
    attempts = Counter()
    for c in cases():
        for mode in MODES:
            row = json.loads((p/f"{key(c)}-{mode}.json").read_text())
            attempts[row["accepted_reset_attempt"]] += 1
    timeout = json.loads((out/"native-timeout-fixture.json").read_text())
    rc, start, end = map(int, (out/"exit.txt").read_text().split())
    require(rc == 0 and timeout["stationary_controls"] == 300 and timeout["native_rewards"] == [0.]*300, "timeout fixture")
    expected = dict(original=0, policy_only=5, adapter_only=0, joint_zero=5, joint_lagged=5)
    require(all({m: x["first_different_issued_control"] for m, x in r["first_changes"].items()} == expected
                for r in rows), "known policy or history intervention did not reach the actuator")
    return dict(kind="CPU simulator interface fixture, deterministic actuator driver", run=str(out),
                pin=result["manifest"]["pin"], host=result["manifest"]["host"], cpu_affinity=result["manifest"]["cpu_affinity"],
                gpu_hours=0, wall_seconds=end-start, renderer=timeout["renderer"], registered_starts=len(rows),
                paired_short_episodes=result["episodes"], controls_per_short_episode=15,
                controls_exercised=result["episodes"]*15+300, accepted_reset_attempts=attempts,
                independent_noise_and_history_reconstructions=result["episodes"], exact_fixture_stock_noop_openings=len(rows),
                known_joint_first_issued_change_by_comparator=expected, historical_timeout_controls=300,
                historical_timeout_native_return=sum(timeout["native_rewards"]), frozen_fixture=result["source_lock"],
                scoring_test=check_scoring(), corrupt_tapes_rejected=check_tamper(row),
                cpu_summary_sha256=sha(out/"cpu-summary.json"), per_start=[
                    dict(case=r["case"], released_row=r["released_row"], physics=r["physics"], noise=r["noise"])
                    for r in rows])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    save(args.receipt, readback(args.out))
