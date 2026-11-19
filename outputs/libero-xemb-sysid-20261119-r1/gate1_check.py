"""Gate 1 realizability check: nominal vs. privileged-oracle frozen π0.5 across 6 embodiments."""

import argparse
from functools import partial
import json
from pathlib import Path

from fluffy_bassoon.xemb.embodiment import EVAL_REGIMES, get_embodiment_spec
from motor_iface import SPLITS_PATH, load_frozen_pi05, run_episode, write_json
from sim_worker import XEmbSimulator


GATE1_EMBODIMENT_NAMES = (
    "fam_nom",
    "fam_yaw90",
    "comp_180_lo_inv",
    "comp_m90_hi_inv",
    "cont_45_nom",
    "cont_135_075_inv",
)


def check_paired_reset(reference, actual, *, output, case, embodiment, arm):
    if reference == actual:
        return
    differences = {key: dict(expected=reference.get(key), observed=actual.get(key))
                   for key in reference.keys() | actual.keys() if reference.get(key) != actual.get(key)}
    write_json(output / "gate1-reset-mismatch.json", dict(
        case=case, embodiment=embodiment, arm=arm, reference="fam_nom/nominal", differences=differences,
    ))
    raise ValueError(f"Unpaired frame 0 on {case} / {embodiment} / {arm}: {list(differences)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--panel", choices=("smoke", "development"), default="development")
    parser.add_argument("--init-state-id", type=int, default=13)
    args = parser.parse_args()

    splits = json.loads(SPLITS_PATH.read_text())
    if args.panel == "smoke":
        tasks = {"libero_goal": [4]}
        emb_names = ("fam_nom", "comp_180_lo_inv")
    else:
        tasks = splits["development"]
        emb_names = GATE1_EMBODIMENT_NAMES
        task_keys = [(s, t) for s, ids in tasks.items() for t in ids]
        if len(task_keys) != 8 or len(set(task_keys)) != 8:
            raise ValueError("Gate 1 requires eight distinct development tasks")

    policy = load_frozen_pi05()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []

    with XEmbSimulator() as simulator:
        for suite_idx, (suite, task_ids) in enumerate(sorted(tasks.items())):
            for task_id in task_ids:
                seed = 20300000 + 10_000 * suite_idx + 100 * int(task_id)
                case = dict(suite=suite, task_id=int(task_id), variant="BASE",
                            init_state_id=args.init_state_id, seed=seed)
                reference = None
                for emb_name in emb_names:
                    spec = get_embodiment_spec(emb_name)
                    arms = {}
                    for arm, estimates in (("nominal", (0.0, 1.0, 1)),
                                           ("privileged_oracle", (spec.yaw_deg, spec.xy_scale, spec.gripper_sign))):
                        yaw, scale, grip = estimates
                        check_reset = None if reference is None else partial(
                            check_paired_reset, reference, output=args.output, case=case, embodiment=emb_name, arm=arm,
                        )
                        arms[arm] = run_episode(simulator, policy, case, spec, yaw_hat=yaw,
                                                scale_hat=scale, grip_hat=grip, check_reset=check_reset,
                                                audit_base_views=emb_name == "fam_nom")
                        if reference is None:
                            reference = arms[arm]["reset"]
                    nom_rec, ora_rec = arms["nominal"], arms["privileged_oracle"]
                    row = dict(
                        suite=suite,
                        task_id=int(task_id),
                        init_state_id=args.init_state_id,
                        seed=seed,
                        embodiment=spec.name,
                        regime=spec.regime,
                        yaw_deg=spec.yaw_deg,
                        xy_scale=spec.xy_scale,
                        gripper_sign=spec.gripper_sign,
                        nominal=nom_rec,
                        privileged_oracle=ora_rec,
                    )
                    if emb_name == "fam_nom":
                        # BASE arms issue the same control law. Report independently rendered views
                        # at common query times, even when their measured endpoints agree.
                        nom_queries, ora_queries = nom_rec["motor_queries"], ora_rec["motor_queries"]
                        row["base_drift"] = dict(
                            nominal_queries=len(nom_queries), oracle_queries=len(ora_queries),
                            mismatched_controls={
                                field: [n["control"] for n, o in zip(nom_queries, ora_queries) if n[field] != o[field]]
                                for field in ("agent", "wrist", "state", "commands")
                            },
                            same_success=nom_rec["success"] == ora_rec["success"],
                            same_native_return=nom_rec["native_return"] == ora_rec["native_return"],
                        )
                    rows.append(row)
                    write_json(args.output / "gate1-progress.json", rows)
                    print(
                        json.dumps(
                            dict(
                                done=len(rows),
                                suite=suite,
                                task_id=int(task_id),
                                emb=spec.name,
                                nom=nom_rec["success"],
                                oracle=ora_rec["success"],
                            )
                        ),
                        flush=True,
                    )

    by_regime = {}
    for regime in (*EVAL_REGIMES, "shifted", "overall"):
        if regime == "shifted":
            grp = [r for r in rows if r["embodiment"] != "fam_nom"]
        elif regime == "overall":
            grp = rows
        else:
            grp = [r for r in rows if r["regime"] == regime]
        if not grp:
            continue
        nom_s = sum(r["nominal"]["success"] for r in grp)
        ora_s = sum(r["privileged_oracle"]["success"] for r in grp)
        by_regime[regime] = dict(
            trials=len(grp),
            nominal_successes=nom_s,
            oracle_successes=ora_s,
            nominal_rate=nom_s / len(grp),
            oracle_rate=ora_s / len(grp),
            oracle_minus_nominal=(ora_s - nom_s) / len(grp),
        )

    by_embodiment = {}
    for emb_name in emb_names:
        grp = [r for r in rows if r["embodiment"] == emb_name]
        nom_s = sum(r["nominal"]["success"] for r in grp)
        ora_s = sum(r["privileged_oracle"]["success"] for r in grp)
        by_embodiment[emb_name] = dict(
            trials=len(grp),
            nominal_successes=nom_s,
            oracle_successes=ora_s,
            nominal_rate=nom_s / len(grp),
            oracle_rate=ora_s / len(grp),
        )

    overall = by_regime["overall"]
    base = by_embodiment["fam_nom"]
    predicates = dict(
        overall_oracle_rate=overall["oracle_rate"] >= 0.75,
        overall_gain=overall["oracle_minus_nominal"] >= 0.50,
        no_base_regression=base["oracle_successes"] >= base["nominal_successes"],
    )
    passed = None
    if args.panel == "development":
        if overall["trials"] != 48 or any(v["trials"] != 8 for v in by_embodiment.values()):
            raise ValueError("Incomplete Gate 1 panel")
        passed = all(predicates.values())
    summary = dict(
        panel=args.panel,
        passed=passed,
        thresholds=dict(min_overall_oracle_rate=0.75, min_overall_gain=0.50, no_base_regression=True),
        predicates=predicates,
        by_regime=by_regime,
        by_embodiment=by_embodiment,
        rows=rows,
    )
    write_json(args.output / "gate1-summary.json", summary)
    print(json.dumps(dict(gate1_passed=passed, by_regime=by_regime, by_embodiment=by_embodiment), indent=2), flush=True)
    if args.panel == "development" and not passed:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
