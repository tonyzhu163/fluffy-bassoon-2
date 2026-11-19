"""Gate 1 realizability check: nominal vs. privileged-oracle frozen π0.5 across 6 embodiments."""

import argparse
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

    policy = load_frozen_pi05()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    prompts = {}

    with XEmbSimulator() as simulator:
        for suite_idx, (suite, task_ids) in enumerate(sorted(tasks.items())):
            for task_id in task_ids:
                for emb_idx, emb_name in enumerate(emb_names):
                    spec = get_embodiment_spec(emb_name)
                    seed = 20300000 + 10_000 * suite_idx + 100 * int(task_id) + emb_idx
                    case = dict(
                        suite=suite,
                        task_id=int(task_id),
                        variant="BASE",
                        init_state_id=args.init_state_id,
                        seed=seed,
                    )
                    key = (suite, int(task_id))
                    nom_rec = run_episode(
                        simulator,
                        policy,
                        case,
                        spec,
                        prompt=prompts.get(key),
                        yaw_hat=0.0,
                        scale_hat=1.0,
                        grip_hat=1,
                    )
                    prompts.setdefault(key, nom_rec["reset"]["prompt"])
                    ora_rec = run_episode(
                        simulator,
                        policy,
                        case,
                        spec,
                        prompt=prompts[key],
                        yaw_hat=spec.yaw_deg,
                        scale_hat=spec.xy_scale,
                        grip_hat=spec.gripper_sign,
                    )
                    if nom_rec["reset"] != ora_rec["reset"]:
                        raise ValueError(f"Gate-1 paired reset mismatch on {case} / {emb_name}")
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

    shifted = by_regime.get("shifted", dict(oracle_rate=0.0, oracle_minus_nominal=0.0))
    nom_emb = by_embodiment.get("fam_nom")
    nom_ok = True if nom_emb is None else nom_emb["oracle_successes"] >= nom_emb["nominal_successes"]
    passed = (
        shifted["oracle_rate"] >= 0.75
        and shifted["oracle_minus_nominal"] >= 0.50
        and nom_ok
    )
    summary = dict(
        panel=args.panel,
        passed=passed,
        thresholds=dict(min_shifted_oracle_rate=0.75, min_shifted_gain=0.50, no_nom_regression=True),
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
