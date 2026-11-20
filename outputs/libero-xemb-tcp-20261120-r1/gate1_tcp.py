"""Gate 1 motor realizability check for physical MuJoCo finger-TCP offsets."""

import argparse
from functools import partial
import json
from pathlib import Path

from fluffy_bassoon.xemb.tcp import make_tcp_catalogue
from tcp_motor import SPLITS_PATH, load_frozen_pi05, run_tcp_episode, write_json
from tcp_sim_worker import TcpSimulator


ARM_NAMES = ("nominal", "oracle_grasp_only", "oracle_grasp_and_release")


def check_paired_reset(reference, actual, *, output, case, embodiment, arm):
    if reference == actual:
        return
    differences = {
        key: dict(expected=reference.get(key), observed=actual.get(key))
        for key in reference.keys() | actual.keys()
        if reference.get(key) != actual.get(key)
    }
    write_json(
        output / "gate1-reset-mismatch.json",
        dict(
            case=case,
            embodiment=embodiment,
            arm=arm,
            reference="tcp_nom/nominal",
            differences=differences,
        ),
    )
    raise ValueError(f"Unpaired frame 0 on {case} / {embodiment} / {arm}: {list(differences)}")


def aggregate_group(rows: list) -> dict:
    n = len(rows)
    out = dict(trials=n)
    for arm in ARM_NAMES:
        succ = sum(int(bool(r[arm]["success"])) for r in rows)
        ret = sum(float(r[arm]["native_return"]) for r in rows) / n if n else 0.0
        out[f"{arm}_successes"] = succ
        out[f"{arm}_rate"] = succ / n if n else 0.0
        out[f"{arm}_mean_return"] = ret
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--panel", choices=("smoke", "development"), default="development")
    parser.add_argument("--radius-m", type=float, default=0.035)
    parser.add_argument("--init-state-id", type=int, default=13)
    args = parser.parse_args()

    splits = json.loads(SPLITS_PATH.read_text())
    _, eval_specs = make_tcp_catalogue(args.radius_m)

    if args.panel == "smoke":
        tasks = {"libero_goal": [4]}
        specs = [s for s in eval_specs if s.name in ("tcp_nom", "tcp_px")]
    else:
        tasks = splits["development"]
        specs = list(eval_specs)
        task_keys = [(s, int(t)) for s, ids in tasks.items() for t in ids]
        if len(task_keys) != 8 or len(set(task_keys)) != 8:
            raise ValueError("Gate 1 requires eight distinct development tasks")

    policy = load_frozen_pi05()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []

    with TcpSimulator() as simulator:
        for suite_idx, (suite, task_ids) in enumerate(sorted(tasks.items())):
            for task_id in task_ids:
                tid = int(task_id)
                seed = 20400000 + 10_000 * suite_idx + 100 * tid
                case = dict(
                    suite=suite,
                    task_id=tid,
                    variant="BASE",
                    init_state_id=args.init_state_id,
                    seed=seed,
                )
                reference = None
                for spec in specs:
                    arms = {}
                    arm_configs = (
                        ("nominal", 0.0, 0.0, False),
                        ("oracle_grasp_only", spec.dx_m, spec.dy_m, False),
                        ("oracle_grasp_and_release", spec.dx_m, spec.dy_m, True),
                    )
                    for arm, dx_hat, dy_hat, comp_rel in arm_configs:
                        check_fn = (
                            None
                            if reference is None
                            else partial(
                                check_paired_reset,
                                reference,
                                output=args.output,
                                case=case,
                                embodiment=spec.name,
                                arm=arm,
                            )
                        )
                        arms[arm] = run_tcp_episode(
                            simulator,
                            policy,
                            case,
                            spec,
                            dx_hat_m=dx_hat,
                            dy_hat_m=dy_hat,
                            compensate_release=comp_rel,
                            check_reset=check_fn,
                        )
                        if reference is None:
                            reference = arms[arm]["reset"]

                    row = dict(
                        suite=suite,
                        task_id=tid,
                        init_state_id=args.init_state_id,
                        seed=seed,
                        embodiment=spec.name,
                        regime=spec.regime,
                        dx_m=spec.dx_m,
                        dy_m=spec.dy_m,
                        dz_m=spec.dz_m,
                        nominal=arms["nominal"],
                        oracle_grasp_only=arms["oracle_grasp_only"],
                        oracle_grasp_and_release=arms["oracle_grasp_and_release"],
                    )
                    rows.append(row)
                    write_json(args.output / "gate1-progress.json", rows)
                    print(
                        json.dumps(
                            dict(
                                done=len(rows),
                                suite=suite,
                                task_id=tid,
                                emb=spec.name,
                                nom=arms["nominal"]["success"],
                                ora_g=arms["oracle_grasp_only"]["success"],
                                ora_gr=arms["oracle_grasp_and_release"]["success"],
                                triggers_g=arms["oracle_grasp_only"]["grasp_triggers"],
                                triggers_gr=(
                                    arms["oracle_grasp_and_release"]["grasp_triggers"],
                                    arms["oracle_grasp_and_release"]["release_triggers"],
                                ),
                            )
                        ),
                        flush=True,
                    )

    groups = {
        "familiar_nom": [r for r in rows if r["embodiment"] == "tcp_nom"],
        "familiar_axis": [r for r in rows if r["regime"] == "familiar" and r["embodiment"] != "tcp_nom"],
        "compositional_ood": [r for r in rows if r["regime"] == "compositional_ood"],
        "shifted": [r for r in rows if r["embodiment"] != "tcp_nom"],
        "overall": rows,
    }
    by_regime = {name: aggregate_group(grp) for name, grp in groups.items() if grp}
    by_embodiment = {
        spec.name: aggregate_group([r for r in rows if r["embodiment"] == spec.name])
        for spec in specs
    }

    shifted = by_regime["shifted"]
    nom_emb = by_embodiment["tcp_nom"]
    winning_arm = (
        "oracle_grasp_and_release"
        if shifted["oracle_grasp_and_release_successes"] > shifted["oracle_grasp_only_successes"]
        else "oracle_grasp_only"
    )
    shifted_oracle_rate = shifted[f"{winning_arm}_rate"]
    shifted_gain = shifted_oracle_rate - shifted["nominal_rate"]
    predicates = dict(
        shifted_oracle_rate=shifted_oracle_rate >= 0.75,
        shifted_gain=shifted_gain >= 0.45,
        no_base_regression=nom_emb[f"{winning_arm}_successes"] >= nom_emb["nominal_successes"],
    )

    passed = None
    if args.panel == "development":
        if by_regime["overall"]["trials"] != 56 or shifted["trials"] != 48 or any(v["trials"] != 8 for v in by_embodiment.values()):
            raise ValueError("Incomplete Gate 1 panel")
        passed = all(predicates.values())

    summary = dict(
        panel=args.panel,
        radius_m=args.radius_m,
        passed=passed,
        winning_arm=winning_arm,
        thresholds=dict(min_shifted_oracle_rate=0.75, min_shifted_gain=0.45, no_base_regression=True),
        predicates=predicates,
        by_regime=by_regime,
        by_embodiment=by_embodiment,
        rows=rows,
    )
    write_json(args.output / "gate1-summary.json", summary)
    print(
        json.dumps(
            dict(
                gate1_passed=passed,
                winning_arm=winning_arm,
                predicates=predicates,
                by_regime=by_regime,
                by_embodiment=by_embodiment,
            ),
            indent=2,
        ),
        flush=True,
    )
    if args.panel == "development" and not passed:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
