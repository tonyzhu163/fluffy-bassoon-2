"""Compare frozen R2 source-candidate replan strides on the eight native Goal 5 development starts."""

import argparse
from itertools import combinations
import os
from pathlib import Path
import sys
from time import perf_counter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", required=True, type=Path, help="completed R2 fit containing dev/report.json")
    parser.add_argument("--out", required=True, type=Path, help="fresh experiment directory")
    parser.add_argument("--weights", type=Path, default=os.environ.get("DUAL_WEIGHTS"),
                        required=not os.environ.get("DUAL_WEIGHTS"))
    parser.add_argument("--revision", default=os.environ.get("MOTOR_CODE_REVISION"),
                        required=not os.environ.get("MOTOR_CODE_REVISION"))
    parser.add_argument("--step", type=int, default=1600)
    parser.add_argument("--strides", nargs="+", type=int, choices=(1, 2, 5), default=[1, 2],
                        help="fresh executions, each once. The original stride-five tapes are always read.")
    args = parser.parse_args()
    if len(set(args.strides)) != len(args.strides):
        parser.error("each fresh stride can be requested only once")
    run(args)


def held_panel(fit, fit_id, contract, candidate, prompt):
    from stride_rollout import ev
    from motor_eval import PANELS, check_episode, read

    dev = fit / "dev"
    panel, report = read(dev / "contract.json"), read(dev / "report.json")
    poses = PANELS["dev"][0]
    if (report["status"] != "complete" or report["stage"] != "dev" or panel["stage"] != "dev"
            or report["contract_sha256"] != ev.digest_file(dev / "contract.json")
            or report["fit_id"] != fit_id or panel["fit_id"] != fit_id or candidate not in panel["candidates"]
            or panel["sources"] != contract["sources"] or panel["prompt"] != prompt
            or panel["poses"] != list(poses) or panel["execute"] != 5 or panel["horizon"] != ev.HORIZON
            or panel["source_bodies"] != list(ev.SOURCES) or panel["base_weights"] != contract["weights"]):
        raise ValueError("Expected the original, completed stride-five native development panel")
    rows = {}
    layouts = {}
    used = {r["released_row_fingerprint"] for r in contract["episodes"]}
    for row in report["episodes"]:
        if row["step"] != candidate["step"]:
            continue
        body, pose = key = row["body"], row["pose"]
        if key in rows or body not in ev.SOURCES or pose not in poses:
            raise ValueError(f"Unexpected held development tape {key}")
        path = dev / f"step-{candidate['step']:06d}" / f"query-{body}-{pose}-raw.npz"
        verified = check_episode(path, dev, body, pose, prompt, candidate, fit_id, row["tape_sha256"])
        if verified != row:
            raise ValueError(f"Held tape and original report disagree on {key}")
        phys = row["opening"]["physics"]
        pair = (phys["released_row_fingerprint"], phys["scene_fingerprint"])
        if pair[0] in used or layouts.setdefault(pose, pair) != pair:
            raise ValueError(f"Repeated training layout or unpaired arms on {key}")
        rows[key] = {**row, "tape": str(path.resolve()), "stride": 5,
                     "model_calls": (row["steps"] + 4) // 5}
    if (set(rows) != {(b, p) for b in ev.SOURCES for p in poses}
            or len({r[0] for r in layouts.values()}) != len(poses)):
        raise ValueError("Expected eight held episodes from four distinct development layouts")
    return rows


def summarize(reference, rows):
    from stride_rollout import ev
    summary = {}
    for body in ev.SOURCES:
        arm = [r for r in rows if r["body"] == body]
        pairs = [(reference[body, r["pose"]]["success"], r["success"]) for r in arm]
        summary[body] = dict(starts=len(arm), successes=sum(r["success"] for r in arm),
                             controls=sum(r["steps"] for r in arm), model_calls=sum(r["model_calls"] for r in arm),
                             wins_vs_held=sum(b and not a for a, b in pairs),
                             losses_vs_held=sum(a and not b for a, b in pairs))
    return summary


def run(args):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import numpy as np
    from stride_rollout import HERE, PairedRollout, compare_plans, ev
    from motor_train import base, runtime, source_ids, torch
    from motor_eval import FrozenMotor, audit_fit, read

    base.require_gpu()
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    fit = args.fit.resolve(strict=True)
    manifest = read(fit / "manifest.json")
    historical = manifest["contract"]["sources"]
    contract, fit_id, shortlist, prompt = audit_fit(fit, historical)
    live = source_ids()
    # Trainer and evaluation orchestration have evolved since the original fit.
    # The numerical motor, verified base loader, and native simulator must agree.
    numerical = ["motor.py", "motor_data.py", "motor_weights.py", "adapt_eval.py", "sim.py", "arm_env.py", "sim_reset.py"]
    if contract.get("architecture", "chunk_huber_v1") != "chunk_huber_v1":
        numerical.append("servo_motor.py")
    if any(historical[k] != live[k] for k in numerical) or contract["recipe"]["horizon"] != 5:
        raise ValueError("Historical inference, native simulator or prediction horizon differs")
    candidates = [c for c in shortlist if c["step"] == args.step]
    if len(candidates) != 1:
        raise ValueError(f"Update {args.step} must have exactly one candidate in the original development shortlist")
    candidate = candidates[0]
    held = held_panel(fit, fit_id, contract, candidate, prompt)
    args.out.mkdir(parents=True, exist_ok=False)
    report_path = args.out / "report.json"
    report = dict(status="running", stage="dev-stride", revision=args.revision, argv=sys.argv,
                  fit=str(fit), fit_id=fit_id, training_revision=contract["revision"], weights=contract["weights"],
                  runtime=runtime(), candidate=candidate, prompt=prompt, source_bodies=list(ev.SOURCES),
                  poses=list(range(32, 36)), fresh_strides=args.strides, horizon=ev.HORIZON, proposal_shape=[5, 7],
                  code_sha256={**live, **{f.name: ev.digest_file(f) for f in (HERE / "stride.py", HERE / "stride_rollout.py")}},
                  historical_sources=historical, input_sha256={str(p.relative_to(fit)): ev.digest_file(p) for p in
                      (fit / "manifest.json", fit / "candidates.json", fit / "dev/contract.json", fit / "dev/report.json")},
                  held=list(held.values()), fresh=[], paired_plans=[],
                  summaries={"held-5": summarize(held, held.values())})
    ev.write_json(report_path, report)
    try:
        model = FrozenMotor(args.weights, contract, fit_id).select(fit, candidate)
        for stride in args.strides:
            runner = PairedRollout(model, stride, candidate["sha256"], fit_id, prompt)
            dest = args.out / f"stride-{stride}"
            dest.mkdir()
            current = []
            for (body, pose), original in sorted(held.items()):
                start = perf_counter()
                row = runner.episode(dest, body, pose, expected=original["opening"])
                row.update(tape=str((dest / row["tape"]).relative_to(args.out)),
                           episode_seconds=perf_counter() - start)
                for previous in report["fresh"]:
                    if (previous["body"], previous["pose"]) == (body, pose):
                        comparison = compare_plans(args.out / previous["tape"], args.out / row["tape"])
                        report["paired_plans"].append(dict(body=body, pose=pose, stride_a=previous["stride"],
                                                           stride_b=stride, **comparison))
                if stride == 5:
                    with np.load(original["tape"], allow_pickle=False) as old:
                        if any(ev.digest_array(old[k]) != row["array_sha256"][k] for k in
                               ("actions", "proposals", "motion", "rewards", "done", "success")):
                            raise ValueError(f"Fresh stride-five trajectory differs from the original on {body}/{pose}")
                current.append(row)
                report["fresh"].append(row)
                ev.write_json(report_path, report)
                print(f"{body}/{pose} stride={stride} native={int(row['success'])} "
                      f"controls={row['steps']} calls={row['model_calls']}", flush=True)
            report["summaries"][f"fresh-{stride}"] = summarize(held, current)
        pairs = {key: {"held-5": row} for key, row in held.items()}
        for row in report["fresh"]:
            pairs[row["body"], row["pose"]][f"fresh-{row['stride']}"] = row
        report["paired_success"] = {
            f"{second}_vs_{first}": {
                body: ev.paired_change([pairs[body, pose] for pose in report["poses"]], second, first)
                for body in ev.SOURCES}
            for first, second in combinations(report["summaries"], 2)}
        if ev.digest_file(fit / candidate["file"]) != candidate["sha256"]:
            raise ValueError("Candidate changed during evaluation")
        report["status"] = "complete"
    except Exception as exc:
        report.update(status="error", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        ev.write_json(report_path, report)
    print(f"{ev.digest_file(report_path)}  {report_path}", flush=True)


if __name__ == "__main__":
    main()
