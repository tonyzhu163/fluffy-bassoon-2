"""Post hoc native geometry from replaying recorded raw actions on source development only."""

import argparse
import json
from pathlib import Path

DEV = tuple(range(32, 36))
ARMS = ("Panda", "Sawyer")
WINDOW = 12
PLATE_TOL_M = 0.0025


def replay(path, digest, body, pose, checkpoint, fit_id, stride, *, legacy):
    import numpy as np
    from stride_rollout import ev, observed, verify_reset

    if body not in ARMS or pose not in DEV:
        raise ValueError("Geometry replay is restricted to Panda/Sawyer development layouts 32..35")
    if ev.digest_file(path) != digest:
        raise ValueError(f"Original tape changed: {path}")
    with np.load(path, allow_pickle=False) as tape:
        row = json.loads(tape["receipt"].item())
        if ((row["body"], row["pose"], row["condition"], row["seed"]) != (body, pose, "raw", ev.seed(pose))
                or (row["motor_checkpoint_sha256"], row["fit_id"]) != (checkpoint, fit_id)
                or row.get("stride", 5) != stride or not 1 <= row["steps"] <= ev.HORIZON):
            raise ValueError(f"Wrong source action tape: {path}")
        commands = ev.real_array(tape["actions"], (row["steps"], 7), "replay controls", bounded=True)
        if not np.array_equal(tape["proposals"], commands):
            raise ValueError("Expected identical raw proposals and executed controls")
        motions, rewards, dones, successes = (tape[k] for k in ("motion", "rewards", "done", "success"))
        checks = {0: row["opening"]["observation"]} if legacy else {
            p["control"]: p["observation"] for p in json.loads(tape["plans"].item())}
    with ev.session(body) as env:
        obs, physics, horizon = ev.reset(env, body, pose)
        verify_reset(env.reset_info, body, pose, ev.seed(pose))
        if (physics != row["opening"]["physics"] or horizon != ev.HORIZON
                or checks[0] != observed(obs) or row["opening"]["horizon"] != horizon):
            raise ValueError(f"Different native replay opening: {body}/{pose}")
        plate, tcp, goal, contact = [], [], [], []

        def snapshot():
            scene = env.expert_scene()
            plate.append(scene["plate"])
            tcp.append(scene["tcp"])
            goal.append(scene["goal"])
            contact.append(scene["contact"])

        snapshot()
        for t, action in enumerate(commands):
            if t in checks and observed(ev.public_obs(obs)) != checks[t]:
                raise ValueError(f"Replay public observation differs at {body}/{pose}/{t}")
            obs, delta, reward, done, success = ev.step(env, action)
            if (not np.array_equal(delta, motions[t]) or (reward, done, success) != (rewards[t], dones[t], successes[t])
                    or done != (t + 1 == len(commands))):
                raise ValueError(f"Replay differs from recorded native transition at {body}/{pose}/{t}")
            snapshot()
    trace = dict(control=np.arange(len(commands) + 1), plate_xyz_m=np.asarray(plate), tcp_xyz_m=np.asarray(tcp),
                 goal_site_xyz_m=np.asarray(goal), finger_plate_contact=np.asarray(contact))
    p, tcp, g = (trace[k] for k in ("plate_xyz_m", "tcp_xyz_m", "goal_site_xyz_m"))
    if not all(np.isfinite(x).all() for x in (p, tcp, g)):
        raise ValueError("Nonfinite replay geometry")
    tcp_plate = np.linalg.norm(tcp - p, axis=1)
    goal_plate = np.linalg.norm((p - g)[:, :2], axis=1)
    # Maximum pairwise XY separation prevents a plate that travelled out and back counting as stationary.
    plateau = np.zeros(len(p), dtype=bool)
    for t in range(WINDOW, len(p)):
        xy = p[t - WINDOW:t + 1, :2]
        plateau[t] = np.linalg.norm(xy[:, None] - xy[None, :], axis=2).max() <= PLATE_TOL_M
    starts = np.flatnonzero(plateau)
    trace.update(tcp_plate_m=tcp_plate, plate_goal_xy_m=goal_plate, plateau_window_end=plateau)
    metrics = dict(success=row["success"], controls=len(commands), closest_tcp_plate_m=float(tcp_plate.min()),
                   closest_tcp_control=int(tcp_plate.argmin()), closest_plate_goal_xy_m=float(goal_plate.min()),
                   closest_goal_control=int(goal_plate.argmin()), final_plate_goal_xy_m=float(goal_plate[-1]),
                   initial_plate_goal_xy_m=float(goal_plate[0]),
                   first_plateau_window_end=int(starts[0]) if len(starts) else None,
                   plateau_windows=int(plateau.sum()), eligible_windows=max(0, len(p) - WINDOW),
                   finger_contact_controls=int(sum(contact[1:])))
    return trace, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path, help="complete stride.py report.json")
    parser.add_argument("--out", required=True, type=Path, help="fresh post hoc output directory")
    args = parser.parse_args()
    import numpy as np
    from stride_rollout import HERE, PRIOR, ev

    report = json.loads(args.report.read_bytes())
    if (report["status"] != "complete" or report["stage"] != "dev-stride" or report["poses"] != list(DEV)
            or report["source_bodies"] != list(ARMS)):
        raise ValueError("Replay needs the completed source development stride comparison")
    code = [HERE / "stride_replay.py", HERE / "stride_rollout.py", HERE / "motor_bank.py",
            PRIOR / "sim.py", PRIOR / "adapt_eval.py", PRIOR / "pretrain_reader.py"]
    code.extend(HERE.parent / "libero-crossarm-20261120-r1" / name for name in ("arm_env.py", "sim_reset.py"))
    hashes = {p.name: ev.digest_file(p) for p in code}
    if any(h != report["code_sha256"][k] for k, h in hashes.items() if k != "stride_replay.py"):
        raise ValueError("Replay and original native implementations differ")
    args.out.mkdir(parents=True, exist_ok=False)
    out = dict(status="running", stride_report_sha256=ev.digest_file(args.report), code_sha256=hashes,
               plateau_window_controls=WINDOW, plateau_max_xy_diameter_m=PLATE_TOL_M, episodes=[], by_arm={})
    records = [("held-5", r) for r in report["held"]]
    records.extend((f"fresh-{r['stride']}", r) for r in report["fresh"])
    expected = {(name, b, p) for name in ("held-5", *(f"fresh-{s}" for s in report["fresh_strides"]))
                for b in ARMS for p in DEV}
    keys = [(name, r["body"], r["pose"]) for name, r in records]
    if set(keys) != expected or len(keys) != len(expected):
        raise ValueError("Expected one raw source tape per arm, layout and execution setting")
    held = {}
    fields = ("closest_tcp_plate_m", "closest_plate_goal_xy_m", "final_plate_goal_xy_m", "plateau_windows")
    path = args.out / "report.json"
    try:
        for name, row in records:
            body, pose = row["body"], row["pose"]
            source = args.report.parent / row["tape"]
            trace, scores = replay(source, row["tape_sha256"], body, pose, report["candidate"]["sha256"],
                                   report["fit_id"], row["stride"], legacy=name == "held-5")
            dest = args.out / f"{name}-{body}-{pose}.npz"
            np.savez_compressed(dest, **trace)
            if name == "held-5":
                held[body, pose] = scores
            diff = {k: scores[k] - held[body, pose][k] for k in fields}
            out["episodes"].append(dict(setting=name, body=body, pose=pose, action_tape_sha256=row["tape_sha256"],
                                        trace=dest.name, trace_sha256=ev.digest_file(dest), **scores,
                                        difference_vs_held=diff))
            ev.write_json(path, out)
        for name, body in sorted({(r["setting"], r["body"]) for r in out["episodes"]}):
            arm = [r for r in out["episodes"] if (r["setting"], r["body"]) == (name, body)]
            out["by_arm"].setdefault(name, {})[body] = dict(
                episodes=len(arm), median={k: float(np.median([r[k] for r in arm])) for k in fields},
                median_paired_difference_vs_held={k: float(np.median([r["difference_vs_held"][k] for r in arm]))
                                                  for k in fields},
                plateau_windows=sum(r["plateau_windows"] for r in arm),
                eligible_windows=sum(r["eligible_windows"] for r in arm))
        out["status"] = "complete"
    except Exception as exc:
        out.update(status="error", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        ev.write_json(path, out)
    print(path, flush=True)


if __name__ == "__main__":
    main()
