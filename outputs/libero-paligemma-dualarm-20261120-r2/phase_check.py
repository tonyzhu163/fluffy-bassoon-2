"""Ordered offline controls and native replay of a frozen source phase checkpoint."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from phase_fit import BODIES, ledger, load_episodes, prepare
from phase_policy import Memory, PhaseMotor
from motor_weights import sha256
from recover import reset_identity


def summary(rows):
    """Average tapes within layout, then layouts within each arm; preserve missing metrics."""
    result = {}
    for body in sorted({r["body"] for r in rows}):
        layouts = {}
        for pose in sorted({r["pose"] for r in rows if r["body"] == body}):
            subset = [r["metrics"] for r in rows if (r["body"], r["pose"]) == (body, pose)]
            layouts[pose] = {}
            for key in sorted(set().union(*subset)):
                vals = [s[key] for s in subset if s.get(key) is not None]
                if vals:
                    layouts[pose][key] = float(np.mean(vals))
        result[body] = dict(layouts=sorted(layouts), means={
            k: float(np.mean([m[k] for m in layouts.values() if k in m]))
            for k in sorted(set().union(*layouts.values()))})
    return result


@torch.inference_mode()
def tape_metrics(model, ep, donor, micro, mean_action):
    # The VLM input never receives the tape phase. Advance the predicted phase
    # floor in order even though independent observation encodings are batched.
    train_plate = model.head.plate_origin.detach().cpu().numpy()
    stats = {k: [] for k in ("matched", "swapped_rgb")}
    for kind in stats:
        memory = Memory()
        for start in range(0, len(ep.actions), micro):
            ticks = list(range(start, min(start + micro, len(ep.actions))))
            obs = [ep.observation(t) for t in ticks]
            if kind == "swapped_rgb":
                for o, t in zip(obs, ticks):
                    d = donor.observation(min(t, len(donor.actions)-1))
                    o.update(rgb=d["rgb"], wrist=d["wrist"])
            pred = model.predict(obs, ticks)
            data = {k: v.detach().cpu().numpy() for k, v in pred.items()}
            for i, t in enumerate(ticks):
                single = {key: value[i:i+1] for key, value in pred.items()}
                _, decoded = model.decode(single, [memory.phase], [memory.tick])
                phase = int(decoded[0, 0])
                memory.advance([phase], 1)
                cmd = data["actions"][i, 0, phase]
                wrong = data["actions"][i, 0, (phase+1) % 3]
                reference = ep.actions[t]
                if not np.isfinite(cmd).all():
                    raise FloatingPointError("Nonfinite offline control")
                row = dict(translation_mae=float(abs(cmd[:3]-reference[:3]).mean()),
                           angular_mae=float(abs(cmd[3:6]-reference[3:6]).mean()),
                           jaw_mae=float(abs(cmd[6]-reference[6])),
                           action_mae=float(abs(cmd-reference).mean()),
                           wrong_phase_translation_mae=float(abs(wrong[:3]-reference[:3]).mean()),
                           mean_action_translation_mae=float(abs(mean_action[:3]-reference[:3]).mean()))
                replay = donor.actions[min(t, len(donor.actions)-1), :3]
                row["donor_replay_translation_mae"] = float(abs(replay-reference[:3]).mean())
                if ep.stage[t] >= 0:
                    row["phase_correct"] = float(phase == ep.stage[t])
                    row[f"phase_{ep.stage[t]}_translation_mae"] = row["translation_mae"]
                if np.isfinite(ep.plate[t]).all():
                    row["constant_plate_xy_error_m"] = float(np.linalg.norm(train_plate-ep.plate[t]))
                    row["plate_xy_error_m"] = float(np.linalg.norm(data["plate"][i]-ep.plate[t]))
                if np.isfinite(ep.error[t]).all():
                    row["wrist_target_error_m"] = float(np.linalg.norm(data["error"][i, 0, phase]-ep.error[t]))
                row["predicted_phase"] = phase
                stats[kind].append(row)
    metrics = {}
    for condition, seq in stats.items():
        for k in sorted(set().union(*(r.keys() for r in seq)) - {"predicted_phase"}):
            vals = [r[k] for r in seq if k in r]
            metrics[f"{condition}/{k}"] = float(np.mean(vals))
    events = []
    for new_phase in (1, 2):
        # Sparse legacy traces may identify an interval but not an exact switch.
        prior = np.flatnonzero((ep.stage >= 0) & (ep.stage < new_phase))
        later = np.flatnonzero(ep.stage >= new_phase)
        predicted = next((t for t, r in enumerate(stats["matched"])
                          if r["predicted_phase"] >= new_phase), None)
        bounds = [int(prior[-1])+1 if len(prior) else 0, int(later[0])] if len(later) else None
        events.append(dict(enter=new_phase, predicted=predicted, teacher_interval=bounds))
    return dict(body=ep.body, pose=ep.pose, seed=ep.seed, steps=len(ep.actions),
                trajectory=ep.trajectory_sha256, donor=donor.trajectory_sha256,
                teacher_phase_ticks=int((ep.stage >= 0).sum()), switches=events, metrics=metrics)


def offline(args, model, contract, scratch):
    trained = load_episodes(args, "fit", contract["route"], scratch)
    held = load_episodes(args, "holdout", contract["route"], scratch)
    prepare(trained)
    prepare(held)
    expected = [r for r in contract["train"] if r["body"] in args.bodies]
    if ledger(trained) != expected:
        raise ValueError("Offline donors and training receipts differ from checkpoint")
    if not held:
        raise ValueError("No positive held-layout tapes for this route; capture layouts 24–27")
    rows = []
    for ep in held:
        candidates = [e for e in trained if e.body == ep.body and e.pose != ep.pose]
        donor = candidates[ep.pose % len(candidates)]
        mean = np.concatenate([e.actions for e in candidates]).mean(0)
        rows.append(tape_metrics(model, ep, donor, args.microbatch, mean))
    return dict(held=ledger(held), positive_coverage={b: sorted({e.pose for e in held if e.body == b})
                for b in args.bodies}, by_arm=summary(rows), episodes=rows)


def native(args, model):
    from sim import Session
    episodes = []
    for body in args.bodies:
        with Session(body) as env:
            for pose in range(24, 28):
                seed = args.seed + pose
                obs = env.reset(pose, seed)
                physics = reset_identity(env.reset_info, body, pose, seed)
                memory = Memory()
                done = False
                stages, rewards = [], []
                while not done and memory.tick < 300:
                    proposals, phases = model.plan([obs], [memory])
                    commands = proposals[0].float().cpu().numpy()
                    plan = phases[0].cpu().tolist()
                    executed = 0
                    for a in commands[:min(args.stride, 300-memory.tick)]:
                        obs, _, reward, done, success = env.step(a)
                        if reward not in (0, 1) or bool(success) != (reward == 1) or (success and not done):
                            raise ValueError("Native success and reward disagree")
                        rewards.append(float(reward))
                        stages.append(plan[executed])
                        executed += 1
                        if done:
                            break
                    memory.advance(plan, executed)
                if not done or (not success and memory.tick != 300):
                    raise ValueError("Unexpected source episode termination")
                row = dict(body=body, pose=pose, seed=seed, steps=memory.tick, success=bool(success),
                           native_return=sum(rewards), phases=stages, initial_state=physics["initial_state_fingerprint"])
                episodes.append(row)
                print(json.dumps({k: row[k] for k in ("body", "pose", "steps", "success")}), flush=True)
    return dict(stride=args.stride, attempts=episodes, successes={b:sum(r["success"] for r in episodes
                if r["body"] == b) for b in args.bodies}, starts_per_arm=4)


def evaluate(args):
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    contract, step = saved["contract"], saved["step"]
    del saved
    args.bodies = args.bodies or contract["bodies"]
    if (len(args.bodies) != len(set(args.bodies)) or not set(args.bodies) <= set(contract["bodies"])
            or not set(args.bodies) <= set(BODIES)):
        raise ValueError("Use trained source bodies")
    model = PhaseMotor.load(args.checkpoint, args.weights)
    args.output.mkdir(parents=True, exist_ok=False)
    if args.command == "check":
        scratch = args.scratch or args.output
        scratch.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=scratch) as tmp:
            results = offline(args, model, contract, Path(tmp))
    else:
        results = native(args, model)
    report = dict(command=args.command, checkpoint_sha256=sha256(args.checkpoint), step=step, **results)
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
