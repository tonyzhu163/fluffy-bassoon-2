"""Fixed native source panels, resumable episode receipts and checkpoint promotion."""

import json
import shutil

import numpy as np
import torch

import adapt_eval as ae
import motor as base
from motor_bank import BODIES, NATIVE_RESET, OFFLINE, TRAIN, verify_reset
from motor_weights import sha256

PANELS = {"dev": (tuple(range(32, 36)), 3), "confirm": (tuple(range(40, 48)), 6)}


def read(path):
    return json.loads(path.read_text())


def bind(path, value):
    """Publish the stage inputs before its first reset and preserve them across retries."""
    if path.exists():
        if read(path) != value:
            raise ValueError(f"Immutable stage inputs differ at {path}")
    else:
        ae.write_json(path, value)


def audit_fit(out, sources):
    manifest = out / "manifest.json"
    fit_id = sha256(manifest)
    contract = read(manifest)["contract"]
    candidates = read(out / "candidates.json")
    if (contract["protocol"] != "dualarm-native-goal5-v2" or contract["source_bodies"] != list(BODIES)
            or contract["sources"] != sources or candidates["fit_id"] != fit_id
            or candidates["status"] != "complete"):
        raise ValueError("Need a completed dual-arm fit with the same motor and evaluation implementation")
    rows = contract["episodes"]
    prompts = {r["prompt"] for r in rows}
    for r in rows:
        split = "train" if r["pose"] in TRAIN else "validation" if r["pose"] in OFFLINE else None
        if r["body"] not in BODIES or split is None or r["split"] != split:
            raise ValueError("Motor used an unauthorized arm or pose")
    if len(prompts) != 1 or any(not any(r["body"] == b and r["split"] == s for r in rows)
                                for b in BODIES for s in ("train", "validation")):
        raise ValueError("Incomplete dual-arm fit or inconsistent prompt")
    snaps = candidates["snapshots"]
    recipe = contract["recipe"]
    if [s["step"] for s in snaps] != recipe["candidate_steps"]:
        raise ValueError("Candidate index differs from registered update schedule")
    shortlist = sorted(snaps, key=lambda s: (s["validation"]["mae"], s["step"]))[:recipe["online_shortlist"]]
    if [s["step"] for s in shortlist] != candidates["shortlist"]:
        raise ValueError("Candidate shortlist differs from offline action error selection")
    for s in shortlist:
        if s["file"] != f"candidates/step-{s['step']:06d}.pt" or sha256(out / s["file"]) != s["sha256"]:
            raise ValueError("Candidate checkpoint digest disagrees with fit")
    return contract, fit_id, shortlist, prompts.pop()


class FrozenMotor:
    """Load the verified 3B base once. Swap only the supervised deltas between candidates."""

    def __init__(self, weights, contract, fit_id):
        from servo_motor import architecture_name, build_motor
        self.model = build_motor(weights, architecture_name(contract))
        if self.model.identity != contract["weights"]:
            raise ValueError("Evaluation base weights differ from training")
        self.trained = [p for p in self.model.parameters() if p.requires_grad]
        self.contract = contract
        self.fit_id = fit_id
        self.current = None

    @torch.no_grad()
    def select(self, out, candidate):
        if self.current == candidate["sha256"]:
            return self.model
        saved = torch.load(out / candidate["file"], map_location="cpu", weights_only=True, mmap=True)
        best = saved["best"]
        if (saved["schema"] != base.SCHEMA or saved["contract"] != self.contract or saved["fit_id"] != self.fit_id
                or saved["step"] != best["step"] or best["step"] != candidate["step"]
                or best["mae"] != candidate["validation"]["mae"]):
            raise ValueError("Candidate contract or selected update differs")
        for p in self.trained:
            p.requires_grad_(True)
        self.model.restore_delta(best["model"])
        self.model.requires_grad_(False).eval()
        self.current = candidate["sha256"]
        return self.model


@torch.inference_mode()
def rollout(path, env, model, body, pose, prompt, checkpoint, fit_id):
    obs, physics, horizon = ae.reset(env, body, pose)
    verify_reset(env.reset_info, body, pose, ae.seed(pose))
    if horizon != ae.HORIZON or obs["prompt"] != prompt:
        raise ValueError("Native horizon or instruction differs from the registered task")
    first = {k: (v if k == "prompt" else ae.digest_array(v)) for k, v in obs.items()}
    opening = dict(observation=first, physics=physics, horizon=horizon)
    actions, proposals, motions, rewards, dones, successes = [], [], [], [], [], []
    while len(actions) < horizon:
        raw = ae.real_array(model.act(obs), (ae.EXECUTE, 7), "frozen proposal", bounded=True)
        if not actions:
            opening.update(chunk=raw.tolist(), chunk_sha256=ae.digest_array(raw))
        for action in raw[:horizon - len(actions)]:
            proposals.append(action.copy())
            issued = action.copy()
            nxt, delta, reward, done, success = ae.step(env, issued)
            actions.append(issued)
            motions.append(delta)
            rewards.append(reward)
            dones.append(done)
            successes.append(success)
            if done or success:
                break
        if done or success or len(actions) == horizon:
            break
        obs = ae.public_obs(nxt)
        if obs["prompt"] != prompt:
            raise ValueError("Goal instruction changed during a closed-loop episode")
    row = dict(body=body, pose=pose, seed=ae.seed(pose), condition="raw", fit_id=fit_id,
               motor_checkpoint_sha256=checkpoint, opening=opening,
               steps=len(actions), success=bool(successes[-1]), done=bool(dones[-1]),
               native_return=float(sum(rewards)), terminal_reward=float(rewards[-1]))
    tmp = path.with_suffix(".npz.tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, receipt=json.dumps(row, sort_keys=True, allow_nan=False), proposals=proposals,
                            actions=actions, motion=motions, rewards=rewards, done=dones, success=successes)
    tmp.replace(path)


def check_episode(path, stage_dir, body, pose, prompt, candidate, fit_id, prior_digest):
    digest = sha256(path)
    if prior_digest is not None and digest != prior_digest:
        raise ValueError(f"Previously committed native tape changed: {path}")
    with np.load(path, allow_pickle=False) as tape:
        row = json.loads(tape["receipt"].item())
        n = row["steps"]
        actions = ae.real_array(tape["actions"], (n, 7), "issued actions", bounded=True)
        proposals = ae.real_array(tape["proposals"], (n, 7), "proposals", bounded=True)
        ae.real_array(tape["motion"], (n, 7), "measured motion")
        rewards = ae.real_array(tape["rewards"], (n,), "rewards")
        success, done = tape["success"], tape["done"]
        opening = row["opening"]
        chunk = ae.real_array(opening["chunk"], (ae.EXECUTE, 7), "opening proposal", bounded=True)
        first = min(ae.EXECUTE, n)
        if (not 1 <= n <= ae.HORIZON or any(a.shape != (n,) or a.dtype != np.bool_ for a in (success, done))
                or ae.digest_array(chunk) != opening["chunk_sha256"]
                or not np.array_equal(chunk[:first], proposals[:first])
                or not np.array_equal(actions, proposals) or not np.array_equal(success, rewards > 0)
                or np.any(rewards[:-1] != 0) or done[:-1].any() or (success[-1] and not done[-1])
                or not done[-1] or (n < ae.HORIZON and not success[-1]) or row["success"] != bool(success[-1])
                or row["done"] != bool(done[-1]) or row["terminal_reward"] != float(rewards[-1])
                or row["native_return"] != float(rewards.sum())):
            raise ValueError(f"Invalid native trace or assisted action in {path}")
    if ((row["body"], row["pose"], row["seed"], row["condition"]) != (body, pose, ae.seed(pose), "raw")
            or row["fit_id"] != fit_id or row["motor_checkpoint_sha256"] != candidate["sha256"]
            or row["opening"]["observation"]["prompt"] != prompt or row["opening"]["horizon"] != ae.HORIZON
            or any(row["opening"]["physics"].get(k) != v for k, v in NATIVE_RESET.items())):
        raise ValueError(f"Native tape provenance differs from registered panel: {path}")
    return {**row, "step": candidate["step"], "tape": str(path.relative_to(stage_dir)), "tape_sha256": digest}


def score_panel(args, stage_dir, stage_contract, candidates, prompt, contract):
    poses, threshold = PANELS[args.stage]
    bind(stage_dir / "contract.json", stage_contract)
    report_path = stage_dir / "report.json"
    old = read(report_path) if report_path.exists() else {}
    if old and old.get("contract_sha256") != sha256(stage_dir / "contract.json"):
        raise ValueError("Existing panel report belongs to another contract")
    previous = {r["tape"]: r["tape_sha256"] for r in old.get("episodes", [])}
    report = dict(stage=args.stage, status="running", source_gate=False, fit_id=stage_contract["fit_id"],
                  contract_sha256=sha256(stage_dir / "contract.json"), revision=args.revision,
                  source_bodies=list(BODIES), poses=list(poses), required_per_body=threshold,
                  episodes=[], scores=[])
    if args.stage == "confirm":
        report.update(motor_checkpoint_sha256=candidates[0]["sha256"], selected_step=candidates[0]["step"])
    backend = None
    openings = {}
    used_rows = {r["released_row_fingerprint"]: r["pose"] for r in contract["episodes"]}
    used_rows.update(stage_contract.get("development_reset_rows", {}))
    panel_rows = {}
    for c in candidates:
        dest = stage_dir / f"step-{c['step']:06d}"
        dest.mkdir(exist_ok=True)
        counts = {}
        for body in BODIES:
            env = None
            rows = []
            try:
                for pose in poses:
                    path = dest / f"query-{body}-{pose}-raw.npz"
                    rel = str(path.relative_to(stage_dir))
                    if not path.exists():
                        if rel in previous or old.get("status") == "complete":
                            raise ValueError(f"Missing committed tape {path}")
                        if backend is None:
                            backend = FrozenMotor(args.weights, contract, report["fit_id"])
                        model = backend.select(args.output, c)
                        if env is None:
                            from sim import Session
                            env = Session(body, task_id=ae.TASK)
                        rollout(path, env, model, body, pose, prompt, c["sha256"], report["fit_id"])
                    row = check_episode(path, stage_dir, body, pose, prompt, c, report["fit_id"], previous.get(rel))
                    initial = {k: row["opening"][k] for k in ("observation", "physics", "horizon")}
                    if openings.setdefault((body, pose), initial) != initial:
                        raise ValueError(f"Unpaired initial observation/physics on {body}/{pose}")
                    released = initial["physics"]["released_row_fingerprint"]
                    if used_rows.setdefault(released, pose) != pose or panel_rows.setdefault(pose, released) != released:
                        raise ValueError(f"Repeated or inconsistent released reset row on {body}/{pose}")
                    rows.append(row)
                    report["episodes"].append(row)
                    if not old or (rel not in previous and old.get("status") != "complete"):
                        ae.write_json(report_path, report)
                        print(json.dumps(dict(stage=args.stage, step=c["step"], body=body, pose=pose,
                                              success=row["success"], controls=row["steps"])), flush=True)
            finally:
                if env is not None:
                    env.close()
            counts[body] = sum(int(r["success"]) for r in rows)
        report["scores"].append(dict(step=c["step"], sha256=c["sha256"], native_successes=counts,
                                     offline_mae=c["validation"]["mae"], passed=min(counts.values()) >= threshold))
    return report


def promote(out, candidate):
    selected = out / "selected"
    selected.mkdir(exist_ok=True)
    target = selected / "checkpoint.pt"
    if not target.exists():
        tmp = target.with_suffix(".pt.tmp")
        shutil.copyfile(out / candidate["file"], tmp)
        if sha256(tmp) != candidate["sha256"]:
            raise ValueError("Selected checkpoint copy differs from development winner")
        tmp.replace(target)
    if sha256(target) != candidate["sha256"]:
        raise ValueError("Existing selected checkpoint differs from development winner")
    (selected / "motor.lock").touch(exist_ok=True)


def rank(row):
    counts = row["native_successes"].values()
    return -min(counts), -sum(counts), row["offline_mae"], row["step"]


def finish(args, stage_dir, stage_contract, candidates, prompt, contract):
    report = score_panel(args, stage_dir, stage_contract, candidates, prompt, contract)
    winner = min(report["scores"], key=rank)
    report.update(status="complete", source_gate=winner["passed"], native_successes=winner["native_successes"])
    if args.stage == "dev" and winner["passed"]:
        candidate = next(c for c in candidates if c["step"] == winner["step"])
        promote(args.output, candidate)
        report.update(motor_checkpoint_sha256=winner["sha256"], selected_step=winner["step"])
    path = stage_dir / "report.json"
    if path.exists() and read(path).get("status") == "complete" and read(path) != report:
        raise ValueError("Completed report changed on readback")
    ae.write_json(path, report)
    print(json.dumps(dict(stage=args.stage, source_gate=report["source_gate"],
                          native_successes=report["native_successes"]), sort_keys=True), flush=True)
    return 0 if report["source_gate"] else 2


def run(args, sources, runtime, exclusive):
    base.require_gpu()
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    contract, fit_id, candidates, prompt = audit_fit(args.output, sources)
    stage_dir = args.output / args.stage
    stage_dir.mkdir(exist_ok=True)
    stage_contract = dict(stage=args.stage, fit_id=fit_id, revision=args.revision, sources=sources,
                          runtime=runtime, source_bodies=list(BODIES), prompt=prompt, seed_base=ae.seed(0),
                          poses=list(PANELS[args.stage][0]), required_per_body=PANELS[args.stage][1],
                          execute=ae.EXECUTE, horizon=ae.HORIZON, base_weights=contract["weights"])
    if args.stage == "dev":
        stage_contract["candidates"] = candidates
        return finish(args, stage_dir, stage_contract, candidates, prompt, contract)
    dev_dir = args.output / "dev"
    dev = read(dev_dir / "report.json")
    dev_contract = read(dev_dir / "contract.json")
    if (dev["status"] != "complete" or not dev["source_gate"] or dev["stage"] != "dev"
            or dev["fit_id"] != fit_id or dev["contract_sha256"] != sha256(dev_dir / "contract.json")
            or dev_contract["candidates"] != candidates or dev_contract["sources"] != sources):
        raise ValueError("Sealed confirmation requires a completed passing source development panel")
    # Re-derive the development winner from every registered native tape before opening the seal.
    from argparse import Namespace
    dev_args = Namespace(**{**vars(args), "stage": "dev", "revision": dev_contract["revision"]})
    audited = score_panel(dev_args, dev_dir, dev_contract, candidates, prompt, contract)
    winner = min(audited["scores"], key=rank)
    if (not winner["passed"] or audited["scores"] != dev["scores"] or audited["episodes"] != dev["episodes"]
            or winner["sha256"] != dev["motor_checkpoint_sha256"] or winner["step"] != dev["selected_step"]
            or sha256(args.output / "selected" / "checkpoint.pt") != winner["sha256"]):
        raise ValueError("Development winner, native evidence, and selected checkpoint disagree")
    selected = next(c for c in candidates if c["step"] == winner["step"])
    selected = {**selected, "file": "selected/checkpoint.pt"}
    stage_contract.update(candidates=[selected], development_report_sha256=sha256(dev_dir / "report.json"),
                          development_reset_rows={r["opening"]["physics"]["released_row_fingerprint"]: r["pose"]
                                                  for r in dev["episodes"]},
                          seal_dir=str(args.seal_dir.resolve()))
    with exclusive(args.seal_dir):
        seal = dict(fit_dir=str(args.output), fit_id=fit_id, selected_step=winner["step"],
                    motor_checkpoint_sha256=winner["sha256"], confirmation=stage_contract)
        bind(args.seal_dir / "selection.json", seal)
        return finish(args, stage_dir, stage_contract, [selected], prompt, contract)
