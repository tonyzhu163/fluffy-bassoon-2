"""Source-only reader initialization, expert proposal banks and joint fitting."""

import json
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

import adapt_eval as ev
import pretrain_reader as s1
import motor_bank


TRAIN = range(28)
VALID = range(28, 32)
STEPS = 150
PER_BODY = 16
LR = 0.003


def load_stage1(directory, reserved):
    path = directory / "report.json"
    record = json.loads(path.read_bytes())
    checkpoint = directory / "reader-stage1.pt"
    digest = ev.digest_file(checkpoint)
    expected = {
        "status": "complete", "stage": "source_motion", "task_id": ev.TASK,
        "train_bodies": list(ev.SOURCES), "fit_seed": ev.FIT_SEED,
        "support_poses": list(ev.SUPPORT), "motion_query_poses": list(ev.MOTION_QUERY),
        "validation_pose": s1.CHECK_POSE, "practice_transitions": ev.K, "steps": s1.STEPS,
        "batch_size": s1.BATCH_SIZE, "checkpoint": checkpoint.name, "checkpoint_sha256": digest,
        "source_histories": {
            b: {"supports": [f"practice-{b}-{p}.npz" for p in ev.SUPPORT],
                "motion_queries": [f"practice-{b}-{p}.npz" for p in ev.MOTION_QUERY]}
            for b in ev.SOURCES
        },
    }
    if any(record.get(k) != v for k, v in expected.items()):
        raise ValueError("Stage 1 provenance differs from the completed two-source motion fit")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if any(saved.get(k) != v for k, v in {
        "stage": "source_motion", "version": 1, "revision": record["revision"],
        "steps": s1.STEPS, "batch_size": s1.BATCH_SIZE, "fit_seed": ev.FIT_SEED,
    }.items()):
        raise ValueError("Stage 1 checkpoint metadata disagrees with its report")
    net = ev.ReaderAdapter()
    net.load_state_dict(saved["state_dict"])
    if (not all(torch.isfinite(v).all() for v in net.state_dict().values())
            or any(torch.count_nonzero(p) for p in net.adapter.parameters())
            or not (net.motion_scale > 0).all()
            or not np.array_equal(net.motion_scale.numpy(), record["motion_scale"])):
        raise ValueError("Expected a finite motion-pretrained reader and an untouched action head")
    net.response.requires_grad_(False)
    evidence, histories = {}, {}
    for body in ev.SOURCES:
        histories[body] = {}
        for pose in (*ev.SUPPORT, *ev.MOTION_QUERY, s1.CHECK_POSE):
            name = f"practice-{body}-{pose}.npz"
            evidence[name] = ev.digest_file(directory / name)
            if evidence[name] != record["tapes"][name]["sha256"]:
                raise ValueError(f"Stage 1 source tape changed: {name}")
            with np.load(directory / name, allow_pickle=False) as tape:
                physics = json.loads(tape["reset_json"].item())
                if physics["released_row_fingerprint"] in reserved:
                    raise ValueError(f"Stage 1 tape overlaps the released confirmation layouts: {name}")
            histories[body][pose] = s1.read_tape(directory, body, pose)
    if set(record["tapes"]) != set(evidence):
        raise ValueError("Unexpected tape in the Stage 1 provenance")
    for pose in (*ev.SUPPORT, *ev.MOTION_QUERY, s1.CHECK_POSE):
        if not np.array_equal(*(histories[b][pose].commands for b in ev.SOURCES)):
            raise ValueError(f"Source probes differ across bodies on layout {pose}")
    motion = np.concatenate([histories[b][p].motion for b in ev.SOURCES for p in ev.SUPPORT])
    scale = np.maximum(np.sqrt(np.mean(motion.astype(np.float64) ** 2, axis=0)), 1e-6)
    if not np.array_equal(scale.astype(np.float32), net.motion_scale.numpy()):
        raise ValueError("Reader normalization differs from the measured source supports")
    return net, histories, {
        "directory": str(directory.resolve()), "checkpoint_sha256": digest, "revision": record["revision"],
        "report_sha256": ev.digest_file(path), "tape_sha256": evidence,
    }


def check_reserved(records, reserved):
    for row in records:
        identities = {motor_bank.source_fingerprint(row), row["seed"]}
        if "layout_origin" in row:
            identities.update(row["layout_origin"][k] for k in ("scene_hash", "movable_xy_hash"))
        if identities & reserved:
            raise ValueError("A source expert overlaps a reserved evaluation layout")


@torch.inference_mode()
def cache_experts(out, directories, motor, prompt, reserved):
    # Use the two-arm trainer's loader for validation of native reset and reward receipts,
    # exact trajectory deduplication, and aligned public pre-action camera streams.
    banks = {split: {b: defaultdict(list) for b in ev.SOURCES} for split in ("train", "validation")}
    receipts = []
    with TemporaryDirectory(dir=out, prefix="expert-frames-") as scratch:
        episodes, records, indices = motor_bank.load_bank(directories, Path(scratch))
        coverage = motor_bank.coverage(records)
        check_reserved(records, reserved)
        unique = [r for r in records if "duplicate_of" not in r]
        for split in banks:
            if {r["body"] for r in unique if r["split"] == split} != set(ev.SOURCES):
                raise ValueError(f"Need native expert labels from both bodies in {split}")
        for i, (ep, row) in enumerate(zip(episodes, unique, strict=True)):
            body, pose = ep.body, int(ep.pose)
            if (row["body"], row["pose"], row["prompt"]) != (body, pose, prompt):
                raise ValueError("Source episode metadata differs from the confirmed task")
            raw = []
            for t in range(0, len(ep.actions), ev.EXECUTE):
                chunk = ev.proposal(motor, ev.public_obs(ep.observation(t)), ev.plan_seed(pose, t))
                if chunk.shape != (ev.EXECUTE, 7):
                    raise ValueError(f"Unexpected frozen motor horizon: {chunk.shape}")
                raw.append(chunk[:len(ep.actions) - t])
            raw = np.concatenate(raw)
            name = f"expert-{i:04d}-{body}-{pose}.npz"
            np.savez_compressed(out / name, proposals=raw, expert=ep.actions)
            banks[row["split"]][body][pose].append(ev.Query(raw, ep.actions, name))
            receipts.append({**row, "proposal_bank": name, "proposal_bank_sha256": ev.digest_file(out / name)})
            if i == 0 or (i + 1) % 10 == 0 or i + 1 == len(unique):
                print(f"Cached {i + 1}/{len(unique)} expert trajectories", flush=True)
    source_indices = [{"path": str((d / "index.json").resolve()), "sha256": h}
                      for d, h in zip(directories, indices, strict=True)]
    return banks, {"indices": source_indices, "coverage": coverage, "episodes": receipts}


def verify_expert_data(directory, data, prompt, reserved):
    """Reopen the exact collector indices and match source labels and proposal caches."""
    roots = []
    for index in data["indices"]:
        path = Path(index["path"])
        if (path.name != "index.json" or str(path.resolve(strict=True)) != index["path"]
                or ev.digest_file(path) != index["sha256"] or path.parent in roots):
            raise ValueError("Changed, redirected or repeated source expert index")
        roots.append(path.parent)
    with TemporaryDirectory(dir=directory, prefix="expert-readback-") as scratch:
        episodes, records, hashes = motor_bank.load_bank(roots, Path(scratch))
        if (hashes != [r["sha256"] for r in data["indices"]]
                or motor_bank.coverage(records) != data["coverage"]):
            raise ValueError("Source bank indices or training/validation coverage changed")
        check_reserved(records, reserved)
        unique = [r for r in records if "duplicate_of" not in r]
        for i, (ep, row, saved) in enumerate(zip(episodes, unique, data["episodes"], strict=True)):
            name = f"expert-{i:04d}-{ep.body}-{int(ep.pose)}.npz"
            path = motor_bank.bank_path(directory.resolve(), name, ".npz")
            if (row["prompt"] != prompt or saved != {**row, "proposal_bank": name,
                                                     "proposal_bank_sha256": ev.digest_file(path)}):
                raise ValueError("Saved source identity, native tape or frozen proposals changed")
            with np.load(path, allow_pickle=False) as cache:
                proposals = motor_bank.numeric(cache, "proposals", ep.actions.shape)
                if np.abs(proposals).max() > 1 or not np.array_equal(cache["expert"], ep.actions):
                    raise ValueError("Invalid proposal bank or changed native expert actions")
        if {(r["body"], r["split"]) for r in unique} != {
                (body, split) for body in ev.SOURCES for split in ("train", "validation")}:
            raise ValueError("Source fit needs both arms and released validation")


def joint_fit(net, histories, banks):
    rng = np.random.default_rng(ev.FIT_SEED)
    modules = {"reader": net.reader, "adapter": net.adapter}
    original = {k: [p.detach().clone() for p in mod.parameters()] for k, mod in modules.items()}
    params = [p for mod in modules.values() for p in mod.parameters()]
    optimizer = torch.optim.Adam(params, lr=LR)
    trace = []
    for step in range(1, STEPS + 1):
        losses = []
        # Share the random support choices and dropout across arms. Exclude the
        # query layout, including when a source demonstration is on 20 or 21.
        support_draws = rng.integers(len(ev.SUPPORT), size=PER_BODY)
        keep = torch.from_numpy((rng.random(PER_BODY) >= 0.125).astype(np.float32))[:, None]
        for body in ev.SOURCES:
            bank = banks["train"][body]
            x, y, practice = [], [], []
            for pose, draw in zip(rng.choice(sorted(bank), PER_BODY), support_draws, strict=True):
                demos = bank[pose]
                demo = demos[rng.integers(len(demos))]
                t = rng.integers(len(demo.x))
                x.append(demo.x[t])
                y.append(demo.y[t])
                supports = [p for p in ev.SUPPORT if p != pose]
                practice.append(histories[body][supports[draw % len(supports)]].tensor())
            z = net.encode(torch.stack(practice))
            pred = net.adapt(z * keep, torch.from_numpy(np.stack(x)))
            losses.append((pred - torch.from_numpy(np.stack(y))).square().mean())
        optimizer.zero_grad(set_to_none=True)
        torch.stack(losses).mean().backward()
        norm = torch.nn.utils.clip_grad_norm_(params, 5.0, error_if_nonfinite=True)
        optimizer.step()
        trace.append({"step": step, "grad_norm": norm.item(),
                      "action_mse": {b: v.item() for b, v in zip(ev.SOURCES, losses, strict=True)}})
    changes = {k: sum((p.detach().double() - start.double()).square().sum()
                      for p, start in zip(mod.parameters(), original[k], strict=True)).sqrt().item()
               for k, mod in modules.items()}
    if (not all(np.isfinite(v) and v > 0 for v in changes.values())
            or not all(t.isfinite().all() for t in net.state_dict().values())):
        raise ValueError("Stage 2 requires finite weights and updates to both reader and residual")
    return {"seed": ev.FIT_SEED, "steps": STEPS, "per_body_per_update": PER_BODY,
            "optimizer": "Adam", "lr": LR, "context_dropout": 0.125, "clip_norm": 5.0,
            "parameter_change_l2": changes, "curve": trace}


@torch.inference_mode()
def crossed_scores(net, histories, valid):
    """Offline expert action MSE, equal weight per episode within layout, then per layout."""
    rows, by_body = [], {}
    for body in ev.SOURCES:
        layout_scores = []
        for pose, demos in sorted(valid[body].items()):
            sums = {k: [] for k in ("raw", "empty", *ev.SOURCES)}
            for demo in demos:
                # Stage 1 did not update on 28. Avoid same-layout support for query 28.
                support_ids = ev.SUPPORT if pose == s1.CHECK_POSE else (s1.CHECK_POSE,)
                for support in support_ids:
                    transforms = {b: ev.learned(net, histories[b][support]) for b in ev.SOURCES}
                    transforms["empty"] = ev.learned(net, None)
                    actions = {"raw": demo.x, **{k: fn(demo.x) for k, fn in transforms.items()}}
                    scores = {k: float(np.square(x - demo.y).mean()) for k, x in actions.items()}
                    if not np.isfinite(list(scores.values())).all():
                        raise ValueError("Nonfinite crossed-body expert score")
                    rows.append({"body": body, "pose": pose, "support_pose": support,
                                 "bank": demo.path, "action_mse": scores})
                    for k, value in scores.items():
                        sums[k].append(value)
            layout_scores.append({k: np.mean(values) for k, values in sums.items()})
        by_body[body] = {k: float(np.mean([r[k] for r in layout_scores])) for k in layout_scores[0]}
    return {"rows": rows, "per_query_body": by_body}
