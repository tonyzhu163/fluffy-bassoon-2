"""Cache and reopen frozen proposals from the extended native source bank."""

from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

import adapt_eval as ev
import motor_bank
from large_panel_sim import mobile_joints


@contextmanager
def source_episodes(directory, roots, prompt, reserved):
    with TemporaryDirectory(dir=directory, prefix="expert-frames-") as scratch:
        episodes, records, hashes = motor_bank.load_bank(roots, Path(scratch))
        coverage = motor_bank.coverage(records)
        for row in records:
            # Recovery suffixes share the released reset. Generated replays carry
            # a raw donor, a settled arm state, and sampled scene/placement hashes.
            identities = {
                motor_bank.source_fingerprint(row),
                row["seed"],
                row["initial_state_fingerprint"],
            }
            if "layout_origin" in row:
                origin = row["layout_origin"]
                identities.update((
                    origin["scene_hash"],
                    origin["movable_xy_hash"],
                    mobile_joints(origin["sampled_scene"]),
                ))
            if identities & reserved:
                raise ValueError("Source expert overlaps a reserved evaluation layout")
        unique = [r for r in records if "duplicate_of" not in r]
        if {(r["split"], r["body"]) for r in unique} != {
            (split, body) for split in ("train", "validation") for body in ev.SOURCES
        }:
            raise ValueError("Both source arms need training and released validation experts")
        pairs = list(zip(episodes, unique, strict=True))
        for ep, row in pairs:
            if (ep.body, int(ep.pose), ep.prompt) != (row["body"], row["pose"], prompt) or row["prompt"] != prompt:
                raise ValueError("Source episode and native task receipt disagree")
        indices = [
            {"path": str(root / "index.json"), "sha256": digest}
            for root, digest in zip(roots, hashes, strict=True)
        ]
        yield pairs, {"indices": indices, "coverage": coverage}


def proposal_path(directory, i, row):
    return directory / f"expert-{i:04d}-{row['body']}-{row['pose']}.npz"


def cache_experts(directory, roots, motor, prompt):
    banks = {s: {b: defaultdict(list) for b in ev.SOURCES} for s in ("train", "validation")}
    with source_episodes(directory, roots, prompt, set()) as (pairs, evidence):
        receipts = []
        for i, (ep, row) in enumerate(pairs):
            chunks = []
            for t in range(0, len(ep.actions), ev.EXECUTE):
                obs = ev.public_obs(ep.observation(t))
                chunk = ev.proposal(motor, obs, ev.plan_seed(row["pose"], t))
                if chunk.shape != (ev.EXECUTE, 7):
                    raise ValueError(f"Unexpected frozen motor horizon: {chunk.shape}")
                chunks.append(chunk[:len(ep.actions) - t])
            proposals = np.concatenate(chunks)
            path = proposal_path(directory, i, row)
            np.savez_compressed(path, proposals=proposals, expert=ep.actions)
            banks[row["split"]][ep.body][row["pose"]].append(ev.Query(proposals, ep.actions, path.name))
            receipts.append({**row, "proposal_bank": path.name, "proposal_bank_sha256": ev.digest_file(path)})
            if i == 0 or (i + 1) % 10 == 0 or i + 1 == len(pairs):
                print(f"Cached {i + 1}/{len(pairs)} expert trajectories", flush=True)
    return banks, {**evidence, "episodes": receipts}


def verify_expert_data(directory, data, prompt, reserved):
    roots = []
    for index in data["indices"]:
        path = Path(index["path"])
        if (path.name != "index.json" or str(path.resolve(strict=True)) != index["path"]
                or path.parent in roots or ev.digest_file(path) != index["sha256"]):
            raise ValueError("Changed, redirected or repeated source index")
        roots.append(path.parent)
    with source_episodes(directory, roots, prompt, reserved) as (pairs, evidence):
        if any(data[key] != value for key, value in evidence.items()):
            raise ValueError("Source index hashes or split coverage changed")
        for i, ((ep, row), saved) in enumerate(zip(pairs, data["episodes"], strict=True)):
            path = proposal_path(directory, i, row)
            if (path.resolve(strict=True).parent != directory.resolve()
                    or saved != {**row, "proposal_bank": path.name, "proposal_bank_sha256": ev.digest_file(path)}):
                raise ValueError("Source episode receipt or frozen proposals changed")
            with np.load(path, allow_pickle=False) as cache:
                proposals = motor_bank.numeric(cache, "proposals", ep.actions.shape)
                expert = motor_bank.numeric(cache, "expert", ep.actions.shape)
                if np.abs(proposals).max() > 1 or not np.array_equal(expert, ep.actions):
                    raise ValueError("Invalid proposal or mismatched source expert controls")
