"""Join auxiliary targets to their original cameras by tape digest before sampling."""

import hashlib
import json
from dataclasses import dataclass

import numpy as np
from motor_bank import load_bank
from motor_weights import sha256
from waypoint_motor import LOOKAHEAD, SLOTS


@dataclass
class LabeledEpisode:
    episode: object
    plate: np.ndarray
    motion: np.ndarray
    record: dict

    @property
    def body(self):
        return self.episode.body

    @property
    def pose(self):
        return self.episode.pose

    @property
    def actions(self):
        return self.episode.actions

    def sample(self, t):
        obs, commands, mask = self.episode.sample(t)
        assert commands.shape == (SLOTS, 7)
        idx = np.minimum(t + np.arange(SLOTS), len(self.actions) - 1)
        return obs, commands, mask, self.plate[t], self.motion[idx]


def load_labeled(roots, scratch):
    mappings = []
    hashes = []
    for root in roots:
        raw = (root / "index.json").read_bytes()
        idx = json.loads(raw)
        if (idx.get("status"), idx.get("label_schema"), idx.get("lookahead")) != (
            "complete",
            1,
            LOOKAHEAD,
        ):
            raise ValueError(f"Expected a completed waypoint bank: {root}")
        accepted = [row for row in idx["attempts"] if row["accepted"]]
        if len(accepted) != idx["accepted"]:
            raise ValueError("Completed label count differs from accepted receipts")
        mappings.append({row["file"]: row for row in accepted})
        hashes.append(hashlib.sha256(raw).hexdigest())

    episodes, records, read_hashes = load_bank(roots, scratch)
    if hashes != read_hashes:
        raise ValueError("Source bank changed while staging cameras")
    by_tape = {}
    labeled = []
    for r in records:
        bank = r["bank"]
        row = mappings[bank][r["file"]]
        root = roots[bank].resolve()
        sidecar = (root / row["aux_file"]).resolve(strict=True)
        if (
            not sidecar.is_relative_to(root)
            or sha256(sidecar) != row["aux_sha256"]
            or row["sha256"] != r["sha256"]
        ):
            raise ValueError("Auxiliary tape binding or content changed")
        r.update(
            aux_sha256=row["aux_sha256"],
            aux_file=row["aux_file"],
            label_origin=row["waypoint_provenance"]["mode"],
        )
        key = (r["body"], r["trajectory_sha256"])
        if by_tape.setdefault(key, r["aux_sha256"]) != r["aux_sha256"]:
            raise ValueError("Same camera/state/action sequence has conflicting labels")
    unique = [r for r in records if "duplicate_of" not in r]
    for ep, r in zip(episodes, unique, strict=True):
        root = roots[r["bank"]]
        with np.load(root / r["aux_file"], allow_pickle=False) as aux:
            plate, motion = aux["plate_world_m"], aux["motion_m"]
            n = len(ep.actions)
            indices = np.minimum(np.arange(n) + LOOKAHEAD, n)
            with np.load(root / r["file"], allow_pickle=False) as tape:
                states = np.vstack((ep.state, tape["terminal_state"]))
            waypoint = states[indices, :3]
            if (
                any(
                    a.dtype != np.float32
                    or a.shape != (n, 3)
                    or not np.isfinite(a).all()
                    for a in (plate, motion)
                )
                or not np.array_equal(aux["pre_state"], ep.state)
                or not np.array_equal(aux["future_state_index"], indices)
                or not np.array_equal(waypoint, aux["waypoint_world_m"])
                or not np.array_equal(waypoint - plate, aux["waypoint_minus_plate_m"])
                or not np.array_equal(waypoint - states[:-1, :3], motion)
            ):
                raise ValueError(
                    f"Pre-action label/waypoint binding failed for {r['file']}"
                )
        labeled.append(LabeledEpisode(ep, plate, motion, r))
    labeled.sort(key=lambda e: (e.body, int(e.pose), e.record["trajectory_sha256"]))
    return labeled, records, hashes
