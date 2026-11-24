"""Replay donor command tapes on Kinova3 with cameras on and keep the public observations.

Runs on the fit host through the same simulator interface as the panel, in five-control
chunks, so rows are the states at which the frozen policy would replan. A replay is fit
data only if it earns native reward on this host. Not executed on the authoring VM.
"""
import hashlib
import json
from pathlib import Path
import tarfile

import numpy as np

from harness import TASK, array_digest, public

ARCHIVE = Path(__file__).resolve().parents[1] / "frozen-transfer-donor-20261124-r1/inputs/raw-tapes.tar.gz"
ARCHIVE_SHA = "5b8b787f7715a5c5ba5eb3a1f16a25d4463222589ca99cac1fd17901d37ee750"
# (Kinova3 layout, IIWA donor layout). All earned reward in the CPU replays.
FIT = ((34, 34), (34, 35), (35, 35), (37, 35), (14, 34))
HELDOUT = ((31, 35),)
HORIZON = 10


def tapes():
    blob = ARCHIVE.read_bytes()
    if hashlib.sha256(blob).hexdigest() != ARCHIVE_SHA:
        raise ValueError("Donor archive differs from the recorded one")
    out = {}
    with tarfile.open(ARCHIVE) as archive:
        for layout in (34, 35):
            tape = json.loads(archive.extractfile(f"gpu01/episodes/IIWA-{layout}-raw.json").read())
            commands = np.array([row for call in tape["calls"] for row in call["sent"]], dtype=np.float32)
            if not tape["native"]["success"] or commands.shape != (tape["native"]["controls"], 7):
                raise ValueError("Donor tape is not a complete successful episode")
            out[layout] = commands
    return out


def replay(sim, layout, commands):
    """Returns (rows, receipt). Rows hold public observations only."""
    case = dict(arm="Kinova3", suite="libero_goal", task_id=3, variant="BASE",
                init_state_id=layout, seed=42710300 + layout)
    _, first = sim.call("create", dict(case, embodiment="nom"))
    if first["language_instruction"] != TASK:
        raise ValueError("Unexpected native prompt")
    raw, frames, rewards, t, success = first["observation"], [], [], 0, False
    while t < len(commands) and not success:
        frames.append((t, public(raw)))
        step = sim.call("chunk", commands[t:t + 5])
        n = int(step["steps"])
        rewards.extend(float(r) for r in step["rewards"])
        t += n
        success = bool(step["success"])
        raw = step["observation"]
        if step["terminated"] or step["truncated"]:
            break
    rows = [dict(t=at, public=seen, actions=commands[at:at + HORIZON].copy())
            for at, seen in frames if at + HORIZON <= len(commands)]
    receipt = dict(case=case, controls=t, native_rewards=rewards, success=success, rows=len(rows),
                   donor_commands_sha256=array_digest(commands), consumed_sha256=array_digest(commands[:t]),
                   reset=first["environment_reproducibility"])
    return rows, receipt


def collect(sim, out, log=print):
    """Reward gates episodes here, offline. It is never an adapter input."""
    donors, sets, receipts = tapes(), dict(fit=[], heldout=[]), []
    for split, pairs in (("fit", FIT), ("heldout", HELDOUT)):
        for layout, donor in pairs:
            rows, receipt = replay(sim, layout, donors[donor])
            episode = f"Kinova3-{layout}-from-IIWA-{donor}"
            receipt.update(episode=episode, split=split, used=receipt["success"])
            receipts.append(receipt)
            log(json.dumps({k: receipt[k] for k in ("episode", "split", "controls", "success", "rows")}))
            if receipt["success"]:
                sets[split].extend(dict(row, episode=episode) for row in rows)
    for index, row in enumerate(sets["fit"] + sets["heldout"]):
        row["index"] = index
    (Path(out) / "collection.json").write_text(json.dumps(receipts, indent=2, allow_nan=False) + "\n")
    return sets, receipts
