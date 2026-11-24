"""Independently verify the sealed source manifests and the two equal-budget training arms."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np


def load(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def rows(part):
    return [(t["layout_seed"], t["scene_hash"], phase, t["body"], step)
            for t in part["trajectories"] for phase, step in zip(t["phase_slots"], t["t"], strict=True)]


def check(test, message):
    if not test:
        raise ValueError(message)


def audit(root, seal_hash):
    sealfile = root / "seal.json"
    check(sha(sealfile) == seal_hash, "Dataset SHA differs from native publication")
    seal = load(sealfile)
    attempt = root.parent / seal["attempt"]
    rc, started, finished = map(int, (attempt / "exit.txt").read_text().split())
    check(rc == 0, "Collection failed")
    selection = load(attempt / "selection.json")
    check(sha(attempt / "selection.json") == seal["selection_sha256"] and
          sha(attempt / "enrollment.json") == seal["enrollment_sha256"], "Enrollment changed")
    ledger = [json.loads(line) for line in (attempt / "attempts.jsonl").read_text().splitlines()]
    enrollment = load(attempt / "enrollment.json")
    accepted = [r for r in ledger if r["accepted"]]
    check(len(accepted) == len(enrollment) == len(seal["layouts"]) == 20, "Enrollment size")
    next_reserve, slot, rank = 20, 0, 0
    for row in ledger:
        candidate = selection["candidates"][rank]
        check((row["slot"], row["rank"], row["scene_hash"]) == (slot, rank, candidate["scene_hash"]),
              "Non-deterministic source replacement")
        if row["accepted"]:
            slot += 1
            rank = slot
        else:
            rank, next_reserve = next_reserve, next_reserve + 1
    counts, steps, lengths, teacher = Counter(), Counter(), Counter(), set()
    max_abs_symmetric_error = [0.0, 0.0]
    for i, (enr, public, admitted) in enumerate(zip(enrollment, seal["layouts"], accepted, strict=True)):
        check(all(enr[k] == v for k, v in public.items()) and enr["slot"] == i
              and public["split"] == ("train" if i < 16 else "heldout") and enr["rank"] == admitted["rank"],
              "Public and executed layouts differ")
        cand = selection["candidates"][enr["rank"]]
        for body_index, body in enumerate(("Panda", "Sawyer")):
            e = enr["episodes"][body]
            native = attempt / "tapes" / e["file"]
            check(e["archive_sha256"] == cand["tapes"][body] and sha(native) == e["sequence_sha256"] and
                  all(e[k] == public[k] for k in ("layout_seed", "scene_hash", "movable_xy_hash", "generated_row_fingerprint"))
                  and e["success"] and e["native_success"] and e["terminated"] and not e["truncated"]
                  and e["terminal_reward"] == e["native_return"] == 1 and e["exact_issued_prefix"], "Native source receipt")
            with h5py.File(native, "r") as f:
                n = e["steps"]
                check(f["actions"].shape == (n, 7) and f["rewards"].shape == (n,) and f["rewards"][-1] == 1
                      and not f["rewards"][:-1].any() and f["history"].shape == (n + 1, 52)
                      and all(f[k].shape == (n + 1, 256, 256, 3) for k in ("agentview_image", "robot0_eye_in_hand_image")),
                      "Native full-tape readback")
                fingers = f["robot0_gripper_qpos"][:]
                check(fingers.shape == (n + 1, 2), "Native dual finger dimensions")
                error = float(np.max(np.abs(fingers.sum(axis=1))) / 2)
                max_abs_symmetric_error[body_index] = max(max_abs_symmetric_error[body_index], error)
            counts[body] += 1
            steps[body] += n
            lengths[body] += int(n != e["archive_steps"])
            teacher.add(e["teacher_revision"])
    for key in ("layout_seed", "scene_hash", "movable_xy_hash", "generated_row_fingerprint"):
        check(len({r[key] for r in seal["layouts"]}) == 20, f"Duplicate held-out/training {key}")

    roots = {**seal["variants"], **{f"diag-{k}": v for k, v in seal["diagnostics"].items()}}
    tables, summaries = {}, {}
    for key, ref in roots.items():
        base = root / ref["root"]
        path = base / "manifest.json"
        check(sha(path) == ref["manifest_sha256"], "Manifest digest")
        manifest = load(path)
        for split, part in manifest["splits"].items():
            filename = base / part["file"]
            table = rows(part)
            check(sha(filename) == part["sha256"] and len(table) == len(set(table)) == part["examples"],
                  "Packed file digest or duplicate example")
            tables[key, split] = (table, filename)
            summaries.setdefault(key, {})[split] = dict(examples=len(table), bodies=dict(Counter(r[3] for r in table)),
                                                        history_ages=part["history_lengths"])
    same_rows = {}
    for split, layouts, per in (("train", 16, 12), ("heldout", 4, 8)):
        a, f1 = tables["panda", split]
        b, f2 = tables["mixed", split]
        check(len(a) == len(b) == layouts * per and [r[:3] for r in a] == [r[:3] for r in b],
              "Phase-slot or layout mismatched across source conditions")
        c = Counter((r[0], r[3]) for r in b)
        check(len(c) == layouts * 2 and set(c.values()) == {per // 2}, "Arm imbalance within layout")
        check({r[3] for r in a} == {"Panda"} and all(x == y for x, y in zip(a, b) if y[3] == "Panda"),
              "Panda shared row identity")
        common = [i for i, y in enumerate(b) if y[3] == "Panda"]
        with h5py.File(f1, "r") as p, h5py.File(f2, "r") as m:
            check(set(p) == set(m) and all(np.array_equal(p[k][common], m[k][common]) for k in p), "Shared Panda row bytes")
        same_rows[split] = len(common)
    for body in ("Panda", "Sawyer"):
        diag, _ = tables[f"diag-{body}", "heldout"]
        check(len(diag) == 32 and {r[3] for r in diag} == {body} and
              {r[0] for r in diag} == {r["layout_seed"] for r in enrollment[16:]}, "Common arm diagnostic split")
    return dict(seal_sha256=seal_hash, code_revision=seal["code_revision"], archive_sha256=selection["archive_sha256"],
                selection_sha256=seal["selection_sha256"], execution_pin=seal["code_revision"], dataset_root=str(root),
                gpu_host=(attempt / "launch.txt").read_text().splitlines()[0], cpu_affinity=sorted(os.sched_getaffinity(0)),
                launcher_seconds=finished - started, eligible_source_layouts=len(selection["candidates"]),
                attempted_pairs=len(ledger), excluded_pairs=[r for r in ledger if not r["accepted"]], accepted_pairs=len(enrollment),
                full_episodes=dict(counts), native_commands=dict(steps), different_length_than_archive=dict(lengths),
                teacher_revisions=sorted(teacher), shared_identical_panda_windows=same_rows,
                max_finger_error_if_symmetric_m=dict(zip(("Panda", "Sawyer"), max_abs_symmetric_error)), windows=summaries)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("seal_sha256")
    p.add_argument("output", type=Path)
    args = p.parse_args()
    assert 0 < len(os.sched_getaffinity(0)) <= 2
    result = audit(args.root, args.seal_sha256)
    with args.output.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    print("Read back", args.output, sha(args.output), flush=True)
