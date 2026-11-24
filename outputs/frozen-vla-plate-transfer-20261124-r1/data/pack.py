"""Pack and read back native expert windows with a common layout/phase draw index."""
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from source_data import FINGERS, HORIZON, POS, PublicHistory, SourceDataset

from collect import BODIES, KEYS, PROMPT, check, sha, write


def starts(n, count, slot):
    first = 2 * (slot % 2)
    early = [first, first + 1]
    later = [i for i in range(4, n - HORIZON + 1)]
    if len(later) < count - 2:
        later = [i for i in range(n - HORIZON + 1) if i not in early]
    positions = np.linspace(0, len(later) - 1, count - 2, dtype=int)
    result = early + [later[i] for i in positions]
    check(len(set(result)) == count and max(result) + HORIZON <= n, "Incomplete/duplicate action window")
    return result


def groups(item, mode, count):
    # Identical minibatch row -> identical layout and phase slot. Block ordering
    # preserves one trajectory per body in the existing manifest reader.
    parity = item["slot"] // 2 % 2
    a = list(range(parity, count, 2))
    b = list(range(1 - parity, count, 2))
    if mode == "mixed":
        return [("Panda", a), ("Sawyer", b)]
    return [("Sawyer" if mode == "diagnostic-sawyer" else "Panda", a + b)]


def write_split(root, split, layouts, mode, tapes):
    count = 12 if split == "train" else 8
    meta, hashes, ages = [], [], {}
    path = root / f"{split}.h5"
    with h5py.File(path, "x") as f:
        rows = 0
        for item in layouts:
            for body, phases in groups(item, mode, count):
                r = item["episodes"][body]
                check(r["success"] and r["terminated"] and r["terminal_reward"] == 1, "Nonpositive admitted teacher")
                times = [starts(r["steps"], count, item["slot"])[i] for i in phases]
                trajectory = f"{body}/generated-{r['layout_seed']}"
                meta.append(dict(id=trajectory, prompt=PROMPT, body=body, layout_seed=r["layout_seed"],
                                 scene_hash=r["scene_hash"], movable_xy_hash=r["movable_xy_hash"],
                                 generated_row_fingerprint=r["generated_row_fingerprint"], steps=r["steps"],
                                 t=times, phase_slots=phases, sequence_sha256=r["sequence_sha256"],
                                 archive_sha256=r["archive_sha256"], terminal_reward=r["terminal_reward"]))
                with h5py.File(tapes / r["file"], "r") as tape:
                    check(tape.attrs["scene_hash"] == item["scene_hash"] and tape.attrs["body"] == body
                          and tape["rewards"][-1] == 1 and not tape["rewards"][:-1].any()
                          and tape["actions"].shape == (r["steps"], 7), "Native replay readback")
                    actions, tools, fingers = (tape[k][:] for k in ("actions", POS, FINGERS))
                    for t in times:
                        raw = {key: tape[key][t] for key in KEYS}
                        history = PublicHistory()
                        for j in range(max(0, t - 4), t):
                            history.append(actions[j], {POS: tools[j], FINGERS: fingers[j]},
                                           {POS: tools[j + 1], FINGERS: fingers[j + 1]})
                        history = history.vector()
                        check(np.array_equal(history, tape["history"][t]), "Offline and native histories differ")
                        values = dict(**raw, actions=actions[t:t + HORIZON], history=history)
                        digest = hashlib.sha256()
                        for key, value in values.items():
                            if key not in f:
                                f.create_dataset(key, shape=(len(layouts) * count, *value.shape), dtype=value.dtype,
                                                 chunks=(1, *value.shape), compression="lzf", track_times=False)
                            f[key][rows] = value
                            digest.update(value.tobytes())
                        hashes.append(digest.hexdigest())
                        age = int(sum(history[12::13]))
                        ages.setdefault(body, []).append(age)
                        rows += 1
        check(rows == count * len(layouts), "Expected count per layout")
    return dict(file=path.name, sha256=sha(path), examples=rows, trajectories=meta,
                history_lengths={k: {str(i): v.count(i) for i in range(5)} for k, v in ages.items()}), hashes


def publish(out, enrolled, pin):
    target = out.parent / "dataset-v1"
    check(not target.exists() and len(enrolled) == 20, "Fresh complete source dataset required")
    tmp = out / "staging"
    tmp.mkdir()
    public = [{k: item[k] for k in ("slot", "rank", "split", "layout_seed", "movable_xy_hash", "scene_hash",
                                     "generated_row_fingerprint")} for item in enrolled]
    seal = dict(code_revision=pin, attempt=out.name, selection_sha256=sha(out / "selection.json"),
                enrollment_sha256=sha(out / "enrollment.json"), layouts=public, variants={}, diagnostics={},
                source_manifest_sha256=json.loads((out / "selection.json").read_text())["archive_sha256"])
    for item in enrolled:
        for r in item["episodes"].values():
            check(sha(out / "tapes" / r["file"]) == r["sequence_sha256"], "Native tape bytes changed")
    for mode in ("panda", "mixed", "diagnostic-panda", "diagnostic-sawyer"):
        root = tmp / mode
        root.mkdir()
        manifest = dict(schema="libero-panda-openpi-v1", code_revision=pin, split_unit="generated physical layout",
                        variant=mode, horizon=HORIZON, history_shape=[52], splits={})
        row_digests = {}
        for split in (("heldout",) if mode.startswith("diagnostic") else ("train", "heldout")):
            layouts = [row for row in enrolled if row["split"] == split]
            part, row_digests[split] = write_split(root, split, layouts, mode, out / "tapes")
            check(all(all(v > 0 for v in counts.values()) for counts in part["history_lengths"].values()),
                  "Missing history-age coverage")
            manifest["splits"][split] = part
        write(root / "manifest.json", manifest)
        for split, hashes in row_digests.items():
            ds = SourceDataset(root, split)
            check(len(ds) == len(hashes), "Loader count")
            for i, expected in enumerate(hashes):
                row = ds[i]
                arrays = [row["raw"][k] for k in KEYS] + [row["actions"], row["history"]]
                actual = hashlib.sha256(b"".join(a.tobytes() for a in arrays)).hexdigest()
                check(actual == expected and row["prompt"] == PROMPT, "Packed native row bytes changed")
        receipt = dict(root=mode, manifest_sha256=sha(root / "manifest.json"))
        if mode.startswith("diagnostic"):
            seal["diagnostics"][mode.split("-")[1].capitalize()] = receipt
        else:
            seal["variants"][mode] = receipt
    write(tmp / "seal.json", seal)
    digest = sha(tmp / "seal.json")
    tmp.rename(target)
    print("Sealed", target / "seal.json", digest, flush=True)
