"""CPU-only inspection of verified saved DIT evidence; never runs a policy."""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import platform
import tarfile

import numpy as np
import torch


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def compare(left, right, path=""):
    """Describe exact differences without inventing a numerical acceptance gate."""
    if isinstance(left, torch.Tensor):
        left = left.detach().cpu().numpy()
    if isinstance(right, torch.Tensor):
        right = right.detach().cpu().numpy()
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        row = dict(path=path, shape_left=list(left.shape), shape_right=list(right.shape),
                   dtype_left=str(left.dtype), dtype_right=str(right.dtype),
                   exact=left.dtype == right.dtype and np.array_equal(left, right))
        if left.shape == right.shape and left.size and left.dtype.kind in "biuf" and right.dtype.kind in "biuf":
            delta = np.abs(left.astype(np.float64) - right.astype(np.float64))
            row.update(max_absolute_delta=float(delta.max()), mean_absolute_delta=float(delta.mean()),
                       unequal_elements=int(np.count_nonzero(left != right)), elements=int(left.size))
        return [row]
    if isinstance(left, dict) and isinstance(right, dict):
        rows = []
        for key in sorted(set(left) | set(right)):
            sub = path + "/" + str(key)
            if key not in left or key not in right:
                rows.append(dict(path=sub, exact=False, missing="left" if key not in left else "right"))
            else:
                rows.extend(compare(left[key], right[key], sub))
        return rows
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return [dict(path=path, exact=False, length_left=len(left), length_right=len(right))]
        return [row for i, (a, b) in enumerate(zip(left, right)) for row in compare(a, b, path + f"/{i}")]
    exact = type(left) is type(right) and left == right
    return [dict(path=path, exact=bool(exact), **({} if exact else dict(left=repr(left), right=repr(right))))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert platform.system() == "Linux", "Run evidence computations on allocated Linux"
    args.output.mkdir(parents=True, exist_ok=False)
    probe = args.output / ".write-probe"
    probe.write_text("CPU evidence audit")
    assert probe.read_text() == "CPU evidence audit"
    probe.unlink()
    torch.set_num_threads(1)
    index = json.loads(args.comparison.read_text())
    manifest = {}
    first = {}
    metadata_receipts = []
    for source in index["source_banks"]:
        if source["method"] != "dit":
            continue
        tail = source["bank"].split("/readouts/", 1)[1]
        bank = args.metadata / "readouts" / tail
        assert sha(bank / "RESULT.json") == source["result_sha256"]
        marker = json.loads((bank / "PACKED-ARCHIVE.json").read_text())
        assert sha(marker["manifest"]) == source["manifest_sha256"]
        assert marker["archive_sha256"] == source["archive_sha256"]
        manifest[tail] = (marker, json.loads(Path(marker["manifest"]).read_text()))
        result = json.loads((bank / "RESULT.json").read_text())
        metadata_receipts.append(dict(bank=tail, result_sha256=source["result_sha256"],
                                      manifest_sha256=source["manifest_sha256"]))
        for episode in result["episodes"]:
            if episode["attempt"] == 1:
                identity = json.dumps(episode["case"], sort_keys=True)
                first.setdefault(identity, {})[result["mode"]] = episode
    paired = []
    for identity, modes in sorted(first.items()):
        a, b = modes["full"], modes["episode"]
        paired.append(dict(case=json.loads(identity), entry_exact=a["memory_entry"] == b["memory_entry"],
            full_entry=a["memory_entry"], reset_entry=b["memory_entry"],
            full_success=a["success_once"], reset_success=b["success_once"]))
    (args.output / "metadata.json").write_text(json.dumps(dict(banks=metadata_receipts, cases=paired), indent=2) + "\n")
    raw = {}
    receipts = []
    # Selection fixed before examining raw arrays: all eight cases in Coffee bank zero.
    for mode in ("full", "episode"):
        tail = f"kv_bridge/dit/ID/{mode}/CoffeeSetupMug/bank-000"
        marker, members = manifest[tail]
        archive = Path(marker["cold"])
        assert archive.stat().st_size == members["archive_bytes"]
        actual_sha = sha(archive)
        assert actual_sha == marker["archive_sha256"]
        chosen = {item["relative"]: item for item in members["files"]
                  if "/attempt-1/" in item["relative"] and
                  (item["relative"].endswith(("initial.pt", "complete.pt", ".pkl.gz")))}
        staged = args.output / "staged" / mode
        with tarfile.open(archive, "r:") as stream:
            found = set()
            for item in stream:
                name = item.name.removeprefix("./")
                if name not in chosen:
                    continue
                assert item.isfile() and not Path(name).is_absolute() and ".." not in Path(name).parts
                data = stream.extractfile(item).read()
                expected = chosen[name]
                assert len(data) == expected["bytes"] and hashlib.sha256(data).hexdigest() == expected["sha256"]
                target = staged / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                found.add(name)
            assert found == set(chosen)
        receipts.append(dict(mode=mode, archive=str(archive), archive_sha256=actual_sha,
                             archive_bytes=members["archive_bytes"], selected_members=list(chosen.values()),
                             stageback="Fresh diagnostic directory; original paths/archives remain unchanged"))
        raw[mode] = {}
        for path in sorted(staged.rglob("complete.pt")):
            complete = torch.load(path, map_location="cpu", weights_only=False)
            slot = complete["case"]["slot"]
            initial = torch.load(path.with_name("initial.pt"), map_location="cpu", weights_only=False)
            chunks = []
            for chunk in sorted(path.parent.glob("chunk-*.pkl.gz")):
                with gzip.open(chunk, "rb") as handle:
                    chunks.append(pickle.load(handle))
            assert len(chunks) * 50 == complete["horizon"]
            raw[mode][slot] = (complete, initial, chunks)
        print(json.dumps(dict(mode=mode, archive_verified=True, slots=sorted(raw[mode]))), flush=True)
    assert set(raw["full"]) == set(raw["episode"]) == set(range(8))
    evidence = []
    fields = ("raw_observation", "observation", "features", "control", "native_noise",
              "native_noise_provenance", "requested_commands", "executed_commands", "reward", "done",
              "raw_next_observation", "next_observation", "execution_provenance")
    for slot in range(8):
        a, ia, ca = raw["full"][slot]
        b, ib, cb = raw["episode"][slot]
        assert a["case"] == b["case"] and a["policy_weights_sha256"] == b["policy_weights_sha256"]
        chunks = []
        for left, right in zip(ca, cb):
            assert left["chunk"] == right["chunk"]
            fields_result = {field: compare(left.get(field), right.get(field)) for field in fields}
            chunks.append(dict(chunk=left["chunk"], fields=fields_result))
        evidence.append(dict(slot=slot, case=a["case"], full_success=a["success_once"], reset_success=b["success_once"],
            policy_weights_sha256=a["policy_weights_sha256"], memory_entry_exact=a["memory_entry"] == b["memory_entry"],
            initial_evidence=compare(ia["evidence"], ib["evidence"]), chunks=chunks))
    output = dict(checked_utc=datetime.now(timezone.utc).isoformat(),
        scope="DIT all first-attempt metadata; raw CoffeeSetupMug bank-000 eight cases, all first-attempt chunks",
        comparison_sha256=sha(args.comparison), audit_script_sha256=sha(__file__), GPU_hours=0,
        paired_metadata_cases=len(paired), entry_mismatches=sum(not row["entry_exact"] for row in paired),
        first_success_discordances=sum(row["full_success"] != row["reset_success"] for row in paired),
        archives=receipts, raw_cases=evidence)
    (args.output / "RESULT.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({key: value for key, value in output.items() if key not in ("archives", "raw_cases")}), flush=True)


if __name__ == "__main__":
    main()
