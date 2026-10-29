"""Read-only completed-artifact audit; standard library, no model imports."""
import hashlib
import json
from pathlib import Path
import zipfile


ROOT = Path("/mowenjie/fluffy-bassoon-2/outputs")
RUNS = ("libero-rlinf130-corpus-clean-20261029-r1",
        "libero-rlinf130-corpus-plus-20261029-r1",
        "libero-rlinf130-plus-dev-20261029-r1", "libero-rlinf130-pro-dev-20261029-r1")


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def receipt(path, expected=None):
    actual = digest(path)
    if expected is not None:
        assert actual == expected, str(path)
    return dict(path=str(path.relative_to(ROOT)), bytes=path.stat().st_size,
                sha256=actual, expected_matched=expected is not None)


report = dict(read_only=True, model_computation=False, runs={})
for run in RUNS:
    root, checked = ROOT / run, []
    assert (root / "exit-code.txt").read_text().strip() == "0"
    summary = json.loads((root / "experiment/summary.json").read_text())
    assert summary["completed"] is True
    for path in sorted(root.rglob("*")):
        if path.is_file() and (path.suffix in (".json", ".jsonl") or path.name == "exit-code.txt"):
            checked.append(receipt(path))
    if "trials" in summary:
        for trial in summary["trials"]:
            data_paths = {}
            for objective, item in trial["datasets"].items():
                path = root / "experiment" / item["path"]
                checked.append(receipt(path, item["sha256"]))
                assert json.loads(path.with_suffix(".json").read_text())["sha256"] == item["sha256"]
                data_paths[objective] = path
            with zipfile.ZipFile(data_paths["ad"]) as ad, zipfile.ZipFile(data_paths["dpt"]) as dpt:
                assert set(ad.namelist()) == set(dpt.namelist())
                for field in ad.namelist():
                    if field != "target_noise.npy":
                        assert ad.read(field) == dpt.read(field), (trial["trial_id"], field)
                assert ad.read("target_noise.npy") == ad.read("executed_noise.npy")
            for episode in trial["episodes"]:
                trace = root / "experiment" / Path(trial["datasets"]["ad"]["path"]).parent / f"episode-{episode['attempt']:03d}.pkl.gz"
                checked.append(receipt(trace, episode["trace_sha256"]))
            checked.append(receipt(root / "experiment" / trial["source_checkpoint"], trial["source_checkpoint_sha256"]))
        for bank in summary.get("pilot_datasets", {}).values():
            for item in bank.values():
                checked.append(receipt(root / "experiment" / item["path"], item["sha256"]))
    else:
        rows = [json.loads(line) for line in (root / "experiment/episodes.jsonl").read_text().splitlines()]
        for row in rows:
            trace = root / "experiment" / f"case-{row['case_index']:02d}" / row["method"] / f"episode-{row['attempt']:03d}.pkl.gz"
            checked.append(receipt(trace, row["trace_sha256"]))
        for path in sorted((root / "experiment").rglob("recovery.pkl")):
            checked.append(receipt(path))
    focused = root / "focused-tests.log"
    report["runs"][run] = dict(completed=True, exit_code=0, checked_files=checked,
        expected_hash_checks=sum(item["expected_matched"] for item in checked),
        focused_test_summary=focused.read_text().splitlines()[-1] if focused.exists() else None,
        paired_contexts_bitwise_equal=True if "trials" in summary else None)
print(json.dumps(report, indent=2))
