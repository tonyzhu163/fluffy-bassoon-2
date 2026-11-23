"""Filesystem adapter for the sealed prepare01pro attempt. Run on Baige Linux."""
import argparse
from pathlib import Path
import re
import subprocess
import sys
import tempfile

ROOT = Path("/mowenjie/fluffy-bassoon-2")
RUN = "symmeta-perceiver-retry-20261123-r1"
CODE = ROOT / "setups/symmeta-perceiver-followups-20261123-r1/src-d568535b758f"
PIN = "d568535b758f2b97b382f9b3e280376e769b5942"
PREP = ROOT / "outputs" / RUN / "prepare01pro"
CONTRACT = "e559a9d57c1ba73e8bb8a06cb5b6b1c7e69d252aaa539718b99b212ca8669441"
HERE = CODE / "outputs" / RUN
sys.path.insert(0, str(HERE))
import paired_records as r
from paired_devices import hardware
from paired_inputs import check_tree


def inputs(src):
    raw = r.checked(src / "contract.json", CONTRACT)
    contract = r.parse(raw)
    yield Path("contract.json"), raw
    yield Path("predictions.json"), r.checked(src / "predictions.json", contract["predictions_sha256"])
    for panel in r.PANELS:
        path = Path(panel, "panel.json")
        yield path, r.checked(src / path, contract["panels"][panel])


def snapshot(out):
    for path, raw in inputs(PREP):
        dest = out / path
        dest.parent.mkdir(exist_ok=True)
        with dest.open("xb") as f:
            f.write(raw)
        dest.chmod(0o444)
        r.checked(dest, r.digest(raw))
    # Recompute the complete four-panel seal, including current evaluator file hashes.
    r.seal(out, r.contract(out)[1]["predictions_sha256"])


def origin(src, panel):
    sha = (src / "execution.sha256").read_text().strip()
    rec = r.receipt(src, sha)
    host = rec["hostname"]
    r.require(re.fullmatch(r"job-[a-z0-9]+-master-0", host) and src.name == f"{panel}-{host}"
              and rec["schema"] == 1 and rec["attempt"] == src.name and rec["deployment"] == PIN
              and rec["contract_sha256"] == CONTRACT and len(rec["devices"]) == 1, f"Job receipt {src}")
    probe_sha = rec["gpu_probe_sha256"]
    probe_raw = r.checked(src / "gpu.json", probe_sha)
    probe = r.parse(probe_raw)
    dev = rec["devices"][0]
    r.require(probe["hostname"] == host and probe["status"] == "passed" and probe["purpose"] == "evaluation"
              and probe["allocation_type"] == "baige_container" and dev["allocation_ordinal"] == 0
              and type(dev["egl_device"]) is int and dev["egl_device"] >= 0
              and probe["devices"] == [{k: v for k, v in dev.items() if k != "egl_device"}]
              and probe["gpu_identity_mapping"]["assigned_nvml_uuids"] == [dev["uuid"]], f"Job probe {src}")
    return sha, rec, probe_raw


def finish(src, panel):
    s = r.reduce_attempt(src, partial=True)
    r.require(s["completed_pairs"] == 288 and s["panels"][panel]["overall"]["complete"]
              and s["panels"][panel]["ood"]["complete"], f"Incomplete panel {src}")


def admit(src, panel, stage):
    for _ in inputs(src):  # Verify every frozen input, including the three unassigned panels.
        pass
    sha, rec, probe = origin(src, panel)
    rc, start, end = map(int, (src / "exit.txt").read_text().split())
    r.require(rc == 0 and end >= start, f"Unsuccessful job {src}")
    for shard in range(8):
        r.require((src / panel / f"shard-{shard}.rc").read_text().strip() == "0", f"Shard exit {src}/{shard}")
    data = r.parse((stage / panel / "panel.json").read_bytes())
    paths = {r.pair_path(src, row) for row in data["rows"]}
    r.require(len(paths) == 288 and set((src / "pairs").rglob("*.json")) == paths, f"Panel coverage {src}")
    r.atomic(stage / "receipts" / (sha + ".json"), r.checked(src / "receipts" / (sha + ".json"), sha))
    r.atomic(stage / "probes" / (rec["gpu_probe_sha256"] + ".json"), probe)
    receipts = {sha: rec}
    for row in data["rows"]:
        raw, body = r.read_pair(src, row, data["refs"][row["shard"]], CONTRACT, receipts)
        r.require(body["execution_sha256"] == sha and type(body["lane"]) is int and body["lane"] == 0,
                  f"Foreign execution at {r.pair_path(src, row)}")
        r.atomic(r.pair_path(stage, row), raw)
    return rec


def inventory(path):
    r.require(path.is_dir() and not path.is_symlink(), path)
    result = {}
    for child in path.rglob("*"):
        r.require(not child.is_symlink() and (child.is_dir() or child.is_file()), child)
        if child.is_file():
            result[child.relative_to(path)] = r.digest(child.read_bytes())
    return result


def combine(batch, jobs):
    r.require(len(jobs) == len(set(jobs)) == len(r.PANELS)
              and all(re.fullmatch(r"job-[a-z0-9]+", job) for job in jobs), "Four distinct Baige job IDs")
    dest = batch / "combined"
    with tempfile.TemporaryDirectory(prefix=".admission-", dir=batch) as tmp:
        stage = Path(tmp) / "combined"
        stage.mkdir()
        snapshot(stage)
        devices = []
        sources = {}
        for panel, job in zip(r.PANELS, jobs, strict=True):
            src = batch / f"{panel}-{job}-master-0"
            rec = admit(src, panel, stage)
            devices.append(hardware(rec, 0))
            sources[panel] = src.name
        r.require(all(d == devices[0] for d in devices), "Mixed GPU model, memory, driver or software")
        r.atomic(stage / "sources.json", r.encode(sources))
        subprocess.run([sys.executable, "-B", HERE / "paired_records.py", "reduce", stage], check=True)
        # Rename a fully audited tree. Concurrent publishers cannot replace a nonempty tree.
        if not dest.exists() and not dest.is_symlink():
            try:
                stage.rename(dest)
                return
            except OSError:
                if not dest.exists():
                    raise
        r.require(inventory(dest) == inventory(stage), f"Conflicting previous combine at {dest}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    cmds = p.add_subparsers(dest="mode", required=True)
    for name in ("snapshot", "selector", "finish", "combine"):
        sub = cmds.add_parser(name)
        sub.add_argument("directory", type=Path)
        if name in {"selector", "finish"}:
            sub.add_argument("panel", choices=r.PANELS)
        if name == "combine":
            sub.add_argument("jobs", nargs=4, help="IDs in gpu09test, gpu09iiwa, gpu06kinova3, gpu08pi0fix order")
    a = p.parse_args()
    batch = a.directory if a.mode == "combine" else a.directory.parent
    r.require(a.directory.resolve() == a.directory and a.directory.is_dir() and
              (batch.is_relative_to(Path("/tmp")) and batch != Path("/tmp")
               or batch.parent == PREP.parent and re.fullmatch(r"fragment-[A-Za-z0-9_-]+", batch.name)),
              "Use a fresh fragment batch on the run volume or a private /tmp scratch directory")
    if a.mode in {"snapshot", "finish"}:
        r.require(not (a.directory / "exit.txt").exists(), "Closed job directory")
    check_tree(CODE, PIN)
    if a.mode == "snapshot":
        snapshot(a.directory)
    elif a.mode == "selector":
        _, rec, _ = origin(a.directory, a.panel)
        d = rec["devices"][0]
        print(d["visibility_token"], d["egl_device"])
    elif a.mode == "finish":
        finish(a.directory, a.panel)
    else:
        combine(a.directory, a.jobs)


if __name__ == "__main__":
    main()
