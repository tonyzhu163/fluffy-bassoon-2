"""SOURCE-PREPARED, NOT QUALIFIED: byte-preserving view for the existing exporter.

Run only in an owner-admitted Linux CPU allocation. No tensors, policy labels,
simulator, ledger constructors, source controls, or new converter are used.
The qualified c62 ancestry implementation is imported by its exact file SHA;
only corpus_chain is called. Unequal path collisions halt before publication.
"""
import argparse
import errno
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import sys

READER_SHA = "67db52fa8f6fa4e8bb56bc6e5cea9f531b5fe41716845350f3372226c5de77a0"
LINEAGE = "db0184089fa88a5f96c153f9b80029bb91b77def"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def bound(ref):
    p = Path(ref["path"]).resolve()
    require(sha(p) == ref["sha256"], "Bound input changed: " + str(p))
    return p


def contained(root, path, *, directory=False):
    require(not path.is_symlink() and path.resolve().is_relative_to(root),
            "Symlink or path escape in source corpus: " + str(path))
    require(path.is_dir() if directory else path.is_file(),
            "Missing regular source artifact: " + str(path))


def corpus_files(root):
    for folder in ("episodes", "frames", "policies"):
        directory = root / folder
        contained(root, directory, directory=True)
        for path in sorted(directory.rglob("*")):
            require(not path.is_symlink() and path.resolve().is_relative_to(root),
                    "Symlink or path escape in source inventory: " + str(path))
            if path.is_dir():
                continue
            contained(root, path)
            yield path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--admission", type=Path, required=True)
    p.add_argument("--admission-sha256", required=True)
    p.add_argument("--chain-reader", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--expected-hostname", required=True)
    p.add_argument("--copy-cross-device", action="store_true")
    p.add_argument("--purpose", choices=("finalbank", "reference"), default="finalbank")
    a = p.parse_args()
    require(sys.platform == "linux" and socket.gethostname() == a.expected_hostname,
            "Use the exact allocated Linux CPU host")
    require(sha(a.admission) == a.admission_sha256, "Admission changed")
    admission = json.loads(a.admission.read_text())
    if a.purpose == "finalbank":
        require(admission.get("assembly_cpu_gate_passed") is True
                and admission.get("closed_source100_authorized") is True,
                "Owner CPU qualification and closed-source100 admission required")
        rounds = 100
    else:
        require(admission.get("closed_reference_authorized") is True,
                "Owner admission required for allocated compatibility reference")
        rounds = admission["expected_source_rounds"]
        require(type(rounds) is int and 1 <= rounds < 100, "Invalid closed reference")
    require(admission["lineage_commit"] == LINEAGE, "Wrong scientific lineage")
    ancestry = json.loads(bound(admission["ancestry"]).read_text())
    tip = json.loads(bound(ancestry["tip_manifest"]).read_text())
    expected = tip["expected"]
    require(tip["kind"] == "whole_trial_boundary"
            and tip["method"] == admission["method"]
            and expected["completed_source_iterations"] == rounds
            and expected["completed_source_trials"] == 6 * rounds
            and not expected["pending_rollout_digests"],
            "Require the final100-round/600-K4 complete source boundary")
    bound(tip["checkpoint"])
    require(sha(a.chain_reader) == READER_SHA, "Use the exact c62 reader")
    spec = importlib.util.spec_from_file_location("pinned_c62_corpus_reader", a.chain_reader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    roots = [Path(x).resolve() for x in module.corpus_chain(ancestry["tip_manifest"])]
    require([str(x) for x in roots] == ancestry["ordered_corpus_roots"]
            and len(roots) == ancestry["root_count"], "Owner and c62 ancestry disagree")
    require(len(roots) == len(set(roots)), "Duplicate corpus root")
    output = a.output.resolve()
    require(not output.exists() and output.parent.is_dir(), "Fresh output with existing parent required")
    require(all(not output.is_relative_to(r) and not r.is_relative_to(output) for r in roots),
            "Projection must not contain or overwrite original roots")
    files, root_evidence, inventories, origin_roots = {}, [], {}, {}
    base_contract = base_runtime = None
    total_unique_bytes = 0
    for root in roots:
        cp, rp = root / "contract.json", root / "runtime-contract.json"
        contained(root, cp)
        contained(root, rp)
        contract, runtime = json.loads(cp.read_text()), json.loads(rp.read_text())
        scientific = {k: v for k, v in contract.items()
                      if k not in ("execution_source_commit", "resume_trial_manifest_sha256")}
        runtime_scientific = {k: v for k, v in runtime.items() if k != "actual_render_pci"}
        if base_contract is None:
            base_contract, base_runtime = scientific, runtime_scientific
        require(scientific == base_contract and runtime_scientific == base_runtime,
                "Producer scientific contracts differ; do not relax or rewrite them")
        root_evidence.append({"root": str(root), "contract_sha256": sha(cp),
                              "runtime_sha256": sha(rp)})
        inventories[str(root)] = set()
        for path in corpus_files(root):
            require(".partial" not in path.name and not path.name.endswith(".tmp"),
                    "Uncommitted file in closed corpus: " + str(path))
            rel, digest = str(path.relative_to(root)), sha(path)
            inventories[str(root)].add(rel)
            origin_roots[str(path)] = root
            if rel in files:
                require(files[rel]["sha256"] == digest,
                        "Unequal path collision; needs reviewed multi-root reader, not overwrite: " + rel)
                files[rel]["origins"].append(str(path))
            else:
                size = path.stat().st_size
                files[rel] = {"sha256": digest, "bytes": size, "origins": [str(path)]}
                total_unique_bytes += size
    # Contract/runtime files are copied byte-for-byte from the original first
    # root. All producer contracts remain separately pinned; none is rewritten.
    require(base_contract.get("source_commit") == LINEAGE, "Wrong corpus lineage")
    if a.copy_cross_device:
        require(shutil.disk_usage(output.parent).free > total_unique_bytes + (1 << 30),
                "Insufficient capacity for worst-case copies")
    output.mkdir()
    try:
        for name in ("contract.json", "runtime-contract.json"):
            contained(roots[0], roots[0] / name)
            shutil.copyfile(roots[0] / name, output / name)
            require(sha(output / name) == sha(roots[0] / name), "Top-level metadata copy differs")
        for rel, item in files.items():
            target, original = output / rel, Path(item["origins"][0])
            contained(origin_roots[str(original)], original)
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(original, target)
            except OSError as exc:
                if exc.errno != errno.EXDEV or not a.copy_cross_device:
                    raise
                shutil.copyfile(original, target)
            # Full file SHA, including RGB/blob bytes, unlike header-only audit.
            require(sha(target) == item["sha256"], "Projection bytes differ")
            for origin in item["origins"]:
                contained(origin_roots[origin], Path(origin))
                require(sha(origin) == item["sha256"], "Source changed during assembly")
        for record in root_evidence:
            root = Path(record["root"])
            contained(root, root / "contract.json")
            contained(root, root / "runtime-contract.json")
            require(sha(root / "contract.json") == record["contract_sha256"]
                    and sha(root / "runtime-contract.json") == record["runtime_sha256"],
                    "Producer metadata changed during assembly")
            current = {str(p.relative_to(root)) for p in corpus_files(root)}
            require(current == inventories[str(root)], "Closed source inventory changed during assembly")
        require(sha(a.admission) == a.admission_sha256 and sha(a.chain_reader) == READER_SHA,
                "Admission or pinned reader changed during assembly")
        bound(admission["ancestry"])
        bound(ancestry["tip_manifest"])
        result = dict(status="byte_projection_only_not_export_admission", purpose=a.purpose,
                      source_rounds=rounds, method=tip["method"],
                      lineage_commit=LINEAGE, ancestry=admission['ancestry'],
                      source_checkpoint=tip['checkpoint'],
                      admission_sha256=a.admission_sha256, reader_sha256=READER_SHA,
                      tip_manifest=ancestry["tip_manifest"], roots=root_evidence, files=files,
                      unique_files=len(files), unique_file_bytes=total_unique_bytes,
                      selected_contract_origin=str(roots[0]),
                      scientific_input_rule="existing exporter selects source/validation phases; raw views retain all phases",
                      additional_robot_actions=0, additional_vla_queries=0)
        (output / "projection.json").write_text(json.dumps(result, indent=2) + "\n")
    except BaseException:
        # Preserve partial evidence. Never delete or retry automatically.
        (output / "ASSEMBLY-INCOMPLETE").write_text("Not admitted; inspect before any retry.\n")
        raise


if __name__ == "__main__":
    main()
