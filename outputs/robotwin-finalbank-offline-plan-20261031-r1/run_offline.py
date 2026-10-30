"""STATIC executable command plan. Defaults to printing commands, never launch.

--execute is for the execution owner inside a separately admitted allocation.
It invokes the existing qualified exporter/trainer; no alternate algorithm.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import time


def digest(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify_final_indexes(b, projection):
    """Reject an old valid bank paired with unrelated valid source100 refs."""
    corpus = Path(b["projection_root"]).resolve()
    exports = Path(b["exports_root"]).resolve()
    source = Path(b["source100_stage"]["path"]).resolve()
    source_sha = b["source100_stage"]["sha256"]
    tip_ref = projection["tip_manifest"]
    if digest(tip_ref["path"]) != tip_ref["sha256"]:
        raise ValueError("Projection tip changed")
    tip = json.loads(Path(tip_ref["path"]).read_text())
    if (tip["method"] != "sac" or tip["kind"] != "whole_trial_boundary"
            or tip["expected"]["completed_source_iterations"] != 100
            or tip["expected"]["completed_source_trials"] != 600):
        raise ValueError("Projection is not a final source100 boundary")
    metadata, cases = {}, {}
    for partition, key, count, objectives in (
            ("train", "train_index", 600, {"ad", "dpt"}),
            ("validation", "validation_index", 6, {"behavior_cloning", "dpt"})):
        path = Path(b[key]["path"]).resolve()
        if path != exports / partition / "index.json":
            raise ValueError("Index is outside the admitted export output")
        m = json.loads(path.read_text())
        if (m.get("format") != "robotwin_native_history_dataset_v1"
                or m.get("completed") is not True or m.get("partition") != partition
                or m.get("source_method") != "sac" or m.get("teacher_method") != "sac"
                or m.get("source_recovery_sha256") != source_sha
                or m.get("teacher_recovery_sha256") != b["teacher100_source"]["sha256"]
                or m.get("policy_input_dim") != 3090
                or set(m.get("datasets", {})) != objectives
                or m.get("validation_iteration") != (100 if partition == "validation" else None)
                or m.get("counts", {}).get("complete_K4_trials") != count
                or m["counts"].get("episodes") != 4 * count
                or m["counts"].get("committed_source_actions") != tip["expected"]["source_steps"]
                or len(m.get("source_trials", [])) != count
                or len(m.get("case_ids", [])) != count):
            raise ValueError("Index does not describe the admitted complete source100 bank")
        if partition == "train" and m["counts"]["selected_actions"] != tip["expected"]["source_steps"]:
            raise ValueError("Final source action coverage differs")
        iterations = m["source_contract"].get("source_iterations")
        if (m.get("additional_simulator_actions") != 0 or m.get("additional_vla_queries") != 0
                or m["source_contract"].get("source_commit") != "db0184089fa88a5f96c153f9b80029bb91b77def"
                or not isinstance(iterations, dict) or iterations.get("count") != 100
                or iterations.get("trials_per_iteration") != 6
                or iterations.get("episodes_per_trial") != 4):
            raise ValueError("Wrong source contract or undeclared new data")
        inputs = m["input_sha256"]
        if inputs.get(str(source)) != source_sha:
            raise ValueError("Source100 checkpoint is absent from exporter inputs")
        for name in ("contract.json", "runtime-contract.json"):
            if inputs.get(str(corpus / name)) != digest(corpus / name):
                raise ValueError("Index is not linked to the admitted corpus projection")
        allowed_provenance = {str(source): source_sha}
        for ref in (projection['ancestry'], projection['tip_manifest'], projection['source_checkpoint']):
            allowed_provenance[str(Path(ref['path']).resolve())] = ref['sha256']
        ancestry_ref = projection['ancestry']
        if digest(ancestry_ref['path']) != ancestry_ref['sha256']:
            raise ValueError('Projection ancestry changed')
        ancestry = json.loads(Path(ancestry_ref['path']).read_text())
        for entry in ancestry.get('ordered_ancestry', []):
            refs = [entry['manifest'], entry['bound_corpus_contract'], entry['canonical_corpus_contract']]
            if entry['execution_migration'] is not None:
                refs.append(entry['execution_migration']['ref'])
            for ref in refs:
                allowed_provenance[str(Path(ref['path']).resolve())] = ref['sha256']
        for root in projection['roots']:
            for name, field in (('contract.json', 'contract_sha256'), ('runtime-contract.json', 'runtime_sha256')):
                allowed_provenance[str((Path(root['root']) / name).resolve())] = root[field]
        for relative, item in projection['files'].items():
            if Path(relative).parts[0] == 'policies':
                for origin in item['origins']:
                    allowed_provenance[str(Path(origin).resolve())] = item['sha256']
        if any(not Path(x).resolve().is_relative_to(corpus)
               and allowed_provenance.get(str(Path(x).resolve())) != sha for x, sha in inputs.items()):
            raise ValueError("Index contains unbound input paths outside the admitted source projection")
        for objective, dataset in m["datasets"].items():
            if (dataset.get("objective") != objective
                    or Path(dataset["path"]).resolve() != exports / partition / (objective + ".npz")
                    or digest(dataset["path"]) != dataset["sha256"]):
                raise ValueError("Dataset path/objective/SHA differs from admitted export")
        trials = m["source_trials"]
        if sorted(t["number"] for t in trials) != list(range(1, count + 1)):
            raise ValueError("Source trial cursor coverage differs")
        flat = [(t["task"], case["actual_seed"]) for t in trials for case in t["cases"]]
        if len(flat) != 4 * count:
            raise ValueError("Incomplete source episode cases")
        if any(sorted(c["episode_index"] for c in t["cases"]) != [0, 1, 2, 3] for t in trials):
            raise ValueError("Source K4 episode identities differ")
        if sum(t["decisions"] for t in trials) != m["counts"]["decisions"]:
            raise ValueError("Source decision coverage differs")
        cases[partition], metadata[partition] = set(flat), m
    if (cases["train"] & cases["validation"]
            or metadata["train"]["interface"] != metadata["validation"]["interface"]
            or metadata["train"]["source_contract"] != metadata["validation"]["source_contract"]
            or set(metadata["train"]["case_ids"]) & set(metadata["validation"]["case_ids"])):
        raise ValueError("Source/validation cases or scientific interfaces disagree")


def main():
    invocation_started = time.monotonic()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bindings", type=Path, required=True)
    p.add_argument("--bindings-sha256", required=True)
    p.add_argument("--phase", choices=("export", "fit"), required=True)
    p.add_argument("--execute", action="store_true")
    a = p.parse_args()
    if digest(a.bindings) != a.bindings_sha256:
        raise ValueError("Bindings changed")
    b = json.loads(a.bindings.read_text())
    common = ["--source-commit", b["code_commit"], "--expected-hostname", b["allocated_hostname"]]
    commands = []
    if a.phase == "export":
        for partition in ("train", "validation"):
            command = [b["python"], "-m", "fluffy_bassoon.meta_rl.robotwin_history_corpus",
                       "--corpus-root", b["projection_root"],
                       "--source-recovery", b["source100_stage"]["path"],
                       "--source-sha256", b["source100_stage"]["sha256"], "--source-method", "sac",
                       "--teacher-recovery", b["teacher100_source"]["path"],
                       "--teacher-sha256", b["teacher100_source"]["sha256"], "--teacher-method", "sac",
                       "--output", str(Path(b["exports_root"]) / partition),
                       "--device", "cpu", "--threads", "4", "--objectives"]
            command += ["ad", "dpt"] if partition == "train" else ["behavior_cloning", "dpt"]
            command += ["--partition", partition]
            if partition == "validation":
                command += ["--validation-iteration", "100"]
            commands.append(command + common)
    else:
        for objective, mode in (("ad", "full"), ("ad", "static"), ("dpt", "full"), ("dpt", "static")):
            commands.append([b["python"], "-m", "fluffy_bassoon.meta_rl.robotwin_history_train",
                "--train-index", b["train_index"]["path"], "--train-sha256", b["train_index"]["sha256"],
                "--validation-index", b["validation_index"]["path"],
                "--validation-sha256", b["validation_index"]["sha256"],
                "--objective", objective, "--mode", mode,
                "--output", str(Path(b["fits_root"]) / (objective + "-" + mode)),
                "--device", "cuda:0", "--seed", "0", "--updates", "500", "--milestones", "25", "100", "500",
                "--batch-size", "8", "--max-window", "128", "--width", "128", "--layers", "2",
                "--heads", "4", "--context-length", "300", "--learning-rate", "0.0003",
                "--eval-every", "25", "--validation-batches", "4", "--threads", "4",
                "--max-seconds", "600"] + common)
    if not a.execute:
        for command in commands:
            print(shlex.join(command))
        return
    if sys.platform != "linux" or socket.gethostname() != b["allocated_hostname"]:
        raise ValueError("Only the exact admitted allocated Linux host")
    gate = b["admission"]
    required = ["closed_source100", "chain_projection_verified", "assembly_cpu_gate_passed",
                "stage100_state_matches_closed_tip", "disjoint_source_cases_verified",
                "teacher_source_qualification_reviewed", "owner_compute_reservation_admitted"]
    if a.phase == "fit":
        required += ["actual_gpu_probe_passed", "export_cpu_admission_passed"]
    if any(gate.get(k) is not True for k in required):
        raise ValueError("Missing exact owner admission: " + ",".join(k for k in required if gate.get(k) is not True))
    for key in required:
        ref = b["admission_receipts"][key]
        if digest(ref["path"]) != ref["sha256"]:
            raise ValueError("Changed owner admission receipt: " + key)
    cwd = b["code_root"]
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=cwd, text=True).strip()
    if head != b["code_commit"] or dirty:
        raise ValueError("Require reviewed exact clean code")
    if b["source100_stage"] != b["teacher100_source"]:
        raise ValueError("Primary plan preselects the same source100 SAC snapshot as teacher")
    for key in ("source100_stage", "teacher100_source", "projection_receipt") + (("train_index", "validation_index") if a.phase == "fit" else ()):
        if digest(b[key]["path"]) != b[key]["sha256"]:
            raise ValueError("Changed bound input: " + key)
    projection = json.loads(Path(b["projection_receipt"]["path"]).read_text())
    if (projection.get("status") != "byte_projection_only_not_export_admission"
            or projection.get("purpose") != "finalbank" or projection.get("source_rounds") != 100
            or projection.get("method") != "sac"
            or Path(b["projection_root"], "ASSEMBLY-INCOMPLETE").exists()
            or Path(b["projection_receipt"]["path"]).resolve() != Path(b["projection_root"], "projection.json").resolve()):
        raise ValueError("Require the exact complete final-SAC100 projection receipt")
    if a.phase == "fit":
        verify_final_indexes(b, projection)
    # Time includes subprocess loads/exports; future scheduler receipts remain
    # authoritative. No automatic continuation, retry, or bound enlargement.
    cap = 6900 if a.phase == "export" else 3300
    maximum = b["admitted_parent_seconds"]
    if type(maximum) is not int or not 0 < maximum <= cap:
        raise ValueError("Remaining parent time exceeds admitted cap")
    deadline = invocation_started + maximum
    for command in commands:
        remaining = deadline - time.monotonic()
        minimum = 120 if a.phase == "export" else 750
        if remaining < minimum:
            raise RuntimeError("Insufficient time; preserve committed outputs and report incomplete")
        subprocess.run(command, cwd=cwd, check=True, timeout=min(remaining - 60, 750) if a.phase == "fit" else remaining - 60)
        if a.phase == "fit":
            output = Path(command[command.index("--output") + 1])
            result = json.loads((output / "summary.json").read_text())
            if not result["completed"] or result["update"] != 500:
                raise RuntimeError("Controlled partial fit; no endpoint completion or automatic retry")


if __name__ == "__main__":
    main()
