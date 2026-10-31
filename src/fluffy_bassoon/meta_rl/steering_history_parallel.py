"""Four isolated scientific fit processes using the existing history trainer.

The deployment owner supplies qualified allocated resources. No scheduler,
resource observer, learner, automatic retry, or profile reduction is added.
"""
import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time

from . import steering_history as history


FORMAT = "frozen_vla_history_parallel_fit_manifest_v1"
FIT_IDS = ("ad-full", "ad-static", "dpt-full", "dpt-static")
HOST = "__allocated_fit_host__"
SCIENTIFIC_FIT = dict(seed=0, updates=2500, width=128, layers=2, heads=4,
    context_length=256, batch_size=8, eval_every=250, validation_batches=16,
    learning_rate=3e-4, max_seconds_per_fit=360)


def file_ref(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=history.digest(path))


def write_json(path, value):
    from fluffy_bassoon.rl.robotwin_corpus import _atomic_json
    _atomic_json(Path(path), value)


def check_source(source_commit):
    root = Path(__file__).resolve().parents[3]
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()
    if actual != source_commit or dirty:
        raise ValueError("Use the exact clean reviewed fit commit")


def build_manifest(assembly_refs, *, output_root, source_commit, run_id, expected_hostname):
    history._linux(expected_hostname)
    if (set(assembly_refs) != {"ad", "dpt"} or not re.fullmatch(r"[0-9a-f]{40}", source_commit)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", run_id)):
        raise ValueError("Bind both admitted objectives, the reviewed commit and a fresh run identity")
    output_root = Path(output_root)
    if not output_root.is_absolute():
        raise ValueError("Fit outputs need one explicit absolute root")
    output_root = output_root.resolve()
    assemblies = {key: history.read_json(ref) for key, ref in assembly_refs.items()}
    profile = assemblies["ad"]["profile"]
    if (profile["fit"] != SCIENTIFIC_FIT or profile["prefix_episodes"] != 24
            or any(a["objective"] != key or a["profile"] != profile
                   or history.content_hash(a["prefix"]) != history.content_hash(assemblies["ad"]["prefix"])
                   for key, a in assemblies.items())):
        raise ValueError("Require the fixed scientific profile and identical whole-lineage source contexts")
    fits = {}
    for objective in ("ad", "dpt"):
        # Existing command generation re-admits original source/A2/teacher gates
        # and checks immutable TRAIN/VAL dataset and manifest references.
        commands = history.fit_commands(assemblies[objective], output=output_root,
            source_commit=source_commit, expected_hostname=HOST)
        for partition, ref in assemblies[objective]["datasets"][objective].items():
            metadata = history.read_json(dict(path=str(Path(ref["path"]).with_suffix(".json")),
                sha256=ref["manifest_sha256"]))
            declared = metadata["metadata"]
            expected_ids = sorted(r["lineage_id"] for r in assemblies[objective]["prefix"]["lineages"]
                                  if r["partition"] == partition)
            if (metadata.get("objective") != objective or declared.get("partition") != partition
                    or declared.get("trial_ids") != expected_ids
                    or declared.get("profile_sha256") != history.content_hash(profile)
                    or declared.get("prefix_sha256") != history.content_hash(assemblies[objective]["prefix"])):
                raise ValueError("Fit datasets changed their admitted whole-lineage TRAIN/VAL split or provenance")
        for mode, command in zip(("full", "static"), commands):
            key = objective+"-"+mode
            fits[key] = dict(objective=objective, mode=mode, argv_template=command,
                output=str(output_root/key), control=str(output_root/"processes"/key))
    return dict(format=FORMAT, run_id=run_id, source_commit=source_commit,
        assemblies=deepcopy(assembly_refs), output_root=str(output_root), fits=fits,
        profile_sha256=history.content_hash(profile), prefix_sha256=history.content_hash(assemblies["ad"]["prefix"]),
        scientific_fit=deepcopy(SCIENTIFIC_FIT),
        allocation_owner="existing execution parent / finish_assembly",
        allocation_rule="Parent chooses qualified GPU placement and accounts actual allocation GPU time; no new grant",
        total_per_fit_time_limits_seconds=4*SCIENTIFIC_FIT["max_seconds_per_fit"])


def read_manifest(reference, *, expected_hostname):
    manifest = history.read_json(reference)
    expected = build_manifest(manifest["assemblies"], output_root=manifest["output_root"],
        source_commit=manifest["source_commit"], run_id=manifest["run_id"], expected_hostname=expected_hostname)
    if manifest != expected:
        raise ValueError("Fit manifest changed the admitted trainer commands or scientific schedule")
    check_source(manifest["source_commit"])
    return manifest


def bound_command(template, *, hostname, device, threads, execution_batch_size=None):
    if (not re.fullmatch(r"cuda:[0-9]+", device) or type(threads) is not int or threads < 1
            or (execution_batch_size is not None and
                (type(execution_batch_size) is not int or not 1 <= execution_batch_size <= 8))):
        raise ValueError("Bind an allocated CUDA device, positive CPU threads and optional execution batch <=8")
    command = list(template)
    command[command.index("--expected-hostname")+1] = hostname
    command[command.index("--device")+1] = device
    command += ["--threads", str(threads)]
    if execution_batch_size is not None:
        command += ["--execution-batch-size", str(execution_batch_size)]
    return command


def run_fits(reference, *, placements, expected_hostname, threads=1, execution_batches=None):
    """Start ALL selected processes before waiting for any; independent failures.

    A parent may select all four on one qualified GPU, several GPUs, or one
    fit per independent allocated job. Every process owns a distinct control
    directory and existing trainer output. Scheduler GPU visibility is retained.
    """
    manifest = read_manifest(reference, expected_hostname=expected_hostname)
    execution_batches = execution_batches or {}
    if not placements or set(placements)-set(FIT_IDS) or set(execution_batches)-set(placements):
        raise ValueError("Select a nonempty distinct subset of the four declared fits")
    keys = [key for key in FIT_IDS if key in placements]
    commands = {key: bound_command(manifest["fits"][key]["argv_template"],
        hostname=expected_hostname, device=placements[key], threads=threads,
        execution_batch_size=execution_batches.get(key)) for key in keys}
    controls = {key: Path(manifest["fits"][key]["control"]) for key in keys}
    if any(Path(manifest["fits"][key]["output"]).exists() or controls[key].exists() for key in keys):
        raise ValueError("Never reuse another fit process, checkpoint or output identity")
    # Atomic per-fit ownership also protects independent jobs sharing this root.
    for path in controls.values():
        path.mkdir(parents=True, exist_ok=False)
    children, streams, records = {}, {}, {}
    stopping = False

    def stop(signum, _frame):
        nonlocal stopping
        stopping = True
        for child in children.values():
            if child.poll() is None:
                try:
                    child.send_signal(signum)
                except ProcessLookupError:
                    pass

    previous = {s: signal.signal(s, stop) for s in (signal.SIGINT, signal.SIGTERM)}
    root = Path(__file__).resolve().parents[3]
    try:
        for key in keys:
            intent = dict(format="history_fit_process_v1", fit_id=key, run_id=manifest["run_id"],
                manifest=reference, source_commit=manifest["source_commit"], hostname=expected_hostname,
                device=placements[key], threads=threads, execution_batch_size=execution_batches.get(key),
                argv=commands[key], output=manifest["fits"][key]["output"], started_unix=time.time())
            write_json(controls[key]/"command.json", intent)
            records[key] = dict(command=file_ref(controls[key]/"command.json"), pid=None,
                returncode=None, status="not_started")
            if stopping:
                continue
            environment = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads))
            streams[key] = (controls[key]/"trainer.log").open("x")
            try:
                # Exec a fresh interpreter; optimizer, sampler and all global /
                # private RNGs live only in this child, with the matched seed.
                child = subprocess.Popen(commands[key], cwd=root, env=environment,
                    stdout=streams[key], stderr=subprocess.STDOUT, start_new_session=True)
                children[key] = child
                records[key].update(pid=child.pid, status="running")
            except OSError as error:
                records[key].update(status="launch_failed", error=dict(type=type(error).__name__, message=str(error)))
        # No child wait occurs in the launch loop above.
        for key, child in children.items():
            records[key].update(returncode=child.wait(), status="exited")
    except BaseException:
        stop(signal.SIGTERM, None)
        for key, child in children.items():
            records[key].update(returncode=child.wait(), status="interrupted")
        raise
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
        for stream in streams.values():
            stream.close()
        for key, record in records.items():
            record.update(ended_unix=time.time(), interrupted=stopping,
                candidate_admission="pending separate final-artifact collection")
            output = Path(manifest["fits"][key]["output"])
            record["artifacts"] = {name: file_ref(output/name) for name in
                ("candidate.pt", "latest.pt", "metrics.jsonl", "contract.json") if (output/name).is_file()}
            if (controls[key]/"trainer.log").is_file():
                record["log"] = file_ref(controls[key]/"trainer.log")
            write_json(controls[key]/"process-result.json", record)
    return records


def verify_finished_fit(manifest, key, assembly):
    """Admit the existing final candidate AND its complete independent recovery."""
    from .warmstart import checked_load
    from .robotwin_history_corpus import parameter_digest
    row = manifest["fits"][key]
    result = history.read_json(file_ref(Path(row["control"])/"process-result.json"))
    command = history.read_json(result["command"])
    if (result.get("status") != "exited" or result.get("returncode") != 0 or result.get("interrupted")
            or command.get("fit_id") != key or command.get("run_id") != manifest["run_id"]
            or command.get("source_commit") != manifest["source_commit"] or command.get("output") != row["output"]
            or history.read_json(command["manifest"]) != manifest
            or command["argv"] != bound_command(row["argv_template"], hostname=command["hostname"],
                device=command["device"], threads=command["threads"], execution_batch_size=command["execution_batch_size"])):
        raise ValueError("A fit is missing, failed, interrupted, or changed its immutable command identity")
    expected_names = {"candidate.pt", "latest.pt", "metrics.jsonl", "contract.json"}
    if set(result["artifacts"]) != expected_names:
        raise ValueError("Partial fits retain recoveries but cannot enter the four-fit aggregate")
    refs = result["artifacts"]
    for name, ref in refs.items():
        if history.reference(ref) != Path(row["output"])/name:
            raise ValueError("Fit artifacts cannot alias another process output")
    policy, candidate = history.load_student(refs["candidate.pt"], assembly, mode=row["mode"],
        source_commit=manifest["source_commit"], device="cpu")
    recovery = checked_load(history.reference(refs["latest.pt"]), refs["latest.pt"]["sha256"], weights_only=False)
    if (not {"format", "contract", "update", "policy", "optimizer", "rng", "best_validation_nll", "elapsed_seconds"} <= recovery.keys()
            or recovery.get("format") != "frozen_vla_history_resume_v1"
            or recovery.get("update") != SCIENTIFIC_FIT["updates"]
            or recovery.get("contract") != candidate["contract"]
            or not recovery.get("optimizer", {}).get("state")
            or not recovery.get("optimizer", {}).get("param_groups")
            or not {"sampler", "python", "numpy", "torch", "cuda"} <= recovery.get("rng", {}).keys()
            or parameter_digest(recovery["policy"]) != parameter_digest(policy.state_dict())
            or history.read_json(refs["contract.json"]) != candidate["contract"]):
        raise ValueError("Require the same final weights and complete optimizer/sampler/global/private RNG recovery")
    metrics = [json.loads(line) for line in history.reference(refs["metrics.jsonl"]).read_text().splitlines()]
    if (not metrics or metrics[-1]["update"] != SCIENTIFIC_FIT["updates"]
            or not all(math.isfinite(metrics[-1][key]) for key in ("nll", "validation_nll"))
            or metrics[-1]["validation_nll"] != candidate["validation_nll"]):
        raise ValueError("Final candidate lacks its actual endpoint training/validation measurements")
    for ref in refs.values():
        history.reference(ref)
    return dict(candidate=refs["candidate.pt"], recovery=refs["latest.pt"],
        metrics=refs["metrics.jsonl"], process=file_ref(Path(row["control"])/"process-result.json"),
        endpoint=metrics[-1], validation_diagnostic=candidate["validation_nll"])


def collect_fits(reference, *, expected_hostname, output):
    manifest = read_manifest(reference, expected_hostname=expected_hostname)
    assemblies = {key: history.read_json(ref) for key, ref in manifest["assemblies"].items()}
    # No aggregate file is created until ALL four actual final artifacts pass.
    results = {key: verify_finished_fit(manifest, key, assemblies[manifest["fits"][key]["objective"]]) for key in FIT_IDS}
    comparisons = {}
    for objective in ("ad", "dpt"):
        full, static = (results[objective+"-"+mode]["endpoint"] for mode in ("full", "static"))
        comparisons[objective] = dict(full_minus_static_train_nll=full["nll"]-static["nll"],
            full_minus_static_validation_nll=full["validation_nll"]-static["validation_nll"],
            full_validation_minus_train_nll=full["validation_nll"]-full["nll"],
            static_validation_minus_train_nll=static["validation_nll"]-static["nll"])
    result = dict(format="frozen_vla_history_four_final_fits_v1", manifest=reference,
        source_commit=manifest["source_commit"], results=results, comparisons=comparisons,
        nll_measurements="Train NLL is the final sampled global batch; validation NLL averages the fixed16 batches. Compare full/static within an objective, with complete curves.",
        interpretation="Completed imitation fits; robot generalization, reward causality and ICRL remain untested",
        analysis_rule="Compare full/static within each objective. Check capacity, underfit and data/teacher issues before architecture claims; use the genuine history/reward robot controls for mechanism conclusions.")
    write_json(output, result)
    return result


def assignments(values, *, integer=False):
    result = {}
    for value in values:
        key, separator, item = value.partition("=")
        if not separator or key in result or key not in FIT_IDS:
            raise ValueError("Use distinct FIT_ID=value assignments")
        result[key] = int(item) if integer else item
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    for key in ("ad", "dpt"):
        prepare.add_argument("--"+key+"-assembly", type=Path, required=True)
        prepare.add_argument("--"+key+"-assembly-sha256", required=True)
    prepare.add_argument("--output-root", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--source-commit", required=True)
    prepare.add_argument("--run-id", required=True)
    run = commands.add_parser("run")
    run.add_argument("--placement", action="append", required=True)
    run.add_argument("--execution-batch", action="append", default=[])
    run.add_argument("--threads", type=int, default=1)
    collect = commands.add_parser("collect")
    collect.add_argument("--output", type=Path, required=True)
    for command in (run, collect):
        command.add_argument("--manifest", type=Path, required=True)
        command.add_argument("--manifest-sha256", required=True)
    for command in (prepare, run, collect):
        command.add_argument("--expected-hostname", required=True)
    args = parser.parse_args(argv)
    history._linux(args.expected_hostname)
    if args.command == "prepare":
        check_source(args.source_commit)
        assemblies = {key: dict(path=str(getattr(args, key+"_assembly").resolve()),
            sha256=getattr(args, key+"_assembly_sha256")) for key in ("ad", "dpt")}
        value = build_manifest(assemblies, output_root=args.output_root, source_commit=args.source_commit,
            run_id=args.run_id, expected_hostname=args.expected_hostname)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, value)
    else:
        reference = dict(path=str(args.manifest.resolve()), sha256=args.manifest_sha256)
        if args.command == "run":
            value = run_fits(reference, placements=assignments(args.placement), expected_hostname=args.expected_hostname,
                threads=args.threads, execution_batches=assignments(args.execution_batch, integer=True))
            if any(r["status"] != "exited" or r["returncode"] != 0 or r["interrupted"] for r in value.values()):
                raise SystemExit(1)
        else:
            collect_fits(reference, expected_hostname=args.expected_hostname, output=args.output)


if __name__ == "__main__":
    main()
