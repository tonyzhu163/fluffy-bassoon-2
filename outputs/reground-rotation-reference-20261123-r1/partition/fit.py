"""One seed in one already-reserved Baige allocation; no submission or reservation.

From the qualified runtime with its detached reference directory on PYTHONPATH:
  python -Bu fit.py --seed 0 --store "$store" --attempt "$attempt"

The parent explicitly selects an eligible, sealed seed and supplies a fresh attempt
in reference-native/attempts (including its exclusive training probe and reservation).
The store is reference-native or partition-namespaces/NAME under REGROUND_RUNTIME_ROOT.
Use the same store and a fresh attempt to resume. The parent owns the wall timeout,
ledger admission and exit charge. A busy seed returns 75. No other seed is scanned.

Published shards are immutable after eligibility: legacy replace-capable producers
for this seed must have finished, including any original fit writer. Hard links
pin the input inodes for this fit. Legacy partial caches are trusted under that
same contract.
All attempt and store paths must share a filesystem supporting atomic hard links
and rename of nonempty directories. The original runner can read every fit result.
"""
from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import sys
import tempfile
import time
from unittest.mock import patch

import reference as ref


REFERENCE_SHA256 = "f4013e6f00264fcd36505ab6feeaafd4db378d1c2f77f5a62abdfd7c11e0d162"
GRID_SHA256 = "d9a615ce1aa85c2602809e955dfa3d9524cc5bfc70a84d0bb16b627dd915cb58"


def validate_admission(attempt, seed):
    host = socket.gethostname()
    ref.require(attempt.parent == ref.ROOT.resolve() / "reference-native/attempts" and
                not (attempt / "exit.json").exists(), "Expected a current canonical ledger attempt")
    r = ref.read(attempt / "reservation.json")
    ref.require(re.fullmatch(r"job-[a-z0-9]+-master-0", host) and os.environ.get("KUBERNETES_SERVICE_HOST")
                and os.environ.get("WORLD_SIZE", "1") == "1", "Expected a Baige Distributed Training master")
    ref.require(r["host"] == host and r["phase"] == "fit" and r["gpus"] == 1 and r["seeds"] == [seed]
                and r["upstream"] == ref.PIN and r["policy"] == ref.POLICY
                and r["code"] == os.environ["REFERENCE_CODE_PIN"], "Fit reservation identity differs")
    cap = float(os.environ["REFERENCE_CAP_GPUH"])
    namespace = "canonical" if ref.STORE == ref.ROOT / "reference-native" else ref.STORE.name
    expected_store = (ref.ROOT / "reference-native" if namespace == "canonical" else
                      ref.ROOT / "partition-namespaces" / namespace)
    ref.require(re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,95}", namespace)
                and ref.STORE.resolve() == expected_store, "Invalid fit namespace")
    exact_store = ref.ROOT / ("reference-native" if namespace == "canonical" else f"partition-namespaces/{namespace}")
    ref.require(ref.STORE == exact_store, "Fit namespace path differs from its reservation")
    contract = dict(phase="fit", seeds=[seed], shards=None, descending=False,
                    namespace=namespace, gpus=1, wall_seconds=r["wall_seconds"], cap_gpuh=cap,
                    transport=r["code"], reference="23a6641562a4d833837f27ab0535a20ad94e8ec8",
                    upstream=ref.PIN, policy=ref.POLICY)
    ref.require(ref.read(attempt / "partition.json") == contract, "Fit selection differs from the reservation")
    ref.require(0 < r["cap_gpuh"] == cap <= 40 and r["wall_seconds"] == r["max_gpu_seconds"] > 0
                and 0 <= r["prior_gpu_seconds"] <= 3600 * cap - r["max_gpu_seconds"]
                and r["started_utc"] <= time.time() < r["started_utc"] + r["wall_seconds"],
                "Fit reservation exceeds its budget or deadline")
    probe = ref.read(attempt / "gpu.json")
    ref.require(probe["status"] == "passed" and probe["hostname"] == host
                and probe["purpose"] == "training" and probe["allocation_type"] == "baige_container"
                and probe["exclusive_training_admission"] is True and len(probe["devices"]) == 1,
                "Expected this allocation's one-GPU exclusive training probe")
    row = probe["devices"][0]
    name = row["cuda_name"].upper()
    ref.require(row["allocation_ordinal"] == 0 and row["uuid"] == os.environ.get("NVIDIA_VISIBLE_DEVICES")
                and (name in ("NVIDIA GEFORCE RTX 4090", "NVIDIA RPBZZZ6") or "RTX PRO 6000" in name),
                "Probe and scheduler allocation differ")
    return row


def check_science(seed):
    for path, digest in ((Path(ref.__file__), REFERENCE_SHA256),
                         (ref.HERE / "evidence/rotation-jobs.json.gz", GRID_SHA256)):
        ref.require(hashlib.sha256(path.read_bytes()).hexdigest() == digest, f"Wrong reference bytes: {path}")
    job = ref.grid()[seed]
    cfg = ref.read(ref.SOURCE / "configs/paper.json")["rot_z/90"]
    ref.require((job["seed"], job["cell"], job["method"], job["policy"], job["cfg"]) ==
                (seed, "rot_z/90", "reground", "pi05", cfg), "Wrong native rotation recipe")
    return cfg


def elect_file(source, target):
    """link(2) exposes one complete winner, including across independent hosts."""
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except FileExistsError:
        return False
    return True


def snapshot_inputs(seed, private):
    from dataclasses import asdict
    from reground.data.babble import BabbleCfg

    snapshot = private / "input-runs"
    hashes = {}
    for phase, count in (("babble", 50), ("collect", 25)):
        jobs = ref.units(phase, [seed])
        for cell in ref.CELLS:
            selected = [j for j in jobs if j["cell"] == cell]
            paths = [ref.output(j) for j in selected]
            ref.require(len(paths) == count and set(paths) == set(paths[0].parent.glob("*.npz")),
                        f"Expected exactly {count} sealed {cell} {phase} shards for seed {seed}")
            for job, path in zip(selected, paths):
                rel = path.relative_to(ref.NATIVE)
                target = snapshot / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                seal = ref.STORE / "partition-seals" / ref.label(job)
                for suffix in (".json", ".npz"):
                    pinned = target.with_suffix(suffix)
                    os.link(path.with_suffix(suffix), pinned)
                    if seal.exists():
                        ref.require(pinned.samefile((seal / path.name).with_suffix(suffix)),
                                    f"Shard differs from its elected publication: {path}")
                meta = ref.read(target.with_suffix(".json"))
                ref.require(meta.get("cfg") == asdict(BabbleCfg()) if phase == "babble"
                            else meta.get("x0_fixed") is True, f"Wrong native collection metadata: {path}")
                hashes[str(rel)] = hashlib.sha256(target.read_bytes()).hexdigest()
    return snapshot, hashes


def fit_seed(seed, device, private, cfg, log):
    """Called under the reference seed claim. Atomic publications remain safe if two hosts enter."""
    import torch
    from reground.data import babble
    from reground.method import build
    from reground.models import cache

    ref.require(babble.RUNS == cache.RUNS == ref.NATIVE == ref.STORE / "runs",
                "Configure the namespace before importing native consumers")
    snapshot, hashes = snapshot_inputs(seed, private)
    receipt_path = ref.output(dict(kind="fit", seed=seed))
    with patch.object(ref, "NATIVE", snapshot), patch.object(babble, "RUNS", snapshot):
        data = ref.audit_data(seed)
        identity = private / "input-sha256.json"
        ref.write(identity, hashes)
        seal = ref.STORE / "partition-seals" / f"seed{seed}" / "fit-inputs.json"
        elect_file(identity, seal)
        ref.require(ref.read(seal) == hashes, f"Calibration bytes changed since the first fit: {seal}")

        fingerprint = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
        expected = []
        specs = []
        for role, cell, budget in (("nom", "none/0", 100000), ("dep", "rot_z/90", 10000)):
            for kind in ("fwd", "idm"):
                spec = cache.make_spec(kind, role, cell, seed, budget, 0.6, policy="pi05")
                specs.append(spec)
                expected.append(dict(dir=str(cache.spec_dir(spec).relative_to(ref.STORE)), kind=kind, role=role))
        receipt = dict(models=expected, data=data)
        complete = receipt_path.exists()
        if complete:
            ref.require(ref.read(receipt_path) == receipt, f"Invalid fitted seed receipt: {receipt_path}")

        def validate(directory, spec):
            provenance = directory / "partition-inputs.json"
            ref.require(provenance.is_file() and ref.read(provenance) == fingerprint,
                        f"Fit cache lacks matching calibration provenance: {directory}")
            metrics = ref.read(directory / "fit_metrics.json")
            ref.require(ref.read(directory / "spec.json") == spec
                        and (directory / "model.pt").stat().st_size > 0
                        and all(isinstance(metrics[k], dict) for k in ("heldout", "fit", "onpolicy"))
                        and metrics["n_train_eps"] > 0 and metrics["n_val_eps"] > 0,
                        f"Incomplete native model: {directory}")

        loaded = []

        def load(spec, device, **kw):
            ref.require(len(loaded) < len(specs) and spec == specs[len(loaded)], "Unexpected native fit spec")
            dest = cache.spec_dir(spec)
            loaded.append(spec)
            if dest.exists() or complete:
                validate(dest, spec)
                return cache.load_or_fit(spec, device, log=log, fit_ok=False)
            # Preserve the native pre-construction CPU RNG across deaths and contenders.
            state = ref.STORE / "fit-starts" / f"{dest.name}.pt"
            if not state.exists():
                tmp = private / f"{dest.name}.start.pt"
                torch.save(torch.get_rng_state(), tmp)
                elect_file(tmp, state)
            torch.set_rng_state(torch.load(state, map_location="cpu", weights_only=True))
            with patch.object(cache, "RUNS", private):
                model = cache.load_or_fit(spec, device, **kw)
                src = cache.spec_dir(spec)
            ref.write(src / "partition-inputs.json", fingerprint)
            validate(src, spec)
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                src.rename(dest)
            except OSError as exc:
                if exc.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                    raise
                validate(dest, spec)
                return cache.load_or_fit(spec, device, log=log, fit_ok=False)
            log(f"[checkpoint] {dest}")
            return model

        build._MEMO.clear()
        try:
            with patch.object(build, "load_or_fit", load):
                build.nominal_models(seed, device, log=log)
                build.make_controller("reground", cfg, ref.CELLS[1], seed, device, log=log, policy="pi05")
        finally:
            build._MEMO.clear()
        ref.require(loaded == specs, "Incomplete nominal/deployed model set")
    if not complete:
        staged = private / "receipt.json"
        ref.write(staged, receipt)
        complete = not elect_file(staged, receipt_path)
        ref.require(ref.read(receipt_path) == receipt, "Conflicting fitted seed receipt")
    return "existing" if complete else "published"


def run_lane(args):
    """Parent entry point after ref.STORE/NATIVE selection and ref.configure()."""
    ref.require(args.gpus == 1 and args.rank == 0 and len(args.seeds) == 1 and args.seeds[0] in ref.SEEDS,
                "Fit requires one allocated GPU and one explicitly selected seed in 0..4")
    seed = args.seeds[0]
    attempt = args.attempt.resolve(strict=True)
    row = validate_admission(attempt, seed)
    cfg = check_science(seed)
    import torch
    from fluffy_bassoon.baige_gpu import cuda_pci, pci
    ref.require(torch.cuda.device_count() == 1 and pci(row["pci_bus_id"]) == cuda_pci(0)
                and str(torch.cuda.get_device_properties(0).uuid) == row["cuda_uuid"], "Changed CUDA device")
    torch.cuda.set_device(0)
    job = dict(kind="fit", seed=seed)
    unit = ref.label(job)
    with (attempt / "fit-gpu0.jsonl").open("x", buffering=1) as events, ref.claim(job) as mine:
        if not mine:
            print(json.dumps(dict(event="busy", unit=unit, utc=time.time())), file=events)
            return 75
        started = time.monotonic()
        print(json.dumps(dict(event="start", unit=unit, utc=time.time())), file=events)
        with tempfile.TemporaryDirectory(dir=attempt, prefix=f"partial-fit-seed{seed}-") as private:
            event = fit_seed(seed, "cuda:0", Path(private), cfg, lambda msg: print(msg, flush=True))
        seconds = time.monotonic() - started
        result = dict(event=event, unit=unit, utc=time.time(), wall_s=seconds,
                      models=4, models_per_s=4 / seconds, receipt=str(ref.output(job)))
        print(json.dumps(result), file=events)
    print(json.dumps(result), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=ref.SEEDS, required=True)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--attempt", type=Path, required=True)
    args = parser.parse_args()
    args.seeds, args.rank, args.gpus = [args.seed], 0, 1
    ref.STORE = args.store.resolve()
    ref.NATIVE = ref.STORE / "runs"
    # Direct invocations also end at the gross one-card reservation deadline.
    reservation = ref.read(args.attempt / "reservation.json")
    remaining = reservation["started_utc"] + reservation["wall_seconds"] - time.time()
    ref.require(remaining > 0, "Fit reservation expired")
    signal.signal(signal.SIGALRM, signal.SIG_DFL)
    signal.setitimer(signal.ITIMER_REAL, remaining)
    ref.configure()
    return run_lane(args)


if __name__ == "__main__":
    sys.exit(main())
