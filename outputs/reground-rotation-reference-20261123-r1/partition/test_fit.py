"""Linux CPU fixtures. Use the qualified Python and upstream with CUDA_VISIBLE_DEVICES="".
With the detached reference directory on PYTHONPATH, run python -B test_fit.py -v.
All writes are temporary. Model loading, training and controller construction are doubles.
"""
from contextlib import ExitStack
from dataclasses import asdict
from multiprocessing import get_context
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import fit
import reference as ref


def fixture_fit(store, private, fail_at=None, barrier=None):
    """Use the real native spec builder, shard audit, holdout and RNG serialization."""
    from reground.data import babble
    from reground.method import build
    from reground.models import cache

    trained = []
    reloaded = []

    def cache_double(spec, device, log, fit_ok):
        if device != "cpu":
            raise AssertionError("Fixture expected a CPU device")
        directory = cache.spec_dir(spec)
        key = directory.name
        if not fit_ok:
            reloaded.append(key)
            return (directory / "model.pt").read_bytes()
        if directory.parent != private / "models":
            raise AssertionError("Model must be staged privately")
        trained.append(key)
        if len(trained) == fail_at:
            raise InterruptedError("Fixture death before model publication")
        if barrier is not None and len(trained) == 1:
            barrier.wait(timeout=60)
        directory.mkdir(parents=True)
        ref.write(directory / "spec.json", spec)
        ref.write(directory / "fit_metrics.json", dict(heldout={}, fit={}, onpolicy={}, n_train_eps=2, n_val_eps=1))
        (directory / "model.pt").write_bytes(private.name.encode())
        return (directory / "model.pt").read_bytes()

    with ExitStack() as stack:
        for module, name, value in ((ref, "STORE", store), (ref, "NATIVE", store / "runs"),
                                    (babble, "RUNS", store / "runs"), (cache, "RUNS", store / "runs"),
                                    (cache, "load_or_fit", cache_double), (build, "ReGround", None)):
            if name == "ReGround":
                stack.enter_context(patch.object(module, name))
            else:
                stack.enter_context(patch.object(module, name, value))
        private.mkdir()
        event = fit.fit_seed(0, "cpu", private, fit.check_science(0), lambda _: None)
    return event, trained, reloaded


def contender(store, private, barrier):
    ref.configure()
    event, _, _ = fixture_fit(store, private, barrier=barrier)
    (private / "event").write_text(event)


class FitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if sys.platform != "linux" or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise RuntimeError("Run on Linux with CUDA explicitly hidden")
        cls.work = cls.enterClassContext(tempfile.TemporaryDirectory(dir="/tmp", prefix="reground-fit-fixture-"))
        cls.root = Path(cls.work)
        ref.configure()
        from reground.data.babble import BabbleCfg
        import reference_cpu
        # The reference fixture generates 150 full-size synthetic NPZ/metadata pairs.
        native = cls.root / "template/runs"
        with patch.object(ref, "NATIVE", native), patch("reground.data.babble.RUNS", native):
            reference_cpu.pools()
            for phase in ("babble", "collect"):
                for job in ref.units(phase, [0]):
                    path = ref.output(job).with_suffix(".json")
                    extra = {"cfg": asdict(BabbleCfg())} if phase == "babble" else {"x0_fixed": True}
                    ref.write(path, {**ref.read(path), **extra})

    def setUp(self):
        self.base = self.root / self._testMethodName
        self.store = self.base / "partition-namespaces/fixture"
        shutil.copytree(self.root / "template", self.store)

    def test_partial_model_retry_and_existing_receipt(self):
        with self.assertRaises(InterruptedError):
            fixture_fit(self.store, self.base / "dead", fail_at=2)
        self.assertFalse((self.store / "fitted/seed0.json").exists())
        model, = (self.store / "runs/models").glob("*/model.pt")
        start, = (self.store / "fit-starts").glob(f"{model.parent.name}.pt")
        before = model.stat(), start.read_bytes()
        starts = {p.name: p.read_bytes() for p in start.parent.iterdir()}
        self.assertEqual(len(starts), 2)
        event, trained, reloaded = fixture_fit(self.store, self.base / "retry")
        self.assertEqual((event, len(trained), reloaded), ("published", 3, [model.parent.name]))
        self.assertEqual((model.stat().st_mtime_ns, start.read_bytes()), (before[0].st_mtime_ns, before[1]))
        self.assertEqual(starts, {name: (start.parent / name).read_bytes() for name in starts})
        receipt = self.store / "fitted/seed0.json"
        result = ref.read(receipt)
        self.assertEqual(set(result), {"models", "data"})
        self.assertEqual([(m["role"], m["kind"]) for m in result["models"]],
                         [("nom", "fwd"), ("nom", "idm"), ("dep", "fwd"), ("dep", "idm")])
        self.assertEqual(result["data"]["none/0"]["used_babble"], 60000)
        self.assertEqual(result["data"]["rot_z/90"]["used_vla"], 4000)
        self.assertEqual(result["data"]["rot_z/90"]["heldout_vla_episodes"], 5)
        stamp = receipt.stat().st_mtime_ns
        event, trained, reloaded = fixture_fit(self.store, self.base / "done")
        self.assertEqual((event, trained, len(reloaded)), ("existing", [], 4))
        self.assertEqual(receipt.stat().st_mtime_ns, stamp)
        provenance = model.parent / "partition-inputs.json"
        original = provenance.read_bytes()
        provenance.unlink()
        with self.assertRaisesRegex(ValueError, "calibration provenance"):
            fixture_fit(self.store, self.base / "unattributed-cache")
        provenance.write_bytes(original)
        model.unlink()
        with self.assertRaises(FileNotFoundError):
            fixture_fit(self.store, self.base / "corrupt")
        self.assertFalse(model.exists())

    def test_incomplete_and_unsealed_inputs(self):
        path = self.store / "runs/data/rot_z/90/seed0/vla_pi05/shard024.npz"
        sidecar = path.with_suffix(".json")
        saved = sidecar.read_bytes()
        sidecar.unlink()
        with self.assertRaises(FileNotFoundError):
            fixture_fit(self.store, self.base / "orphan")
        sidecar.write_bytes(saved)
        ref.write(sidecar, {**ref.read(sidecar), "vla_revision": "foreign"})
        with self.assertRaises(ValueError):
            fixture_fit(self.store, self.base / "foreign-policy")
        sidecar.write_bytes(saved)
        seal = self.store / "partition-seals/seed0/rot_z/90/collect/shard024"
        seal.mkdir(parents=True)
        shutil.copyfile(sidecar, seal / sidecar.name)
        with self.assertRaisesRegex(ValueError, "elected publication"):
            fixture_fit(self.store, self.base / "mismatched-seal")
        path.unlink()
        with self.assertRaisesRegex(ValueError, "25 sealed"):
            fixture_fit(self.store, self.base / "missing")
        self.assertFalse((self.store / "fit-starts").exists())
        self.assertFalse((self.store / "fitted").exists())

    def test_changed_pool_cannot_reuse_model(self):
        import numpy as np
        with self.assertRaises(InterruptedError):
            fixture_fit(self.store, self.base / "dead", fail_at=2)
        path = self.store / "runs/data/none/0/seed0/babble/shard000.npz"
        # Legacy publishers replace paths. Keep all shapes, counts and metadata valid.
        with np.load(path) as data:
            arrays = dict(data)
        arrays["X"][0, 0] += 1
        replacement = self.base / "replacement.npz"
        np.savez_compressed(replacement, **arrays)
        replacement.replace(path)
        with self.assertRaisesRegex(ValueError, "Calibration bytes changed"):
            fixture_fit(self.store, self.base / "changed")
        self.assertFalse((self.store / "fitted/seed0.json").exists())

    def test_concurrent_model_start_and_receipt_publication(self):
        # Bypass the advisory seed claim on purpose: some PFS clients admit both.
        ctx = get_context("spawn")
        barrier = ctx.Barrier(2)
        dirs = [self.base / "one", self.base / "two"]
        workers = [ctx.Process(target=contender, args=(self.store, d, barrier)) for d in dirs]
        for p in workers:
            p.start()
        try:
            for p in workers:
                p.join(timeout=120)
                self.assertEqual(p.exitcode, 0)
        finally:
            for p in workers:
                if p.is_alive():
                    p.kill()
                p.join()
        self.assertEqual(sorted((d / "event").read_text() for d in dirs), ["existing", "published"])
        receipt = ref.read(self.store / "fitted/seed0.json")
        for m in receipt["models"]:
            directory = self.store / m["dir"]
            self.assertIn((directory / "model.pt").read_text(), ("one", "two"))
            self.assertTrue((self.store / "fit-starts" / f"{directory.name}.pt").is_file())
        self.assertFalse((self.base / "reference-native/runs/models").exists())

    def test_reservation_binds_seed_namespace_probe_and_caps(self):
        import partition
        host = "job-fixture-master-0"
        attempt = self.base / "reference-native/attempts/admitted"
        args = SimpleNamespace(phase="fit", seed=0, seeds=[0], gpus=1, rank=0,
                               namespace="fixture", wall=60, cap=1.0, shards=None, descending=False)
        env = dict(KUBERNETES_SERVICE_HOST="fixture", WORLD_SIZE="1", NVIDIA_VISIBLE_DEVICES="fixture-uuid",
                   REFERENCE_CODE_PIN="fixture-pin", REFERENCE_CAP_GPUH="1")
        row = dict(uuid=env["NVIDIA_VISIBLE_DEVICES"], cuda_name="NVIDIA GeForce RTX 4090", allocation_ordinal=0)
        row.update(pci_bus_id="0000:01:00.0", cuda_uuid="cuda-fixture")
        with patch.object(ref, "ROOT", self.base), patch.object(ref, "STORE", self.store), \
                patch("socket.gethostname", return_value=host), patch.dict(os.environ, env):
            reservation = dict(host=host, phase="fit", gpus=1, seeds=[0], upstream=ref.PIN, policy=ref.POLICY,
                               code="fixture-pin", wall_seconds=60, max_gpu_seconds=60, prior_gpu_seconds=3500,
                               cap_gpuh=1, started_utc=time.time() - 1)
            probe = dict(status="passed", hostname=host, purpose="training", allocation_type="baige_container",
                         exclusive_training_admission=True, devices=[row])
            ref.write(attempt / "partition.json", partition.contract(args))
            ref.write(attempt / "reservation.json", reservation)
            ref.write(attempt / "gpu.json", probe)
            self.assertEqual(fit.validate_admission(attempt, 0), row)
            mutations = [
                ("reservation.json", reservation, dict(seeds=[0, 1])),
                ("reservation.json", reservation, dict(prior_gpu_seconds=3541)),
                ("reservation.json", reservation, dict(started_utc=time.time() - 100)),
                ("partition.json", partition.contract(args), dict(namespace="canonical")),
                ("gpu.json", probe, dict(devices=[row, row])),
            ]
            for name, original, delta in mutations:
                with self.subTest(delta=delta):
                    ref.write(attempt / name, {**original, **delta})
                    with self.assertRaises(ValueError):
                        fit.validate_admission(attempt, 0)
                    ref.write(attempt / name, original)
            # Drive the same entry used by the parent's lane, with no CUDA calls or models.
            args.attempt = attempt
            with patch("fluffy_bassoon.baige_gpu.cuda_pci", return_value=row["pci_bus_id"]), \
                    patch("torch.cuda.device_count", return_value=1), patch("torch.cuda.set_device"), \
                    patch("torch.cuda.get_device_properties", return_value=SimpleNamespace(uuid=row["cuda_uuid"])), \
                    patch.object(fit, "fit_seed", return_value="existing") as payload:
                self.assertEqual(fit.run_lane(args), 0)
                seed, device, private, cfg, _ = payload.call_args.args
                self.assertEqual((seed, device, private.parent, cfg), (0, "cuda:0", attempt, fit.check_science(0)))
                events = [json.loads(line) for line in (attempt / "fit-gpu0.jsonl").read_text().splitlines()]
                self.assertEqual([e["event"] for e in events], ["start", "existing"])
                args.seeds = [0, 1]
                with self.assertRaisesRegex(ValueError, "one explicitly selected seed"):
                    fit.run_lane(args)


if __name__ == "__main__":
    unittest.main()
