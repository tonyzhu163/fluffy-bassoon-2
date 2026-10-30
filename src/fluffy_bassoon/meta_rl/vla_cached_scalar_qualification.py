"""Funded, zero-physics recorded-input gate for the optional scalar motor cache."""
from __future__ import annotations

import argparse
from copy import deepcopy
import gzip
import hashlib
import json
import os
from pathlib import Path
import pickle
import re
import signal
import socket
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[3]
FUNDING_SCHEMA = "libero_cached_scalar_gate_funding_v1"
INPUT_SCHEMA = "libero_cached_scalar_inputs_v1"
OPENPI_COMMIT = "215abfb217dbac7d5f1273282331b9b1866c0479"
PI05_RECEIPT = "ec064cce32f5003e2e3979c5712d7d6385018f6bfcc59cb2161105881e70d170"
SOURCE_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
FEATURE_TOLERANCE, ACTION_TOLERANCE = 2e-4, 1e-5
INNER_SECONDS, PAIR_RESERVE, SIGNATURE_RESERVE = 540, 120, 45


def require(condition, message):
    if not condition:
        raise ValueError(message)


class DeadlineReached(ValueError):
    pass


class GateFailure(ValueError):
    def __init__(self, message, metrics=None):
        super().__init__(message)
        self.metrics = metrics


class Deadline:
    def __init__(self, seconds=INNER_SECONDS, *, started=None):
        require(0 < seconds <= INNER_SECONDS, "Invalid inner qualification deadline")
        self.started = time.monotonic() if started is None else started
        self.end = self.started + seconds

    def remaining(self):
        return self.end - time.monotonic()

    def require(self, reserve):
        if self.remaining() < reserve:
            raise DeadlineReached("Insufficient time for the predeclared operation and final audit")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def reference(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def check_ref(ref):
    require(isinstance(ref, dict) and set(ref) == {"path", "sha256"}
            and isinstance(ref["path"], str) and isinstance(ref["sha256"], str)
            and Path(ref["path"]).is_absolute()
            and re.fullmatch(r"[0-9a-f]{64}", ref["sha256"]), "Invalid hash-bound artifact reference")
    return Path(ref["path"])


def read_ref(ref):
    path = check_ref(ref)
    require(sha(path) == ref["sha256"], "Artifact checksum differs: " + str(path))
    result = json.loads(path.read_text())
    require(sha(path) == ref["sha256"], "Artifact changed while reading")
    return result


def _jsonable(value):
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w") as stream:
        json.dump(_jsonable(value), stream, indent=2, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def validate_funding(record, *, run_id, source_commit, inputs_ref):
    require(record.get("schema") == FUNDING_SCHEMA and record.get("state") == "funded"
            and record.get("funded_by") == "root", "ROOT-funded record required; template is unfunded")
    require(record.get("run_id") == run_id and record.get("source_commit") == source_commit
            and re.fullmatch(r"[0-9a-f]{40}", source_commit), "Funded run/source identity differs")
    require(type(record.get("max_gpu_seconds")) is int and record["max_gpu_seconds"] == 600
            and type(record.get("robot_actions")) is int and record["robot_actions"] == 0
            and type(record.get("sgd_updates")) is int and record["sgd_updates"] == 0
            and record.get("original_compute_budget_hours") == 32
            and record.get("budget_expanded") is False, "Qualification funding scope differs")
    require(record.get("input_index") == inputs_ref, "Funded input index differs")
    for name in ("input_index", "original_32h_ledger_authority", "canonical_reservation"):
        path = check_ref(record[name])
        require(sha(path) == record[name]["sha256"], "Funding evidence changed: " + name)


def select_records(index):
    require(index.get("schema") == INPUT_SCHEMA and index.get("population_count") == 795,
            "Require the declared 795-record source population")
    goals, records = index.get("goal_keys", []), index.get("records", [])
    require(len(goals) == 14 and len(set(goals)) == 14 and len(records) == 795,
            "Require fourteen distinct source goals and all population references")
    for goal in goals:
        suite, separator, number = goal.partition(":")
        require(separator and suite in SOURCE_SUITES and re.fullmatch(r"0[0-9]", number),
                "Only original source goal keys are eligible")
    grouped = {goal: [] for goal in goals}; paths = set()
    for row in records:
        require(set(row) == {"path", "sha256", "goal_key", "rank"} and row["goal_key"] in grouped
                and type(row["rank"]) is int and row["rank"] >= 0, "Invalid source population rank/goal")
        path = check_ref({key: row[key] for key in ("path", "sha256")})
        require(str(path) not in paths, "Duplicate population record path")
        paths.add(str(path)); grouped[row["goal_key"]].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: row["rank"])
        require(len(rows) >= 3 and [row["rank"] for row in rows] == list(range(len(rows))),
                "Per-goal population ranks must be complete and unique")
    return [deepcopy(grouped[goal][position(len(grouped[goal]))])
            for position in (lambda count: 0, lambda count: count // 2, lambda count: count - 1)
            for goal in goals]


def load_index(ref, expected_sha256=None):
    if expected_sha256 is not None:
        ref = dict(path=str(Path(ref).resolve()), sha256=expected_sha256)
    index = read_ref(ref)
    selected = select_records(index)
    provenance = index["provenance"]
    require(Path(provenance["input_root"]).is_absolute()
            and re.fullmatch(r"[0-9a-f]{40}", provenance["source_commit"])
            and provenance["openpi_commit"] == OPENPI_COMMIT, "Input/model source provenance differs")
    for name in ("source_receipt", "root_qualification_inputs", "model_receipt", "model_parameter_signature"):
        path = check_ref(provenance[name])
        require(sha(path) == provenance[name]["sha256"], "Input provenance changed: " + name)
    root = Path(provenance["input_root"]).resolve()
    for row in index["records"]:
        require(Path(row["path"]).resolve().is_relative_to(root), "Record escaped its declared source root")
    for row in selected:
        require(sha(row["path"]) == row["sha256"], "Selected actual input record changed")
    return index


def verify_cpu_report(xml_path, returncode):
    root = ET.parse(xml_path).getroot()
    cases = list(root.iter("testcase"))
    counts = {name: sum(len(list(case.iter(tag))) for case in cases)
              for name, tag in (("failures", "failure"), ("errors", "error"), ("skipped", "skipped"))}
    require(returncode == 0 and len(cases) == 43 and not any(counts.values()),
            "Mandatory CPU gate requires exactly43 passing cases and zero failures/errors/skips")
    for suite in root.iter("testsuite"):
        require(all(int(suite.get(name, 0)) == 0 for name in counts), "CPU suite reports an unsuccessful case")
        require(int(suite.get("tests", -1)) == len(list(suite.iter("testcase"))), "CPU XML case count is inconsistent")
    return dict(passed=True, tests=43, passed_cases=43, **counts)


def _observed_cpu_report(path):
    if not Path(path).is_file():
        return None
    try:
        root = ET.parse(path).getroot(); cases = list(root.iter("testcase"))
        return dict(cases=len(cases), passed_cases=sum(not any(case.find(tag) is not None
                    for tag in ("failure", "error", "skipped")) for case in cases),
                    failures=sum(len(list(case.iter("failure"))) for case in cases),
                    errors=sum(len(list(case.iter("error"))) for case in cases),
                    skipped=sum(len(list(case.iter("skipped"))) for case in cases),
                    declared_suites=[dict(suite.attrib) for suite in root.iter("testsuite")])
    except ET.ParseError:
        return dict(incomplete_xml=True)


def pair_comparison(native_features, cached_features, native_actions, cached_actions,
                    native_keys, cached_keys, expected_keys):
    import numpy as np
    metrics = {}
    for name, left, right, shape, tolerance in (
        ("feature", native_features, cached_features, (4104,), FEATURE_TOLERANCE),
        ("action", native_actions, cached_actions, (10, 7), ACTION_TOLERANCE)):
        left, right = np.asarray(left), np.asarray(right)
        require(left.shape == right.shape == shape and np.isfinite(left).all() and np.isfinite(right).all(),
                "Invalid paired " + name + " shape/finite values")
        error = float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))
        metrics[name + "_maximum_absolute_error"] = error
        metrics[name + "_tolerance"] = tolerance
    metrics["exact_expected_rng_sequence"] = (
        bool(expected_keys) and _jsonable(native_keys) == _jsonable(cached_keys) == _jsonable(expected_keys))
    if (metrics["feature_maximum_absolute_error"] > FEATURE_TOLERANCE
            or metrics["action_maximum_absolute_error"] > ACTION_TOLERANCE
            or not metrics["exact_expected_rng_sequence"]):
        raise GateFailure("Original feature/action/RNG gate failed", metrics)
    return dict(passed=True, **metrics)


def _run_process(command, *, cwd, env, log, timeout):
    with Path(log).open("x") as stream:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            return process.wait(timeout=timeout)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL); process.wait(timeout=2)


def run_cpu_gate(args, deadline):
    deadline.require(90)
    xml = args.output / "cpu-tests.xml"; log = args.output / "cpu-tests.log"
    command = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--junitxml", str(xml),
               "tests/test_vla_cached_scalar_motor.py"]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
               PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(ROOT / "src"))
    started = time.monotonic()
    code = None; error = None
    try:
        code = _run_process(command, cwd=ROOT, env=env, log=log, timeout=55)
        result = verify_cpu_report(xml, code)
        require(time.monotonic() - started <= 60, "CPU fake gate exceeded its sixty-second stage deadline")
    except BaseException as failure:
        result = dict(passed=False); error = dict(type=type(failure).__name__, reason=str(failure))
    receipt = dict(result, command=command, elapsed_seconds=time.monotonic() - started,
        cuda_visible_devices="", returncode=code, deadline_seconds=60, subprocess_wait_seconds=55,
        xml=reference(xml) if xml.is_file() else None, log=reference(log) if log.is_file() else None,
        test_source=reference(ROOT / "tests/test_vla_cached_scalar_motor.py"))
    receipt.update(error=error, observed=_observed_cpu_report(xml))
    write_json(args.output / "cpu-receipt.json", receipt)
    if not receipt["passed"]:
        raise GateFailure("Mandatory CPU gate failed before model construction", receipt)
    return receipt


def run_gpu_probe(args, deadline):
    deadline.require(120)
    path = args.output / "gpu-probe.json"
    command = [sys.executable, str(ROOT / "scripts/[train]gpu_probe.py"), "--expected-count", "1",
               "--purpose", "evaluation", "--baige-container", "--output", str(path)]
    code = _run_process(command, cwd=ROOT, env=os.environ.copy(), log=args.output / "gpu-probe.log", timeout=45)
    probe = json.loads(path.read_text())
    require(code == 0 and probe.get("status") == "passed" and len(probe.get("devices", [])) == 1,
            "The one allocated GPU probe failed")
    return probe


def validate_context(args):
    require(sys.platform == "linux" and socket.gethostname() == args.expected_hostname
            and args.expected_hostname.startswith("job-"), "Require the exact allocated Linux hostname")
    require(re.fullmatch(r"[0-9a-f]{40}", args.source_commit)
            and subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, timeout=10).strip() == args.source_commit
            and not subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True, timeout=10).strip(),
            "Require the full clean reviewed execution pin")
    require(re.fullmatch(r"libero-cached-scalar-[A-Za-z0-9-]+", args.run_id)
            and args.output.is_absolute() and args.output.name == args.run_id
            and not args.output.is_relative_to(ROOT) and not args.output.exists(), "Fresh external run output required")
    from fluffy_bassoon.gpu_execution import validate_execution
    validate_execution("baige", os.environ)


class NativeReplayEngine:
    strict_decode_counts = True

    def __init__(self, args, index):
        from .vla_libero import FrozenOpenPI
        from .vla_cached_scalar_motor import CachedScalarMotor
        provenance = index["provenance"]
        ready = args.checkpoint / "READY.json"
        require(ready.resolve() == Path(provenance["model_receipt"]["path"]).resolve()
                and sha(ready) == provenance["model_receipt"]["sha256"] == PI05_RECEIPT,
                "Official frozen pi05_libero receipt differs")
        description = json.loads(ready.read_text())
        require(description["status"] == "verified" and description["checkpoint"] == "gs://openpi-assets/checkpoints/pi05_libero"
                and all((args.checkpoint / row["name"]).stat().st_size == row["bytes"] for row in description["files"]),
                "Frozen checkpoint readiness/file sizes differ")
        require(sha(args.checkpoint / "assets/physical-intelligence/libero/norm_stats.json")
                == "b3a44bb2810436fb62917decaea58bd4d9110255df527dea21e8fd40c960bd84",
                "Original motor normalization differs")
        require(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.openpi_root, text=True, timeout=10).strip() == OPENPI_COMMIT
                and not subprocess.check_output(["git", "status", "--porcelain"], cwd=args.openpi_root, text=True, timeout=10).strip(),
                "Require the exact clean qualified OpenPI source")
        sys.path.insert(0, str(args.openpi_root / "src"))
        self.native = FrozenOpenPI(SimpleNamespace(openpi_root=args.openpi_root, checkpoint=args.checkpoint,
            model_config="pi05_libero", feature_interface="vla-prefix", output=args.output))
        require(self.native.horizon == 10 and self.native.metadata["receipt_sha256"] == PI05_RECEIPT,
                "Frozen H10 motor identity differs")
        self.cached = CachedScalarMotor(self.native)
        saved = read_ref(provenance["model_parameter_signature"])
        self.expected_signature = saved.get("motor", saved)
        self.native_qualification = None

    def signature(self):
        from .vla_motor_replay import parameter_signature
        return parameter_signature(self.native.policy._model)

    def counts(self):
        return dict(native_decode_attempts=self.native.calls - self.cached.cached_infer_calls,
            cached_suffix_attempts=self.cached.cached_infer_calls, total_decode_attempts=self.native.calls,
            cached_native_fallback_attempts=self.cached.native_infer_calls,
            feature_encoder_calls=self.cached.feature_encoder.calls, feature_cache_hits=self.cached.feature_encoder.cache_hits,
            original_qualification_extra_reference_prefix_queries=int(self.native_qualification is not None),
            robot_actions=0, sgd_updates=0)

    def clear_cache(self):
        self.cached.clear_cache()

    def provenance_receipt(self):
        return dict(native_motor=self.native.metadata, native_feature_contract=self.native.feature_encoder.contract,
                    adapter_execution=self.cached.execution_metadata, native_feature_qualification=self.native_qualification)

    def _key(self):
        from .vla_source40_corpus import save_motor_rng
        return dict(_jsonable(save_motor_rng(self.native.policy._rng)),
                    native_dtype=str(self.native.policy._rng.dtype), native_shape=list(self.native.policy._rng.shape))

    def _restore(self, key):
        # Pairing restores this exact immutable native key, including its type;
        # it does not introduce a seed, key implementation or training stream.
        self.cached.restore_motor_rng(key)

    def _record(self, ref):
        import numpy as np
        require(sha(ref["path"]) == ref["sha256"], "Actual source inference record changed")
        with gzip.open(ref["path"], "rb") as stream:
            record = pickle.load(stream)
        require(sha(ref["path"]) == ref["sha256"], "Actual source record changed while reading")
        require(isinstance(record["observation"], dict) and isinstance(record["instruction"], str), "Invalid captured input")
        for name, shape in (("public_features", (4104,)), ("native_initial_noise", (10, 32)), ("generated_actions", (10, 7))):
            array = np.asarray(record[name])
            require(array.shape == shape and np.isfinite(array).all(), "Invalid captured " + name)
        require(np.asarray(record["native_initial_noise"]).dtype == np.float32
                and float(np.abs(record["native_initial_noise"]).max()) <= 1.5 + 1e-6,
                "Recorded bounded PPO noise contract differs")
        return record

    def pair(self, ref, *, unbounded=False, repeat=False):
        import jax
        import numpy as np
        from .vla_source40_corpus import save_motor_rng
        record = self._record(ref); obs, instruction = record["observation"], record["instruction"]
        noise = (np.repeat(np.linspace(-4, 4, 32, dtype=np.float32)[None], 10, axis=0)
                 if unbounded else np.asarray(record["native_initial_noise"]))
        self.cached.clear_cache()
        if self.native_qualification is None:
            key = self._key()
            self.native_qualification = self.cached.feature_encoder.qualify(obs, instruction)
            require(self.native_qualification.get("passed") is True and self._key() == key,
                    "Original native feature qualification/RNG gate failed")
        saved_key = self.native.policy._rng; saved = self._key(); expected = []; cursor = saved_key
        for _ in range(2 if repeat else 1):
            cursor, _ = jax.random.split(cursor)
            expected.append(dict(_jsonable(save_motor_rng(cursor)), native_dtype=str(cursor.dtype), native_shape=list(cursor.shape)))
        timing = {}; native_keys = []; cached_keys = []
        self._restore(saved_key); tick = time.monotonic()
        native_features = self.native.feature_encoder(obs, instruction)
        require(self._key() == saved, "Native feature computation advanced motor RNG")
        timing["native_feature_seconds"] = time.monotonic() - tick
        tick = time.monotonic(); native_actions = self.native.infer(obs, instruction, noise)
        timing["native_action_seconds"] = time.monotonic() - tick; native_keys.append(self._key())
        if repeat:
            native_repeat = self.native.infer(obs, instruction, noise); native_keys.append(self._key())
        self._restore(saved_key); tick = time.monotonic()
        cached_features = self.cached.feature_encoder(obs, instruction)
        reread = self.cached.feature_encoder(deepcopy(obs), instruction)
        require(np.array_equal(cached_features, reread) and self._key() == saved and self.cached.cache_pending,
                "Actual repeated feature/cache preparation failed")
        timing["cached_feature_seconds"] = time.monotonic() - tick
        tick = time.monotonic(); cached_actions = self.cached.infer(obs, instruction, noise)
        timing["cached_action_seconds"] = time.monotonic() - tick; cached_keys.append(self._key())
        if repeat:
            post = self.cached.feature_encoder(obs, instruction)
            require(np.array_equal(post, cached_features) and not self.cached.cache_pending, "Consumed KV was revived")
            attempts = self.cached.cached_infer_calls
            cached_repeat = self.cached.infer(obs, instruction, noise); cached_keys.append(self._key())
            require(self.cached.cached_infer_calls == attempts, "Consumed prefix decoded a second time")
        metrics = pair_comparison(native_features, cached_features, native_actions, cached_actions,
                                  native_keys, cached_keys, expected)
        if repeat:
            metrics["repeat"] = pair_comparison(native_features, cached_features, native_repeat, cached_repeat,
                                                 native_keys, cached_keys, expected)
        metrics.update(timing=timing, noise_kind="qualification_fixed_ramp_minus4_plus4" if unbounded else "recorded_bounded_PPO",
            noise_maximum_absolute_value=float(np.abs(noise).max()), native_keys=native_keys,
            cached_keys=cached_keys, expected_keys=expected,
            native_vs_recorded_diagnostic=dict(
                feature_maximum_absolute_error=float(np.max(np.abs(np.asarray(native_features) - record["public_features"]))),
                action_maximum_absolute_error=(None if unbounded else float(np.max(np.abs(native_actions - record["generated_actions"]))))))
        return metrics

    def stale(self, ref, kind):
        import numpy as np
        record = self._record(ref); obs, instruction = record["observation"], record["instruction"]
        self.cached.clear_cache(); before = self._key(); calls = self.native.calls
        self.cached.feature_encoder(obs, instruction)
        changed = deepcopy(obs)
        if kind == "content":
            image = np.asarray(changed["image"]).copy()
            require(image.dtype == np.uint8 and image.size, "Expected an actual uint8 source image")
            image.flat[0] = (int(image.flat[0]) + 1) % 256; changed["image"] = image
        elif kind == "instruction":
            instruction += "\n[stale-cache identity qualification probe]"
        else:
            raise ValueError("Unknown predeclared stale probe")
        try:
            self.cached.infer(changed, instruction, record["native_initial_noise"])
        except ValueError as error:
            require("Prepared scalar prefix is stale" in str(error), "Unrelated failure cannot pass a stale-cache probe")
        else:
            raise ValueError("Stale actual input unexpectedly decoded")
        require(self._key() == before and self.native.calls == calls and not self.cached.cache_pending,
                "Stale input consumed RNG or a decode")
        return dict(passed=True, kind=kind, native_decode_increment=0, cached_decode_increment=0, rng_unchanged=True)


def run_replay(index, engine, deadline, output):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    selected = select_records(index); first = selected[:14]
    rows = []; before = after = None; failure = None; status = "passed"; phase = {}
    try:
        deadline.require(SIGNATURE_RESERVE); tick = time.monotonic(); before = engine.signature()
        phase["full_parameter_signature_before"] = time.monotonic() - tick
        expected = getattr(engine, "expected_signature", before)
        require(before == expected, "Loaded full parameters differ from original source signature")
        operations = [("record", row, number < 14) for number, row in enumerate(selected)]
        operations += [("noise_probe", row, False) for row in first]
        operations += [(kind, row, False) for row in first for kind in ("content", "instruction")]
        for kind, ref, repeat in operations:
            deadline.require(PAIR_RESERVE)
            tick = time.monotonic()
            result = (engine.pair(ref, unbounded=kind == "noise_probe", repeat=repeat)
                      if kind in ("record", "noise_probe") else engine.stale(ref, kind))
            require(result.get("passed") is True, "An actual operation did not pass")
            row = dict(kind=kind, record=ref, result=result, elapsed_seconds=time.monotonic() - tick); rows.append(row)
            with (output / "operations.jsonl").open("a") as stream:
                stream.write(json.dumps(_jsonable(row), allow_nan=False) + "\n"); stream.flush(); os.fsync(stream.fileno())
    except BaseException as error:
        status = "partial" if isinstance(error, DeadlineReached) else "failed"
        failure = dict(type=type(error).__name__, reason=str(error), metrics=getattr(error, "metrics", None))
    try:
        if hasattr(engine, "clear_cache"): engine.clear_cache()
        deadline.require(SIGNATURE_RESERVE); tick = time.monotonic(); after = engine.signature()
        phase["full_parameter_signature_after"] = time.monotonic() - tick
        if before is not None and before != after:
            status = "failed"; failure = dict(type="FullParameterMismatch", reason="Full frozen model signature changed")
    except BaseException as error:
        if status == "passed": status = "partial" if isinstance(error, DeadlineReached) else "failed"
        failure = failure or dict(type=type(error).__name__, reason=str(error))
    counts = {kind: sum(row["kind"] == kind for row in rows) for kind in ("record", "noise_probe", "content", "instruction")}
    complete = counts == dict(record=42, noise_probe=14, content=14, instruction=14)
    actual_counts = engine.counts()
    expected_counts = dict(native_decode_attempts=84, cached_suffix_attempts=56, total_decode_attempts=140,
                           cached_native_fallback_attempts=14)
    if (actual_counts.get("robot_actions", 0) != 0 or actual_counts.get("sgd_updates", 0) != 0
            or (complete and getattr(engine, "strict_decode_counts", False)
                and any(actual_counts.get(name) != value for name, value in expected_counts.items()))):
        status = "failed"; failure = dict(type="DecodeOrScopeCountMismatch", reason="Actual calls or zero-action scope differs")
    qualified = status == "passed" and complete and before is not None and before == after
    result = dict(status=status, qualified=qualified, adoption_allowed=qualified,
        population_records=795, expected_records=42, completed_records=counts["record"],
        expected_noise_probes=14, completed_noise_probes=counts["noise_probe"],
        expected_stale_probes=28, completed_stale_probes=counts["content"] + counts["instruction"],
        expected_repeat_checks=14, completed_repeat_checks=min(counts["record"], 14),
        tested_goal_keys=[goal for goal in index["goal_keys"] if any(row["kind"] == "record" and row["record"]["goal_key"] == goal for row in rows)],
        selection=selected, parameters_before=before, parameters_after=after,
        weights_unchanged=before is not None and before == after, counts=actual_counts,
        expected_full_scope_decode_counts=expected_counts, phase_seconds=phase, failure=failure,
        recorded_model_phase_medians={name: statistics.median(values) for name in
            ("native_feature_seconds", "native_action_seconds", "cached_feature_seconds", "cached_action_seconds")
            if (values := [row["result"]["timing"][name] for row in rows
                           if row["kind"] == "record" and "timing" in row["result"]])},
        timing_limit="First pair includes compilation; cached feature phase includes an extra duplicate-read validation. These medians are diagnostic, not a matched-work speedup. They exclude loading, input I/O, key audit, repeat-consumption probes and final hashes",
        runtime_provenance=engine.provenance_receipt() if hasattr(engine, "provenance_receipt") else None,
        robot_actions=0, sgd_updates=0, no_automatic_retry=True,
        limits="Recorded-input B1 compatibility only; not all795 records, live throughput, reset or robot-benefit qualification")
    write_json(output / "replay-report.json", result)
    return result


def execute_gate(args, *, cpu_runner=None, probe_runner=None, engine_factory=None):
    deadline = Deadline(started=getattr(args, "started", None))
    phase = {}; tick = time.monotonic()
    validate_context(args)
    inputs_ref = dict(path=str(args.inputs.resolve()), sha256=args.inputs_sha256)
    funding_ref = dict(path=str(args.funding.resolve()), sha256=args.funding_sha256)
    funding = read_ref(funding_ref)
    validate_funding(funding, run_id=args.run_id, source_commit=args.source_commit, inputs_ref=inputs_ref)
    index = load_index(inputs_ref)
    phase["context_admission_and_input_index"] = time.monotonic() - tick
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "admission.json", dict(funding=funding_ref, input_index=inputs_ref,
        source_provenance=index["provenance"],
        source_commit=args.source_commit, run_id=args.run_id, state="funded", robot_actions=0, sgd_updates=0,
        predeclared_records=42, unbounded_noise_probes=14, stale_probes=28, max_gpu_seconds=600,
        inner_deadline_seconds=INNER_SECONDS, no_reservation_created_here=True))
    tick = time.monotonic(); cpu = (cpu_runner or run_cpu_gate)(args, deadline)
    phase["cpu_gate"] = time.monotonic() - tick
    require(cpu.get("passed") is True, "CPU gate did not pass before model construction")
    tick = time.monotonic(); probe = (probe_runner or run_gpu_probe)(args, deadline)
    phase["gpu_probe"] = time.monotonic() - tick
    require(probe.get("status") == "passed" and len(probe.get("devices", [])) == 1, "Allocated GPU probe failed")
    os.environ["LYCO_BAIGE_GPU_MAP"] = str(args.output / "gpu-probe.json")
    os.environ["LYCO_BAIGE_ISOLATED_UUID"] = probe["devices"][0]["uuid"]
    deadline.require(180)
    tick = time.monotonic(); engine = (engine_factory or NativeReplayEngine)(args, index)
    phase["model_load"] = time.monotonic() - tick
    result = run_replay(index, engine, deadline, args.output)
    result.update(run_id=args.run_id, source_commit=args.source_commit, funding=funding_ref,
        input_index=inputs_ref, cpu_gate=cpu, gpu_probe=probe,
        setup_phase_seconds=phase,
        elapsed_seconds=time.monotonic() - deadline.started, scheduler_gpu_seconds_requires_owner_terminal_receipt=True)
    write_json(args.output / "gate-report.json", result)
    return result


def main(argv=None):
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("funding", "inputs", "output", "openpi-root", "checkpoint"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("funding-sha256", "inputs-sha256", "source-commit", "expected-hostname", "run-id"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(argv); args.started = started
    for name in ("funding", "inputs", "output", "openpi_root", "checkpoint"):
        setattr(args, name, getattr(args, name).resolve())
    def stop(signum, frame):
        raise DeadlineReached("Qualification deadline/termination; no retry or adoption")
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGALRM, stop); signal.setitimer(signal.ITIMER_REAL, INNER_SECONDS)
    try:
        return 0 if execute_gate(args)["qualified"] else 2
    except BaseException as error:
        if args.output.exists():
            write_json(args.output / "failure.json", dict(status="failed", qualified=False, adoption_allowed=False,
                type=type(error).__name__, reason=str(error), robot_actions=0, sgd_updates=0, no_automatic_retry=True))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == "__main__":
    raise SystemExit(main())
