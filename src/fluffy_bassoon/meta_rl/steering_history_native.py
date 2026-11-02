"""Durable actual-attempt readout through the qualified native steering motor.

The allocation runner owns model/source admission, deadlines and GPU probes.
This backend uses the same physical environment as frozen steering PPO, saves
an intent before every real dispatch, and admits only complete native truth.
"""
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import time

import numpy as np
import torch

from .frozen_steering_history import checked_reference
from .joint_steering_ppo_worker import _RecordingEnvironment, _raw, _save
from .steering_history_eval import collect_case_attempts, weights_digest


def case_key(case):
    encoded = json.dumps(case, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def attempt_noise(case, attempt, chunk):
    """Common across methods/interfaces; independent fixed streams per attempt."""
    if type(attempt) is not int or not 1 <= attempt <= 5 or type(chunk) is not int or chunk < 0:
        raise ValueError("Bind one actual attempt and prospective chunk")
    identity = dict(version="frozen_steering_actual_attempt_noise_v1", case=case,
                    attempt=attempt, chunk=chunk)
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    seed = int.from_bytes(hashlib.sha256(encoded).digest()[:8], "little")
    noise = np.random.Generator(np.random.PCG64(seed)).standard_normal((50, 32)).astype(np.float32)
    return noise, dict(identity, seed=seed, law="PCG64 standard_normal float64 cast float32")


class NativeAttemptBackend:
    def __init__(self, *, output, canonical_root, motor, spec, native_factory, journal,
                 writer_factory, binding, rank=0, environment_type=_RecordingEnvironment,
                 maximum_rgb_delta=1):
        if (not binding or maximum_rgb_delta is not None
                and (type(maximum_rgb_delta) is not int or maximum_rgb_delta < 0)):
            raise ValueError("Bind immutable source/model identity and qualified render contract")
        self.output, self.canonical_root = Path(output), Path(canonical_root)
        self.motor, self.spec, self.native_factory = motor, spec, native_factory
        self.journal, self.writer_factory, self.binding = journal, writer_factory, deepcopy(binding)
        protocol = dict(version="frozen_steering_actual_attempt_v1", interface=spec.record(),
            maximum_rgb_delta=maximum_rgb_delta, actual_attempts=5,
            feedback="Observed first-success once; full physical boundary",
            noise="frozen_steering_actual_attempt_noise_v1")
        if "native_attempt_protocol" in self.binding and self.binding["native_attempt_protocol"] != protocol:
            raise ValueError("Do not change a bound native attempt protocol")
        self.binding["native_attempt_protocol"] = protocol
        self.rank, self.environment_type, self.maximum_rgb_delta = rank, environment_type, maximum_rgb_delta
        self.environments, self.chunks, self.initials = {}, {}, {}
        self.references = []

    def _directory(self, case, attempt):
        return self.output / ("case-" + case_key(case)) / f"attempt-{attempt}"

    def _factory(self, case, attempt):
        directory = self._directory(case, attempt)
        # A partial attempt is preserved for an explicit fresh-attempt recovery,
        # never overwritten or silently reset in the same output identity.
        directory.mkdir(parents=True, exist_ok=False)
        environment = self.environment_type(self.native_factory, [deepcopy(case)], horizon=None,
            output=directory / "videos", monitoring_root=directory / "monitoring", journal=self.journal,
            writer_factory=self.writer_factory, rank=self.rank,
            episode_id_prefix=f"actual-{attempt}-", canonical_root=self.canonical_root,
            source=dict(self.binding, actual_attempt=attempt))
        key = (case_key(case), attempt)
        self.environments[key], self.chunks[key] = environment, []
        return environment

    def _before(self, row, environment):
        key = (case_key(row["case"]), row["attempt"])
        evidence = environment.initial_evidence
        if row["chunk"] == 0:
            self.initials[key] = _save(self._directory(row["case"], row["attempt"]) / "initial.pt",
                dict(binding=self.binding, case=row["case"], evidence=evidence))
            report = evidence["pairing"]
            if (report.get("matched") is not True or report.get("physical_state_exact") is not True
                    or report.get("public_state_exact") is not True
                    or set(report.get("images", {})) != {"main_images", "wrist_images", "extra_view_images"}
                    or self.maximum_rgb_delta is not None and any(
                        image["max_absolute_delta"] > self.maximum_rgb_delta for image in report["images"].values())):
                raise ValueError("Native pairing differs from the phase's qualified render contract")
        path = self._directory(row["case"], row["attempt"]) / f"chunk-{row['chunk']:04d}.pkl.gz"
        if path.exists():
            raise ValueError("Never overwrite an existing physical dispatch")
        record = dict(row, format="frozen_steering_actual_chunk_v1", status="before_dispatch",
            binding=self.binding, interface=self.spec.record(), rank=self.rank,
            raw_observation=_raw(environment), native_counts_before=environment._counts())
        _save(path, record, compressed=True)

    def _after(self, row, environment):
        directory = self._directory(row["case"], row["attempt"])
        path = directory / f"chunk-{row['chunk']:04d}.pkl.gz"
        with gzip.open(path, "rb") as stream:
            record = pickle.load(stream)
        executed = np.asarray(environment.completed_commands)
        dispatched = np.asarray(environment.dispatched_commands)
        counts = environment._counts()
        if (record["status"] != "before_dispatch" or record["binding"] != self.binding
                or not np.array_equal(record["control"], row["control"])
                or executed.shape != (50, 12) or not np.isfinite(executed).all()
                or not np.array_equal(dispatched, executed)
                or counts["controls"] - record["native_counts_before"]["controls"] != 50
                or counts["reset_controls"] != record["native_counts_before"]["reset_controls"]):
            raise ValueError("Actual native commands or immutable dispatch identity differ")
        record.update(row, status="completed", raw_next_observation=_raw(environment),
            executed_commands=executed, dispatched_commands=dispatched,
            native_counts_after=counts, first_success_step=(
                environment.success_steps[0] if environment.success_steps else None))
        self.chunks[(case_key(row["case"]), row["attempt"])].append(_save(path, record, compressed=True))

    def _completed(self, result, controller):
        key = (case_key(result["case"]), result["attempt"])
        environment = self.environments[key]
        directory = self._directory(result["case"], result["attempt"])
        truth_path = environment.monitoring_root / (environment.key + ".json")
        video_path = environment.output / (environment.key + ".json")
        truth, video = json.loads(truth_path.read_text()), json.loads(video_path.read_text())
        counts, horizon = truth["native_counts"], result["horizon"]
        if (truth["case"] != result["case"] or not video["completed"]
                or video["horizon"] != horizon or video["frame_clock"]["frame_count"] != horizon + 1
                or counts["controls"] - counts["reset_controls"] != horizon
                or truth["simulator_success_once"] != result["success_once"]
                or truth["simulator_final_success"] != result["success_at_end"]
                or truth["first_success_step"] != result["first_success_step"]
                or controller.pending or not controller.previous_done):
            raise ValueError("Only complete native simulator truth admits an attempt result")
        from .vla_history import file_sha256
        reference = lambda path: dict(path=str(path), sha256=file_sha256(path))
        for item in video["videos"]:
            checked_reference(item)
        saved = {name: value for name, value in result.items() if name not in ("records", "pairing")}
        saved.update(format="frozen_steering_actual_attempt_v1", binding=self.binding,
            interface=self.spec.record(), initial=self.initials[key], chunks=self.chunks[key],
            truth=reference(truth_path), video=reference(video_path), policy_weights_sha256=weights_digest(controller.policy))
        self.references.append(_save(directory / "complete.pt", saved))

    def recover_prefix(self, references, *, case, horizon, policy):
        prefix = []
        for attempt, reference in enumerate(references, 1):
            saved = torch.load(checked_reference(reference), map_location="cpu", weights_only=False)
            if (saved.get("format") != "frozen_steering_actual_attempt_v1"
                    or saved.get("binding") != self.binding or saved.get("interface") != self.spec.record()
                    or saved["case"] != case or saved["attempt"] != attempt or saved["horizon"] != horizon
                    or not saved["completed"] or saved["policy_weights_sha256"] != weights_digest(policy)
                    or len(saved["chunks"]) * 50 != horizon):
                raise ValueError("Recovery prefix changed native/model/case contract")
            checked_reference(saved["initial"])
            truth = json.loads(checked_reference(saved["truth"]).read_text())
            video = json.loads(checked_reference(saved["video"]).read_text())
            for item in video["videos"]:
                checked_reference(item)
            if (truth["case"] != case or truth["simulator_success_once"] != saved["success_once"]
                    or truth["simulator_final_success"] != saved["success_at_end"]
                    or truth["first_success_step"] != saved["first_success_step"]):
                raise ValueError("Recovered success differs from retained native truth")
            rows = []
            for chunk, item in enumerate(saved["chunks"]):
                with gzip.open(checked_reference(item), "rb") as stream:
                    row = pickle.load(stream)
                noise, provenance = attempt_noise(case, attempt, chunk)
                if (row["status"] != "completed" or row["binding"] != self.binding
                        or row["case"] != case or row["attempt"] != attempt or row["chunk"] != chunk
                        or row["interface"] != self.spec.record()
                        or row["native_noise_provenance"] != provenance
                        or not np.array_equal(row["native_noise"], noise)):
                    raise ValueError("Recovered native dispatch changed case/source/draw contract")
                rows.append(row)
            prefix.append(dict(saved, records=rows))
        return prefix

    def run_case(self, policy, case, *, horizon, attempts=5, prefix_references=()):
        prefix = self.recover_prefix(prefix_references, case=case, horizon=horizon, policy=policy)
        try:
            result = collect_case_attempts(policy, self.spec, case, horizon=horizon,
                environment_factory=self._factory, motor=self.motor, noise_for=attempt_noise,
                before_dispatch=self._before, after_dispatch=self._after, completed_attempt=self._completed,
                attempts=attempts, completed_prefix=prefix)
        except BaseException as error:
            failed = []
            for key, environment in self.environments.items():
                if key[0] == case_key(case):
                    failed.append(dict(attempt=key[1], observed_steps=getattr(environment, "steps", None),
                        dispatched_commands=np.asarray(environment.dispatched_commands),
                        completed_commands=np.asarray(environment.completed_commands)))
            _save(self.output / ("case-" + case_key(case)) / f"failure-{time.time_ns()}.pt",
                  dict(binding=self.binding, error=repr(error), physical_attempts=failed))
            raise
        return result
