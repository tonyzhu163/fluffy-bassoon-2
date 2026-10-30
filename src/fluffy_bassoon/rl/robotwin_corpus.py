"""Lossless public RoboTwin decision corpus, independent of learner replay.

The recorder writes a decision intent before the simulator is called. A native
crash can therefore leave an explicitly unresolved decision, never a fabricated
transition. Chunk-boundary RGB/proprioception is content addressed so a next
observation is reused as the following decision's current observation. All
numeric arrays use NPZ without object dtypes; metadata and language use JSON.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import time
import uuid

import numpy as np


SCHEMA = "robotwin_public_decisions_v1"
_REQUIRED_CONTEXT = ("phase", "method", "trial_id", "policy_version")


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False) + "\n").encode()


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ensure_directory(path):
    path = Path(path)
    missing = []
    current = path
    while not current.exists():
        missing.append(current)
        current = current.parent
    path.mkdir(parents=True, exist_ok=True)
    # A durable file entry also needs its newly created directory hierarchy
    # reachable after a crash (e.g. a fresh episode/chunks directory).
    for directory in reversed(missing):
        _fsync_directory(directory.parent)


def _atomic_json(path, value, *, replace=False):
    path = Path(path)
    _ensure_directory(path.parent)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".partial")
    try:
        with temporary.open("xb") as stream:
            stream.write(_json_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)  # Atomic creation; never replace a record.
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    value = np.ascontiguousarray(value)
    if value.dtype.hasobject or value.dtype.kind not in "buifc":
        raise ValueError("Corpus arrays must have a numeric non-object dtype")
    if value.dtype.kind in "fc" and not np.isfinite(value).all():
        raise ValueError("Corpus arrays must be finite")
    return value


def _array_hash(value):
    digest = hashlib.sha256(str(value.dtype).encode() + str(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _metadata(value, arrays, prefix="metadata"):
    """Preserve typed tensors while retaining readable nested JSON metadata."""
    if isinstance(value, np.ndarray) or hasattr(value, "detach"):
        key = prefix
        arrays[key] = _array(value)
        return {"numeric_array": key}
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("Corpus metadata keys must be strings")
        return {key: _metadata(item, arrays, f"{prefix}/{key}")
                for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_metadata(item, arrays, f"{prefix}/{index}")
                for index, item in enumerate(value)]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError(f"Corpus metadata is not finite JSON/tensor data: {type(value).__name__}")


class RobotwinCorpusWriter:
    """Append-only episode and chunk records with explicit crash ambiguity.

    The root handle and optional lane handles share one run output. Each handle
    owns its serial episode/context/intent state; all file operations remain on
    the parent calling thread. Reopening preserves existing records and reports
    unresolved intents; it never replays or adopts them into another lane.
    Campaign-wide action accounting is owned by the independent budget ledger.
    ``contract`` should identify source code and exact frozen motor receipts.
    """

    def __init__(self, root, contract):
        self.root = Path(root).resolve()
        _ensure_directory(self.root)
        self.contract = {**deepcopy(contract), "schema": SCHEMA}
        _json_bytes(self.contract)
        contract_path = self.root / "contract.json"
        if contract_path.exists():
            if json.loads(contract_path.read_text()) != json.loads(_json_bytes(self.contract)):
                raise ValueError("Raw RoboTwin corpus contract changed")
        else:
            _atomic_json(contract_path, self.contract)
        self.context = {}
        self.episode_id = None
        self.chunk_index = 0
        self.previous_frame = None
        self.pending = None
        self._episode_manifest = None
        self._frame_refs = {}
        self._root_writer = self
        self._lane_writers = {}
        self._stats = self._inspect_existing()
        _atomic_json(self.root / "recovery-inspection.json", self._stats, replace=True)

    def for_lane(self, lane):
        """Return a stable lane-local handle without changing the corpus schema.

        Frames, contract and aggregate counters are shared. Context, chronology,
        current episode and pending decisions are independent, including from
        the root handle used for serial evaluation. Creation does not reopen
        the corpus or overwrite its original recovery-inspection receipt.
        Statistics use corpus-wide counters with this handle's local active
        episode/pending fields; explicit inspection reconciles disk contents.
        """
        if type(lane) is not int or lane < 0:
            raise ValueError("Corpus lane must be a nonnegative integer")
        root = self._root_writer
        if lane not in root._lane_writers:
            child = object.__new__(type(root))
            child.root = root.root
            child.contract = deepcopy(root.contract)
            child.context = deepcopy(root.context)
            child.episode_id = None
            child.chunk_index = 0
            child.previous_frame = None
            child.pending = None
            child._episode_manifest = None
            child._frame_refs = root._frame_refs
            child._stats = root._stats
            child._root_writer = root
            child._lane_writers = root._lane_writers
            root._lane_writers[lane] = child
        return root._lane_writers[lane]

    def _inspect_existing(self):
        stats = dict(schema=SCHEMA, completed_chunks=0, native_completed_chunks=0,
                     unresolved_decisions=[], executed_primitive_steps=0,
                     episode_count=0, frame_count=0, frame_bytes=0, decision_array_bytes=0)
        stats["episode_count"] = sum(1 for _ in (self.root / "episodes").glob("*/reset.json"))
        for path in (self.root / "frames").glob("*.json"):
            ref = json.loads(path.read_text())
            stats["frame_count"] += 1
            stats["frame_bytes"] += int(ref["bytes"])
        for path in (self.root / "episodes").glob("*/chunks/*.json"):
            record = json.loads(path.read_text())
            stats["decision_array_bytes"] += int(record["decision_arrays"]["bytes"])
            if "native_result" in record:
                stats["decision_array_bytes"] += int(record["native_result"]["arrays"]["bytes"])
            if record["status"] == "complete":
                stats["completed_chunks"] += 1
                stats["executed_primitive_steps"] += record["native_result"]["executed_primitive_steps"]
            elif record["status"] == "native_complete":
                stats["native_completed_chunks"] += 1
                stats["unresolved_decisions"].append(str(path.relative_to(self.root)))
            else:
                stats["unresolved_decisions"].append(str(path.relative_to(self.root)))
        return stats

    def set_context(self, metadata):
        # Context remains readable JSON. Array-valued policy inputs belong to
        # record_decision_metadata and are stored in the decision NPZ instead.
        merged = {**self.context, **deepcopy(metadata)}
        _json_bytes(merged)
        self.context = merged

    def register_runtime(self, contract):
        """Record exact adapter/source/motor receipts alongside the corpus."""
        path = self.root / "runtime-contract.json"
        value = json.loads(_json_bytes(contract))
        if path.exists():
            if json.loads(path.read_text()) != value:
                raise ValueError("Raw corpus runtime contract changed")
        else:
            _atomic_json(path, value)

    def _checked_context(self):
        missing = [key for key in _REQUIRED_CONTEXT if key not in self.context]
        if missing:
            raise ValueError("Missing corpus context: " + ", ".join(missing))
        return deepcopy(self.context)

    def _save_arrays(self, relative, arrays, *, recover_existing=False):
        path = self.root / relative
        _ensure_directory(path.parent)
        if path.exists() and recover_existing:
            # A crash may leave a fully written content-addressed blob before
            # its metadata sidecar. Verify every array before adopting it.
            with np.load(path, allow_pickle=False) as saved:
                if (set(saved.files) != set(arrays)
                        or any(_array_hash(_array(saved[key])) != _array_hash(value)
                               for key, value in arrays.items())):
                    raise ValueError("Orphan corpus frame content does not match its identity")
            return self._array_ref(relative, arrays)
        temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".partial")
        try:
            with temporary.open("xb") as stream:
                np.savez_compressed(stream, **arrays)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary, path)
            _fsync_directory(path.parent)
        finally:
            temporary.unlink(missing_ok=True)
        return self._array_ref(relative, arrays)

    def _array_ref(self, relative, arrays):
        path = self.root / relative
        return dict(path=str(relative), bytes=path.stat().st_size, sha256=_sha256(path),
                    arrays={name: dict(dtype=str(value.dtype), shape=list(value.shape),
                                       sha256=_array_hash(value)) for name, value in arrays.items()})

    def _frame(self, raw, features):
        required = ("main_images", "wrist_images", "states", "task_descriptions")
        if any(key not in raw or raw[key] is None for key in required):
            raise ValueError("Need public main/wrist RGB, proprioception and language")
        arrays = {key: _array(raw[key]) for key in required[:-1]}
        for key in ("main_images", "wrist_images"):
            if arrays[key].dtype != np.uint8 or arrays[key].shape[-1] != 3:
                raise ValueError("Public camera arrays must be original uint8 RGB")
        arrays["features"] = _array(np.asarray(features, dtype=np.float32))
        language = list(raw["task_descriptions"])
        if not all(isinstance(item, str) for item in language):
            raise ValueError("Public task descriptions must be strings")
        semantic = dict(language=language, arrays={name: _array_hash(value)
                                                  for name, value in arrays.items()})
        identity = hashlib.sha256(_json_bytes(semantic)).hexdigest()
        if identity in self._frame_refs:
            return self._frame_refs[identity]
        relative = Path("frames") / (identity + ".npz")
        metadata_path = self.root / "frames" / (identity + ".json")
        if metadata_path.exists():
            ref = json.loads(metadata_path.read_text())
            if ref["sha256"] != _sha256(self.root / relative):
                raise ValueError("Existing corpus frame failed SHA256 verification")
        else:
            ref = self._save_arrays(relative, arrays, recover_existing=True)
            ref.update(content_sha256=identity, language=language)
            _atomic_json(metadata_path, ref)
            self._stats["frame_count"] += 1
            self._stats["frame_bytes"] += ref["bytes"]
        self._frame_refs[identity] = ref
        return ref

    def begin_episode(self, raw, features, reset_info):
        context = self._checked_context()
        if self.pending is not None:
            raise RuntimeError("Cannot reset while a recorded decision is unresolved")
        self.episode_id = "episode-" + uuid.uuid4().hex
        self.chunk_index = 0
        self.previous_frame = self._frame(raw, features)
        record = dict(schema=SCHEMA, episode_id=self.episode_id, context=context,
                      reset_info=deepcopy(reset_info), initial_observation=self.previous_frame,
                      previous_episode_id=getattr(self, "_previous_episode_id", None))
        _atomic_json(self.root / "episodes" / self.episode_id / "reset.json", record)
        self._episode_manifest = dict(schema=SCHEMA, episode_id=self.episode_id,
            reset="reset.json", chunks=[], pending_decision=None,
            completed_chunks=0, executed_primitive_steps=0)
        self._write_episode_manifest()
        self._previous_episode_id = self.episode_id
        self._stats["episode_count"] += 1
        self._write_stats()
        return self.episode_id

    def begin_decision(self, noise, issued_actions, metadata):
        if self.episode_id is None:
            raise RuntimeError("Record a reset before a decision")
        if self.pending is not None:
            raise RuntimeError("Previous recorded decision has not been finalized")
        arrays = {"sampled_noise": _array(noise), "issued_motor_joint_targets": _array(issued_actions)}
        if arrays["issued_motor_joint_targets"].ndim != 2:
            raise ValueError("Issued motor actions must have shape [actions, joints]")
        self.chunk_index += 1
        stem = Path("episodes") / self.episode_id / "chunks" / f"{self.chunk_index:06d}"
        encoded_metadata = _metadata(metadata, arrays)
        ref = self._save_arrays(stem.with_suffix(".npz"), arrays)
        record = dict(schema=SCHEMA, status="intent", episode_id=self.episode_id,
                      chunk_index=self.chunk_index, context=self._checked_context(),
                      previous_chunk_index=self.chunk_index - 1 if self.chunk_index > 1 else None,
                      observation=self.previous_frame, decision_metadata=encoded_metadata,
                      decision_arrays=ref, issued_primitive_steps=len(arrays["issued_motor_joint_targets"]),
                      intent_created_unix_ns=time.time_ns())
        path = self.root / stem.with_suffix(".json")
        _atomic_json(path, record)
        self.pending = dict(path=path, record=record)
        self._episode_manifest["pending_decision"] = str(path.relative_to(path.parent.parent))
        self._write_episode_manifest()
        self._stats["decision_array_bytes"] += ref["bytes"]
        self._stats["unresolved_decisions"].append(str(path.relative_to(self.root)))
        self._write_stats()
        return str(path.relative_to(self.root))

    def complete_native(self, *, next_raw, next_features, raw_rewards, raw_terminations,
                        raw_truncations, valid_action_mask, reward, terminated,
                        truncated, executed_steps, success, duration_seconds,
                        adapter_rewards=None, adapter_terminations=None,
                        adapter_truncations=None, native_info=None):
        if self.pending is None:
            raise RuntimeError("No recorded simulator intent to resolve")
        record = self.pending["record"]
        mask = _array(valid_action_mask)
        if (not np.isin(mask, (0, 1)).all() or int(mask.sum()) != executed_steps
                or not 0 <= executed_steps <= record["issued_primitive_steps"]):
            raise ValueError("Native valid-action mask disagrees with executed-action count")
        stem = self.pending["path"].relative_to(self.root).with_suffix("")
        arrays = {"raw_rewards": _array(raw_rewards),
                  "raw_terminations": _array(raw_terminations),
                  "raw_truncations": _array(raw_truncations),
                  "valid_action_mask": mask}
        for key, value in (("adapter_rewards", adapter_rewards),
                           ("adapter_terminations", adapter_terminations),
                           ("adapter_truncations", adapter_truncations)):
            if value is not None:
                arrays[key] = _array(value)
        encoded_info = _metadata(native_info or {}, arrays, "native_info")
        native_arrays = self._save_arrays(stem.with_name(stem.name + "-native").with_suffix(".npz"), arrays)
        self._stats["decision_array_bytes"] += native_arrays["bytes"]
        next_frame = self._frame(next_raw, next_features)
        record.update(status="native_complete", next_observation=next_frame,
                      native_result=dict(reward=float(reward), terminated=bool(terminated),
                          truncated=bool(truncated), success=bool(success),
                          executed_primitive_steps=int(executed_steps),
                          action_count_semantics="native valid-action mask; terminal last action may be partly executed",
                          duration_seconds=float(duration_seconds), info=encoded_info,
                          arrays=native_arrays))
        _atomic_json(self.pending["path"], record, replace=True)
        self._stats["native_completed_chunks"] += 1
        self.previous_frame = next_frame
        self._write_stats()
        return str(self.pending["path"].relative_to(self.root))

    def finalize_learning_flags(self, *, reward, terminated, truncated, budget_truncated,
                                episode_primitive_steps):
        if self.pending is None or self.pending["record"]["status"] != "native_complete":
            raise RuntimeError("Native result must precede learning-boundary finalization")
        record = self.pending["record"]
        record.update(status="complete", learning_result=dict(reward=float(reward),
                      terminated=bool(terminated), truncated=bool(truncated),
                      budget_truncated=bool(budget_truncated),
                      episode_primitive_steps=int(episode_primitive_steps)))
        _atomic_json(self.pending["path"], record, replace=True)
        self._stats["completed_chunks"] += 1
        self._stats["native_completed_chunks"] -= 1
        self._stats["executed_primitive_steps"] += record["native_result"]["executed_primitive_steps"]
        path = str(self.pending["path"].relative_to(self.root))
        self._stats["unresolved_decisions"].remove(path)
        self._episode_manifest["chunks"].append(dict(
            path=str(self.pending["path"].relative_to(self.pending["path"].parent.parent)),
            sha256=_sha256(self.pending["path"])))
        self._episode_manifest["pending_decision"] = None
        self._episode_manifest["completed_chunks"] += 1
        self._episode_manifest["executed_primitive_steps"] += record["native_result"]["executed_primitive_steps"]
        self._write_episode_manifest()
        self.pending = None
        self._write_stats()
        return path

    def _write_stats(self):
        # Lane callbacks execute serially on the parent, so shared counters can
        # be persisted without rescanning an ever-growing corpus per chunk.
        root = self._root_writer
        _atomic_json(self.root / "statistics.json", root._statistics_snapshot(), replace=True)

    def _write_episode_manifest(self):
        _atomic_json(self.root / "episodes" / self.episode_id / "manifest.json",
                     self._episode_manifest, replace=True)

    def statistics(self):
        # Explicit inspection (e.g. after a parallel group or during recovery)
        # independently reconciles all on-disk records, including unresolved
        # intents from another/reopened handle. Keep the shared dict identity.
        root = self._root_writer
        actual = root._inspect_existing()
        root._stats.clear()
        root._stats.update(actual)
        return self._statistics_snapshot()

    def _statistics_snapshot(self):
        return deepcopy(dict(self._stats, active_episode_id=self.episode_id,
                             pending_decision=(str(self.pending["path"].relative_to(self.root))
                                               if self.pending is not None else None)))
