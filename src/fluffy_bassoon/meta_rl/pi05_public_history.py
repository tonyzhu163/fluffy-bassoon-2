"""Immutable public support histories, independent of the collector's memory.

The collector supplies exactly the native last-layer KV/mask and eleven public
feedback fields consumed before each action chunk. A reader reconstructs its own
state over all 48 support chunks; no collector state, labels or task plan is
serialized. This module does not query a motor or execute simulator actions.
"""

import hashlib
import json
from pathlib import Path
import re

import torch

from fluffy_bassoon.evaluation_stages import sha256
from .pi05_fresh_probe import ProbeBoundary
from .warmstart import atomic_save, checked_load


FORMAT = "native_pi05_public_support_history_v1"
PARTIAL_FORMAT = "native_pi05_public_support_partial_v1"
MANIFEST_FORMAT = "native_pi05_public_support_manifest_v1"
PUBLIC_RECORD_KEYS = ("kv_key", "kv_value", "prefix_mask", "feedback")
BOUNDARY_KEYS = {"previous_command", "pending_reward", "attempt_index", "attempts",
                 "chunk_index", "attempt_end"}
SUPPORT_ATTEMPTS, ATTEMPTS, CHUNKS = 3, 4, 16
HISTORY_KEYS = {"format", "motor_sha256", "interface_sha256", "history_ids", "records",
                "boundaries", "content_sha256"}
MANIFEST_KEYS = {"format", "source_commit", "collector_id", "history_file", "history_sha256",
                 "content_sha256", "motor_sha256", "interface_sha256", "history_ids",
                 "lanes", "support_attempts", "attempts", "chunks_per_attempt",
                 "policy_chunks", "simulator_steps", "collector_checkpoint_sha256"}


def _hash(value, length=64):
    if not isinstance(value, str) or re.fullmatch(f"[0-9a-f]{{{length}}}", value) is None:
        raise ValueError("Require an explicit lowercase SHA256/source digest")
    return value


def _cpu(value):
    if not isinstance(value, torch.Tensor):
        raise ValueError("Public record fields must be tensors")
    return value.detach().to(device="cpu", copy=True).contiguous()


def _content_hash(history):
    digest = hashlib.sha256()
    def add(value):
        if isinstance(value, torch.Tensor):
            digest.update(b"tensor")
            digest.update(json.dumps([str(value.dtype), list(value.shape)]).encode())
            raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
            digest.update(len(raw).to_bytes(8, "big"))
            digest.update(raw)
        elif isinstance(value, dict):
            digest.update(b"dict")
            for key in sorted(value):
                add(key)
                add(value[key])
        elif isinstance(value, (list, tuple)):
            digest.update(b"list" + len(value).to_bytes(8, "big"))
            for entry in value:
                add(entry)
        else:
            raw = json.dumps(value, allow_nan=False, separators=(",", ":")).encode()
            digest.update(b"json" + len(raw).to_bytes(8, "big") + raw)
    add({key: value for key, value in history.items() if key != "content_sha256"})
    return digest.hexdigest()


def _ids(values):
    if (not isinstance(values, (list, tuple)) or not values
            or any(not isinstance(value, str) or not value or len(value) > 256 for value in values)
            or len(set(values)) != len(values)):
        raise ValueError("Require unique opaque support history identities")
    return len(values)


def _tensor(value, shape, *, dtype=None):
    if (not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape)
            or value.device.type != "cpu" or value.requires_grad
            or (dtype is not None and value.dtype != dtype) or not torch.isfinite(value).all()):
        raise ValueError("Require finite detached CPU public tensors with the declared shape/dtype")


def _record(record, lanes, index):
    if not isinstance(record, dict) or set(record) != set(PUBLIC_RECORD_KEYS):
        raise ValueError("Public records accept only KV, mask and feedback; no labels/private fields")
    key, value, mask, feedback = (record[name] for name in PUBLIC_RECORD_KEYS)
    if (not isinstance(key, torch.Tensor) or key.ndim != 4 or key.shape[0] != lanes
            or key.shape[1] < 1 or tuple(key.shape[2:]) != (1, 256)
            or key.dtype not in (torch.float16, torch.bfloat16, torch.float32)):
        raise ValueError("Require native last-layer KV [B,P,1,256]")
    _tensor(key, key.shape)
    _tensor(value, key.shape, dtype=key.dtype)
    _tensor(mask, key.shape[:2], dtype=torch.bool)
    _tensor(feedback, (lanes, 11), dtype=torch.float32)
    attempt, chunk = divmod(index, CHUNKS)
    if (not mask.any(1).all() or not ((feedback[:, 7] == 0) | (feedback[:, 7] == 1)).all()
            or not feedback[:, 8].eq(float(attempt > 0 and chunk == 0)).all()
            or not feedback[:, 9].eq(attempt / ATTEMPTS).all()
            or not feedback[:, 10].eq(chunk / CHUNKS).all()
            or ((attempt == 0 or chunk > 0) and feedback[:, 7].ne(0).any())
            or (index == 0 and feedback[:, :7].ne(0).any())):
        raise ValueError("Public feedback must follow all 48 chronological K4 support clocks")


def _boundary(boundary, lanes, attempt):
    if not isinstance(boundary, dict) or set(boundary) != BOUNDARY_KEYS:
        raise ValueError("Public boundaries exclude collector memory, noise, labels and task metadata")
    _tensor(boundary["previous_command"], (lanes, 7), dtype=torch.float32)
    _tensor(boundary["pending_reward"], (lanes,), dtype=torch.float32)
    if (type(boundary["attempt_index"]) is not int or boundary["attempt_index"] != attempt
            or type(boundary["attempts"]) is not int or boundary["attempts"] != ATTEMPTS
            or type(boundary["chunk_index"]) is not int or boundary["chunk_index"] != 0
            or boundary["attempt_end"] is not True
            or not ((boundary["pending_reward"] == 0) | (boundary["pending_reward"] == 1)).all()):
        raise ValueError("Require a complete public boundary after each support attempt")


def validate_public_history(history, *, expected_motor_sha256, expected_interface_sha256):
    """Validate complete public fields, native shapes, chronology and content hash."""
    _hash(expected_motor_sha256)
    _hash(expected_interface_sha256)
    if (not isinstance(history, dict) or set(history) != HISTORY_KEYS or history["format"] != FORMAT
            or history["motor_sha256"] != expected_motor_sha256
            or history["interface_sha256"] != expected_interface_sha256):
        raise ValueError("Public history format or motor/interface hash differs")
    lanes = _ids(history["history_ids"])
    if (not isinstance(history["records"], list) or len(history["records"]) != SUPPORT_ATTEMPTS * CHUNKS
            or not isinstance(history["boundaries"], list) or len(history["boundaries"]) != SUPPORT_ATTEMPTS):
        raise ValueError("Require exactly three complete support attempts / 48 chunks")
    for index, record in enumerate(history["records"]):
        _record(record, lanes, index)
    for attempt, boundary in enumerate(history["boundaries"], 1):
        _boundary(boundary, lanes, attempt)
        if attempt < SUPPORT_ATTEMPTS:
            next_feedback = history["records"][attempt * CHUNKS]["feedback"]
            if (not torch.equal(boundary["previous_command"], next_feedback[:, :7])
                    or not torch.equal(boundary["pending_reward"], next_feedback[:, 7])):
                raise ValueError("Pending terminal action/reward must be consumed at the next support boundary")
    if history["content_sha256"] != _content_hash(history):
        raise ValueError("Public history content hash differs")
    return dict(lanes=lanes, support_attempts=SUPPORT_ATTEMPTS, attempts=ATTEMPTS,
                chunks_per_attempt=CHUNKS, policy_chunks=lanes * SUPPORT_ATTEMPTS * CHUNKS,
                simulator_steps=lanes * SUPPORT_ATTEMPTS * 80, content_sha256=history["content_sha256"])


class PublicHistoryRecorder:
    """Callbacks for a live collector; records and boundaries are copied to CPU.

    on_chunk expects an explicit four-field projection, not a full flow-policy
    rollout record. on_boundary deliberately ignores the collector's recurrent
    memory and native RNG. Call after each of the first three support attempts.
    """

    def __init__(self, *, motor_sha256, interface_sha256, history_ids):
        self._lanes = _ids(history_ids)
        self._history = dict(format=FORMAT, motor_sha256=_hash(motor_sha256),
            interface_sha256=_hash(interface_sha256), history_ids=list(history_ids),
            records=[], boundaries=[])
        self._finished = False

    def on_chunk(self, record):
        index = len(self._history["records"])
        if self._finished or index >= SUPPORT_ATTEMPTS * CHUNKS:
            raise ValueError("Public support collection is already complete")
        if index // CHUNKS != len(self._history["boundaries"]):
            raise ValueError("Record the previous complete attempt boundary before its next chunk")
        if not isinstance(record, dict) or set(record) != set(PUBLIC_RECORD_KEYS):
            raise ValueError("Public records accept only KV, mask and feedback; no labels/private fields")
        copied = {key: _cpu(value) for key, value in record.items()}
        _record(copied, self._lanes, index)
        self._history["records"].append(copied)

    def on_boundary(self, state):
        attempt = len(self._history["boundaries"]) + 1
        if (self._finished or attempt > SUPPORT_ATTEMPTS
                or len(self._history["records"]) != attempt * CHUNKS
                or not isinstance(state, ProbeBoundary) or tuple(state.history_ids) != tuple(self._history["history_ids"])):
            raise ValueError("Require the collector's matching complete support boundary")
        # Do not serialize or transplant state.memory, state.noise_state, or any
        # private simulator metadata. The reader will recompute its own memory.
        public = dict(previous_command=_cpu(state.previous_command), pending_reward=_cpu(state.pending_reward),
            attempt_index=state.attempt_index, attempts=state.attempts,
            chunk_index=state.chunk_index, attempt_end=state.attempt_end)
        _boundary(public, self._lanes, attempt)
        self._history["boundaries"].append(public)

    def finish(self):
        if self._finished:
            raise ValueError("Public history was already finalized")
        self._history["content_sha256"] = _content_hash(self._history)
        validate_public_history(self._history, expected_motor_sha256=self._history["motor_sha256"],
                                expected_interface_sha256=self._history["interface_sha256"])
        self._finished = True
        return self._history

    def partial_snapshot(self):
        """Copy the accumulated public prefix for a separate boundary checkpoint.

        This is deliberately a different, incomplete format, even if all three
        supports have accumulated. It is not accepted by final history loading
        or replay. The caller separately saves collector policy/state and RNG;
        this API neither provides nor promises a driver/CLI resume mechanism.
        """
        for index, record in enumerate(self._history["records"]):
            _record(record, self._lanes, index)
        for attempt, boundary in enumerate(self._history["boundaries"], 1):
            _boundary(boundary, self._lanes, attempt)
        partial = dict(format=PARTIAL_FORMAT, complete=False,
            motor_sha256=self._history["motor_sha256"],
            interface_sha256=self._history["interface_sha256"],
            history_ids=list(self._history["history_ids"]),
            completed_support_attempts=len(self._history["boundaries"]),
            collected_chunks=len(self._history["records"]),
            records=[{key: _cpu(record[key]) for key in PUBLIC_RECORD_KEYS}
                     for record in self._history["records"]],
            boundaries=[{key: _cpu(boundary[key]) if isinstance(boundary[key], torch.Tensor) else boundary[key]
                         for key in sorted(BOUNDARY_KEYS)}
                        for boundary in self._history["boundaries"]])
        partial["content_sha256"] = _content_hash(partial)
        return partial


@torch.no_grad()
def replay_public_history(reader, history, *, expected_motor_sha256, expected_interface_sha256):
    """Reconstruct this reader's memory, then copy only the public probe boundary.

    reader is a Pi05FeatureBridge or its StochasticPi05BridgePolicy wrapper.
    No motor inference, action generation, gradient update or collector state
    transplant occurs. All 48 recurrent transitions start from fresh state.
    """
    receipt = validate_public_history(history, expected_motor_sha256=expected_motor_sha256,
                                     expected_interface_sha256=expected_interface_sha256)
    bridge = getattr(reader, "bridge", reader)
    if bridge.kv_layer != -1 or bridge.kv_dim != 512 or bridge.feedback_dim != 11:
        raise ValueError("Reader differs from the native last-layer public history interface")
    reader.eval()
    reference = next(bridge.parameters())
    memory = None
    for record in history["records"]:
        _, memory = bridge.read_context(((record["kv_key"].to(reference.device),
            record["kv_value"].to(reference.device)),), record["prefix_mask"].to(reference.device),
            record["feedback"].to(reference), memory)
    final = history["boundaries"][-1]
    boundary = ProbeBoundary({key: value.detach().clone() for key, value in memory.items()},
        final["previous_command"].to(reference).clone(), final["pending_reward"].clone(),
        SUPPORT_ATTEMPTS, tuple(history["history_ids"]), attempt_end=True)
    if boundary.validate() != receipt["lanes"]:
        raise ValueError("Reader reconstruction changed the support lane count")
    return boundary


def save_public_history(directory, history, *, source_commit, collector_id, collector_checkpoint_sha256=None):
    """Write a fresh immutable directory; return its manifest and file digest."""
    _hash(source_commit, 40)
    if collector_checkpoint_sha256 is not None:
        _hash(collector_checkpoint_sha256)
    if not isinstance(collector_id, str) or re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", collector_id) is None:
        raise ValueError("Require an opaque collector identity, not private task metadata")
    receipt = validate_public_history(history, expected_motor_sha256=history["motor_sha256"],
                                     expected_interface_sha256=history["interface_sha256"])
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    digest = atomic_save(directory / "history.pt", history)
    manifest = dict(format=MANIFEST_FORMAT, source_commit=source_commit, collector_id=collector_id,
        collector_checkpoint_sha256=collector_checkpoint_sha256,
        history_file="history.pt", history_sha256=digest, motor_sha256=history["motor_sha256"],
        interface_sha256=history["interface_sha256"], history_ids=list(history["history_ids"]), **receipt)
    with (directory / "manifest.json").open("x") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return dict(manifest=manifest, manifest_sha256=sha256(directory / "manifest.json"))


def load_public_history(directory, expected_manifest_sha256, *, expected_motor_sha256, expected_interface_sha256):
    """Verify externally pinned manifest, file bytes and validated public content."""
    directory = Path(directory)
    _hash(expected_manifest_sha256)
    manifest_path = directory / "manifest.json"
    if sha256(manifest_path) != expected_manifest_sha256:
        raise ValueError("Public manifest hash differs")
    manifest = json.loads(manifest_path.read_text())
    if (not isinstance(manifest, dict) or set(manifest) != MANIFEST_KEYS
            or manifest["format"] != MANIFEST_FORMAT or manifest["history_file"] != "history.pt"):
        raise ValueError("Invalid public manifest schema")
    _hash(manifest["source_commit"], 40)
    if (not isinstance(manifest["collector_id"], str)
            or re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", manifest["collector_id"]) is None):
        raise ValueError("Invalid collector identity in public manifest")
    if manifest["collector_checkpoint_sha256"] is not None:
        _hash(manifest["collector_checkpoint_sha256"])
    history = checked_load(directory / "history.pt", manifest["history_sha256"])
    receipt = validate_public_history(history, expected_motor_sha256=expected_motor_sha256,
                                     expected_interface_sha256=expected_interface_sha256)
    if (any(manifest[key] != value for key, value in receipt.items())
            or any(manifest[key] != history[key] for key in ("motor_sha256", "interface_sha256", "history_ids"))
            or sha256(manifest_path) != expected_manifest_sha256):
        raise ValueError("Public manifest and validated support history differ")
    return history
