"""Reviewed production integration candidate; no scalar or motor semantics change.

One parent-owned scalar motor runs unchanged B1 encode/decode calls. Native
simulators execute concurrently in isolated processes. Complete groups are
admitted against a live exclusive campaign ledger; per-step reservations and
all corpus intents precede any physical dispatch. This is not an automatic
migration or resume path for an existing run.
"""
from __future__ import annotations

from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pickle
import threading
import time
import uuid

import numpy as np

from .robotwin_production_process import ProductionSimulatorPool, ProductionRPCError, _cpu_tree


BASE_PIN = "db0184089fa88a5f96c153f9b80029bb91b77def"
RESUME_BLOCKER = "ORDINARY-RESUME-BLOCKED.json"


class ProductionAdmissionError(RuntimeError):
    """No physical dispatch has started; the caller may restore its sampler."""


class ProductionVectorError(RuntimeError):
    def __init__(self, message, recovery_metadata=None):
        super().__init__(message)
        self.recovery_metadata = recovery_metadata


def _durable_json(path, value):
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _durable_pickle(path, value):
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(value,stream); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent,os.O_RDONLY)
    try: os.fsync(descriptor)
    finally: os.close(descriptor)


class ProductionRobotwinVectorEnv:
    """Parallel native RPCs with a shared unchanged motor and lane-local corpus.

    The caller sets every lane context, calls admit_group(tasks, episode_seeds),
    collects complete trials, then calls finish_group before replay admission.
    There are no external partial action cuts. Failures expose recovery metadata
    and durably block ordinary resume; only the caller owns learner recovery.
    """
    def __init__(self, scalar_env, budget, run_id, lanes, corpus_writer, *,
                 output_dir=None, _pool_factory=ProductionSimulatorPool):
        if type(lanes) is not int or lanes < 1:
            raise ValueError("lanes must be a positive integer")
        if corpus_writer is None or not callable(getattr(corpus_writer, "for_lane", None)):
            raise ValueError("Production vector collection requires lane-local raw corpus capture")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("run_id must be explicit")
        self.scalar_env, self.backend, self.budget = scalar_env, scalar_env.backend, budget
        self.run_id, self.lanes, self._pool_factory = run_id, lanes, _pool_factory
        self.output_dir = Path(output_dir if output_dir is not None else self.backend.output_dir.parent).resolve()
        if (self.output_dir / RESUME_BLOCKER).exists():
            raise ProductionAdmissionError("Run already blocks ordinary resume")
        self.root = self.output_dir / "production-vector"
        self.root.mkdir(parents=True, exist_ok=False)
        self._owner = (os.getpid(), threading.get_ident())
        self.observation_dim = scalar_env.observation_dim
        self.noise_shape = tuple(scalar_env.noise_shape)
        self.native_horizon = scalar_env.native_horizon
        declared = getattr(scalar_env, "task_episode_limits", None)
        if not declared:
            raise ValueError("Production parallel collection requires explicit per-task native horizons")
        self.task_episode_limits = dict(declared)
        self.episode_limit = max(self.task_episode_limits.values())
        self.writers = [corpus_writer.for_lane(lane) for lane in range(lanes)]
        for writer in self.writers:
            writer.register_runtime(self.backend.contract)
        self.contexts = [{} for _ in range(lanes)]
        self.metadata = [None] * lanes
        self.cache = [None] * lanes
        self.raw = [None] * lanes
        self.features = np.zeros((lanes, self.observation_dim), np.float32)
        self.active = np.zeros(lanes, bool)
        self.elapsed = np.zeros(lanes, np.int64)
        self.chunks = np.zeros(lanes, np.int64)
        self.episode_positions = np.full(lanes, -1, np.int64)
        self.last_reset_info = [None] * lanes
        self._group = self._last_group = None
        self._pool = None
        self._sequence = self._group_sequence = 0
        self.resume_blocker_metadata = None
        self.closed = False
        self.execution_contract = dict(base_pin=BASE_PIN, source_commit=budget.source_commit,
            lanes=lanes, simulator_execution="isolated processes; production native hooks",
            motor_execution="shared unchanged scalar encode/decode; microbatch 1",
            task_episode_limits=self.task_episode_limits, external_partial_cuts=False,
            device_memory=self._device_memory())
        _durable_json(self.root / "execution-contract.json", self.execution_contract)
        self._budget_snapshot()

    def _check_owner(self):
        if self.closed or self.resume_blocker_metadata is not None:
            raise ProductionVectorError("Vector is closed/blocked; no automatic resume", self.group_recovery_metadata())
        if self._owner != (os.getpid(), threading.get_ident()):
            raise RuntimeError("Production vector and shared motor require their owning parent thread")

    def _budget_snapshot(self):
        self.budget._require_open()
        if self.budget.run_id != self.run_id:
            raise ProductionAdmissionError("Vector and live campaign run identities differ")
        # The existing ledger holds a process-exclusive flock. Never pass this
        # object or descriptor to simulator workers.
        fcntl.flock(self.budget._lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        snapshot = self.budget.snapshot()
        if self._group is not None and snapshot["journal_events"] != self._expected_budget_events:
            raise ProductionVectorError("Another budget consumer interleaved an active execution group")
        return snapshot

    def _event(self, kind, **values):
        path = self.root / f"event-{self._sequence:08d}.json"
        _durable_json(path, dict(kind=kind, sequence=self._sequence,
                                unix_ns=time.time_ns(), **values))
        self._sequence += 1
        return str(path)

    def _persist(self):
        if self._group is not None:
            _durable_json(Path(self._group["journal_path"]), self._group)

    def episode_limit_for_task(self, task):
        return self.scalar_env.episode_limit_for_task(task)

    def _device_memory(self):
        if getattr(self.backend,"model",None) is None:
            return dict(scope="CPU fixture; no allocated-device query")
        free,total = self.backend.torch.cuda.mem_get_info()
        return dict(scope="instantaneous whole allocated device; includes other processes",
                    total_bytes=int(total),used_bytes=int(total-free),free_bytes=int(free),
                    name=self.backend.torch.cuda.get_device_properties(0).name,
                    pci=getattr(self.backend,"render_pci",None))

    def set_lane_context(self, lane, metadata):
        self._check_owner()
        if type(lane) is not int or not 0 <= lane < self.lanes:
            raise ValueError("Invalid collector lane")
        merged = {**self.contexts[lane], **deepcopy(metadata), "collector_lane": lane,
                  "motor_microbatch": 1}
        json.dumps(merged, allow_nan=False)
        if self._group is not None:
            for key in ("phase", "method", "trial_id", "policy_version", "checkpoint_sha256",
                        "policy_checkpoint_sha256"):
                if merged.get(key) != self.contexts[lane].get(key):
                    raise RuntimeError("A lane's behavior identity cannot change during an admitted group")
        self.contexts[lane] = merged
        self.writers[lane].set_context(merged)

    def record_decision_metadata(self, lane, metadata):
        self._check_owner()
        if type(lane) is not int or not 0 <= lane < self.lanes:
            raise ValueError("Invalid collector lane")
        self.metadata[lane] = _cpu_tree(metadata)

    def admit_group(self, tasks, episode_seeds):
        self._check_owner()
        if self._group is not None:
            raise ProductionAdmissionError("Finish or cancel the prior group before admission")
        tasks, episode_seeds = list(tasks), [list(seeds) for seeds in episode_seeds]
        if (len(tasks) != self.lanes or len(episode_seeds) != self.lanes
                or any(not seeds or any(type(seed) is not int or seed < 0 for seed in seeds)
                       for seeds in episode_seeds)):
            raise ProductionAdmissionError("Require one complete explicit trial plan per lane")
        for context, task in zip(self.contexts, tasks):
            if any(not isinstance(context.get(key), str) or not context[key]
                   for key in ("phase", "method", "trial_id", "policy_version")):
                raise ProductionAdmissionError("Every lane requires complete immutable behavior identity")
            if context.get("task", task) != task:
                raise ProductionAdmissionError("Lane context/task plan mismatch")
        identities = {(c["phase"], c["method"], c["policy_version"]) for c in self.contexts}
        if len(identities) != 1:
            raise ProductionAdmissionError("A physical execution group requires one behavior version and budget phase/method")
        before = self._budget_snapshot()
        phase, method, version = next(iter(identities))
        limits = [self.episode_limit_for_task(task) for task in tasks]
        bound = sum(len(seeds)*limit for seeds, limit in zip(episode_seeds, limits))
        remaining = self.budget.remaining(phase=phase, method=method)
        if remaining < bound:
            raise ProductionAdmissionError("Full native-horizon group does not fit the live campaign budget")
        identity = f"{self.run_id}:vector-group:{self._group_sequence}"
        self._group_sequence += 1
        path = self.root / f"group-{self._group_sequence:08d}.json"
        self._group = dict(group_id=identity, status="admitted", journal_path=str(path),
            tasks=tasks, episode_seeds=episode_seeds, episode_limits=limits,
            contexts=deepcopy(self.contexts), worst_case_actions=bound,
            budget_before=before, remaining_before=remaining, phase=phase, method=method,
            policy_version=version, dispatch_count=0, decisions=[], physical_actions=0,
            per_lane_actions=[0]*self.lanes, completed_episodes=[0]*self.lanes,
            phase_seconds=dict(motor_encode=0.,motor_decode=0.,reset_rpc=0.,step_rpc=0.,corpus=0.),
            device_memory_before=self._device_memory())
        self._group_started = time.monotonic()
        self._expected_budget_events = before["journal_events"]
        self.episode_positions[:] = -1
        self.active[:] = False
        self.metadata = [None]*self.lanes
        self._persist()
        self._write_fence()
        return deepcopy(self._group)

    def _write_fence(self):
        _durable_json(self.output_dir / RESUME_BLOCKER, dict(resume_blocked=True,
            group_id=self._group["group_id"], journal_path=self._group["journal_path"],
            vector_output=str(self.root), status=self._group["status"],
            requires_checkpoint_ack=True))

    def _clear_fence(self):
        path = self.output_dir / RESUME_BLOCKER
        if json.loads(path.read_text()).get("group_id") != self._group["group_id"]:
            raise RuntimeError("Cannot clear another execution group's ordinary-resume fence")
        path.unlink()
        descriptor = os.open(path.parent,os.O_RDONLY)
        try: os.fsync(descriptor)
        finally: os.close(descriptor)

    def campaign_journal_fence(self):
        snapshot = self._budget_snapshot()
        return dict(canonical_root=str(self.budget.root.resolve()),
            **{key:snapshot[key] for key in ("journal_events","journal_sha256",
                                            "charged_actions","decision_count")})

    def _pool_instance(self):
        if self._pool is None:
            self._pool = self._pool_factory(self.lanes, rlinf_root=self.backend.rlinf_root,
                robotwin_root=self.backend.robotwin_root, output_dir=self.root / "simulators",
                task_episode_limits=self.task_episode_limits,
                final_observation_only=self.backend.final_observation_only,
                task_config_name=self.backend.task_config_name)
        return self._pool

    def _motor_encode(self, lane, raw):
        started = time.monotonic()
        keys = ("_prepared", "_prefix_mask", "_kv_cache", "_raw_obs")
        saved = tuple(getattr(self.backend, key) for key in keys)
        native = {key: (list(value) if key == "task_descriptions" else self.backend.torch.as_tensor(value))
                  for key, value in raw.items()}
        try:
            features = np.asarray(self.backend._encode(native), np.float32).copy()
            if features.shape != (self.observation_dim,) or not np.isfinite(features).all():
                raise RuntimeError("Shared scalar motor returned invalid lane features")
            self.cache[lane] = tuple(getattr(self.backend, key) for key in keys)
            return features
        finally:
            self._group["phase_seconds"]["motor_encode"] += time.monotonic()-started
            for key, value in zip(keys, saved): setattr(self.backend, key, value)

    def _motor_decode(self, lane, noise):
        started = time.monotonic()
        keys = ("_prepared", "_prefix_mask", "_kv_cache", "_raw_obs")
        saved = tuple(getattr(self.backend, key) for key in keys)
        try:
            for key, value in zip(keys, self.cache[lane]): setattr(self.backend, key, value)
            actions = np.asarray(self.backend.decode(noise), np.float32).copy()
            if actions.shape != (self.native_horizon, 14) or not np.isfinite(actions).all():
                raise RuntimeError("Shared scalar motor returned invalid native actions")
            return actions
        finally:
            self._group["phase_seconds"]["motor_decode"] += time.monotonic()-started
            for key, value in zip(keys, saved): setattr(self.backend, key, value)

    def _writer_call(self, lane, method, *args, **kwargs):
        started = time.monotonic()
        try:
            return getattr(self.writers[lane],method)(*args,**kwargs)
        finally:
            self._group["phase_seconds"]["corpus"] += time.monotonic()-started

    def reset_batch(self, tasks, seeds):
        if self._group is None or list(tasks) != self._group["tasks"]:
            raise ProductionAdmissionError("Admit the complete group before its first reset")
        self.reset_lanes(list(range(self.lanes)), tasks, seeds)
        return self.features.copy()

    def reset_lanes(self, indices, tasks, seeds):
        self._check_owner()
        indices, tasks, seeds = list(indices), list(tasks), list(seeds)
        if (self._group is None or not indices or len(set(indices)) != len(indices)
                or len(tasks) != len(indices) or len(seeds) != len(indices)):
            raise ValueError("Reset requires an admitted group and explicit selected lane plans")
        for lane, task, seed in zip(indices, tasks, seeds):
            position = int(self.episode_positions[lane])+1
            if (self.active[lane] or task != self._group["tasks"][lane]
                    or position >= len(self._group["episode_seeds"][lane])
                    or seed != self._group["episode_seeds"][lane][position]):
                raise RuntimeError("Reset differs from its admitted complete-trial plan")
        returned = {}
        try:
            self._budget_snapshot()
            self._event("reset_intent", group_id=self._group["group_id"], lanes=indices,
                        tasks=tasks, seeds=seeds, policy_action_bound=0,
                        contexts=[self.contexts[lane] for lane in indices])
            pool = self._pool_instance()
            self._group["dispatch_count"] += 1
            self._persist()
            started = time.monotonic()
            try:
                raw, receipts = pool.reset(tasks, seeds, indices=indices)
            finally:
                self._group["phase_seconds"]["reset_rpc"] += time.monotonic()-started
            returned = dict(zip(indices,zip(raw,receipts)))
            for lane, observation, receipt in zip(indices, raw, receipts):
                if receipt.get("reset_policy_actions") != 0:
                    raise RuntimeError("Native reset introduced unreserved policy commands")
                self.raw[lane] = _cpu_tree(observation)
                self.features[lane] = self._motor_encode(lane, self.raw[lane])
                self.episode_positions[lane] += 1
                self.elapsed[lane] = self.chunks[lane] = 0
                self.active[lane] = True
                self.last_reset_info[lane] = dict(receipt)
                self.last_reset_info[lane]["corpus_episode_id"] = self._writer_call(lane,"begin_episode",
                    self.raw[lane], self.features[lane], self.last_reset_info[lane])
            self._event("reset_complete", group_id=self._group["group_id"], lanes=indices,
                        receipts=receipts, policy_actions=0)
            self._persist()
            return self.features[np.asarray(indices)].copy()
        except BaseException as error:
            self._group["failure_operation"] = "reset"
            self._fail(error,returned or getattr(error,"completed",None))
            raise ProductionVectorError(str(error), self.group_recovery_metadata()) from error

    def _resolve_all(self, completed, reservations):
        errors = []
        for lane, result in completed.items():
            try:
                count = result[4]["primitive_steps"]
                if type(count) is not int or count < 0:
                    raise RuntimeError("Worker returned a non-exact physical action count")
                self.budget.resolve(reservations[lane], count)
                self._expected_budget_events = self.budget.snapshot()["journal_events"]
                self._group["physical_actions"] += count
                self._group["per_lane_actions"][lane] += count
            except BaseException as error:
                errors.append(f"lane {lane}: {type(error).__name__}: {error}")
        if errors:
            raise RuntimeError("; ".join(errors))

    def step_batch(self, noises, *, active_mask):
        self._check_owner()
        active = np.asarray(active_mask, bool)
        if (self._group is None or active.shape != (self.lanes,) or not active.any()
                or np.any(active & ~self.active)):
            raise ValueError("Require selected active lanes in an admitted group")
        noises = np.asarray(noises, np.float32).reshape(self.lanes, *self.noise_shape).copy()
        if not np.isfinite(noises).all(): raise ValueError("Nonfinite noise")
        indices = np.flatnonzero(active).tolist()
        reservations, completed = {}, {}
        try:
            self._budget_snapshot()
            counts = [min(self.native_horizon, self.episode_limit_for_task(self._group["tasks"][lane])
                          -int(self.elapsed[lane])) for lane in indices]
            actions = [self._motor_decode(lane, noises[lane])[:count].copy()
                       for lane, count in zip(indices, counts)]
            for lane, count, commands in zip(indices, counts, actions):
                if self.metadata[lane] is None:
                    raise RuntimeError("Record lane policy metadata before every physical decision")
                context = self.contexts[lane]
                identity = f"{self._group['group_id']}:lane:{lane}:episode:{self.episode_positions[lane]}:chunk:{self.chunks[lane]}"
                reservation = self.budget.reserve(identity, count, phase=context["phase"],
                    method=context["method"], task=self._group["tasks"][lane],
                    episode=f"{context['trial_id']}:{int(self.episode_positions[lane])+1}",
                    chunk=int(self.chunks[lane]), policy_version=context["policy_version"],
                    metadata=dict(group_id=self._group["group_id"], collector_lane=lane))
                self._expected_budget_events = self.budget.snapshot()["journal_events"]
                if not reservation.execute or reservation.granted != count:
                    raise RuntimeError("Complete-group action reservation was refused or partially granted")
                reservations[lane] = identity
                self._group["decisions"].append(identity)
                self.writers[lane].set_context(dict(campaign_decision_id=identity,
                    campaign_reserved_actions=count))
                self._writer_call(lane,"begin_decision",noises[lane], commands, self.metadata[lane])
            self._event("step_intent", group_id=self._group["group_id"], lanes=indices,
                        reservations=reservations, issued_actions=counts)
            self._group["dispatch_count"] += 1
            self._persist()
            started = time.monotonic()
            try:
                rows = self._pool.step(actions, indices=indices)
                completed = dict(zip(indices, rows))
            except ProductionRPCError as error:
                completed = error.completed
                self._resolve_all(completed, reservations)
                raise
            finally:
                self._group["phase_seconds"]["step_rpc"] += time.monotonic()-started
            # Resolve every returned physical count before encoding, corpus
            # serialization, or another operation that could reject a prefix.
            self._resolve_all(completed, reservations)
            rewards = np.zeros(self.lanes, np.float32)
            terms = np.zeros(self.lanes, bool); truncs = np.zeros(self.lanes, bool)
            primitives = np.zeros(self.lanes, np.int64); success = np.zeros(self.lanes, bool)
            masks = np.zeros((self.lanes,self.native_horizon), bool)
            chunks = [None]*self.lanes
            for lane, result in completed.items():
                raw, reward, native_term, native_trunc, info = result
                count = info["primitive_steps"]
                self.raw[lane] = _cpu_tree(raw)
                self.features[lane] = self._motor_encode(lane, self.raw[lane])
                writer = self.writers[lane]
                self._writer_call(lane,"complete_native",next_raw=self.raw[lane], next_features=self.features[lane],
                    reward=reward, terminated=native_term, truncated=native_trunc,
                    executed_steps=count, success=info["success"],
                    **{key:info[key] for key in ("valid_action_mask", "duration_seconds",
                       "adapter_rewards", "adapter_terminations", "adapter_truncations",
                       "raw_rewards", "raw_terminations", "raw_truncations", "native_info")})
                self.elapsed[lane] += count
                terminated = bool(native_term)
                truncated = bool(not terminated and (native_trunc or self.elapsed[lane]
                    >= self.episode_limit_for_task(self._group["tasks"][lane])))
                chunks[lane] = self._writer_call(lane,"finalize_learning_flags",reward=reward,
                    terminated=terminated, truncated=truncated, budget_truncated=False,
                    episode_primitive_steps=int(self.elapsed[lane]))
                self.chunks[lane] += 1
                self.metadata[lane] = None
                self.active[lane] = not (terminated or truncated)
                self._group["completed_episodes"][lane] += int(not self.active[lane])
                rewards[lane], terms[lane], truncs[lane] = reward, terminated, truncated
                primitives[lane], success[lane] = count, info["success"]
                mask = np.asarray(info["valid_action_mask"], bool).reshape(-1)
                masks[lane,:len(mask)] = mask
            self._event("step_complete", group_id=self._group["group_id"], lanes=indices,
                        physical_actions=primitives.tolist())
            self._persist()
            return self.features.copy(), rewards, terms, truncs, dict(
                primitive_steps=primitives, success=success, valid_action_mask=masks,
                budget_truncated=np.zeros(self.lanes,bool), corpus_chunk=chunks,
                episode_primitive_steps=self.elapsed.copy())
        except BaseException as error:
            self._group["failure_operation"] = "step"
            self._fail(error, completed)
            raise ProductionVectorError(str(error), self.group_recovery_metadata()) from error

    def finish_group(self):
        self._check_owner()
        if (self._group is None or self.active.any()
                or self._group["completed_episodes"] != [len(x) for x in self._group["episode_seeds"]]
                or any(writer.pending is not None for writer in self.writers)):
            raise RuntimeError("A group must contain every complete native episode and finalized corpus decision")
        snapshot = self._budget_snapshot()
        if any(identity in snapshot["unresolved_decisions"] for identity in self._group["decisions"]):
            raise RuntimeError("Group contains unresolved physical action reservations")
        self._group.update(status="physical_complete_uncheckpointed", budget_after=snapshot,
                           wall_seconds=time.monotonic()-self._group_started,
                           device_memory_after=self._device_memory())
        self._persist()
        self._write_fence()
        return deepcopy(self._group)

    def acknowledge_checkpoint(self, checkpoint_path):
        """Release the fence only for a durable coordinator containing this group."""
        self._check_owner()
        if self._group is None or self._group["status"] != "physical_complete_uncheckpointed":
            raise RuntimeError("Checkpoint acknowledgement requires a physically complete group")
        try:
            path = Path(checkpoint_path).resolve()
            with path.open("rb") as stream: os.fsync(stream.fileno())
            import torch
            checkpoint = torch.load(path,map_location="cpu",weights_only=False)
            group = checkpoint.get("parallel_collection_group") or {}
            expected = self.campaign_journal_fence()
            if (group.get("group_id") != self._group["group_id"]
                    or group.get("status") != "physical_complete_uncheckpointed"
                    or group.get("physical_actions") != self._group["physical_actions"]
                    or group.get("journal_path") != self._group["journal_path"]
                    or checkpoint.get("campaign_journal_fence") != expected):
                raise RuntimeError("Coordinator checkpoint does not commit this complete group and live campaign fence")
            descriptor = os.open(path.parent,os.O_RDONLY)
            try: os.fsync(descriptor)
            finally: os.close(descriptor)
            self._group.update(status="checkpointed",checkpoint_path=str(path),
                checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                checkpoint_campaign_fence=expected)
            self._persist()
            self._clear_fence()
            self._last_group, self._group = deepcopy(self._group), None
            return deepcopy(self._last_group)
        except BaseException as error:
            self._fail(error)
            raise ProductionVectorError(str(error),self.group_recovery_metadata()) from error

    def cancel_group(self, reason):
        self._check_owner()
        if self._group is None or self._group["dispatch_count"]:
            raise RuntimeError("Only a group with no reset/step dispatch can be cancelled")
        self._group.update(status="cancelled_before_dispatch", reason=str(reason))
        self._persist()
        self._clear_fence()
        self._last_group, self._group = deepcopy(self._group), None
        return deepcopy(self._last_group)

    def group_recovery_metadata(self):
        value = deepcopy(self._group or self._last_group)
        if value is not None:
            value.update(active_group=self._group is not None,
                         zero_dispatch_proven=self.can_restore_unstarted_group())
        return value

    def can_restore_unstarted_group(self):
        return (self.resume_blocker_metadata is None and
                (self._group is None or self._group["dispatch_count"] == 0))

    def _fail(self, error, completed=None):
        if self._group is None:
            return
        self._group.update(status="unresolved", error=f"{type(error).__name__}: {error}")
        metadata = dict(resume_blocked=True, group_id=self._group["group_id"],
                        journal_path=self._group["journal_path"],
                        vector_output=str(self.root), error=self._group["error"])
        self.resume_blocker_metadata = metadata
        _durable_json(self.output_dir / RESUME_BLOCKER, metadata)
        self._persist()
        if completed:
            _durable_pickle(self.root / "failure-returned-replies.pkl",completed)

    def close(self):
        if self.closed: return
        try:
            if self._group is not None and self.resume_blocker_metadata is None:
                if self._group["dispatch_count"] == 0:
                    self.cancel_group("closed before physical dispatch")
                else:
                    self._fail(RuntimeError("Closed before completing the admitted group"))
            if self._pool is not None:
                self._pool.close()
        finally:
            self.cache = [None]*self.lanes
            self.raw = [None]*self.lanes
            self.closed = True
