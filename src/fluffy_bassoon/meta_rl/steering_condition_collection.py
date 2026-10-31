"""Source-only 4D coefficient SAC, using the owner's canonical episode runner.

No simulator loop, scheduler, corpus writer or action ledger is created here.
All guards/checkpoint/validation callbacks belong to the one execution owner.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import os
from pathlib import Path
import random
import re

import numpy as np

from .vla_noise_sac import NoiseSAC, SACConfig


SOURCE_METHOD_KEYS = (
    "sac_calibration_minus_train0", "sac_calibration_plus_train0",
    "sac_calibration_minus_train1", "sac_calibration_plus_train1",
    "sac_calibration_minus_validation2", "sac_calibration_plus_validation2",
)
SOURCE_TASKS = ("place_phone_stand", "place_object_stand")
PHASE_CAPS = dict(a1=30000, source=120000, source_validation=30000,
                  condition_cross=15000, student_evaluation=105000)


def _sha(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("Require an actual SHA-256 binding")
    return value


def _ref(value):
    if not isinstance(value, dict) or not value.get("path"):
        raise ValueError("Require a committed canonical artifact reference")
    _sha(value.get("sha256"))
    return deepcopy(value)


def _state_digest(value):
    """Stable full-state check, including tensor contents rather than pickle IDs."""
    digest = hashlib.sha256()

    def add(item):
        if isinstance(item, dict):
            digest.update(b"dict")
            for key in sorted(item, key=repr):
                add(key)
                add(item[key])
        elif isinstance(item, (tuple, list)):
            digest.update(type(item).__name__.encode())
            digest.update(str(len(item)).encode())
            for child in item:
                add(child)
        elif isinstance(item, np.ndarray) or hasattr(item, "detach"):
            array = item.detach().cpu().numpy() if hasattr(item, "detach") else item
            digest.update(str(array.dtype).encode()+str(array.shape).encode())
            digest.update(np.ascontiguousarray(array).tobytes())
        else:
            digest.update((type(item).__name__+":"+repr(item)+";").encode())

    add(value)
    return digest.hexdigest()


@dataclass(frozen=True)
class LineageSpec:
    index: int
    method: str
    sign: int
    partition: str
    learner_seed: int

    @property
    def condition_id(self):
        return "calibration_minus" if self.sign < 0 else "calibration_plus"


def lineage_specs(seed=731031):
    if type(seed) is not int or not 0 <= seed < 2**32 - 6*104729:
        raise ValueError("Invalid source learner seed namespace")
    return tuple(LineageSpec(i, key, -1 if i % 2 == 0 else 1,
                             "train" if i < 4 else "validation", seed+i*104729)
                 for i, key in enumerate(SOURCE_METHOD_KEYS))


@dataclass(frozen=True)
class SourceProfile:
    task: str
    episode_limit: int
    mapping_sha256: str
    interface_sha256: str
    motor_sha256: str
    scope_sha256: str
    task_horizon_sha256: str
    source_selection_sha256: str
    arm: str
    initial_episodes: int = 24
    maximum_episodes: int = 48
    public_dim: int = 2062
    native_horizon: int = 32
    learner_seed: int = 731031
    reset_action_bound: int = 0

    def __post_init__(self):
        from fluffy_bassoon.rl.robotwin_steering import normalize_arm

        object.__setattr__(self, "arm", normalize_arm(self.arm))
        if (self.task not in SOURCE_TASKS or type(self.episode_limit) is not int
                or self.episode_limit < 1 or self.public_dim != 2062 or self.native_horizon != 32
                or self.initial_episodes != 24 or self.maximum_episodes != 48
                or type(self.reset_action_bound) is not int or self.reset_action_bound < 0
                or self.arm not in ("noise", "kv", "action")):
            raise ValueError("Bind selected source task, verified horizon and declared 4D mapping")
        for value in (self.mapping_sha256, self.interface_sha256, self.motor_sha256, self.scope_sha256,
                      self.task_horizon_sha256, self.source_selection_sha256):
            _sha(value)

    def source_caps(self):
        bound = self.maximum_episodes * (self.episode_limit + self.reset_action_bound)
        return {key: bound for key in SOURCE_METHOD_KEYS}

    def contract(self):
        return dict(**asdict(self), source_caps=self.source_caps(), phase_caps=dict(PHASE_CAPS),
                    learner=dict(batch_size=32, warmup_valid_chunks=32, utd=4, action_dim=4,
                        action_limit=1., update_boundary="complete episode only",
                        behavior="bounded uniform until first completed optimizer cycle",
                        gamma=".99 per valid decision; exact primitive duration retained"),
                    teacher_label_selection={"calibration_minus": SOURCE_METHOD_KEYS[2],
                                             "calibration_plus": SOURCE_METHOD_KEYS[3]},
                    source_only=True, target57_forbidden=True, new_ledger_created=False)

    def validate_budget(self, budget):
        header = budget.header
        if (header.get("total_actions") != 300000 or header.get("phase_caps") != PHASE_CAPS
                or header.get("source_caps") != self.source_caps()
                or header.get("contract", {}).get("scope_sha256") != self.scope_sha256):
            raise ValueError("Collector must use the ONE canonical bound phase ledger")


def episode_spec(profile, lineage, episode, *, phase="source", case=0, milestone=None):
    if lineage not in lineage_specs(profile.learner_seed):
        raise ValueError("Unknown independent source lineage")
    if phase == "source":
        if type(episode) is not int or not 0 <= episode < profile.maximum_episodes:
            raise ValueError("Source episode is outside the predeclared maximum")
        seed = 1100000000 + lineage.index*100000 + episode*997
        identity = f"{lineage.method}:episode:{episode:03d}"
    elif phase == "source_validation":
        if case not in (0, 1) or milestone not in (0, 12, 24, 36, 48):
            raise ValueError("Only two fixed source cases at predeclared milestones")
        seed = 2100000000 + lineage.index*10000 + case*997
        identity = f"{lineage.method}:validation:{milestone:03d}:{case}"
    else:
        raise ValueError("Unknown source collection phase")
    return dict(id=identity, task=profile.task, case=case, reset_seed=seed,
                motor_seed=1700000000 + lineage.index*100000 + (episode if phase == "source" else case)*997,
                arm=profile.arm, method=lineage.method, coefficients=[0., 0., 0., 0.],
                horizon=profile.episode_limit, condition_id=lineage.condition_id,
                steering_partition=lineage.partition, phase=phase,
                milestone=milestone, episode=episode)


def make_learner(profile, lineage, *, device="cuda"):
    """Explicit new profile; the existing NoiseSAC/SACConfig defaults stay intact."""
    return NoiseSAC(profile.public_dim, 4, device=device, action_limit=1.,
                    seed=lineage.learner_seed,
                    config=SACConfig(batch_size=32, warmup_chunks=32, utd=4))


class BoundedTeacherRecovery:
    """Collector adapter to existing atomic Torch recovery/corpus primitives.

    Two rolling slots plus protected immutable references; no corpus/ledger or
    simulator ownership. The caller supplies an exclusively owned directory,
    global storage/time guard and actual per-lineage recovery reservation.
    Both pre-cycle and interrupted slots remain available after a failed cycle.
    """
    def __init__(self, root, *, guard, control=None):
        self.root, self.guard, self.control = Path(root), guard, control
        self.guard()
        self.root.mkdir(parents=True, exist_ok=False)
        self.serial, self.slot = 0, 0
        self.latest = None

    def __call__(self, event, payload, *, immutable):
        import torch
        from fluffy_bassoon.rl.robotwin_corpus import _atomic_json, _fsync_directory
        from .vla_history import file_sha256

        self.guard()
        self.serial += 1
        if immutable:
            path = self.root/f"protected-{self.serial:08d}.pt"
        else:
            self.slot = 1-self.slot
            path = self.root/f"latest-{self.slot}.pt"
        data = dict(format="source_coefficient_full_recovery_v1", event=event,
            collector=payload, python_rng=random.getstate(), numpy_rng=np.random.get_state(),
            torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,
            control=None if self.control is None else self.control.snapshot(),
            interpretation="Full coefficient SAC/pending state; physical episode restore needs owner reconciliation")
        temporary = path.with_suffix(".partial")
        with temporary.open("xb") as target:
            torch.save(data, target)
            target.flush()
            os.fsync(target.fileno())
        self.guard()  # Include the temporary file in the actual shared cap.
        reference = dict(path=str(path), sha256=file_sha256(temporary),
                         retention="immutable" if immutable else "rolling",
                         size_bytes=temporary.stat().st_size)
        temporary.replace(path)
        _fsync_directory(self.root)
        # The previously latest slot is never removed by this adapter. The
        # alternating slot is only reused once a newer complete file exists.
        _atomic_json(self.root/"latest.json", dict(current=reference, previous=self.latest), replace=True)
        self.latest = reference
        return deepcopy(reference)

    @staticmethod
    def load(reference):
        import torch
        from .vla_history import file_sha256

        reference = _ref(reference)
        if file_sha256(reference["path"]) != reference["sha256"]:
            raise ValueError("Full recovery bytes differ from the committed reference")
        data = torch.load(reference["path"], map_location="cpu", weights_only=False)
        if data.get("format") != "source_coefficient_full_recovery_v1":
            raise ValueError("Not a full coefficient-collector recovery")
        return data


class CoefficientTeacher:
    """Actual behavior state, including pending action/update recovery.

    checkpoint(event,payload,immutable=bool) atomically saves the FULL payload
    using the owner's recovery path and returns path/SHA/retention. Ordinary
    boundaries replace rolling latest recovery; initial, validation and published
    prefix references are immutable. Never retain a full replay copy per action.
    The sink retains previous latest until replacement verifies and cannot prune
    referenced immutable recoveries. archive_actor returns
    the existing pure-actor archive descriptor, separate from this recovery.
    An interrupted physical episode is preserved and blocked, never silently
    replaced by a fresh reset/controller. Pending boundary updates can resume.
    """
    format = "source_coefficient_teacher_v1"

    def __init__(self, learner, profile, lineage, *, checkpoint, archive_actor, guard):
        if (learner.observation_dim != profile.public_dim or learner.action_dim != 4
                or learner.action_limit != 1. or learner.config.batch_size != 32
                or learner.config.warmup_chunks != 32 or learner.config.utd != 4
                or learner.target_replay_mode is not None or learner.seed != lineage.learner_seed):
            raise ValueError("Teacher dimensions/bounds differ from the actual mapping")
        if lineage not in lineage_specs(profile.learner_seed):
            raise ValueError("Require one of the six independent source lineages")
        self.learner, self.profile, self.lineage = learner, profile, lineage
        self.checkpoint, self.archive_actor, self.guard = checkpoint, archive_actor, guard
        self.active_episode = self.pending = self.initial_recovery = self.last_recovery = None
        self.completed_episodes = []
        self.eligible_chunks = self.pending_update_cycles = self.episode_steps = 0
        self.decision_count = self.observed_count = 0
        self.updating = False
        self.validation_done = []
        self.validation_receipts = {}
        self._metadata = None

    def _persist(self, event, *, immutable=False):
        self.guard()
        reference = _ref(self.checkpoint(event, self.state_dict(), immutable=immutable))
        if reference.get("retention") != ("immutable" if immutable else "rolling"):
            raise ValueError("Recovery sink must distinguish immutable references from rolling latest")
        self.last_recovery = reference
        if self.initial_recovery is None and immutable:
            self.initial_recovery = reference
        return reference

    def _policy(self):
        if self.learner.updates == 0:
            return dict(policy_version="bounded_uniform_warmup", source_warmup=True,
                        policy_checkpoint=None, policy_sha256=None, policy_checkpoint_sha256=None)
        from .robotwin_history_corpus import parameter_digest
        descriptor = deepcopy(self.archive_actor(self.learner))
        digest = parameter_digest(self.learner.actor.state_dict())
        if (descriptor.get("policy_sha256") != digest
                or descriptor.get("policy_version") != f"sac:update:{self.learner.updates}:{digest}"
                or not descriptor.get("policy_checkpoint")):
            raise ValueError("Pure acting weights/version do not match the learner")
        _sha(descriptor.get("policy_checkpoint_sha256"))
        return dict(descriptor, source_warmup=False)

    def begin_episode(self, spec):
        if self.active_episode is not None or self.pending is not None or self.pending_update_cycles:
            raise RuntimeError("Resolve physical prefix/pending updates before starting another episode")
        if (spec["task"] != self.profile.task or spec["method"] != self.lineage.method
                or spec["condition_id"] != self.lineage.condition_id):
            raise ValueError("Source task/lineage/condition changed")
        if self.initial_recovery is None:
            self._persist("lineage_initial_full_state", immutable=True)
        self.active_episode = deepcopy(spec)
        self.eligible_chunks = self.episode_steps = self.decision_count = self.observed_count = 0
        self._persist("episode_begin")

    def act(self, observations):
        if self.active_episode is None or self.pending is not None or self.pending_update_cycles:
            raise RuntimeError("Require one active physical episode and no unresolved decision")
        self.guard()
        features = np.asarray(observations, np.float32)
        if features.shape != (self.profile.public_dim,) or not np.isfinite(features).all():
            raise ValueError("Only the current finite public feature vector enters the actor")
        policy = self._policy()
        before_numpy = deepcopy(self.learner.rng.bit_generator.state)
        before_torch = self.learner.torch_rng.get_state().cpu().numpy().copy()
        # Entire crossing episode stays uniform until its first boundary update.
        action = (self.learner.sample_warmup(features) if policy["source_warmup"]
                  else self.learner.act(features, deterministic=False))
        action = np.asarray(action, np.float32)
        if action.shape != (4,) or not np.isfinite(action).all() or np.any(np.abs(action) > 1):
            raise ValueError("Actor must request a supported finite coefficient vector")
        self.pending = dict(observations=features.copy(), coefficients=action.copy())
        self._metadata = dict(policy, steering_coefficients=action.copy(),
            proposed_steering_coefficients=action.copy(),
            steering_mapping_sha256=self.profile.mapping_sha256,
            steering_interface_sha256=self.profile.interface_sha256,
            steering_lineage_id=self.lineage.method,
            steering_condition_id=self.lineage.condition_id,
            steering_partition=self.lineage.partition,
            optimizer_updates=int(self.learner.updates),
            interaction_numpy_rng=before_numpy, interaction_torch_rng=before_torch)
        self.decision_count += 1
        self._persist("action_intent")
        return action.copy()

    def decision_metadata(self):
        if self.pending is None or self._metadata is None:
            raise RuntimeError("No actual pre-action behavior metadata")
        return deepcopy(self._metadata)

    def observe(self, observations, coefficients, reward, following, terminated,
                truncated, primitive_steps, corpus_ref):
        if (self.pending is None or type(primitive_steps) is not int
                or not 1 <= primitive_steps <= self.profile.native_horizon
                or not np.array_equal(np.asarray(observations, np.float32), self.pending["observations"])
                or not np.array_equal(np.asarray(coefficients, np.float32), self.pending["coefficients"])):
            raise ValueError("Transition does not match the committed requested coefficient/input")
        reference = _ref(corpus_ref)
        self.learner.observe(observations, coefficients, reward, following, terminated,
                             truncated, primitive_steps=primitive_steps)
        self.eligible_chunks += int(self.learner.ready_to_update)
        self.episode_steps += primitive_steps
        self.observed_count += 1
        self.pending = None
        self._metadata = None
        self.last_corpus_ref = reference
        self._persist("transition_committed")

    def finish_episode(self, receipt):
        if (self.active_episode is None or receipt.get("complete") is not True
                or receipt.get("spec") != self.active_episode
                or receipt.get("primitive_steps") != self.episode_steps
                or receipt.get("chunks") != self.decision_count
                or len(receipt.get("corpus_refs", [])) != self.decision_count):
            raise ValueError("Require the complete canonical episode receipt and every chunk ref")
        for reference in receipt["corpus_refs"]:
            _ref(reference)
        _ref(receipt.get("episode_manifest_ref"))
        if self.pending is not None:
            # The canonical runner skips observe() only for a zero-valid final
            # chunk. Equal actual-step sums rule out dropping a valid chunk.
            if self.decision_count != self.observed_count + 1:
                raise RuntimeError("Unresolved physical decision cannot be adopted")
            receipt = dict(receipt, zero_valid_terminal_chunk=True)
            self.pending = None
            self._metadata = None
        self.completed_episodes.append(deepcopy(receipt))
        self.active_episode = None
        self.pending_update_cycles = self.eligible_chunks * self.learner.config.utd
        self._persist("complete_episode_before_update")
        self.finish_updates()
        return self._persist("complete_episode_after_update")

    def finish_updates(self):
        if self.active_episode is not None or self.pending is not None:
            raise RuntimeError("Optimizer updates cannot run inside a physical episode")
        while self.pending_update_cycles:
            self.guard()
            self.updating = True
            self._persist("optimizer_cycle_intent")
            try:
                before = self.learner.updates
                self.learner.update(gradient_steps=1)
                if self.learner.updates != before + 1:
                    raise RuntimeError("SAC did not complete the requested optimizer cycle")
            except BaseException:
                self._persist("optimizer_cycle_interrupted_full_state")
                raise
            self.pending_update_cycles -= 1
            self.updating = False
            self._persist("optimizer_cycle_committed")
        # Clear warmup pending rows without retroactive optimization debt.
        if not self.eligible_chunks:
            self.learner.update(gradient_steps=0)

    def state_dict(self):
        return dict(format=self.format, profile=self.profile.contract(), lineage=asdict(self.lineage),
            learner=self.learner.state_dict(include_replay=True),
            active_episode=deepcopy(self.active_episode), pending=deepcopy(self.pending),
            decision_metadata=deepcopy(self._metadata), completed_episodes=deepcopy(self.completed_episodes),
            eligible_chunks=self.eligible_chunks, pending_update_cycles=self.pending_update_cycles,
            episode_steps=self.episode_steps, decision_count=self.decision_count,
            observed_count=self.observed_count, updating=self.updating,
            validation_done=list(self.validation_done),
            validation_receipts=deepcopy(self.validation_receipts),
            initial_recovery=deepcopy(self.initial_recovery))

    def load_state_dict(self, payload, *, recovery_ref):
        if (payload["format"] != self.format or payload["profile"] != self.profile.contract()
                or payload["lineage"] != asdict(self.lineage)):
            raise ValueError("Source profile/lineage/mapping recovery changed")
        self.learner.load_state_dict(payload["learner"], require_replay=True)
        for key in ("active_episode", "pending", "completed_episodes", "eligible_chunks",
                    "pending_update_cycles", "episode_steps", "decision_count", "observed_count", "updating",
                    "validation_done", "validation_receipts"):
            setattr(self, key, deepcopy(payload[key]))
        self._metadata = deepcopy(payload["decision_metadata"])
        self.last_recovery = _ref(recovery_ref)
        self.initial_recovery = deepcopy(payload["initial_recovery"])
        if self.initial_recovery is None and self.last_recovery.get("retention") == "immutable":
            self.initial_recovery = deepcopy(self.last_recovery)
        if self.updating:
            raise RuntimeError("Interrupted partial optimizer cycle retained; execution owner must reconcile it")


class QualifiedCoefficientTeacher:
    """DPT query-label API, independent of behavior-learning qualification.

    An inverse/search/SAC policy is acceptable only with actual same-mapping
    physical qualification. A flat SAC curve does not invalidate a separately
    qualified oracle, nor does an inverse formula itself qualify a teacher.
    """
    def __init__(self, policy, *, qualification, profile, condition_id):
        q = deepcopy(qualification)
        if (q.get("status") != "physically_qualified_source_coefficient_teacher"
                or q.get("mapping_sha256") != profile.mapping_sha256
                or q.get("motor_sha256") != profile.motor_sha256
                or q.get("condition_id") != condition_id or not q.get("source_only")
                or q.get("new_target_data_used") is not False
                or q.get("coefficient_dim") != 4 or q.get("coefficient_bound") != 1.
                or getattr(policy, "action_limit", None) != 1. or q.get("same_mapping") is not True):
            raise ValueError("DPT needs actual source-only same-mapping teacher qualification")
        for key in ("qualification_receipt", "policy_checkpoint", "validation_case_manifest", "cost_receipt"):
            _ref(q.get(key))
        if q.get("label_sampling_mode") not in ("deterministic_mean", "sampled_gaussian", "fixed_coefficients"):
            raise ValueError("Bind the actual physically qualified label sampling convention")
        from .robotwin_history_corpus import parameter_digest
        if hasattr(policy, "actor"):
            primary = profile.contract()["teacher_label_selection"].get(condition_id)
            if (q.get("teacher_lineage") != primary or q["label_sampling_mode"] == "fixed_coefficients"
                    or parameter_digest(policy.actor.state_dict()) != q.get("policy_sha256")):
                raise ValueError("TRAIN and VAL labels require the same predeclared TRAIN teacher weights")
            self.fixed = None
        else:
            # The restricted source search/inverse is a fixed 4D controller.
            # Bind its actual coefficients; never call an unbound callable.
            fixed = np.asarray(getattr(policy, "coefficients", None), np.float32)
            if (fixed.shape != (4,) or not np.isfinite(fixed).all() or np.any(np.abs(fixed) > 1)
                    or q["label_sampling_mode"] != "fixed_coefficients"
                    or _state_digest(fixed) != q.get("policy_sha256")):
                raise ValueError("Bind the actual qualified same-mapping fixed coefficient policy")
            self.fixed = fixed.copy()
        self.policy, self.qualification, self.profile = deepcopy(policy), q, profile

    def act(self, public_observation):
        features = np.asarray(public_observation, np.float32)
        if features.shape != (self.profile.public_dim,) or not np.isfinite(features).all():
            raise ValueError("DPT teacher only receives public query observations")
        action = (self.fixed.copy() if self.fixed is not None else np.asarray(
            self.policy.act(features, deterministic=self.qualification["label_sampling_mode"] == "deterministic_mean"),
            np.float32))
        if action.shape != (4,) or not np.isfinite(action).all() or np.any(np.abs(action) > 1):
            raise ValueError("Qualified teacher returned an unsupported coefficient label")
        return action.copy()

    def metadata(self):
        return dict(qualification=deepcopy(self.qualification),
                    behavior_improving_history_qualified="separate gate, not implied",
                    label_rule="query-only 4D coefficients; actual behavior histories unchanged")


def collect_lineage(teacher, *, control, budget, run_id, episode_runner,
                    post_command_transform, validate_source, publish_prefix,
                    episode_target=24, one_episode=False):
    """Concrete collection via the SAME complete-episode driver; no second loop.

    validate_source receives only a saved checkpoint ref plus fixed specs and
    must restore a separate frozen actor/private RNG (never observe/update the
    source learner). All its physical actions use source_validation. The
    supplied guard/sink enforce remaining shared GPU/storage/CPU accounting.
    """
    profile, lineage = teacher.profile, teacher.lineage
    profile.validate_budget(budget)
    from fluffy_bassoon.rl.steering_actuator_condition import CalibrationDistribution, contract_sha256

    if (not isinstance(post_command_transform, CalibrationDistribution)
            or getattr(control, "post_command_transform", None) is not post_command_transform
            or contract_sha256(control.condition_contract) != contract_sha256(post_command_transform.contract())
            or control.interface_sha256 != profile.interface_sha256
            or control.episode_limit_for_task(profile.task) != profile.episode_limit
            or any(s.mapping_sha256 != profile.mapping_sha256 or s.motor_sha256 != profile.motor_sha256
                   for s in post_command_transform.specs.values())):
        raise ValueError("Installed calibration, motor, mapping, interface or native horizon differs")
    if control.active:
        raise RuntimeError("Do not change a condition during an active physical episode")
    control.set_coefficients(profile.arm, [0., 0., 0., 0.])
    if control.mapping_sha256() != profile.mapping_sha256:
        raise ValueError("Selected canonical runtime arm does not match the profile mapping")
    post_command_transform.select(lineage.sign)
    if episode_target not in (24, 48):
        raise ValueError("Only the initial24 or explicitly admitted upper48 prefix")
    if teacher.active_episode is not None or teacher.pending is not None:
        raise RuntimeError("Preserved partial episode needs owner recovery; no fresh reset")
    teacher.finish_updates()
    if teacher.initial_recovery is None:
        teacher._persist("lineage_initial_full_state", immutable=True)
    validation_done = set(teacher.validation_done)

    def validate(milestone):
        if milestone in (0, 12, 24, 36, 48) and milestone not in validation_done:
            specs = [episode_spec(profile, lineage, milestone, phase="source_validation", case=i,
                                  milestone=milestone) for i in range(2)]
            # The validation callback restores a separate actor/private RNG.
            # Hash full state around it to catch a closure mutating source state.
            frozen_ref = teacher._persist("source_validation_input", immutable=True)
            before = _state_digest(teacher.learner.state_dict(include_replay=True))
            result = validate_source(frozen_ref, specs)
            after = _state_digest(teacher.learner.state_dict(include_replay=True))
            if after != before:
                raise RuntimeError("Source validation mutated the source learner/replay/RNG")
            reference = _ref(result)
            if (result.get("complete") is not True
                    or result.get("episode_ids") != [s["id"] for s in specs]
                    or result.get("phase") != "source_validation"):
                raise ValueError("Require both actual fixed validation cases and their committed receipt")
            teacher.validation_receipts[str(milestone)] = dict(reference, input_checkpoint=frozen_ref)
            validation_done.add(milestone)
            teacher.validation_done = sorted(validation_done)
            teacher._persist("source_validation_milestone")

    validate(len(teacher.completed_episodes))
    for episode in range(len(teacher.completed_episodes), episode_target):
        spec = episode_spec(profile, lineage, episode)
        bound = profile.episode_limit + profile.reset_action_bound
        if budget.remaining(phase="source", method=lineage.method) < bound:
            return dict(status="shared_action_cap_stop", full_recovery=teacher._persist("budget_stop"),
                        complete_episodes=len(teacher.completed_episodes))
        teacher.guard()
        if control.active:
            raise RuntimeError("Validation did not close its physical episode")
        post_command_transform.select(lineage.sign)
        teacher.begin_episode(spec)
        try:
            receipt = episode_runner(control, spec, budget, run_id=run_id, phase="source",
                                     controller=teacher)
            final_ref = teacher.finish_episode(receipt)
        except BaseException:
            teacher._persist("collection_interrupted_preserved")
            raise
        validate(episode + 1)
        if lineage.partition == "train" and (episode + 1) in (12, 24, 36, 48):
            final_ref = teacher._persist("published_prefix_full_state", immutable=True)
            publish_prefix(dict(lineage=asdict(lineage), condition_id=lineage.condition_id,
                mapping_sha256=profile.mapping_sha256, interface_sha256=profile.interface_sha256,
                complete_episode_manifests=[r["episode_manifest_ref"] for r in teacher.completed_episodes],
                actual_resets=[r["reset_info"] for r in teacher.completed_episodes],
                initial_full_state=teacher.initial_recovery, final_full_state=final_ref,
                source_validation_receipts=deepcopy(teacher.validation_receipts),
                actual_optimizer_updates=teacher.learner.updates,
                actual_primitive_steps=teacher.learner.primitive_steps,
                behavioral_improvement="requires fixed source validation and retry comparison",
                DPT_teacher_qualification="separate physical gate; no automatic permission"))
        if one_episode and episode + 1 < episode_target:
            return dict(status="episode_committed", complete_episodes=len(teacher.completed_episodes),
                        full_recovery=teacher.last_recovery)
    final_ref = teacher._persist("lineage_final_full_state", immutable=True)
    return dict(status="initial_prefix_complete" if episode_target == 24 else "upper_prefix_complete",
                complete_episodes=len(teacher.completed_episodes), full_recovery=final_ref,
                scientist_claim="collection completion is not learning/teacher/ICRL qualification")


def collect_round_robin(teachers, *, episode_target=24, **callbacks):
    """Six real independent learner lineages, balanced at episode boundaries."""
    if set(teachers) != set(SOURCE_METHOD_KEYS):
        raise ValueError("All six predeclared source methods must be present")
    if (len({id(t.learner) for t in teachers.values()}) != 6
            or len({t.learner.seed for t in teachers.values()}) != 6
            or any(t.lineage.method != key for key, t in teachers.items())
            or len({_state_digest(t.profile.contract()) for t in teachers.values()}) != 1):
        raise ValueError("Require six independent learners under one exact source profile")
    if episode_target not in (24, 48):
        raise ValueError("Only the initial24 or explicitly admitted upper48 prefix")
    for episode in range(episode_target):
        for key in SOURCE_METHOD_KEYS:
            teacher = teachers[key]
            completed = len(teacher.completed_episodes)
            if completed >= episode + 1:
                continue
            if completed != episode:
                raise RuntimeError("A saved lineage cursor is inconsistent with balanced collection")
            result = collect_lineage(teacher, episode_target=episode_target, one_episode=True, **callbacks)
            if result["status"] == "shared_action_cap_stop":
                return dict(status="shared_action_cap_stop", stopped_lineage=key,
                            completed={k: len(t.completed_episodes) for k, t in teachers.items()})
    return dict(status="initial_prefix_complete" if episode_target == 24 else "upper_prefix_complete",
                completed={k: len(t.completed_episodes) for k, t in teachers.items()},
                teacher_competence="separate physical qualification",
                behavioral_improvement="separate measured source-validation gate")


def cross_evaluation_specs(profile, context_blind_method):
    """4 fresh scenes x2conditions x2matched/mismatched teachers +ONE fixed control.

    Select the single context-blind method from source-validation data before
    opening this bank; never choose a different control per latent condition.
    Stationary retry is measured by the fixed source-validation warmup baseline.
    """
    if not isinstance(context_blind_method, str) or not context_blind_method:
        raise ValueError("Predeclare one condition-blind controller")
    result = []
    for case in range(4):
        for sign in (-1, 1):
            condition_id = "calibration_minus" if sign < 0 else "calibration_plus"
            for method in (SOURCE_METHOD_KEYS[2], SOURCE_METHOD_KEYS[3], context_blind_method):
                result.append(dict(id=f"A2:{condition_id}:{case}:{method}", task=profile.task,
                    case=case, reset_seed=2300000000+case*997, motor_seed=1730000000+case*997,
                    arm=profile.arm, method=method, coefficients=[0., 0., 0., 0.],
                    horizon=profile.episode_limit, condition_id=condition_id,
                    steering_partition="validation", phase="condition_cross"))
    if len(result)*(profile.episode_limit+profile.reset_action_bound) > PHASE_CAPS["condition_cross"]:
        raise ValueError("The predeclared A2 grid exceeds its shared phase subcap")
    return result
