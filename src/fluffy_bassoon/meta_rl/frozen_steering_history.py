"""Task-preserving AD/DPT views of frozen steering PPO episodes.

The source action is the actual pre-tanh steering vector. Native Gaussian
motor draws remain exogenous provenance, never steering labels. First-success
logical episodes and their full physical suffixes are deliberately distinct.
"""
from dataclasses import dataclass
import gzip
import json
from pathlib import Path
import pickle

import numpy as np
import torch

from .residual_history import relabel_teacher
from .interface_ppo import InterfaceSpec
from .residual_ppo import ResidualActorCritic, ResidualPPOConfig, history_trial
from .vla_history import file_sha256, save_dataset, validate_trial


def interface_record(arguments):
    """Compare constructor and recorded specs without losing mapping metadata."""
    arguments = dict(arguments)
    mapping = arguments.pop("kv_map", None)
    if "motor_shape" in arguments:
        arguments["motor_shape"] = tuple(arguments["motor_shape"])
    try:
        record = InterfaceSpec(**arguments).record()
    except (TypeError, ValueError) as error:
        raise ValueError("Require the exact declared steering interface") from error
    if mapping is not None and mapping != record.get("kv_map"):
        raise ValueError("Steering interface mapping metadata differs")
    return record


@dataclass(frozen=True)
class SteeringSource:
    arm: str
    source_commit: str
    runtime_commit: str
    motor_sha256: str
    normalization_sha256: str
    tasks: tuple[str, ...]
    interface: dict
    episodes_per_task_per_update: int | None
    feature_dim: int = 2064

    def __post_init__(self):
        expected = {"velocity": 32, "kv_all_layers": 1152}
        if (self.arm not in expected or self.interface.get("kind") != self.arm
                or self.interface.get("steering_dim") != expected[self.arm]
                or self.arm == "kv_all_layers" and self.interface.get("kv_layers") != 18
                or not np.isfinite(self.interface.get("fraction", np.nan))
                or self.interface.get("fraction", 0) <= 0
                or not self.tasks or len(set(self.tasks)) != len(self.tasks)
                or self.episodes_per_task_per_update is not None and (
                    type(self.episodes_per_task_per_update) is not int or self.episodes_per_task_per_update < 2)
                or self.feature_dim < 1
                or interface_record(self.interface) != self.interface
                or any(not value for value in (self.source_commit, self.runtime_commit,
                                               self.motor_sha256, self.normalization_sha256))):
            raise ValueError("Bind an exact task set and velocity or independent all-layer K/V source")

    @property
    def steering_dim(self):
        return self.interface["steering_dim"]


def checked_reference(reference):
    path = Path(reference["path"])
    if file_sha256(path) != reference["sha256"]:
        raise ValueError("Source artifact checksum differs")
    return path


def case_identity(case):
    return json.dumps(case, sort_keys=True, separators=(",", ":"), allow_nan=False)


class EpisodeSchedule:
    """Exact prospective case/behavior coverage, including intentional gaps.

    Entries come from the producer's published requests, before outcomes or
    DPT labels. A variable task count must be declared explicitly by setting
    episodes_per_task_per_update=None; it never falls back to a fixed count.
    """
    def __init__(self, entries, source, *, expected_updates):
        self.source, self.versions = source, tuple(expected_updates)
        if (len(self.versions) < 2 or any(type(v) is not int or v < 0 for v in self.versions)
                or tuple(sorted(set(self.versions))) != self.versions):
            raise ValueError("Require multiple chronological source versions")
        self.by_case, self.by_stream = {}, {}
        observed_versions, tasks = set(), set()
        count = source.episodes_per_task_per_update
        for entry in entries:
            case, version, slot = entry["case"], entry["behavior_version"], entry["stream_slot"]
            task = case.get("task")
            if (task not in source.tasks or type(version) is not int or version not in self.versions
                    or type(slot) is not int or slot < 0):
                raise ValueError("Unknown task, stream or behavior in producer schedule")
            if count is not None and (slot >= count or case["slot"] != version * count + slot):
                raise ValueError("Declared uniform source schedule differs from its collection slots")
            identity, stream = (version, case_identity(case)), (task, slot)
            group = self.by_stream.setdefault(stream, {})
            if identity in self.by_case or version in group:
                raise ValueError("Duplicate scheduled source episode")
            self.by_case[identity] = stream
            group[version] = identity[1]
            observed_versions.add(version)
            tasks.add(task)
        if tasks != set(source.tasks) or observed_versions != set(self.versions):
            raise ValueError("Schedule must cover all declared source tasks and versions")
        self.slots = {task: sorted(slot for t, slot in self.by_stream if t == task) for task in source.tasks}
        for task, slots in self.slots.items():
            if len(slots) < 2 or slots != list(range(len(slots))):
                raise ValueError("Require contiguous task-pure source stream identities")
        for group in self.by_stream.values():
            if len(group) < 2 or count is not None and tuple(sorted(group)) != self.versions:
                raise ValueError("Incomplete uniform schedule or insufficient AD behavior versions")


def load_episode(reference, source, *, schedule=None):
    """Read one complete producer episode, including its physical suffix.

    The optimizer's logical prefix ends at first success or physical timeout.
    History training uses actual complete physical episodes: the unchanged
    first-success reward and native physical_done fields from every chunk.
    The optimizer prefix remains separately checked and retained as provenance.
    """
    saved = torch.load(checked_reference(reference), map_location="cpu", weights_only=False)
    if (saved.get("format") != "joint_steering_episode_boundary_v1"
            or saved.get("source_commit") != source.source_commit
            or saved.get("runtime_commit") != source.runtime_commit):
        raise ValueError("Require the exact complete frozen-steering source checkpoint")
    episode, rows = saved["episode"], saved["records"]
    case = episode["case"]
    if (not episode.get("completed") or case.get("task") not in source.tasks
            or not rows or episode["valid_ppo_chunks"] != len(rows)
            or episode["physical_chunks"] * 50 != episode["horizon"]
            or len(episode["chunks"]) != episode["physical_chunks"]
            or episode["success_once"] != (episode["first_success_step"] is not None)):
        raise ValueError("Require an actual complete physical episode and its logical PPO prefix")
    updates = {int(row["policy_iteration"]) for row in rows}
    if len(updates) != 1 or min(updates) < 0:
        raise ValueError("One source behavior version must produce the entire episode")
    update = updates.pop()
    count = source.episodes_per_task_per_update
    # Producer round n collects with policy n-1, then performs update n.
    # Initial behavior version0 is real source data, not a missing update.
    if schedule is None:
        if count is None:
            raise ValueError("Variable task counts require an explicit producer case schedule")
        slot = int(case["slot"]) - update * count
        if not 0 <= slot < count:
            raise ValueError("Case slot does not match its declared chronological source round")
    else:
        if schedule.source != source or (update, case_identity(case)) not in schedule.by_case:
            raise ValueError("Actual source episode is not declared in this producer schedule")
        task, slot = schedule.by_case[(update, case_identity(case))]
        if task != case["task"]:
            raise ValueError("Scheduled source stream crosses tasks")
    for index, row in enumerate(rows):
        if (row["case"] != case or row["chunk_index"] != index
                or bool(row["episode_start"]) != (index == 0)
                or bool(row["done"]) != (index == len(rows) - 1)
                or row["executed_controls"] != 50
                or np.asarray(row["features"]).shape != (source.feature_dim,)
                or np.asarray(row["control"]).shape != (source.steering_dim,)):
            raise ValueError("Source prefix changed case, chronology, feature port or steering coordinates")
    rewards = np.asarray([row["reward"] for row in rows], np.float32)
    expected = np.zeros(len(rows), np.float32)
    expected[-1] = float(episode["success_once"])
    if not np.array_equal(rewards, expected):
        raise ValueError("Preserve the actual first-success logical reward without relabeling context")
    for item in (episode["initial"], episode["final"]):
        checked_reference(item)
    physical_rows = []
    for index, item in enumerate(episode["chunks"]):
        with gzip.open(checked_reference(item), "rb") as stream:
            physical = pickle.load(stream)
        if (physical.get("format") != "joint_steering_physical_chunk_v1"
                or physical.get("status") != "completed" or physical.get("phase") != "train"
                or physical.get("source_commit") != source.source_commit
                or physical.get("runtime_commit") != source.runtime_commit
                or physical.get("interface") != source.interface
                or physical.get("policy_iteration") != update
                or physical.get("iteration") != update + 1
                or physical.get("case") != case or physical.get("chunk_index") != index
                or bool(physical.get("physical_done")) != (index == episode["physical_chunks"] - 1)
                or physical.get("executed_controls") != 50
                or bool(physical.get("valid_ppo")) != (index < len(rows))
                or np.asarray(physical.get("native_noise")).shape != (50, 32)
                or not np.isfinite(physical["native_noise"]).all()
                or np.asarray(physical.get("features")).shape != (source.feature_dim,)
                or np.asarray(physical.get("control")).shape != (source.steering_dim,)):
            raise ValueError("Full physical history changed source, interface, case or chronology")
        if index < len(rows):
            logical = rows[index]
            if (physical.get("reward") != logical["reward"] or physical.get("done") != logical["done"]
                    or not np.array_equal(np.asarray(physical["features"]), np.asarray(logical["features"]))
                    or not np.array_equal(np.asarray(physical["control"]), np.asarray(logical["control"]))):
                raise ValueError("Logical row differs from the actual physical steering dispatch")
        physical_rows.append(dict(physical, done=bool(physical["physical_done"]),
                                  episode_start=index == 0))
    expected_physical = np.zeros(episode["physical_chunks"], np.float32)
    if episode["success_once"]:
        expected_physical[len(rows) - 1] = 1.
    if not np.array_equal([row["reward"] for row in physical_rows], expected_physical):
        raise ValueError("Do not move the first-success reward or invent post-success rewards")
    trial = history_trial([(update, physical_rows)])
    # A single episode is not AD yet: multiple behavior versions are joined
    # only after a task/slot-preserving train/validation split.
    validate_trial(trial, objective="behavior_cloning")
    return dict(task=case["task"], stream_slot=slot, update=update, trial=trial,
                source_episode=reference, physical_episode=episode,
                request_reference=(dict(path=saved["request_path"], sha256=saved["request_sha256"])
                    if saved.get("request_path") and saved.get("request_sha256") else None))


def split_streams(episodes, source, *, validation_slots, expected_updates, schedule=None):
    """Group task/slot streams, preserving every scheduled physical episode."""
    count = source.episodes_per_task_per_update
    versions = tuple(expected_updates)
    if (len(versions) < 2 or any(type(v) is not int or v < 0 for v in versions)
            or tuple(sorted(set(versions))) != versions
            or schedule is None and count is None
            or schedule is not None and (schedule.source != source or schedule.versions != versions)):
        raise ValueError("Require the exact producer schedule and chronological source versions")
    available = ({task: list(range(count)) for task in source.tasks} if schedule is None else schedule.slots)
    heldout = (validation_slots if isinstance(validation_slots, dict)
               else {task: validation_slots for task in source.tasks})
    if set(heldout) != set(source.tasks) or any(
            not slots or len(set(slots)) != len(slots) or len(slots) >= len(available[task])
            or any(type(i) is not int or i not in available[task] for i in slots)
            for task, slots in heldout.items()):
        raise ValueError("Require held-out slots and multiple chronological source versions")
    groups = {}
    for episode in episodes:
        key = (episode["task"], episode["stream_slot"])
        if key[0] not in source.tasks or key[1] not in available[key[0]]:
            raise ValueError("Unknown source task/stream slot")
        if schedule is not None and schedule.by_stream[key].get(episode["update"]) != case_identity(
                episode["physical_episode"]["case"]):
            raise ValueError("Actual episode differs from the scheduled source case")
        by_update = groups.setdefault(key, {})
        if episode["update"] in by_update:
            raise ValueError("Duplicate source episode; successful retries cannot add samples twice")
        by_update[episode["update"]] = episode
    if set(groups) != {(task, slot) for task in source.tasks for slot in available[task]}:
        raise ValueError("Every declared training task needs every source collection slot")
    result = {"train": [], "validation": []}
    for task in source.tasks:
        for slot in available[task]:
            group = groups[(task, slot)]
            expected = versions if schedule is None else tuple(sorted(schedule.by_stream[(task, slot)]))
            if tuple(sorted(group)) != expected:
                raise ValueError("Incomplete or changed behavior-version coverage")
            fields = group[expected[0]]["trial"]
            trial = {key: np.concatenate([group[v]["trial"][key] for v in expected]) for key in fields}
            validate_trial(trial, objective="ad")
            destination = "validation" if slot in heldout[task] else "train"
            result[destination].append((f"{task}:slot-{slot:03d}", trial))
    return result


def load_final_teacher(reference, source):
    """The DPT target is the frozen final steering mean at the same feature."""
    saved = torch.load(checked_reference(reference), map_location="cpu", weights_only=False)
    if (saved.get("format") != "joint_frozen_steering_boundary_v1"
            or saved.get("arm") != source.arm
            or interface_record(saved.get("spec", {})) != source.interface):
        raise ValueError("Final teacher belongs to another steering interface")
    config = ResidualPPOConfig(**saved["policy_config"])
    if config.feature_dim != source.feature_dim or config.residual_dim != source.steering_dim:
        raise ValueError("The teacher must use the same steering-only policy port")
    teacher = ResidualActorCritic(config)
    teacher.load_state_dict(saved["policy"], strict=True)
    if any(not torch.isfinite(value).all() for value in teacher.state_dict().values()):
        raise ValueError("Final teacher or normalization is nonfinite")
    return teacher.eval().requires_grad_(False), int(saved["iteration"])


def save_corpus(directory, streams, source, *, teacher_reference, provenance):
    """Write all four objectives with identical histories and held-out identities."""
    teacher, final_update = load_final_teacher(teacher_reference, source)
    identities = {partition: [identity for identity, _ in streams[partition]]
                  for partition in ("train", "validation")}
    if (any(not values or len(set(values)) != len(values) for values in identities.values())
            or set(identities["train"]) & set(identities["validation"])):
        raise ValueError("Require distinct complete source streams in the two student partitions")
    terminal = max(int(trial["teacher_steps"][-1])
                   for partition in ("train", "validation") for _, trial in streams[partition])
    if (terminal + 1 != final_update or source.episodes_per_task_per_update is not None and any(
            int(trial["teacher_steps"][-1]) + 1 != final_update
            for partition in ("train", "validation") for _, trial in streams[partition])):
        raise ValueError("DPT final teacher must match the declared terminal source version")
    output = Path(directory)
    if output.exists():
        raise ValueError("Never overwrite a bound source corpus")
    output.mkdir(parents=True)
    manifests = {}
    for partition in ("train", "validation"):
        identities, actual = zip(*streams[partition], strict=True)
        labeled = relabel_teacher(actual, teacher, deterministic=True)
        for objective, trials in (("ad", actual), ("dpt", labeled), ("dit", actual), ("ic_iql", actual)):
            metadata = dict(provenance, trial_ids=list(identities), source_tasks=list(source.tasks),
                motor_sha256=source.motor_sha256, normalization_sha256=source.normalization_sha256,
                feature_contract=dict(dimension=source.feature_dim, port="source pre-steering public features"),
                noise_contract=dict(meaning="actual pre-tanh steering coefficients only", interface=source.interface,
                                    native_Gaussian="exogenous; retained in source receipts, never a steering target"),
                source_commit=source.source_commit, runtime_commit=source.runtime_commit,
                teacher=teacher_reference, logical_boundary="source PPO first-success prefix retained separately",
                physical_boundary="actual native physical_done; full physical episodes for history",
                physical_suffix="actual post-success controls included; no reward is moved or invented",
                streams="same task/declared collection slot across scheduled versions; source scenes may change",
                validation="student imitation only; source PPO saw all source states")
            if objective in ("dit", "ic_iql"):
                metadata["validation"] = "held-out offline value/policy objective; source PPO saw all source states"
                metadata["value_contract"] = dict(discount=.99, discount_unit="one H50 steering decision",
                    boundary="physical_done", rewards="observed first-success once; physical suffix retained",
                    next_observation="next row within the same physical episode; terminal bootstrap zero",
                    targets="executed steering actions; no final-teacher relabels",
                    returns="complete observed episode, before sampling windows; critic supervision only")
            path = output / f"{objective}-{partition}.npz"
            manifests[f"{objective}_{partition}"] = save_dataset(path, list(trials), objective=objective,
                                                                 metadata=metadata)
    return manifests
