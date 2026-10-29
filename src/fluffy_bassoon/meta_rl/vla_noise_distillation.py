"""Vintix-inspired continuous noise distillation in frozen-VLA steering space.

This experimental port mixes bounded DSRL coordinates, not physical robot
actions. No source-policy weights are updated while collecting histories.
Allocation, simulator workers and teacher construction are caller-owned.
"""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import pickle
import sys

import numpy as np

from .vla_history import file_sha256, save_dataset


@dataclass(frozen=True)
class NoiseSchedule:
    """Anneal epsilon continuously over real steering decisions, across resets."""
    decisions: int
    power: float = 1.

    def __post_init__(self):
        if (type(self.decisions) is not int or self.decisions < 2
                or not math.isfinite(self.power) or self.power <= 0):
            raise ValueError("Require at least two decisions and a positive finite annealing power")

    def epsilon(self, decision):
        if type(decision) is not int or decision < 0:
            raise ValueError("Decision clock must be a nonnegative integer")
        return (1. - min(1., decision / (self.decisions - 1))) ** self.power


def mix_noise(expert, uniform, epsilon):
    """The actually executed coordinate in Vintix Algorithm 1's convex mixture."""
    expert, uniform = np.asarray(expert, np.float32), np.asarray(uniform, np.float32)
    if (expert.ndim != 1 or expert.shape != uniform.shape or not np.isfinite(expert).all()
            or not np.isfinite(uniform).all() or (np.abs(expert) > 1.).any()
            or (np.abs(uniform) > 1.).any() or not math.isfinite(epsilon) or not 0 <= epsilon <= 1):
        raise ValueError("Require finite matched bounded steering coordinates and epsilon in [0,1]")
    return ((1. - epsilon) * expert + epsilon * uniform).astype(np.float32)


def _json(path, value):
    temporary = Path(path).with_suffix(".json.partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def collect_noise_distillation(*, teacher_action, teacher_fingerprint, teacher_checkpoint,
        teacher_steps, teacher_primitive_steps, teacher_evaluation, motor_sha256, feature_contract, noise_contract, scenes,
        output_dir, min_episodes, max_episodes, episode_horizon, chunk,
        anneal_decisions, anneal_powers=(1.,), seed=0, max_primitive_steps,
        heldout_initial_states=(), min_teacher_success_rate=.7,
        min_teacher_evaluation_episodes=8, noise_dim=32, sim=None, motor=None,
        episode_runner=None, row_feature_encoder=None):
    """Collect complete multi-episode CND trials with a fixed qualified teacher.

    scenes has trial_id/split/initial_state/scene_seed dictionaries. The two
    anneal_decisions bounds and anneal_powers are sampled once per trial from
    the declared random stream. The schedule is NEVER a model input. Continue
    complete episodes until at least min_episodes and the full annealing
    duration have been observed; max_episodes and max_primitive_steps bound cost.
    The final episode may include a zero-noise expert tail. A short schedule
    is not padded with fabricated transitions or silently called complete.

    teacher_action must be deterministic (e.g. OfficialSAC.action with
    deterministic=True). teacher_fingerprint hashes its actual model state
    including optimizer/update state. It must remain identical after collection.
    teacher_evaluation supplies successes/completed_episodes/checkpoint_sha256
    from real episodes on this task/interface; source learning alone is not an
    expert qualification. The caller declares the minimum acceptable competence.
    """
    if sys.platform != "linux":
        raise ValueError("CND collection and tests require allocated Linux resources")
    if (not scenes or min_episodes < 2 or max_episodes < min_episodes
            or episode_horizon < 1 or chunk < 1 or max_primitive_steps < 1
            or noise_dim < 1 or type(teacher_steps) is not int or teacher_steps < 0
            or type(teacher_primitive_steps) is not int or teacher_primitive_steps < 0
            or not motor_sha256 or not feature_contract or not noise_contract or not callable(teacher_action)
            or not callable(teacher_fingerprint) or not 0 <= seed < 2**32):
        raise ValueError("Require explicit complete-trial collection and immutable teacher contracts")
    if (len(anneal_decisions) != 2 or any(type(v) is not int or v < 2 for v in anneal_decisions)
            or anneal_decisions[0] > anneal_decisions[1] or not anneal_powers
            or any(not math.isfinite(v) or v <= 0 for v in anneal_powers)):
        raise ValueError("Require declared annealing duration bounds and finite positive powers")
    if len(scenes) * max_episodes * episode_horizon > max_primitive_steps:
        raise ValueError("Worst-case complete episode cost exceeds the explicit CND interaction budget")
    required = {"trial_id", "split", "initial_state", "scene_seed"}
    if any(not isinstance(s, dict) or set(s) != required for s in scenes):
        raise ValueError("Require exactly trial/split/scene/seed identities")
    ids = [s["trial_id"] for s in scenes]
    if (any(not isinstance(i, str) or not i for i in ids) or len(ids) != len(set(ids))
            or any(s["split"] not in ("train", "validation")
                   or type(s["initial_state"]) is not int or s["initial_state"] < 0
                   or type(s["scene_seed"]) is not int or not 0 <= s["scene_seed"] < 2**31 for s in scenes)):
        raise ValueError("Invalid unique CND identities, split or scene seed")
    banks = {split: {s["initial_state"] for s in scenes if s["split"] == split}
             for split in ("train", "validation")}
    if (not all(banks.values()) or banks["train"] & banks["validation"]
            or set.union(*banks.values()) & set(heldout_initial_states)):
        raise ValueError("CND train/validation/evaluation scene indices must be disjoint")
    teacher_checkpoint, output_dir = Path(teacher_checkpoint), Path(output_dir)
    checkpoint_digest = file_sha256(teacher_checkpoint)
    if (not math.isfinite(min_teacher_success_rate) or not 0 <= min_teacher_success_rate <= 1
            or min_teacher_evaluation_episodes < 1
            or teacher_evaluation.get("checkpoint_sha256") != checkpoint_digest
            or type(teacher_evaluation.get("completed_episodes")) is not int
            or type(teacher_evaluation.get("successes")) is not int
            or teacher_evaluation["completed_episodes"] < min_teacher_evaluation_episodes
            or not 0 <= teacher_evaluation["successes"] <= teacher_evaluation["completed_episodes"]
            or teacher_evaluation["successes"] / teacher_evaluation["completed_episodes"] < min_teacher_success_rate):
        raise ValueError("The fixed teacher needs matching measured robot competence before CND collection")
    if episode_runner is None:
        from .vla_libero import run_episode
        episode_runner = run_episode
    fingerprint = teacher_fingerprint()
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError("Require an actual source-state fingerprint")
    contract = dict(format="frozen_vla_cnd_collection_v1", teacher_checkpoint_sha256=checkpoint_digest,
        teacher_state_sha256=fingerprint, teacher_steps=teacher_steps,
        teacher_primitive_steps=teacher_primitive_steps, teacher_evaluation=teacher_evaluation,
        teacher_qualification=dict(min_success_rate=min_teacher_success_rate,
                                   min_completed_episodes=min_teacher_evaluation_episodes),
        motor_sha256=motor_sha256, feature_contract=feature_contract, noise_contract=noise_contract,
        seed=seed, scenes=scenes,
        min_episodes=min_episodes, max_episodes=max_episodes, episode_horizon=episode_horizon,
        chunk=chunk, anneal_decisions=list(anneal_decisions), anneal_powers=list(anneal_powers),
        noise_dim=noise_dim, noise_bounds=[-1., 1.], max_primitive_steps=max_primitive_steps,
        heldout_initial_states=list(heldout_initial_states),
        adaptation="Vintix CND convex mixture in frozen-VLA steering coordinates, not physical actions",
        target="actual executed noisy steering coordinate; schedule never enters student inputs",
        schedule="epsilon=(1-min(1,decision/(duration-1)))**power; clock spans episode resets")
    output_dir.mkdir(parents=True, exist_ok=False)
    _json(output_dir / "contract.json", contract)
    datasets = {split: [] for split in banks}
    trial_ids = {split: [] for split in banks}
    rng = np.random.default_rng(seed)
    receipts, total_steps, settling_steps = [], 0, 0
    try:
        for ordinal, scene in enumerate(scenes):
            schedule = NoiseSchedule(int(rng.integers(anneal_decisions[0], anneal_decisions[1] + 1)),
                                     float(rng.choice(anneal_powers)))
            trial_dir = output_dir / f"trial-{ordinal:05d}"
            trial_dir.mkdir()
            rows, starts, epsilons, expert_actions, uniforms = [], [], [], [], []
            first_hash, episode_receipts, decision = None, [], 0
            for episode in range(max_episodes):
                def choose(obs):
                    nonlocal decision
                    expert = np.asarray(teacher_action(obs), np.float32)
                    if expert.shape != (noise_dim,):
                        raise ValueError("Teacher steering dimension differs from the declared interface")
                    uniform = rng.uniform(-1., 1., noise_dim).astype(np.float32)
                    epsilon = schedule.epsilon(decision)
                    action = mix_noise(expert, uniform, epsilon)
                    expert_actions.append(expert.copy())
                    uniforms.append(uniform.copy())
                    epsilons.append(epsilon)
                    decision += 1
                    return action
                actual, summary = episode_runner(sim, motor, choose,
                    initial_state=scene["initial_state"], scene_seed=scene["scene_seed"], chunk=chunk, bounded=True)
                if (not actual or not actual[-1]["done"] or any(r["done"] for r in actual[:-1])
                        or summary["primitive_steps"] > episode_horizon
                        or summary["primitive_steps"] != sum(r["primitive_steps"] for r in actual)
                        or len(rows) + len(actual) != decision):
                    raise ValueError("CND requires a complete episode and every actual steering transition")
                for index, row in enumerate(actual, start=len(rows)):
                    expected = mix_noise(expert_actions[index], uniforms[index], epsilons[index])
                    if not np.array_equal(np.asarray(row["executed_noise"], np.float32), expected):
                        raise ValueError("Executed noise differs from the declared CND mixture")
                current_hash = summary["initial_observation_sha256"]
                if first_hash is None:
                    first_hash = current_hash
                elif current_hash != first_hash:
                    raise ValueError("Repeated CND scene changed its actual initial observation")
                rows.extend(actual)
                starts.extend([True] + [False] * (len(actual) - 1))
                total_steps += int(summary["primitive_steps"])
                settling_steps += int(getattr(sim, "metadata", {}).get("settle_steps", 0))
                if total_steps > max_primitive_steps:
                    raise RuntimeError("CND exceeded its declared physical interaction budget")
                path = trial_dir / f"episode-{episode + 1:03d}.pkl"
                with path.open("xb") as stream:
                    pickle.dump(dict(rows=actual, summary=summary,
                        epsilon=epsilons[-len(actual):], expert_noise=expert_actions[-len(actual):],
                        uniform_noise=uniforms[-len(actual):], schedule=asdict(schedule),
                        teacher_steps=teacher_steps), stream, protocol=5)
                episode_receipts.append(dict(episode=episode + 1, **summary,
                    epsilon_first=epsilons[-len(actual)], epsilon_last=epsilons[-1],
                    trace_sha256=file_sha256(path)))
                _json(trial_dir / "progress.json", dict(episodes=episode_receipts,
                    steering_decisions=decision, primitive_steps=sum(r["primitive_steps"] for r in rows),
                    schedule=asdict(schedule)))
                if episode + 1 >= min_episodes and decision >= schedule.decisions:
                    break
            if decision < schedule.decisions:
                raise RuntimeError("Complete-episode budget ended before epsilon reached zero; no complete CND dataset exported")
            observations = np.stack([row["public_features"] if row_feature_encoder is None
                                     else row_feature_encoder(row) for row in rows]).astype(np.float32)
            executed = np.stack([row["executed_noise"] for row in rows]).astype(np.float32)
            trial = dict(observations=observations, executed_noise=executed, target_noise=executed.copy(),
                rewards=np.asarray([row["reward"] for row in rows], np.float32),
                dones=np.asarray([row["done"] for row in rows], bool),
                episode_starts=np.asarray(starts, bool),
                teacher_steps=np.full(len(rows), teacher_steps, np.int64))
            datasets[scene["split"]].append(trial)
            trial_ids[scene["split"]].append(scene["trial_id"])
            receipts.append(dict(scene=scene, schedule=asdict(schedule), episodes=episode_receipts,
                                 steering_decisions=decision, epsilon_last=epsilons[-1]))
    finally:
        final_fingerprint = teacher_fingerprint()
        unchanged = final_fingerprint == fingerprint and file_sha256(teacher_checkpoint) == checkpoint_digest
        _json(output_dir / "teacher-integrity.json", dict(unchanged=unchanged,
            before_state_sha256=fingerprint, after_state_sha256=final_fingerprint,
            checkpoint_sha256=checkpoint_digest))
        if not unchanged:
            raise RuntimeError("CND changed its fixed source teacher or checkpoint")
    paths = {}
    for split in banks:
        path = output_dir / f"cnd-{split}.npz"
        save_dataset(path, datasets[split], objective="cnd", metadata=dict(
            trial_ids=trial_ids[split], motor_sha256=motor_sha256, feature_contract=feature_contract,
            noise_contract=noise_contract,
            source_contract=contract, scene_receipts=receipts, collection_primitive_steps=total_steps,
            collection_settling_steps=settling_steps,
            teacher_primitive_steps=teacher_primitive_steps,
            total_source_primitive_steps=teacher_primitive_steps + total_steps,
            source_rule="fixed deterministic teacher; annealed uniform noise; targets equal executed mixtures"))
        paths[split] = path
    _json(output_dir / "summary.json", dict(completed=True, collection_primitive_steps=total_steps,
        teacher_primitive_steps=teacher_primitive_steps, total_source_primitive_steps=teacher_primitive_steps + total_steps,
        collection_settling_steps=settling_steps, trials=receipts,
        datasets={split: str(path) for split, path in paths.items()}))
    return paths
