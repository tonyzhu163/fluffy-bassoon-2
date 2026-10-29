"""Same-scene retry teachers branched independently from one shared DSRL prior.

The source learner changes weights between attempts; the deployed student does
not. All students use one fixed shared-prior encoder, including after a teacher
adapts its own encoder. This module does not allocate or launch jobs.
"""

import json
from pathlib import Path
import pickle
import sys

import numpy as np

from .vla_history import file_sha256, save_dataset


def _json(path, value):
    temporary = Path(path).with_suffix(".json.partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def append_episode_to_replay(learner, rows, *, discount=.999):
    """The same replay projection as the ordinary source DSRL path."""
    for row in rows:
        learner.replay.append(dict(observations=row["sac_observation"],
            next_observations=row["sac_next_observation"], actions=row["executed_noise"][None],
            next_actions=row["executed_noise"][None], rewards=np.float32(row["reward"]),
            masks=np.float32(not row["success"]),
            discount=np.float32(discount ** row["primitive_steps"])))


def collect_retry_teachers(*, source_root, shared_prior_path, prior_contract, fixed_encoder,
        scenes, attempts, batch_size, utd, chunk, output_dir, seed=0,
        objectives=("ad", "dpt"), episode_horizon=400, max_primitive_steps,
        heldout_initial_states=(), learner_factory=None, episode_runner=None,
        sim=None, motor=None, row_feature_encoder=None, fixed_feature_contract=None):
    """Collect independent K-attempt learning trials on explicitly split scenes.

    scenes contains dicts with `trial_id`, `split` (train/validation),
    `initial_state`, and `scene_seed`. Every attempt in one trial resets to the
    identical state and seed. `shared_prior_path` must be a complete, immutable
    OfficialSAC recovery checkpoint. The original prior restore contract is
    preserved; each branch's own saved checkpoints use a branch contract.

    The fixed_encoder callback accepts one saved SAC observation and returns
    a public feature vector. It must belong to the shared prior, never a
    scene-adapted teacher. Source models/replay are restored independently.
    Alternatively row_feature_encoder accepts each full saved row (for fixed
    native VLA camera/instruction features); fixed_feature_contract then records
    that frozen representation rather than the shared SAC prior encoder.
    """
    if sys.platform != "linux":
        raise ValueError("Retry teaching requires allocated Linux resources")
    if (not scenes or attempts < 2 or batch_size < 1 or utd < 1 or chunk < 1
            or episode_horizon < 1 or max_primitive_steps < 1
            or not objectives or set(objectives) - {"ad", "dpt"}
            or len(set(objectives)) != len(objectives)):
        raise ValueError("Require valid explicit retry teaching and interaction budgets")
    if len(scenes) * attempts * episode_horizon > max_primitive_steps:
        raise ValueError("Worst-case retry teacher interactions exceed the declared source budget")
    required = {"trial_id", "split", "initial_state", "scene_seed"}
    if any(not isinstance(scene, dict) or set(scene) != required for scene in scenes):
        raise ValueError("Require exactly trial identity, split, initial state and scene seed")
    identities = [scene["trial_id"] for scene in scenes]
    if (any(not isinstance(value, str) or not value for value in identities)
            or len(set(identities)) != len(identities)
            or any(scene["split"] not in ("train", "validation")
                   or type(scene["initial_state"]) is not int or scene["initial_state"] < 0
                   or type(scene["scene_seed"]) is not int or not 0 <= scene["scene_seed"] < 2**31
                   for scene in scenes)):
        raise ValueError("Invalid unique scene identities/splits/seeds")
    splits = {name: {scene["initial_state"] for scene in scenes if scene["split"] == name}
              for name in ("train", "validation")}
    if (not all(splits.values()) or splits["train"] & splits["validation"]
            or (splits["train"] | splits["validation"]) & set(heldout_initial_states)):
        raise ValueError("Require train/validation initial states disjoint from each other and evaluation")
    if learner_factory is None or episode_runner is None:
        from .vla_libero import OfficialSAC, run_episode
        learner_factory = OfficialSAC if learner_factory is None else learner_factory
        episode_runner = run_episode if episode_runner is None else episode_runner
    shared_prior_path, output_dir = Path(shared_prior_path), Path(output_dir)
    prior_digest = file_sha256(shared_prior_path)
    output_dir.mkdir(parents=True, exist_ok=False)
    feature_contract = (dict(kind="fixed shared-prior SAC actor bottleneck and public proprioception",
        shared_prior_sha256=prior_digest) if fixed_feature_contract is None else
        json.loads(json.dumps(fixed_feature_contract, allow_nan=False)))
    if not feature_contract or (row_feature_encoder is not None and fixed_feature_contract is None):
        raise ValueError("A row feature encoder requires its explicit fixed representation contract")
    contract = dict(format="frozen_vla_retry_teachers_v1", shared_prior_sha256=prior_digest,
        prior_contract=prior_contract, fixed_feature_encoder=feature_contract,
        scenes=scenes, attempts=attempts, batch_size=batch_size, utd=utd, chunk=chunk,
        episode_horizon=episode_horizon, max_primitive_steps=max_primitive_steps, seed=seed,
        objectives=list(objectives), heldout_initial_states=list(heldout_initial_states),
        source_rule="fresh full prior restore per scene; update between same-scene attempts",
        source_cost="shared prior training plus actual additional retry-teacher primitive steps",
        reward="recorded source -1/0 sparse transition reward; raw primitive rewards retained in traces")
    _json(output_dir / "contract.json", contract)
    datasets = {objective: {split: [] for split in splits} for objective in objectives}
    trial_ids = {split: [] for split in splits}
    receipts, total_steps, total_updates, prior_primitive_steps = [], 0, 0, None
    for ordinal, scene in enumerate(scenes):
        trial_dir = output_dir / f"trial-{ordinal:05d}"
        trial_dir.mkdir()
        branch_seed = int(np.random.SeedSequence([seed, ordinal, 1853]).generate_state(1)[0])
        sampler = np.random.default_rng(branch_seed)
        learner = learner_factory(source_root, branch_seed)
        prior_progress = learner.restore(shared_prior_path, sampler, prior_contract)
        restored_cost = int(prior_progress["primitive_steps"])
        if restored_cost < 0 or (prior_primitive_steps is not None and restored_cost != prior_primitive_steps):
            raise ValueError("Shared prior training cost differs across independent restores")
        prior_primitive_steps = restored_cost
        # The restore recovers the source sampler. Branch only its explicit
        # minibatch stream; actor sampling starts from the paired prior RNG.
        sampler = np.random.default_rng(branch_seed)
        branch_contract = dict(parent=contract, scene=scene, ordinal=ordinal, branch_seed=branch_seed)
        _json(trial_dir / "contract.json", branch_contract)
        all_rows, boundaries, source_steps, episode_receipts = [], [], [], []
        trial_steps, first_update, initial_hash = 0, learner.updates, None
        for attempt in range(attempts):
            before = learner.updates
            rows, summary = episode_runner(sim, motor, learner.action,
                initial_state=scene["initial_state"], scene_seed=scene["scene_seed"],
                chunk=chunk, bounded=True)
            if (not rows or not rows[-1]["done"] or any(row["done"] for row in rows[:-1])
                    or summary["primitive_steps"] > episode_horizon
                    or int(summary["primitive_steps"]) != sum(int(row["primitive_steps"]) for row in rows)):
                raise ValueError("Retry teacher must collect exactly one complete bounded episode")
            if initial_hash is None:
                initial_hash = summary["initial_observation_sha256"]
            elif summary["initial_observation_sha256"] != initial_hash:
                raise ValueError("Same-state retry initial observation changed")
            append_episode_to_replay(learner, rows)
            for _ in range(len(rows) * utd):
                learner.update(batch_size, sampler)
            trial_steps += int(summary["primitive_steps"])
            total_steps += int(summary["primitive_steps"])
            total_updates += learner.updates - before
            if total_steps > max_primitive_steps:
                raise RuntimeError("Retry teacher exceeded source interaction budget")
            trace_path = trial_dir / f"episode-{attempt + 1:03d}.pkl"
            with trace_path.open("xb") as stream:
                pickle.dump(dict(rows=rows, summary=summary, attempt=attempt + 1,
                    scene=scene, source_updates_before=before, source_updates_after=learner.updates,
                    shared_prior_sha256=prior_digest), stream, protocol=5)
            # A complete learning boundary is saved before proceeding to the
            # next retry. It includes optimizer/replay/sampler/actor RNG state.
            learner.save(trial_dir / "source-recovery.pkl", sampler, branch_contract,
                episodes=attempt + 1, primitive_steps=trial_steps, prior_progress=prior_progress,
                source_updates_before_trial=first_update)
            receipt = dict(attempt=attempt + 1, **summary, source_updates_before=before,
                source_updates_after=learner.updates, trace_sha256=file_sha256(trace_path))
            episode_receipts.append(receipt)
            all_rows.extend(rows)
            boundaries.extend([True] + [False] * (len(rows) - 1))
            source_steps.extend([before] * len(rows))
            _json(trial_dir / "progress.json", dict(episodes=episode_receipts,
                primitive_steps=trial_steps, total_source_steps=total_steps))
        observations = np.stack([fixed_encoder(row["sac_observation"]) if row_feature_encoder is None
                                 else row_feature_encoder(row) for row in all_rows]).astype(np.float32)
        executed = np.stack([row["executed_noise"] for row in all_rows]).astype(np.float32)
        base = dict(observations=observations, executed_noise=executed,
            rewards=np.asarray([row["reward"] for row in all_rows], np.float32),
            dones=np.asarray([row["done"] for row in all_rows], bool),
            episode_starts=np.asarray(boundaries, bool), teacher_steps=np.asarray(source_steps, np.int64))
        for objective in objectives:
            if objective == "ad":
                target = executed.copy()
            else:
                target = np.stack([np.asarray(learner.agent.eval_actions(
                    {key: value[None] for key, value in row["sac_observation"].items()})).reshape(-1)
                    for row in all_rows]).astype(np.float32)
            datasets[objective][scene["split"]].append(dict(**base, target_noise=target))
        trial_ids[scene["split"]].append(scene["trial_id"])
        receipts.append(dict(scene=scene, episodes=episode_receipts, primitive_steps=trial_steps,
            source_update_delta=learner.updates - first_update,
            final_teacher_sha256=file_sha256(trial_dir / "source-recovery.pkl")))
    if file_sha256(shared_prior_path) != prior_digest:
        raise RuntimeError("Shared prior checkpoint changed during retry teaching")
    metadata = dict(motor_sha256=prior_contract["motor"]["receipt_sha256"],
        feature_contract=feature_contract,
        noise_contract=prior_contract["noise"], source_commit=prior_contract["source_commit"],
        source_prior_training_contract=prior_contract, extra_source_primitive_steps=total_steps,
        extra_source_updates=total_updates, shared_prior_primitive_steps=prior_primitive_steps,
        total_source_primitive_steps=prior_primitive_steps + total_steps,
        attempts=attempts, scene_receipts=receipts,
        split="disjoint initial-state indices; same state/seed within each retry trial")
    paths = {}
    for objective in objectives:
        paths[objective] = {}
        for split in splits:
            path = output_dir / f"{objective}-{split}.npz"
            save_dataset(path, datasets[objective][split], objective=objective,
                         metadata=dict(**metadata, trial_ids=trial_ids[split]))
            paths[objective][split] = path
    _json(output_dir / "summary.json", dict(completed=True, extra_source_primitive_steps=total_steps,
        extra_source_updates=total_updates, shared_prior_primitive_steps=prior_primitive_steps,
        total_source_primitive_steps=prior_primitive_steps + total_steps,
        shared_prior_sha256=prior_digest, trials=receipts,
        datasets={objective: {split: str(path) for split, path in split_paths.items()}
                  for objective, split_paths in paths.items()}))
    return paths, metadata
