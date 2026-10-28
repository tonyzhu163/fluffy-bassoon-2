"""Private, deterministic native histories with explicit balanced provenance.

Planner outputs are simulator/operator metadata. They are never policy inputs.
No model, simulator, Torch or GPU initialization is performed by this module.
"""

from collections import Counter
import hashlib
import json
import random


PLAN_FORMAT = "native_pi05_balanced_plan_v1"
MANIFEST_FORMAT = "native_pi05_balanced_history_manifest_v1"
TRAIN_THETAS = (-45., 0., 45.)
EVAL_THETAS = (-30., 30.)
OBJECTS = (0, 1)
TRAIN_OVERLAYS = tuple(range(12))
HELDOUT_OVERLAYS = tuple(range(12, 16))
ATTEMPTS, SUPPORT_ATTEMPTS = 4, 3
EPISODE_STEPS, ACTION_CHUNK, DENOISE_STEPS = 80, 5, 4


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def plan_sha256(plan):
    return canonical_sha256({key: value for key, value in plan.items() if key != "plan_sha256"})


def manifest_sha256(manifest):
    return canonical_sha256({key: value for key, value in manifest.items() if key != "manifest_sha256"})


def _seed(value):
    if type(value) is not int or not 0 <= value < 2**32:
        raise ValueError("Seed must be an integer in [0, 2**32)")
    return value


def _derive_seed(seed, *identity, validation=False):
    # Disjoint parity partitions make train/validation seed overlap impossible.
    digest = canonical_sha256([PLAN_FORMAT, _seed(seed), *identity])
    return (int(digest[:8], 16) & 0x7fffffff) * 2 + int(validation)


def _scene_contract(infos):
    if not isinstance(infos, (list, tuple)) or len(infos) != 8:
        raise ValueError("Require RLinf's eight-field basic_obj_infos")
    objects, _, overlays, _, plates, _, positions, orientations = infos
    if ((objects, overlays, plates) != (25, 21, 1)
            or type(positions) is not int or positions < 1
            or type(orientations) is not int or orientations < 1):
        raise ValueError("Unexpected Main-v3 scene dimensions")
    return dict(objects=objects, overlays=overlays, plates=plates,
                positions=positions, orientations=orientations)


def _episode_id(contract, obj, overlay, position, orientation):
    return ((obj * contract["plates"]) * contract["overlays"] + overlay) * (
        contract["positions"] * contract["orientations"]) + position * contract["orientations"] + orientation


def _draw_scene(contract, obj, overlays, rng, used):
    positions, orientations = contract["positions"], contract["orientations"]
    capacity = len(overlays) * positions * orientations
    occupied = sum(row["object"] == obj and row["overlay"] in overlays for row in used.values())
    if occupied >= capacity:
        raise ValueError("Scene budget exceeds distinct available scenes")
    if occupied * 2 >= capacity:
        # Exact uniform sampling on the remaining pool also handles tiny fixtures.
        available = [(overlay, position, orientation) for overlay in overlays for position in range(positions)
                     for orientation in range(orientations)
                     if _episode_id(contract, obj, overlay, position, orientation) not in used]
        overlay, position, orientation = rng.choice(available)
    else:
        for _ in range(256):
            overlay, position, orientation = rng.choice(overlays), rng.randrange(positions), rng.randrange(orientations)
            if _episode_id(contract, obj, overlay, position, orientation) not in used:
                break
        else:
            raise ValueError("Could not draw a distinct scene within the fixed planning budget")
    row = dict(object=obj, overlay=overlay, position=position, orientation=orientation,
               episode_id=_episode_id(contract, obj, overlay, position, orientation))
    used[row["episode_id"]] = row
    return row


def _trial_id(split, history_seed, theta, obj, scenes):
    return canonical_sha256(dict(split=split, history_seed=history_seed,
                                 theta_degrees=theta, object=obj, scenes=scenes))


def _make_plan(contract, *, seed, split, thetas, repeat, overlays_by_attempt, used,
               cells=None, attempt_roles=None):
    cells = [(theta, obj) for theta in thetas for obj in OBJECTS] if cells is None else cells
    histories, lane_scenes, trial_ids = [], [], []
    for theta, obj in cells:
        history_seed = _derive_seed(seed, split, repeat, theta, obj, "history", validation=split == "validation")
        rng = random.Random(history_seed)
        scenes = [_draw_scene(contract, obj, overlays, rng, used) for overlays in overlays_by_attempt]
        histories.append(history_seed)
        lane_scenes.append(scenes)
        trial_ids.append(_trial_id(split, history_seed, theta, obj, scenes))
    attempts = len(overlays_by_attempt)
    plan = dict(format=PLAN_FORMAT, theta_degrees=[theta for theta, _ in cells],
                object_indices=[obj for _, obj in cells], history_seeds=histories, trial_ids=trial_ids,
                scenes=[[lane_scenes[lane][attempt] for lane in range(len(cells))] for attempt in range(attempts)],
                reset_seeds=[_derive_seed(seed, split, repeat, attempt, "reset", validation=split == "validation")
                             for attempt in range(attempts)],
                split=split, repeat=repeat,
                attempt_roles=(attempt_roles if attempt_roles is not None else
                               ["support"] * (attempts - 1) + ["probe"] if attempts > 1 else ["probe"]),
                overlays_by_attempt=[list(overlays) for overlays in overlays_by_attempt],
                scene_contract=contract, metadata_is_private=True)
    plan["plan_sha256"] = plan_sha256(plan)
    return plan


def _coverage(plans):
    counts = Counter((theta, obj) for plan in plans
                     for theta, obj in zip(plan["theta_degrees"], plan["object_indices"], strict=True))
    return [dict(theta_degrees=theta, object=obj, trials=counts[theta, obj])
            for theta, obj in sorted(counts)]


def make_balanced_history_manifest(infos, *, seed=0, trials_per_cell=1, lanes=6):
    """Return explicit train/validation plans; default is one six-lane batch each.

    Both history splits use overlays 0..11 and all six theta/object cells.
    Complete history seeds and all scene identities are disjoint across splits.
    Validation is an in-distribution history holdout, not a task/angle holdout.
    """
    _seed(seed)
    if type(trials_per_cell) is not int or trials_per_cell < 1 or type(lanes) is not int or lanes != 6:
        raise ValueError("Require positive trials_per_cell and exactly six balanced lanes")
    contract, used, plans = _scene_contract(infos), {}, []
    for split in ("train", "validation"):
        for repeat in range(trials_per_cell):
            plans.append(_make_plan(contract, seed=seed, split=split, thetas=TRAIN_THETAS, repeat=repeat,
                                    overlays_by_attempt=[TRAIN_OVERLAYS] * ATTEMPTS, used=used))
    training, validation = list(range(trials_per_cell)), list(range(trials_per_cell, 2 * trials_per_cell))
    trials = 6 * trials_per_cell
    manifest = dict(format=MANIFEST_FORMAT, seed=seed, plans=plans,
                    train_indices=training, validation_indices=validation,
                    coverage={split: _coverage([plans[index] for index in indices])
                              for split, indices in (("train", training), ("validation", validation))},
                    trial_sets={split: [identity for index in indices for identity in plans[index]["trial_ids"]]
                                for split, indices in (("train", training), ("validation", validation))},
                    sample_budget=dict(lanes=lanes, train_trials=trials, validation_trials=trials,
                        attempts_per_trial=ATTEMPTS, support_attempts=SUPPORT_ATTEMPTS, probe_attempts=1,
                        episode_steps=EPISODE_STEPS, action_chunk=ACTION_CHUNK, denoise_steps=DENOISE_STEPS,
                        train_simulator_steps=trials * ATTEMPTS * EPISODE_STEPS,
                        validation_simulator_steps=trials * ATTEMPTS * EPISODE_STEPS,
                        total_simulator_steps=2 * trials * ATTEMPTS * EPISODE_STEPS,
                        total_policy_chunks=2 * trials * ATTEMPTS * EPISODE_STEPS // ACTION_CHUNK,
                        additional_label_motor_calls=2 * trials * ATTEMPTS * EPISODE_STEPS // ACTION_CHUNK),
                    split_contract="explicit complete-history indices; shared theta/object support; disjoint seeds and scenes")
    manifest["manifest_sha256"] = manifest_sha256(manifest)
    validate_balanced_history_manifest(manifest)
    return manifest


def make_heldout_probe_plan(infos, *, seed=1, repeat=0, probes=1):
    """Return the evaluator's three-support/fresh-probe wrapper at four balanced lanes.

    Opposite-history partners share their object and differ only in theta sign.
    Probe scenes never overlap any support scene or another probe replicate.
    Probe noise seeds are reused across evaluation conditions, not resampled.
    """
    _seed(seed)
    if type(repeat) is not int or repeat < 0 or type(probes) is not int or probes < 1:
        raise ValueError("Require a nonnegative repeat and positive probe count")
    contract, used = _scene_contract(infos), {}
    cells = [(theta, obj) for obj in OBJECTS for theta in EVAL_THETAS]
    support = _make_plan(contract, seed=seed, split="heldout_support", thetas=EVAL_THETAS,
                         repeat=repeat, overlays_by_attempt=[TRAIN_OVERLAYS] * SUPPORT_ATTEMPTS,
                         used=used, cells=cells, attempt_roles=["support"] * SUPPORT_ATTEMPTS)
    probe_plans = []
    for index in range(probes):
        plan = _make_plan(contract, seed=seed, split="heldout_probe", thetas=EVAL_THETAS,
                          repeat=repeat * probes + index, overlays_by_attempt=[HELDOUT_OVERLAYS],
                          used=used, cells=cells)
        probe = dict(scenes=plan["scenes"][0], reset_seed=plan["reset_seeds"][0],
                     theta_degrees=plan["theta_degrees"], object_indices=plan["object_indices"],
                     trial_ids=plan["trial_ids"], history_seeds=plan["history_seeds"],
                     split=plan["split"], repeat=plan["repeat"], metadata_is_private=True)
        probe["plan_sha256"] = plan_sha256(probe)
        probe_plans.append(probe)
    result = dict(format="native_pi05_support_probe_plan_v1", support_plan=support,
                  probe_plans=probe_plans, opposite_indices=[1, 0, 3, 2],
                  probe_seeds=[_derive_seed(seed, "heldout_probe_noise", repeat, index) for index in range(probes)],
                  metadata_is_private=True, theta_split="held-out +/-30-degree interpolation between training angles",
                  sample_budget=dict(support_trials=4, support_attempts=3, support_simulator_steps=4 * 3 * EPISODE_STEPS,
                                     probe_replicates=probes, probe_simulator_steps_per_condition=4 * probes * EPISODE_STEPS))
    result["plan_sha256"] = plan_sha256(result)
    return result


def make_teacher_probe_plan(infos, *, seed=2, theta_degrees=None, repeat=0):
    """One fresh held-out-overlay attempt per cell for paired interface competence.

    theta_degrees=None yields all six cells; selecting a training theta yields
    two object lanes. Private theta may condition only the privileged teacher.
    """
    _seed(seed)
    if theta_degrees is not None and theta_degrees not in TRAIN_THETAS:
        raise ValueError("Teacher qualification requires a declared training theta")
    if type(repeat) is not int or repeat < 0:
        raise ValueError("repeat must be a nonnegative integer")
    thetas = TRAIN_THETAS if theta_degrees is None else (float(theta_degrees),)
    return _make_plan(_scene_contract(infos), seed=seed, split="teacher_interface_probe", thetas=thetas,
                      repeat=repeat, overlays_by_attempt=[HELDOUT_OVERLAYS], used={})


def validate_balanced_history_manifest(manifest, *, expected_sha256=None):
    """Validate signed plans, exact split identities, per-cell balance and sample accounting."""
    if not isinstance(manifest, dict) or manifest.get("format") != MANIFEST_FORMAT:
        raise ValueError("Unknown balanced history manifest")
    digest = manifest_sha256(manifest)
    if manifest.get("manifest_sha256") != digest or (expected_sha256 is not None and expected_sha256 != digest):
        raise ValueError("Balanced history manifest hash differs")
    plans = manifest.get("plans", [])
    train, validation = manifest.get("train_indices", []), manifest.get("validation_indices", [])
    if (not plans or not train or not validation or any(type(index) is not int for index in train + validation)
            or sorted(train + validation) != list(range(len(plans))) or set(train) & set(validation)):
        raise ValueError("Require explicit disjoint exhaustive train/validation batch indices")
    expected_cells = Counter((theta, obj) for theta in TRAIN_THETAS for obj in OBJECTS)
    seen_trials, seen_seeds, seen_resets, seen_scenes = set(), set(), set(), set()
    for split, indices in (("train", train), ("validation", validation)):
        split_plans = [plans[index] for index in indices]
        for plan in split_plans:
            if (plan.get("format") != PLAN_FORMAT or plan.get("split") != split
                    or plan.get("plan_sha256") != plan_sha256(plan) or plan.get("metadata_is_private") is not True
                    or len(plan.get("scenes", [])) != ATTEMPTS
                    or plan.get("attempt_roles") != ["support"] * SUPPORT_ATTEMPTS + ["probe"]
                    or plan.get("overlays_by_attempt") != [list(TRAIN_OVERLAYS)] * ATTEMPTS):
                raise ValueError("Balanced plan hash, split or attempt contract differs")
            if Counter(zip(plan["theta_degrees"], plan["object_indices"], strict=True)) != expected_cells:
                raise ValueError("Every history batch must cover all six theta/object cells exactly once")
            if (len(plan["history_seeds"]) != 6 or len(plan["trial_ids"]) != 6 or len(plan["reset_seeds"]) != ATTEMPTS
                    or any(len(batch) != 6 for batch in plan["scenes"])):
                raise ValueError("Balanced history lane/attempt dimensions differ")
            for value in plan["reset_seeds"]:
                _seed(value)
                if value in seen_resets:
                    raise ValueError("Reset seeds must identify disjoint complete histories")
                seen_resets.add(value)
            for lane, (theta, obj) in enumerate(zip(plan["theta_degrees"], plan["object_indices"], strict=True)):
                seed, identity = plan["history_seeds"][lane], plan["trial_ids"][lane]
                _seed(seed)
                if seed in seen_seeds or identity in seen_trials:
                    raise ValueError("History seeds/trial identities overlap")
                seen_seeds.add(seed)
                seen_trials.add(identity)
                scenes = [batch[lane] for batch in plan["scenes"]]
                for scene in scenes:
                    if (scene["object"] != obj or scene["overlay"] not in TRAIN_OVERLAYS
                            or any(type(scene[key]) is not int or scene[key] < 0 for key in
                                   ("object", "overlay", "position", "orientation", "episode_id"))
                            or scene["position"] >= plan["scene_contract"]["positions"]
                            or scene["orientation"] >= plan["scene_contract"]["orientations"]
                            or scene["episode_id"] != _episode_id(plan["scene_contract"], obj, scene["overlay"],
                                                                  scene["position"], scene["orientation"])
                            or scene["episode_id"] in seen_scenes):
                        raise ValueError("Scene identity overlaps or violates the fixed-object training contract")
                    seen_scenes.add(scene["episode_id"])
                if identity != _trial_id(split, seed, theta, obj, scenes):
                    raise ValueError("Trial identity does not match its complete history")
        if manifest["coverage"].get(split) != _coverage(split_plans):
            raise ValueError("Declared per-cell coverage differs")
        if manifest["trial_sets"].get(split) != [identity for plan in split_plans for identity in plan["trial_ids"]]:
            raise ValueError("Declared split trial sets differ")
    budget = manifest["sample_budget"]
    train_trials, validation_trials = 6 * len(train), 6 * len(validation)
    if (train_trials != validation_trials or budget.get("lanes") != 6
            or budget.get("train_trials") != train_trials or budget.get("validation_trials") != validation_trials
            or budget.get("attempts_per_trial") != ATTEMPTS or budget.get("support_attempts") != SUPPORT_ATTEMPTS
            or budget.get("probe_attempts") != 1 or budget.get("episode_steps") != EPISODE_STEPS
            or budget.get("action_chunk") != ACTION_CHUNK or budget.get("denoise_steps") != DENOISE_STEPS
            or budget.get("train_simulator_steps") != train_trials * ATTEMPTS * EPISODE_STEPS
            or budget.get("validation_simulator_steps") != validation_trials * ATTEMPTS * EPISODE_STEPS
            or budget.get("total_simulator_steps") != (train_trials + validation_trials) * ATTEMPTS * EPISODE_STEPS
            or budget.get("total_policy_chunks") != (train_trials + validation_trials) * ATTEMPTS * EPISODE_STEPS // ACTION_CHUNK
            or budget.get("additional_label_motor_calls") != budget.get("total_policy_chunks")):
        raise ValueError("Balanced history sample budget differs")
    return dict(manifest_sha256=digest, train_trials=train_trials, validation_trials=validation_trials,
                total_simulator_steps=budget["total_simulator_steps"], private_metadata_only=True)
