"""Source-training provenance checks for generated Goal 5 banks, without simulator imports."""

import hashlib
import json
from pathlib import Path
import re

SOURCE_KIND = "native_bddl_sample"
GROUP_BASE = 1_000_000
API_ORIGIN = "ControlEnv.seed/reset/get_sim_state"
BANK_SHA = "794beed679b18bbd7d9fb851c81877a974915ad63addff2d5ac5af0607fafac9"
PUBLIC_LIST_SHA = "52d3d8066748eddea7ac875260cd198d3c49b15d69adb63cd3888e89b7cbadaf"
TASK = dict(suite="libero_goal", task_id=5, case="BASE")
MOVABLES = ("akita_black_bowl_1", "cream_cheese_1", "plate_1", "wine_bottle_1")
FIXTURES = ("flat_stove_1", "wine_rack_1", "wooden_cabinet_1")


def expect(mapping, expected):
    for key, value in expected.items():
        actual = mapping.get(key)
        if type(actual) is not type(value) or actual != value:
            raise ValueError(f"Generated source provenance mismatch for {key}")


def canonical(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def public_denylist(ordered):
    """Accept only the 50 numerical_state_fingerprint digests of the pinned public bank."""
    if len(ordered) != 50 or not all(isinstance(h, str) and re.fullmatch("[a-f0-9]{64}", h) for h in ordered):
        raise ValueError("Expected the 50 public Goal 5 SHA-256 digests in bank order")
    if hashlib.sha256(b"".join(bytes.fromhex(h) for h in ordered)).hexdigest() != PUBLIC_LIST_SHA:
        raise ValueError("Different public Goal 5 row list")
    return frozenset(ordered)


def validate_origin(origin):
    seed = origin["layout_seed"]
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Invalid native layout seed")
    expect(origin, dict(**TASK, split="train", source_kind=SOURCE_KIND, source_index=GROUP_BASE + seed,
                        native_sampler_seed=seed, native_api_origin=API_ORIGIN, bank_row_index=0,
                        released_rows_checked=50, released_bank_sha256=BANK_SHA, collision_free=True))
    if not re.fullmatch("[0-9a-f]{64}", origin["initial_state_sha256"]):
        raise ValueError("Invalid generated row fingerprint")
    positions = origin["movable_xyz_m"]
    if set(positions) != set(MOVABLES) or any(len(xyz) != 3 for xyz in positions.values()):
        raise ValueError("Incomplete movable world positions")
    expect(origin, dict(scene_hash=canonical(origin["sampled_scene"]),
                        movable_xy_hash=canonical({k: v[:2] for k, v in positions.items()})))
    expect(origin["collision_check"], dict(tolerance_m=0.002, deep_contacts=0, non_support_contacts=0,
                                           entities=list(MOVABLES + FIXTURES)))
    return origin


def source_record(row, native_reset):
    origin = validate_origin(row["layout_origin"])
    expect(origin, native_reset)
    expect(row, dict(split="train", source_kind=SOURCE_KIND, pose=origin["source_index"],
                     seed=origin["layout_seed"]))
    if row["body"] not in ("Panda", "Sawyer") or "released_row_fingerprint" in row:
        raise ValueError("Generated records require a source arm and generated row identity")
    return origin


def validate_generated_attempt(index, row, *, native_reset):
    """Validate one accepted camera receipt and return fields to merge into a motor-bank record."""
    origin = source_record(row, native_reset)
    expect(index, dict(**TASK, split="train", source_kind=SOURCE_KIND, source_constructor="native_generated_layout"))
    if row["body"] not in index["bodies"] or set(index["bodies"]) - {"Panda", "Sawyer"}:
        raise ValueError("Generated bank declares inconsistent source arms")
    if row["pose"] not in index["poses"] or any(type(p) is not int or p < GROUP_BASE for p in index["poses"]):
        raise ValueError("Generated index contains an invalid training group")
    layouts = [entry for entry in index["layouts"] if entry["layout_seed"] == row["seed"]]
    if len(layouts) != 1:
        raise ValueError("Expected one native layout receipt per generated seed")
    expect(layouts[0], {**origin, "row_fingerprint": origin["initial_state_sha256"]})
    if origin["initial_state_sha256"] in public_denylist(index["public_row_fingerprints"]):
        raise ValueError("Generated row matches a public row")
    expect(row, dict(accepted=True, done=True, native_success=True, camera=True, same_initial_state=True,
                     layout_seed=origin["layout_seed"]))
    if not (0 < row["steps"] == row["search_steps"] <= 300 and row["terminal_reward"] == row["native_return"] == 1):
        raise ValueError("Incomplete positive command replay")
    reset = row["reset"]
    expect(reset["model_identity"], {"arm": row["body"]})
    expect(reset, {"max_steps": 300})
    vendor, renderer = reset["renderer"]
    if "NVIDIA" not in vendor.upper() or not renderer:
        raise ValueError("Missing allocated camera renderer")
    physics = reset["environment_reproducibility"]
    expect(physics, {**native_reset, "suite": TASK["suite"], "task_id": 5, "init_state_id": row["pose"],
                     "seed": row["seed"], "bank_row_index": 0, "source_kind": SOURCE_KIND,
                     "native_sampler_seed": row["seed"], "native_api_origin": API_ORIGIN,
                     "generated_row_fingerprint": origin["initial_state_sha256"], "scene_fingerprint": origin["scene_hash"],
                     "initial_state_fingerprint": row["search_initial_state_fingerprint"]})
    if "released_row_fingerprint" in physics or not re.fullmatch("[0-9a-f]{64}", physics["initial_state_fingerprint"]):
        raise ValueError("Invalid generated reset fingerprint")
    expect(physics["perturbation_plan"], {"case_id": "BASE", "seed": row["seed"]})
    expect(physics["settled_collision_check"], {"deep_contacts": 0, "non_support_contacts": 0, "tolerance_m": 0.002})
    return dict(**TASK, body=row["body"], pose=row["pose"], seed=row["seed"], split="train", source_kind=SOURCE_KIND,
                layout_origin=origin, generated_row_fingerprint=origin["initial_state_sha256"],
                initial_state_fingerprint=physics["initial_state_fingerprint"], scene_fingerprint=origin["scene_hash"])


def validate_generated_tape(tape, row):
    """Bind an open pickle-free save_tape NPZ to its generated receipt."""
    expect({k: tape[k].item() for k in ("body", "pose", "seed", "layout_seed")},
           {k: row[k] for k in ("body", "pose", "seed", "layout_seed")})
    if json.loads(tape["layout_origin"].item()) != row["layout_origin"]:
        raise ValueError("Generated tape and bank origins differ")


def validate_layout_file(root, layout):
    """Verify the small raw Panda state file once per layout at bank ingestion."""
    import numpy as np

    origin = validate_origin(layout)
    relative = f"layouts/{origin['layout_seed']}.npy"
    expect(layout, {"state_file": relative})
    path = Path(root) / relative
    if hashlib.sha256(path.read_bytes()).hexdigest() != layout["state_sha256"]:
        raise ValueError("Generated Panda state file digest mismatch")
    state = np.load(path, allow_pickle=False)
    if state.shape != (79,) or state.dtype != np.float64 or not np.isfinite(state).all():
        raise ValueError("Expected a finite flat Goal 5 Panda state")
    payload = b'{"dtype":"<f8","shape":[79]}\0' + state.astype("<f8", copy=False).tobytes()
    expect(layout, {"initial_state_sha256": hashlib.sha256(payload).hexdigest(),
                    "row_fingerprint": origin["initial_state_sha256"]})


def validate_generated_groups(records, public_row_fingerprints, *, native_reset):
    """Check generated training records together across banks and bodies at source fit/readback."""
    released = public_denylist(public_row_fingerprints)
    groups, owners, starts = {}, {}, {}
    for row in records:
        origin = source_record(row, native_reset)
        pose, raw = row["pose"], origin["initial_state_sha256"]
        expect(row, {**TASK, "generated_row_fingerprint": raw, "scene_fingerprint": origin["scene_hash"]})
        if raw in released or groups.setdefault(pose, origin) != origin:
            raise ValueError(f"Published or inconsistent generated group {pose}")
        start = row["initial_state_fingerprint"]
        if (not re.fullmatch("[0-9a-f]{64}", start)
                or starts.setdefault((row["body"], pose), start) != start):
            raise ValueError(f"Inconsistent settled state for {row['body']} group {pose}")
        for key in ("initial_state_sha256", "scene_hash", "movable_xy_hash"):
            if owners.setdefault((key, origin[key]), pose) != pose:
                raise ValueError(f"Duplicate {key} across generated groups")
