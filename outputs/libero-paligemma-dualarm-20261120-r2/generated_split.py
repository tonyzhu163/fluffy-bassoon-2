"""Validate native BDDL samples for R2 Goal 5 source training.

For accepted rows of a native_generated_layout index, call
`validate_generated_attempt(index, row, native_reset=NATIVE_RESET)`. Check the
NPZ pose against the collector row and its embedded JSON origin against
row["origin"]. Merge the returned fields into the motor-bank episode record;
its pose is the training group ID. Retain the usual tape, reward, donor and
renderer checks. At fit or transfer readback, call
`validate_generated_groups(generated_records, released_fingerprints)` across
all supplied banks. Supply generated_layouts.released_fingerprints() from the
pinned 50-row bank, or its values saved at ingestion.

Dispatch ordinary released rows through the existing R2 loader (0..27 train,
28..31 validation). A missing origin on a generated pose is an error. The
development and sealed confirmation panels use their existing released resets.
"""

from collections.abc import Mapping
import re


SOURCE_KIND = "native_bddl_sample"
SOURCE_INDEX_BASE = 1_000_000
NATIVE_API_ORIGIN = "ControlEnv.seed/reset/get_sim_state"
BODIES = ("Panda", "Sawyer")
TASK = {"suite": "libero_goal", "task_id": 5, "case": "BASE"}
PIN_FIELDS = ("source_pin", "manifest_sha256", "bddl_sha256")
BANK_SHA = "794beed679b18bbd7d9fb851c81877a974915ad63addff2d5ac5af0607fafac9"


def _expect(mapping, expected, where):
    if not isinstance(mapping, Mapping):
        raise ValueError(f"{where}: expected an object")
    for key, value in expected.items():
        actual = mapping.get(key)
        if type(actual) is not type(value) or actual != value:
            raise ValueError(f"{where}: expected {key}={value!r}, got {actual!r}")


def _seed(mapping, key):
    value = mapping.get(key)
    if type(value) is not int or not 0 <= value < 2**32:
        raise ValueError(f"Expected an unsigned 32-bit {key}, got {value!r}")
    return value


def _sha256(value, name):
    if not isinstance(value, str) or not re.fullmatch("[0-9a-f]{64}", value):
        raise ValueError(f"Invalid SHA-256 for {name}")
    return value


def validate_origin(origin):
    """Return the JSON provenance fields shared by all attempts on one layout."""
    if not isinstance(origin, Mapping):
        raise ValueError("Missing generated layout origin")
    seed = _seed(origin, "layout_seed")
    identity = dict(**TASK, split="train", source_kind=SOURCE_KIND, source_index=SOURCE_INDEX_BASE + seed,
                    layout_seed=seed, native_sampler_seed=seed, bank_row_index=0,
                    native_api_origin=NATIVE_API_ORIGIN, collision_free=True,
                    released_rows_checked=50, released_bank_sha256=BANK_SHA)
    _expect(origin, identity, "generated layout origin")
    for key in ("initial_state_sha256", "scene_hash"):
        identity[key] = _sha256(origin.get(key), key)
    if "movable_xy_hash" in origin:
        _sha256(origin["movable_xy_hash"], "movable XY placement")
    return dict(origin)


def _check_copies(mapping, origin, where):
    # Both the sampler entry and the reset/collector can carry provenance.
    _expect(mapping, {k: v for k, v in origin.items() if k in mapping}, where)
    if "origin" in mapping:
        _expect(mapping["origin"], origin, where)


def _source_row(row, origin):
    body = row.get("body")
    if body not in BODIES:
        raise ValueError(f"Unexpected generated training arm {body!r}")
    _expect(row, {"pose": origin["source_index"], "split": "train"}, "generated training row")
    if "group_id" in row:
        _expect(row, {"group_id": origin["source_index"]}, "generated training group")
    _expect(row, {"seed": origin["layout_seed"]}, "generated rollout")
    _check_copies(row, origin, "generated training row")
    return body


def validate_generated_attempt(index, row, *, native_reset):
    """Bind one collector row and the sampler entry to the witnessed native reset.

    native_reset is the existing pretrain_reader.NATIVE_RESET. The collector
    uses its layout seed for the native rollout and the perturbation plan.
    """
    origin = validate_origin(row.get("origin"))
    seed, group = origin["layout_seed"], origin["source_index"]
    _expect(index, {**TASK, "source_constructor": "native_generated_layout"}, "generated bank")
    if "layout_bank_row" in index:
        _expect(index, {"layout_bank_row": origin["bank_row_index"]}, "generated bank slot")
    _expect(row, {"layout_seed": seed, "seed": seed}, "generated attempt")
    tape_pose = row.get("pose")
    if type(tape_pose) is not int or tape_pose not in (origin["bank_row_index"], group):
        raise ValueError("Generated tape pose disagrees with its bank slot or training group")
    if "split" in row:
        _expect(row, {"split": "train"}, "generated attempt")
    body = _source_row({**row, "pose": group, "split": "train"}, origin)
    layouts = [entry for entry in index["layouts"] if entry["layout_seed"] == seed]
    if len(layouts) != 1:
        raise ValueError(f"Expected one sampler entry for generated group {group}")
    layout = layouts[0]
    _expect(layout, {"layout_seed": seed, "origin": origin, "row_fingerprint": origin["initial_state_sha256"],
                     "scene_fingerprint": origin["scene_hash"]}, "sampler entry")
    _check_copies(layout, origin, "sampler entry")
    declared = index.get("bodies")
    if (not isinstance(declared, (tuple, list)) or body not in declared
            or any(b not in BODIES for b in declared)):
        raise ValueError("Generated bank must declare only its source arms")
    _expect(row, {k: v for k, v in TASK.items() if k in row}, "generated attempt")
    reset = row["reset"]
    _expect(reset["model_identity"], {"arm": body}, "native model")
    physics = reset["environment_reproducibility"]
    pins = {k: native_reset[k] for k in PIN_FIELDS}
    _expect(origin, pins, "native sampler")
    _expect(physics, {**pins, "suite": TASK["suite"], "task_id": TASK["task_id"],
                      "seed": seed, "released_row_fingerprint": origin["initial_state_sha256"],
                      "scene_fingerprint": origin["scene_hash"]},
            "generated reset")
    _expect(physics["perturbation_plan"], {"case_id": TASK["case"], "seed": seed}, "BASE plan")
    _check_copies(reset, origin, "generated reset")
    _check_copies(physics, origin, "generated reset physics")
    # ArmEnv reads the sole sampled row at zero. A collector may relabel the
    # public init_state_id to the logical group if it preserves the native slot.
    if physics.get("init_state_id") == origin["source_index"]:
        _expect(physics, {"init_state_id": origin["source_index"]}, "generated group")
        if "bank_row_index" not in physics and "origin" not in physics:
            raise ValueError("Relabeled native init_state_id lacks its internal bank_row_index")
    else:
        _expect(physics, {"init_state_id": origin["bank_row_index"]}, "internal generated bank")
    settled = _sha256(physics.get("initial_state_fingerprint"), "settled arm state")
    scene = _sha256(physics.get("scene_fingerprint"), "pre-settle task scene")
    return dict(**TASK, body=body, pose=group, seed=seed, split="train", origin=origin,
                released_row_fingerprint=origin["initial_state_sha256"],
                initial_state_fingerprint=settled, scene_fingerprint=scene)


def validate_generated_groups(records, released_fingerprints):
    """Check saved generated records across banks, arms and collector repeats."""
    if not isinstance(released_fingerprints, (set, frozenset, list, tuple)) or len(released_fingerprints) != 50:
        raise ValueError("Supply the raw fingerprints of all 50 released Goal 5 rows")
    released = {_sha256(fp, f"released row {i}") for i, fp in enumerate(released_fingerprints)}
    groups, raw_owners, scene_owners, xy_owners = {}, {}, {}, {}
    for row in records:
        origin = validate_origin(row.get("origin"))
        _source_row(row, origin)
        _expect(row, {**TASK, "released_row_fingerprint": origin["initial_state_sha256"],
                      "scene_fingerprint": origin["scene_hash"]},
                "saved generated record")
        _sha256(row.get("initial_state_fingerprint"), "settled arm state")
        pose, raw = origin["source_index"], origin["initial_state_sha256"]
        if raw in released:
            raise ValueError(f"Generated group {pose} reproduces a released Goal 5 row")
        if groups.setdefault(pose, origin) != origin:
            raise ValueError(f"Inconsistent generated layout provenance for group {pose}")
        identities = [(raw_owners, raw), (scene_owners, origin["scene_hash"])]
        if "movable_xy_hash" in origin:
            identities.append((xy_owners, origin["movable_xy_hash"]))
        for owners, digest in identities:
            owner = owners.setdefault(digest, pose)
            if owner != pose:
                raise ValueError(f"Generated groups {owner} and {pose} identify the same layout")
