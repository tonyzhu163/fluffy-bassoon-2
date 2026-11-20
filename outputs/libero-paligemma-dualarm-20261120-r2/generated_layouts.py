"""Sample native, source-training-only LIBERO Goal 5 scenes. Run in the R2 simulator interpreter."""

import argparse
from contextlib import closing
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import sys

from expert import write_json  # establishes the pinned sibling sim.py import path

SUITE, TASK, CASE = "libero_goal", 5, "BASE"
GROUP_BASE = 1_000_000
BANK_SHA = "794beed679b18bbd7d9fb851c81877a974915ad63addff2d5ac5af0607fafac9"
BDDL_SHA = "c2db20071f58fd43fe029e5bf9176f3c7371ee62cadc311f2b8682c089174c0d"
SOURCE_KIND = "native_bddl_sample"
API_ORIGIN = "ControlEnv.seed/reset/get_sim_state"
MOVABLES = ("akita_black_bowl_1", "cream_cheese_1", "plate_1", "wine_bottle_1")
FIXTURES = ("flat_stove_1", "wine_rack_1", "wooden_cabinet_1")


class LayoutRejected(ValueError):
    """Expected sampler/admission failure for one training seed."""


@lru_cache(maxsize=1)
def native_module():
    from sim import _arm_class

    return sys.modules[_arm_class("Panda").__module__]


@lru_cache(maxsize=1)
def released_fingerprints():
    """Only use the 50 public flat rows as an exact hash denylist."""
    from libero.libero import benchmark, get_libero_path
    from libero_remote.reproducibility import numerical_state_fingerprint

    suite = benchmark.get_benchmark_dict()[SUITE]()
    task = suite.get_task(TASK)
    path = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
    if hashlib.sha256(path.read_bytes()).hexdigest() != BANK_SHA:
        raise ValueError("Unexpected public Goal 5 initial-state bank")
    rows = suite.get_task_init_states(TASK)
    if len(rows) != 50:
        raise ValueError("Expected all 50 published Goal 5 rows")
    return frozenset(numerical_state_fingerprint(row) for row in rows)


def make_sampler():
    """The returned camera-free native Panda wrapper is owned and closed by its caller."""
    from importlib.metadata import version

    native = native_module()
    if (version("robosuite"), version("mujoco")) != ("1.4.0", "3.2.3"):
        raise RuntimeError("Use the qualified R2 simulator interpreter")
    release = Path(os.environ.get("CROSSARM_RELEASE", Path(native.task_manifest.__file__).resolve().parents[2]))
    manifest = native._manifest(release / "assets/task_manifest.json")
    entry = manifest.require(SUITE, TASK, CASE)
    if (entry["bddl_sha256"], entry["max_steps"], entry["reset_stabilization_steps"]) != (BDDL_SHA, 300, 10):
        raise ValueError("Unexpected native Goal 5 task contract")
    return native.ControlEnv(
        bddl_file_name=str(manifest.resolve_bddl(SUITE, TASK, CASE)), robots=["Panda"],
        controller="OSC_POSE", gripper_types="PandaGripper", initialization_noise=None, control_freq=20,
        camera_names=list(native.CAMERAS), camera_heights=256, camera_widths=256,
        use_camera_obs=False, has_offscreen_renderer=False,
    )


def collision_admission(wrapper):
    """Check signed MuJoCo placement contacts at a reset boundary without stepping the simulator."""
    from libero.libero.envs.nonphysical_perturbations import _entity_deep_contacts

    env = wrapper.env
    if set(env.objects_dict) != set(MOVABLES) or set(env.fixtures_dict) != set(FIXTURES):
        raise ValueError("Unexpected Goal 5 object/fixture identities")
    # The release's scene-validation predicate allows shallow support contacts.
    # Include every named sampled fixture (and its articulated descendants).
    for name in MOVABLES + FIXTURES:
        deep = _entity_deep_contacts(env, name, tolerance=0.002)
        if deep:
            raise LayoutRejected(f"Native placement penetration at {name}: {deep}")
    return dict(method="LIBERO _entity_deep_contacts", tolerance_m=0.002,
                entities=list(MOVABLES + FIXTURES), deep_contacts=0, ncon=int(env.sim.data.ncon))


def sample_layout(sampler_env, layout_seed, released_fingerprints):
    """Return a copy of the native pre-settle Panda row plus its complete placement receipt."""
    import numpy as np
    from libero_remote.reproducibility import canonical_fingerprint, numerical_state_fingerprint
    from robosuite.utils.errors import RandomizationError

    if type(layout_seed) is not int or not 0 <= layout_seed < 2**32:
        raise ValueError("The native sampler takes a uint32 seed")
    if sampler_env.env.deterministic_reset:
        raise ValueError("The native BDDL placement initializer is disabled")
    native = native_module()
    try:
        native._reset_wrapper(sampler_env, layout_seed)  # seed, set_perturbation_seed, reset
    except RandomizationError as exc:
        raise LayoutRejected(f"Native placement sampler exhausted its retries: {exc}") from exc
    plan = sampler_env.get_perturbation_plan()
    if (plan["case_id"], plan["seed"]) != (CASE, layout_seed):
        raise RuntimeError("Native case/seed readback disagrees with requested reset")
    state = np.asarray(sampler_env.get_sim_state(), dtype=np.float64).copy()
    model = sampler_env.sim.model
    if model.na or state.shape != (1 + model.nq + model.nv,) or not np.isfinite(state).all():
        raise RuntimeError("Invalid native Panda flat state")
    fingerprint = numerical_state_fingerprint(state)
    if fingerprint in released_fingerprints:
        raise LayoutRejected("Exact duplicate of a public Goal 5 initial-state row")
    admission = collision_admission(sampler_env)
    if sampler_env.check_success():
        raise LayoutRejected("Native Goal 5 already successful at sampler reset")
    scene = native.scene_state(sampler_env)
    positions = {name: sampler_env.sim.data.body_xpos[sampler_env.env.obj_body_id[name]].tolist()
                 for name in MOVABLES}
    return state, dict(
        source_kind=SOURCE_KIND, source_index=GROUP_BASE + layout_seed, split="train",
        layout_seed=layout_seed, native_sampler_seed=layout_seed, bank_row_index=0,
        native_api_origin=API_ORIGIN, suite=SUITE, task_id=TASK, case=CASE,
        source_pin=native.SOURCE_PIN, manifest_sha256=native.MANIFEST_SHA, bddl_sha256=BDDL_SHA,
        initial_state_sha256=fingerprint, scene_hash=canonical_fingerprint(scene), sampled_scene=scene,
        movable_xyz_m=positions, movable_xy_hash=canonical_fingerprint({k: v[:2] for k, v in positions.items()}),
        collision_free=True, collision_check=admission, released_rows_checked=50, released_bank_sha256=BANK_SHA,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-start", type=int, default=283100)
    parser.add_argument("--count", type=int, default=4, help="Distinct training layouts required")
    parser.add_argument("--max-seeds", type=int, help="Maximum consecutive native seeds to try (default: --count)")
    parser.add_argument("--out", type=Path, required=True, help="Fresh directory for native rows and receipts")
    args = parser.parse_args()
    budget = args.count if args.max_seeds is None else args.max_seeds
    if not (0 < args.count <= budget and 0 <= args.seed_start <= 2**32 - budget):
        parser.error("Require a positive count within the finite uint32 seed range and budget")
    import numpy as np

    public = released_fingerprints()
    args.out.mkdir(parents=True, exist_ok=False)
    accepted, rejected = [], []
    index = dict(suite=SUITE, task_id=TASK, case=CASE, source_kind=SOURCE_KIND, split="train",
                 command=sys.argv, revision=os.environ.get("MOTOR_CODE_REVISION"), camera=False,
                 requested=args.count, max_seeds=budget, released_rows_checked=50, released_bank_sha256=BANK_SHA,
                 accepted=accepted, rejected=rejected)
    write_json(args.out / "index.json", index)
    seen = set()
    with closing(make_sampler()) as env:
        for seed in range(args.seed_start, args.seed_start + budget):
            try:
                state, origin = sample_layout(env, seed, public)
                if origin["movable_xy_hash"] in seen:
                    raise LayoutRejected("Same movable world XY placement as a prior admitted training layout")
            except LayoutRejected as exc:
                rejected.append(dict(layout_seed=seed, reason=str(exc)))
            else:
                seen.add(origin["movable_xy_hash"])
                filename = f"panda-{seed}.npy"
                np.save(args.out / filename, state, allow_pickle=False)
                accepted.append(dict(state_file=filename, **origin))
                print(json.dumps(dict(seed=seed, group=origin["source_index"], movable_xyz_m=origin["movable_xyz_m"])),
                      flush=True)
            write_json(args.out / "index.json", index)
            if len(accepted) == args.count:
                break
    print(f"{len(accepted)}/{args.count} distinct, collision-admitted native placements", flush=True)
    return int(len(accepted) != args.count)


if __name__ == "__main__":
    raise SystemExit(main())
