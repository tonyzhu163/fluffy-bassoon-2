"""Sample native, source-training-only LIBERO Goal 5 scenes. Run in the R2 simulator interpreter."""

import argparse
from contextlib import closing
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from expert import write_json  # establishes the pinned sibling sim.py import path
from generated_split import API_ORIGIN, BANK_SHA, FIXTURES, GROUP_BASE, MOVABLES, SOURCE_KIND, public_denylist

SUITE, TASK, CASE = "libero_goal", 5, "BASE"
BDDL_SHA = "c2db20071f58fd43fe029e5bf9176f3c7371ee62cadc311f2b8682c089174c0d"


class LayoutRejected(ValueError):
    """Expected sampler/admission failure for one training seed."""


def code_receipt():
    here = Path(__file__).resolve().parent
    sibling = here.parent / "libero-paligemma-crossarm-20261120-r1"
    arm = here.parent / "libero-crossarm-20261120-r1"
    files = [*(here / n for n in ("generated_layouts.py", "generated_collect.py", "generated_split.py",
                                  "expert.py", "paddle.py", "sweep_bank.py", "expert_sweep.py", "recover.py")),
             sibling / "sim.py", sibling / "demos.py", arm / "arm_env.py", arm / "sim_reset.py"]
    revision = os.environ.get("MOTOR_CODE_REVISION")
    if not revision:
        try:
            revision = subprocess.check_output(["git", "-C", str(here), "rev-parse", "HEAD"],
                                               stderr=subprocess.DEVNULL, text=True).strip()
        except (FileNotFoundError, subprocess.CalledProcessError):
            revision = None
    return dict(revision=revision, code_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files})


@lru_cache(maxsize=1)
def native_module():
    from sim import _arm_class

    return sys.modules[_arm_class("Panda").__module__]


@lru_cache(maxsize=1)
def released_row_fingerprints():
    """The 50 public flat-state hashes in native bank order, for membership checks only."""
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
    hashes = tuple(numerical_state_fingerprint(row) for row in rows)
    public_denylist(hashes)
    return hashes


@lru_cache(maxsize=1)
def released_fingerprints():
    return frozenset(released_row_fingerprints())


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
    from types import SimpleNamespace

    import mujoco
    from libero.libero.envs.nonphysical_perturbations import _entity_body_ids, _entity_deep_contacts

    env = wrapper.env
    if set(env.objects_dict) != set(MOVABLES) or set(env.fixtures_dict) != set(FIXTURES):
        raise ValueError("Unexpected Goal 5 object/fixture identities")
    # The release's scene-validation predicate allows shallow support contacts.
    # Include every named sampled fixture (and its articulated descendants).
    model, live = env.sim.model, env.sim.data
    # A policy boundary can follow mj_step. Recompute contacts at the actual
    # recorded qpos in scratch MjData without perturbing solver warm starts.
    data = mujoco.MjData(model._model)
    data.qpos[:] = live.qpos
    data.mocap_pos[:] = live.mocap_pos
    data.mocap_quat[:] = live.mocap_quat
    mujoco.mj_fwdPosition(model._model, data)
    view = SimpleNamespace(obj_body_id=env.obj_body_id, sim=SimpleNamespace(model=model, data=data))
    owners = {}
    for name in MOVABLES + FIXTURES:
        deep = _entity_deep_contacts(view, name, tolerance=0.002)
        if deep:
            raise LayoutRejected(f"Native placement penetration at {name}: {deep}")
        owners.update((idx, name) for idx in _entity_body_ids(view, name))
    table = model.geom_name2id("table_collision")
    for i in range(data.ncon):
        contact = data.contact[i]
        a, b = int(contact.geom1), int(contact.geom2)
        first, second = owners.get(int(model.geom_bodyid[a])), owners.get(int(model.geom_bodyid[b]))
        if first != second and table not in (a, b) and contact.dist <= 0:
            names = model.geom_id2name(a), model.geom_id2name(b)
            raise LayoutRejected(f"Non-support contact {names} at signed distance {contact.dist} m")
    # MuJoCo's constraint generation filters fixed-fixed pairs. Query the
    # collision geometries of distinct welded fixtures explicitly at this pose.
    fixed = [g for g in range(model.ngeom) if owners.get(int(model.geom_bodyid[g])) in FIXTURES
             and model.body_weldid[model.geom_bodyid[g]] == 0 and (model.geom_contype[g] or model.geom_conaffinity[g])]
    for i, a in enumerate(fixed):
        for b in fixed[i + 1:]:
            if owners[int(model.geom_bodyid[a])] != owners[int(model.geom_bodyid[b])]:
                distance = mujoco.mj_geomDistance(model._model, data, a, b, 0.002, None)
                if distance <= 0:
                    raise LayoutRejected(f"Welded fixture geoms {a}, {b} at signed distance {distance} m")
    return dict(method="LIBERO _entity_deep_contacts", tolerance_m=0.002,
                entities=list(MOVABLES + FIXTURES), deep_contacts=0, non_support_contacts=0,
                welded_fixture_overlaps=0, ncon=int(data.ncon))


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
    native._identity(sampler_env, "Panda")
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


def reset_generated(session, body, camera, state, origin):
    """Reset an R2 session from a private one-row bank. Return its observation and honest receipt."""
    if body not in ("Panda", "Sawyer") or origin["source_kind"] != SOURCE_KIND:
        raise ValueError("Only generated Panda/Sawyer training starts are supported")
    arm = session.native.arm if camera else session.arm
    if arm.arm != body or arm.task_id != TASK or arm.suite != SUITE:
        raise ValueError("Session does not match requested generated source body or task")
    seed, group = origin["native_sampler_seed"], origin["source_index"]
    if origin["bank_row_index"] != 0 or seed != origin["layout_seed"] or group != GROUP_BASE + seed:
        raise ValueError("Inconsistent generated seed, group or private bank slot")
    # ArmEnv rebuilds this arm (and its Panda donor for Sawyer) using the sampler
    # seed, so model-level fixture poses are regenerated too. scene_fingerprint
    # below checks every fixture as well as all non-arm qpos and qvel.
    original_bank, arm._states = arm._states, (state,)
    try:
        observation = session.reset(0, seed)
    finally:
        arm._states = original_bank
    reset = session.reset_info
    physics = reset["environment_reproducibility"]
    if (reset["model_identity"]["arm"] != body or reset["max_steps"] != 300 or
            (physics["init_state_id"], physics["seed"]) != (0, seed) or
            (physics["perturbation_plan"]["case_id"], physics["perturbation_plan"]["seed"]) != (CASE, seed) or
            any(physics[k] != origin[k] for k in ("suite", "task_id", "bddl_sha256", "manifest_sha256", "source_pin")) or
            physics["released_row_fingerprint"] != origin["initial_state_sha256"] or
            physics["scene_fingerprint"] != origin["scene_hash"]):
        raise ValueError("R2 reset did not reproduce the sampled Panda row and full fixture/joint scene")
    # This new receipt is private to the generated session. The legacy keys were
    # constructed by ArmEnv under the assumption of a published bank.
    physics["generated_row_fingerprint"] = physics.pop("released_row_fingerprint")
    physics.update(init_state_id=group, bank_row_index=0, source_kind=SOURCE_KIND,
                   native_sampler_seed=seed, layout_seed=seed, native_api_origin=API_ORIGIN,
                   source_index=group, scene_hash=origin["scene_hash"])
    if body == "Panda":
        physics["alignment"] = dict(method="native_sampled_panda_state")
    physics["settled_collision_check"] = collision_admission(arm._env)
    return observation


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
                 command=sys.argv, **code_receipt(), camera=False,
                 requested=args.count, max_seeds=budget, released_rows_checked=50, released_bank_sha256=BANK_SHA,
                 public_row_fingerprints=released_row_fingerprints(),
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
