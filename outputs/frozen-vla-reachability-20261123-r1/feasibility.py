"""Replay recorded Panda contact configurations as object-relative native BASE waypoint queries."""
import argparse
import copy
import json
import os
import socket
from contextlib import ExitStack
from pathlib import Path

import h5py
import mujoco
import numpy as np

from arm_env import ArmEnv
from geometry import Geometry, body_pose, put_free
from sim_reset import scene_state
from ur5e_sim_worker import _probe_egl

DRAWER = "wooden_cabinet_1_cabinet_top"
BOWL = "akita_black_bowl_1_main"
POTS = ("moka_pot_1_main", "moka_pot_2_main")
TASKS = {
    "goal3": ("libero_goal", 3, (DRAWER, BOWL), DRAWER),
    "long8": ("libero_10", 8, POTS, "flat_stove_1_burner_plate"),
    "goal6": ("libero_goal", 6, ("cream_cheese_1_main",), BOWL),
}
ARMS = ("Panda", "Kinova3")
TURN = np.diag([-1.0, -1.0, 1.0, 1.0])


def state_joints(g):
    free, scalar = {}, []
    for i in range(g.m.njnt):
        j = g.m.joint(i)
        if j.name.startswith(("robot0_", "gripper0_")):
            continue
        if g.m.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
            free[g.m.body(g.m.jnt_bodyid[i]).name] = j.name
        else:
            assert g.m.jnt_type[i] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
            scalar.append(j.name)
    return free, scalar


def donor_model(native, group):
    model = copy.copy(native.sim.model._model)
    if [model.joint(i).name for i in range(model.njnt)] != json.loads(group.attrs["joint_order"]):
        raise ValueError("Recorded state ordering differs from native model")
    world = json.loads(group.attrs["world_bodies"])
    for name in scene_state(native)["fixtures"]:
        i = model.body(name).id
        if model.body_parentid[i] != 0:
            raise ValueError(name)
        model.body_pos[i] = np.fromstring(world[name]["pos"], sep=" ")
        model.body_quat[i] = np.fromstring(world[name]["quat"], sep=" ")
    for name in ("table", "robot0_base"):
        # Some arenas name the table differently. Check the arm base in every arena.
        if name in world:
            assert np.allclose(model.body(name).pos, np.fromstring(world[name]["pos"], sep=" "))
            assert np.allclose(model.body(name).quat, np.fromstring(world[name]["quat"], sep=" "))
    return Geometry(native, model)


def replay(g, state):
    if state.shape != (1 + g.m.nq + g.m.nv,):
        raise ValueError("Invalid full demonstration state")
    g.d.time = state[0]
    g.d.qpos[:] = state[1:1+g.m.nq]
    g.d.qvel[:] = state[1+g.m.nq:]
    g.forward()


def events(g, tape, targets, receiver):
    hits = {t: [] for t in targets}
    tcp = []
    for i, state in enumerate(tape):
        replay(g, state)
        tcp.append(g.tcp()[:3, 3])
        for c in g.contacts(None):
            if "finger" in c["body1"] and c["body2"] in hits:
                hits[c["body2"]].append(i)
    selected = []
    for name, times in hits.items():
        if not times:
            continue
        for phase, t in (("approach", max(0, min(times)-5)), ("first_touch", min(times))):
            selected.append((name, phase, t))
        if name != receiver:
            selected.append((name, "last_touch", max(times)))
    return hits, selected, np.asarray(tcp)


def stage_geometry(src, template, free, scalar, source_start, target_start, obj, phase, hits, receiver, t):
    """Reconstruct a static scene in the query layout, preserving local grasp/receiver geometry."""
    template.d.qpos[:] = template.initial_qpos
    for name in scalar:
        src_idx, dst_idx = src.m.joint(name).qposadr[0], template.m.joint(name).qposadr[0]
        template.d.qpos[dst_idx] = src.d.qpos[src_idx]
    template.forward()
    receiver_map = body_pose(template.m, template.d, receiver) @ np.linalg.inv(body_pose(src.m, src.d, receiver))
    if receiver in free:
        receiver_map = target_start[receiver] @ np.linalg.inv(source_start[receiver])
    if phase == "last_touch":
        tool_map = receiver_map
    elif obj in free:
        tool_map = target_start[obj] @ np.linalg.inv(source_start[obj])
    else:
        tool_map = body_pose(template.m, template.d, obj) @ np.linalg.inv(body_pose(src.m, src.d, obj))
    for body, joint in free.items():
        placed = hits.get(body) and t > max(hits[body])
        if (phase == "last_touch" and body == obj) or placed:
            mapping = receiver_map
        else:
            mapping = target_start[body] @ np.linalg.inv(source_start[body])
        put_free(template.m, template.d, joint, mapping @ body_pose(src.m, src.d, body))
    template.d.qpos[template.fingers] = src.d.qpos[src.fingers]
    template.forward()
    goal = tool_map @ src.tcp()
    # The held object / active contact body must have exactly the same hand-relative pose.
    actual = np.linalg.inv(body_pose(template.m, template.d, obj)) @ goal
    expected = np.linalg.inv(body_pose(src.m, src.d, obj)) @ src.tcp()
    if not np.allclose(actual, expected, atol=1e-9, rtol=0):
        raise ValueError(f"Contact frame not preserved at {obj}, {phase}")
    if phase == "last_touch":
        src_relative = np.linalg.inv(body_pose(src.m, src.d, receiver)) @ src.tcp()
        dst_relative = np.linalg.inv(body_pose(template.m, template.d, receiver)) @ goal
        if not np.allclose(src_relative, dst_relative, atol=1e-9, rtol=0):
            raise ValueError("Placement changed the receiver-relative hand pose")
    return goal


def staged_state(g, free, scalar):
    values = {}
    for name in [*free.values(), *scalar]:
        j = g.m.joint(name)
        n = 7 if name in free.values() else 1
        i = j.qposadr[0]
        values[name] = g.d.qpos[i:i+n].tolist()
    return dict(joints=values, fingers=g.d.qpos[g.fingers].tolist())


def install(g, stage):
    g.d.qpos[:] = g.initial_qpos
    for name, vals in stage["joints"].items():
        i = g.m.joint(name).qposadr[0]
        g.d.qpos[i:i+len(vals)] = vals
    g.d.qpos[g.fingers] = stage["fingers"]
    g.forward()


def multistart(g, goal, obj):
    # Same deterministic perturbations (radians) about each arm's own reset configuration.
    offsets = np.random.default_rng(20261123).normal(size=(4, len(g.adr))) * 1.1
    seeds = [g.qreset, *(g.qreset + offsets)]
    return {mode: [g.solve(pose, seed, obj) for seed in seeds]
            for mode, pose in (("recorded", goal), ("half_turn", goal @ TURN))}


def get_reference(arm, suite, task, layout):
    rows = json.loads((Path(os.environ["NATIVE_PANEL"]) / f"cpu-{arm}.json").read_text())
    return next(r for r in rows if (r["case"]["suite"], r["case"]["task_id"],
                                    r["case"]["init_state_id"]) == (suite, task, layout))


def case(task_key, layout, inputs, out, renderer):
    suite, task, targets, receiver = TASKS[task_key]
    case_id = f"{task_key}-{layout}"
    result = dict(case=case_id, renderer=renderer, host=socket.gethostname(),
                  cpus=sorted(os.sched_getaffinity(0)), native_resets={}, arms={}, demonstrations=[])
    with ExitStack() as stack:
        contexts, geos = {}, {}
        for arm in ARMS:
            env = stack.enter_context(ArmEnv(arm, suite, task))
            ref = get_reference(arm, suite, task, layout)
            first = env.reset(layout, ref["case"]["seed"])
            record = first["environment_reproducibility"]
            if record != ref["reset"] or first["model_identity"] != ref["model"]:
                raise ValueError(f"Independent native reset mismatch: {case_id} {arm}")
            result["native_resets"][arm] = record
            contexts[arm] = env
            geos[arm] = g = Geometry(env._env)
            result["arms"][arm] = dict(model=first["model_identity"], qreset=g.qreset.tolist(),
                                       limited=g.limited.tolist(), ranges=g.m.jnt_range[g.joints].tolist())
        resets = list(result["native_resets"].values())
        for k in ("scene_fingerprint", "released_row_fingerprint", "bddl_sha256", "perturbation_plan_fingerprint"):
            assert resets[0][k] == resets[1][k]
        panda = geos["Panda"]
        free, scalar = state_joints(panda)
        assert state_joints(geos["Kinova3"]) == (free, scalar)
        target_start = {body: body_pose(panda.m, panda.d, body) for body in free}
        panel_bodies = sorted(set(free) | set(targets) | {receiver})
        with h5py.File(inputs / (task_key + ".h5")) as h:
            result["source"] = dict(h.attrs)
            for demo in ("demo_0", "demo_1"):
                group = h[demo]
                tape = group["states"][:]
                src = donor_model(contexts["Panda"]._env, group)
                replay(src, tape[0])
                source_start = {body: body_pose(src.m, src.d, body) for body in free}
                fixture_names = scene_state(contexts["Panda"]._env)["fixtures"]
                fixture_delta = {body: float(np.linalg.norm(src.m.body(body).pos - panda.m.body(body).pos))
                                 for body in fixture_names}
                hits, selected, tcp = events(src, tape, targets, receiver)
                recorded_xyz = group["obs/ee_pos"][:]
                # Stored proprioception and the integrated state can sample different instants.
                # Recompute the separate public joint sample in the same physical TCP frame.
                observed_fk = []
                for joints in group["obs/joint_states"][:]:
                    src.d.qpos[src.adr] = joints
                    mujoco.mj_kinematics(src.m, src.d)
                    observed_fk.append(src.tcp()[:3, 3])
                public_errors = np.linalg.norm(observed_fk-recorded_xyz, axis=1)
                donor = dict(demo=demo, public_joint_fk_to_public_eef_median_m=float(np.median(public_errors)),
                             public_joint_fk_to_public_eef_max_m=float(public_errors.max()), n=len(tape), file_terminal_reward=float(group.attrs["terminal_reward"]),
                             file_terminal_done=bool(group.attrs["terminal_done"]),
                             first_movable_pose_delta_m={k: float(np.linalg.norm(target_start[k][:3, 3] -
                                                                                    v[:3, 3])) for k, v in source_start.items()},
                             fixture_origin_delta_m=fixture_delta,
                             fk_to_stored_eef_median_m=float(np.median(np.linalg.norm(tcp-recorded_xyz, axis=1))),
                             finger_contact_frames={k: sorted(set(v)) for k, v in hits.items()}, waypoints=[])
                result["demonstrations"].append(donor)
                assert donor["file_terminal_reward"] == 1 and donor["file_terminal_done"]
                for obj, phase, t in selected:
                    replay(src, tape[t])
                    goal = stage_geometry(src, panda, free, scalar, source_start, target_start,
                                          obj, phase, hits, receiver, t)
                    stage = staged_state(panda, free, scalar)
                    expected_scene = np.stack([body_pose(panda.m, panda.d, b) for b in panel_bodies])
                    row = dict(body=obj, phase=phase, frame=t, donor_q=src.d.qpos[src.adr].tolist(),
                               donor_tcp=src.tcp().tolist(), donor_contacts=src.contacts(obj),
                               goal=goal.tolist(), stage=stage, search={})
                    donor["waypoints"].append(row)
                    for arm in ARMS:
                        g = geos[arm]
                        install(g, stage)
                        actual_scene = np.stack([body_pose(g.m, g.d, b) for b in panel_bodies])
                        if not np.allclose(actual_scene, expected_scene, atol=1e-12, rtol=0):
                            raise ValueError("Staged scene differs across arms")
                        row["search"][arm] = multistart(g, goal, obj)
                print(case_id, demo, [(n, ph, t) for n, ph, t in selected], flush=True)
        path = out / f"{case_id}.json"
        path.write_text(json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("inputs", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("tasks", nargs="+", choices=TASKS.keys())
    args = ap.parse_args()
    if len(os.sched_getaffinity(0)) > 2 or os.environ["CUDA_VISIBLE_DEVICES"]:
        raise ValueError("This run requires at most two CPU cores and hidden CUDA devices")
    renderer = _probe_egl(True)
    if "llvmpipe" not in renderer[1].lower():
        raise ValueError(renderer)
    args.out.mkdir(parents=True, exist_ok=True)
    for key in args.tasks:
        for idx in (14, 15):
            case(key, idx, args.inputs, args.out, renderer)
