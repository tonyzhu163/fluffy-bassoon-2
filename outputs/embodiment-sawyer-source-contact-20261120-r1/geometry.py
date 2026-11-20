"""Direct plate contacts in the Sawyer Huber source50 replay (R2 pin fc61bf2a).

Use inside the native interpreter, after the physical reset. Pass the public
observation returned by that reset and by each individual executed control.

    with Geometry(env, opening_obs) as geometry:
        for action in issued_actions:
            obs, motion, reward, done, success = env.step(action)
            geometry.sample(obs)
            if done:
                break
        record = geometry.result()

Save record["metrics"] in the case JSON and np.savez_compressed(path,
**record["trace"]) for its numeric traces (replay.py --traces does both).
Geometry also works without a context manager: result() detaches the observer,
and close() detaches it on an abandoned replay.

Trace index 0 is reset. Index k is the endpoint after control k. Pair rows are
[control, plate_geom_id, other_geom_id, sample_mask]. Bit j (0 <= j < N) means an
active constraint in integration substep j+1 of that control. Bit N means an
active constraint at the exact endpoint pose, reconstructed in scratch data.
The same boundary is the next control's pre-action pose. Reset has only bit N.
Simultaneous points on one pair count once. A contact control has at least
one observed active pair for the group during or at the end of that control.
Reset can be first touch but contributes zero executed controls and substeps.
solver_distance_xy_m[k-1, j] is the plate-to-goal XY distance at control k's
j+1 solve. Boundary-only sampling has zero columns and can miss brief touches.
"""

from math import hypot, isclose, isfinite

import mujoco
import numpy as np


HAND = ("palm", "left_finger", "right_finger")
GROUPS = ("hand", *HAND, "arm")


def _ancestry(model, body):
    path = [int(body)]
    while path[-1] != 0:
        path.append(int(model.body_parentid[path[-1]]))
    return path


def _inventory(model, scene, robot, plate):
    """Classify by the nearest owning body root, including unnamed child geoms."""
    from libero.libero.envs.nonphysical_perturbations import _entity_body_ids

    gripper = robot.gripper
    roots = {"arm": model.body(robot.robot_model.root_body).id,
             "palm": model.body(gripper.root_body).id}
    if robot.robot_model.mount is not None:
        roots["mount"] = model.body(robot.robot_model.mount.root_body).id
    # The pinned Sawyer has the Panda hand. Joint 1 drives the left slide and
    # joint 2 the right. Each pad lives on a fixed child of its slide body.
    for side, joint in (("left_finger", "finger_joint1"), ("right_finger", "finger_joint2")):
        jid = model.joint(gripper.correct_naming(joint)).id
        roots[side] = int(model.jnt_bodyid[jid])
    paths = [_ancestry(model, b) for b in range(model.nbody)]
    if (len(set(roots.values())) != len(roots) or roots["arm"] not in paths[roots["palm"]]
            or any(roots["palm"] not in paths[roots[k]] for k in HAND[1:])):
        raise ValueError(f"Unexpected compiled Sawyer gripper ancestry: {roots}")

    body_roles = {v: k for k, v in roots.items()}
    plate_bodies = set(_entity_body_ids(scene, "plate_1"))
    if plate not in plate_bodies:
        raise ValueError("The goal object and its compiled collision subtree disagree")
    roles = []
    catalog = {}
    explicit = set(model.pair_geom1.tolist()) | set(model.pair_geom2.tolist())
    for gid in range(model.ngeom):
        bid = int(model.geom_bodyid[gid])
        role = ("plate" if bid in plate_bodies else
                next((body_roles[b] for b in paths[bid] if b in body_roles), "scene"))
        roles.append(role)
        ct, ca = int(model.geom_contype[gid]), int(model.geom_conaffinity[gid])
        # Either mask can enable collision. Explicit pairs can override both
        # masks. Group numbers and name suffixes are rendering/model conventions.
        catalog[gid] = dict(id=gid, name=model.geom(gid).name or None, body_id=bid, category=role,
                            contype=ct, conaffinity=ca, explicit_pair=gid in explicit)
    if {"plate", *HAND, "arm"} - {g["category"] for g in catalog.values()
                                  if g["contype"] or g["conaffinity"] or g["explicit_pair"]}:
        raise ValueError("Missing compiled plate, palm, finger or arm collision geometry")
    for side in HAND[1:]:
        for name in gripper.important_geoms[side]:
            if roles[model.geom(name).id] != side:
                raise ValueError(f"Finger ancestry disagrees with {side} reference geom {name}")
    return roles, catalog, roots, paths


class Geometry:
    """Episode-local observer for an already open R2 rendered or camera-free arm."""

    def __init__(self, env, opening_obs):
        native = getattr(env, "native", None)
        self.arm = native.arm if native is not None else env.arm
        if self.arm.arm != "Sawyer" or self.arm._step != 0:
            raise ValueError("Construct Geometry at the opening observation of a native Sawyer replay")
        wrapper = self.arm._env
        scene = wrapper.env
        self.sim = wrapper.sim
        self.model, self.live = self.sim.model._model, self.sim.data._data
        robot, = wrapper.robots
        goal, = scene.parsed_problem["goal_state"]
        if [str(x).lower() for x in goal] != ["on", "plate_1", "main_table_stove_front_region"]:
            raise ValueError(f"Unexpected native Goal 5 predicate: {goal!r}")
        # Same BDDL/object references as R2 _PushGeometry.read(). Use the actual
        # site center, independent of the expert's offset pushing waypoint.
        self.plate = int(scene.obj_body_id["plate_1"])
        self.goal = self.model.site(str(goal[2])).id
        joint = int(self.model.body_jntadr[self.plate])
        if (joint < 0 or self.model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE
                or self.model.body(scene.objects_dict["plate_1"].root_body).id != self.plate):
            raise ValueError("Expected the native movable plate's free-jointed root body")
        self.roles, self.catalog, self.roots, self.paths = _inventory(self.model, scene, robot, self.plate)
        self.is_plate = np.array([role == "plate" for role in self.roles] + [False])
        self.dt = float(self.model.opt.timestep)
        self.nsub = int(scene.control_timestep / scene.model_timestep)  # robosuite 1.4.0 step loop
        if not 1 <= self.nsub <= 62:
            raise ValueError("Native control substeps must fit the trace's signed int64 sample mask")
        self.start_time = float(self.live.time)
        self.scratch = mujoco.MjData(self.model)
        self.trace = {k: [] for k in ("time_s", "plate_xyz_m", "goal_xyz_m", "eef_xyz_m",
                                      "plate_goal_xy_m", "plate_pairs")}
        self.first = {}
        self.best = {}
        self.pending = {}
        self.substep_distances = []
        self.solver_trace = []
        self.closed = False
        self._observer = None
        self._endpoint(opening_obs)

        # The pinned MjSim.step is a Python instance method calling mj_step
        # exactly once. After Euler mj_step, contacts and xpos describe its
        # solved position stage, while qpos/time have already advanced.
        # Preserve the full outer observation/controller/render sequence.
        if self.model.opt.integrator != mujoco.mjtIntegrator.mjINT_EULER:
            self.mode = "boundary_only: integrator differs from inspected single-stage Euler"
        elif "step" in vars(self.sim):
            self.mode = "boundary_only: simulator already has an instance step override"
        else:
            self.mode = "all_native_substeps_and_boundaries"
            original = self.sim.step

            def observed_step(*args, **kwargs):
                solve_time = float(self.live.time)
                value = original(*args, **kwargs)
                distance = self._observe(self.live, len(self.substep_distances), solve_time)
                self.substep_distances.append(distance)
                return value

            self.sim.step = self._observer = observed_step

    def _observe(self, data, bit, time):
        # Use the R2 expert_sweep.contact_names criterion (efc_address >= 0).
        # A positive normal-force threshold would change the definition.
        contacts = data.contact
        ids = contacts.geom
        a, b = ids[:, 0], ids[:, 1]
        plate_a, plate_b = self.is_plate[a], self.is_plate[b]
        keep = (contacts.efc_address >= 0) & (a >= 0) & (b >= 0) & (plate_a ^ plate_b)
        seen = set()
        mask = 1 << bit
        for x, y in ids[keep].tolist():
            pair = (x, y) if self.is_plate[x] else (y, x)
            self.pending[pair] = self.pending.get(pair, 0) | mask
            seen.add(self.roles[pair[1]])
        if seen.intersection(HAND):
            seen.add("hand")
        plate, goal = data.xpos[self.plate], data.site_xpos[self.goal]
        distance = hypot(plate[0] - goal[0], plate[1] - goal[1])
        if not isfinite(distance):
            raise ValueError("Nonfinite native plate-to-goal distance")
        for group in GROUPS:
            if group not in self.first and group in seen:
                self.first[group] = dict(control=len(self.trace["time_s"]),
                                         substep=bit + 1 if bit < self.nsub else None,
                                         phase="substep" if bit < self.nsub else "boundary",
                                         time_s=time - self.start_time, distance_xy_m=distance,
                                         plate_xyz_m=plate.tolist(), goal_xyz_m=goal.tolist())
                self.best[group] = distance
            if group in self.first:
                self.best[group] = min(self.best[group], distance)
        return distance

    def _endpoint(self, obs):
        # R2 generated_layouts.collision_admission recomputes the position
        # stage in independent MjData. Calling live sim.forward() here can
        # change the next OSC update and solver warm start.
        d = self.scratch
        d.qpos[:] = self.live.qpos
        d.mocap_pos[:] = self.live.mocap_pos
        d.mocap_quat[:] = self.live.mocap_quat
        d.eq_active[:] = self.live.eq_active
        d.time = self.live.time
        mujoco.mj_fwdPosition(self.model, d)
        public = np.asarray(obs["state"])
        plate, goal = d.xpos[self.plate], d.site_xpos[self.goal]
        if public.shape != (8,) or not np.isfinite([public[:3], plate, goal]).all():
            raise ValueError("Expected finite world-frame plate, goal and public EEF positions")
        distance = self._observe(d, self.nsub, float(d.time))
        self.trace["plate_xyz_m"].append(plate.tolist())
        self.trace["goal_xyz_m"].append(goal.tolist())
        self.trace["eef_xyz_m"].append(public[:3].tolist())
        self.trace["time_s"].append(float(d.time) - self.start_time)
        self.trace["plate_goal_xy_m"].append(distance)
        self.trace["plate_pairs"].append([[p, q, mask] for (p, q), mask in sorted(self.pending.items())])
        self.pending.clear()
        if len(self.trace["time_s"]) > 1:
            self.solver_trace.append(self.substep_distances)
        self.substep_distances = []

    def sample(self, obs):
        """Record the delivered public observation immediately after one env.step."""
        control = len(self.trace["time_s"])
        if self.closed or self.arm._step != control or self.arm._env.sim is not self.sim:
            raise ValueError(f"Expected the same open Sawyer replay at native control {control}")
        nseen = len(self.substep_distances)
        if self._observer is not None and nseen != self.nsub:
            raise RuntimeError(f"Observed {nseen} of {self.nsub} physics substeps at control {control}")
        elapsed = self.live.time - self.start_time - self.trace["time_s"][-1]
        if not isclose(elapsed, self.dt * self.nsub, abs_tol=1e-9):
            raise RuntimeError(f"Native clock advanced by {elapsed} s at control {control}")
        self._endpoint(obs)

    def _summary(self, group):
        categories = HAND if group == "hand" else (group,)
        masks = []
        for row in self.trace["plate_pairs"]:
            mask = 0
            for _, other, observed in row:
                if self.roles[other] in categories:
                    mask |= observed
            masks.append(mask)
        boundary = 1 << self.nsub
        solver = [mask & (boundary - 1) for mask in masks[1:]]
        substeps = sum(mask.bit_count() for mask in solver)
        first = self.first.get(group)
        final = self.trace["plate_goal_xy_m"][-1]
        all_steps = self.mode == "all_native_substeps_and_boundaries"
        return dict(observed=first is not None, reset_contact=bool(masks[0]), first_touch=first,
                    contact_controls=sum(bool(mask) for mask in masks[1:]),
                    boundary_contact_controls=sum(bool(mask & boundary) for mask in masks[1:]),
                    solver_contact_controls=sum(bool(mask) for mask in solver) if all_steps else None,
                    active_substeps=substeps if all_steps else None,
                    sampled_solver_duration_s=substeps * self.dt if all_steps else None,
                    final_distance_xy_m=final, min_distance_since_first_xy_m=self.best.get(group),
                    final_progress_since_first_xy_m=first["distance_xy_m"] - final if first else None,
                    best_progress_since_first_xy_m=first["distance_xy_m"] - self.best[group] if first else None)

    def result(self):
        """Finalize and return per-group summaries, identity tables and aligned traces."""
        self.close()
        controls = len(self.trace["time_s"]) - 1
        if self.substep_distances or self.arm._step != controls:
            raise ValueError("Missing the public observation for an executed control")
        # Include every candidate instrumented geom and every actual opposing
        # geom, including scene/support contacts, without repeating names per step.
        pair_rows = [[k, p, q, m] for k, row in enumerate(self.trace["plate_pairs"]) for p, q, m in row]
        used = {gid for _, p, q, _ in pair_rows for gid in (p, q)}
        geoms = [g for gid, g in self.catalog.items() if gid in used or
                 (g["category"] != "scene" and (g["contype"] or g["conaffinity"] or g["explicit_pair"]))]
        bodies = {b for g in geoms for b in self.paths[g["body_id"]]}
        bodies.update(self.roots.values())
        model = self.model
        goal_body = int(model.site_bodyid[self.goal])
        bodies.update(self.paths[goal_body])
        metrics = dict(schema_version=1, sampling=self.mode, physics_dt_s=self.dt,
                       substeps_per_control=self.nsub, controls=controls, reset_sim_time_s=self.start_time,
                       body_roots=dict(self.roots, plate=self.plate),
                       bodies=[dict(id=b, name=model.body(b).name or None, parent_id=int(model.body_parentid[b]))
                               for b in sorted(bodies)], collision_geoms=geoms,
                       goal_site=dict(id=self.goal, name=model.site(self.goal).name, body_id=goal_body,
                                      size_m=model.site_size[self.goal].tolist()),
                       eef_source="public obs['state'][:3] (robot0_eef_pos)",
                       pair_columns=["control", "plate_geom_id", "other_geom_id", "sample_mask"],
                       boundary_bit=self.nsub, summary={g: self._summary(g) for g in GROUPS})
        trace = {k: np.asarray(v) for k, v in self.trace.items() if k != "plate_pairs"}
        trace["plate_pairs"] = np.asarray(pair_rows, dtype=np.int64).reshape(-1, 4)
        width = self.nsub if self.mode == "all_native_substeps_and_boundaries" else 0
        trace["solver_distance_xy_m"] = np.asarray(self.solver_trace, dtype=np.float64).reshape(controls, width)
        return dict(metrics=metrics, trace=trace)

    def close(self):
        if self._observer is not None and self.sim.step is self._observer:
            del self.sim.step  # reveal the original class method
        self._observer = None
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
