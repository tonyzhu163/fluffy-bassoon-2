"""Outer-edge Goal 5 controller. All feedback is read from the settled live scene."""

from dataclasses import asdict, dataclass
import json
import math

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

LEG_TOLERANCE_M = 0.016


@dataclass(frozen=True)
class Candidate:
    name: str
    tilt: float = 48.0         # inclination of tool +Z from down, toward world -Y
    yaw: float = -90.0
    flank: float = -80.0      # signed angle from the plate's rear, about world +Z
    trailing: float = -28.0
    height: float = 0.006     # lowest finger vertex above table during transport (m)
    gap: float = 0.012        # measured separation at flank descent
    lead: float = 0.028       # tangential feedforward during plate transport (m)
    depth: float = 0.017      # push-in setpoint, measured at leading finger surface (m)
    detour: bool = True       # clearance waypoint on the open side of the cheese

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name or not self.name.replace("-", "").isalnum():
            raise ValueError("Use a unique alphanumeric or hyphenated candidate name")
        numeric = [v for k, v in asdict(self).items() if k not in ("name", "detour")]
        if (type(self.detour) is not bool
                or not all(type(v) in (float, int) and math.isfinite(v) for v in numeric)
                or not (self.height >= 0 and self.gap > 0 and self.lead >= 0 and self.depth > 0
                        and abs(self.trailing) < 90 and abs(self.flank) <= 180 and abs(self.tilt) < 90)):
            raise ValueError(f"Invalid outer-edge candidate {self.name}")


DEFAULTS = (
    Candidate("left-arc"),
    Candidate("right-arc", flank=80, trailing=28),
    Candidate("rear-oblique", flank=45, trailing=4, height=0.008, detour=False),
    Candidate("shallow-side", tilt=32, yaw=0, flank=-65, trailing=-38, height=0.009),
    Candidate("wide-right", tilt=55, yaw=0, flank=96, trailing=46, lead=0.045, depth=0.022),
    Candidate("left-direct", tilt=55, yaw=90, flank=-55, trailing=-8, detour=False),
)


def candidate_list(path, names):
    candidates = list(DEFAULTS)
    if path:
        candidates.extend(Candidate(**spec) for spec in json.loads(path.read_text()))
    by_name = {c.name: c for c in candidates}
    if len(by_name) != len(candidates):
        raise ValueError("Candidate names must be unique")
    return [by_name[n] for n in names] if names else candidates


def unit(v):
    return v / max(np.linalg.norm(v), 1e-9)


def bound(v, length):
    return v * min(1., length / max(np.linalg.norm(v), 1e-9))


def rotate(v, degrees):
    a = np.deg2rad(degrees)
    return np.cos(a) * v + np.sin(a) * np.array([-v[1], v[0]])


def contact_names(geometry):
    """Aggregate solver-active contact points by named geom pair at a control boundary."""
    m, d = geometry.model._model, geometry.data._data
    pairs = {}
    wrench = np.empty(6)
    for i in range(d.ncon):
        c = d.contact[i]
        if c.efc_address < 0:
            continue
        a, b = int(c.geom1), int(c.geom2)
        names = sorted(f"{m.geom(g).name or g}@{m.body(int(m.geom_bodyid[g])).name}" for g in (a, b))
        key = tuple(names)
        mujoco.mj_contactForce(m, d, i, wrench)
        if key not in pairs:
            pairs[key] = dict(geoms=names, points=0, normal_force_n=0., min_distance_m=float(c.dist))
        item = pairs[key]
        item["points"] += 1
        item["normal_force_n"] += float(wrench[0])
        item["min_distance_m"] = min(item["min_distance_m"], float(c.dist))
    return list(pairs.values())


def final_arm_geometry(wrapper, geom):
    robot = wrapper.robots[0]
    m, d = geom.model._model, geom.data._data
    joints = d.qpos[robot._ref_joint_pos_indexes]
    limits = m.jnt_range[robot._ref_joint_indexes]
    fingers = geom._points("fingers")
    # The specific link/fixture pair observed in the pilot. Contact summaries
    # separately include every active pair, even if this pair is clear.
    elbow, cabinet = (m.geom(name).id for name in ("robot0_right_l4_2", "wooden_cabinet_1_g9"))
    return dict(joint_positions_rad=joints.tolist(),
                nearest_joint_limit_rad=np.minimum(joints-limits[:, 0], limits[:, 1]-joints).tolist(),
                finger_bounds_xyz_m=[fingers.min(axis=0).tolist(), fingers.max(axis=0).tolist()],
                elbow_cabinet_gap_capped_10cm_m=float(mujoco.mj_geomDistance(m, d, elbow, cabinet, .1, None)))


class OuterSweep:
    def __init__(self, geom, scene, candidate):
        self.geom = geom
        self.c = c = candidate
        self.orientation = (Rotation.from_euler("x", -c.tilt, degrees=True)
                            * Rotation.from_euler("z", c.yaw, degrees=True)
                            * Rotation.from_euler("x", 180, degrees=True))
        self.start = np.array(scene["plate"][:2])
        self.goal = np.array(scene["goal"][:2]) + .50 * np.array(scene["goal_half"][:2])
        self.waypoints = []
        obstacle = geom._points("obstacle")[:, :2]
        # A staging lane beside the obstacle, parallel to the reset-to-goal line.
        d = unit(self.goal - self.start)
        side = np.array([-d[1], d[0]])
        xy = obstacle - self.start
        along, across = xy @ d, xy @ side
        radius = scene["plate_radius"] + .006 + LEG_TOLERANCE_M
        if (c.detour and along.min() - radius < np.linalg.norm(self.goal - self.start)
                and along.max() + radius > 0 and across.min() < radius and across.max() > -radius):
            left, right = across.max() + radius, across.min() - radius
            offset = left if abs(left) < abs(right) else right
            # First move clear, then pass the obstacle before turning toward the region.
            enter = max(0., along.min() - radius)
            leave = min(np.linalg.norm(self.goal - self.start), along.max() + radius)
            self.waypoints.extend([self.start + enter*d + offset*side, self.start + leave*d + offset*side])
        self.waypoints.append(self.goal)
        self.leg = 0
        self.phase = "hover"
        self.ticks = 0
        self.arc = c.flank
        self.integral = np.zeros(3)
        self.best_distance = np.inf
        self.no_gain = 0
        self.last = {}

    def phase_to(self, phase):
        self.phase = phase
        self.ticks = 0
        self.integral.fill(0)

    def action(self, s):
        c = self.c
        plate = np.array(s["plate"][:2])
        tcp = np.array(s["tcp"])
        fingers = self.geom._points("fingers")
        rim = self.geom._points("plate")
        table = s["table_z"]
        current = Rotation.from_matrix(self.geom.data.site_xmat[self.geom.tcp_id].reshape(3, 3))
        angular = (self.orientation * current.inv()).as_rotvec()
        # At the hover target the fingers are already tilted. Predict the actual
        # leading finger surface at that attitude even during the initial rotation.
        offsets = (self.orientation * current.inv()).apply(fingers - tcp)
        low = offsets[:, 2].min()
        patch = offsets[offsets[:, 2] <= low + max(.004, s["rim_z"] - table - c.height + .003)]
        while True:
            waypoint = self.waypoints[self.leg]
            delta = waypoint - plate
            distance = np.linalg.norm(delta)
            if self.leg < len(self.waypoints) - 1 and distance < LEG_TOLERANCE_M:
                self.leg += 1
                self.arc = c.flank
                self.best_distance, self.no_gain = np.inf, 0
                self.phase_to("lift")
                continue
            direction = unit(delta) if distance > .001 else unit(self.goal - self.start)
            active_angle = self.arc
            normal = rotate(-direction, active_angle)
            tangent = np.array([-normal[1], normal[0]])
            # Behind this support plane, every plate collision vertex is on
            # the inner side. The frontmost low finger surface faces the rim.
            rim_extent = ((rim[:, :2] - plate) @ normal).max()
            projections = patch[:, :2] @ normal
            front = patch[projections <= projections.min() + .003, :2].mean(axis=0)
            gap = c.gap
            clearance = max(.10, s["rim_z"] - table + .075)
            tangent_lead = 0.
            if self.phase in ("lower", "arc", "push"):
                clearance = min(c.height, max(.001, s["rim_z"] - table - .002))
            if self.phase == "arc":
                gap = .002
            elif self.phase == "push":
                gap = -min(c.depth, .5*rim_extent)
                tangent_lead = c.lead * (direction @ tangent)
            target_xy = plate + (rim_extent + gap) * normal + tangent_lead*tangent - front
            if self.phase == "lift":
                target_xy = tcp[:2]
            target = np.r_[target_xy, table + clearance - low]
            error = target - tcp
            # In contact use measured finger height rather than a projected
            # tip offset. The translation cap bounds the corrective command.
            if self.phase in ("lower", "arc", "push"):
                error[2] = table + clearance - s["finger_low"]
                target[2] = tcp[2] + error[2]
            angle_ok = np.linalg.norm(angular) < .19
            xy_ok = np.linalg.norm(error[:2]) < .014
            low_ok = s["finger_low"] <= table + clearance + .004
            if self.phase == "lift" and s["finger_low"] >= table + clearance - .016:
                self.phase_to("hover")
            elif self.phase == "hover" and xy_ok and abs(error[2]) < .020 and angle_ok:
                self.phase_to("lower")
            elif self.phase == "lower" and xy_ok and angle_ok and (low_ok or s["contact"]):
                self.phase_to("arc")
            elif self.phase == "arc" and abs(self.arc - c.trailing) < 1e-6:
                self.phase_to("push")
            else:
                break
        if self.phase == "arc" and (np.linalg.norm(error[:2]) < .021 or s["contact"]):
            self.arc += np.clip(c.trailing - self.arc, -4., 4.)
        self.ticks += 1
        if self.phase == "push":
            if distance < self.best_distance - .003:
                self.no_gain = 0
                self.best_distance = distance
            else:
                self.no_gain += 1
        self.integral = np.clip(self.integral + .055*error, -.018, .018)
        linear = bound(error + self.integral, .025 if self.phase in ("arc", "push") else .040)
        command = np.clip(np.r_[1.5*linear/.05, bound(angular, .32)/.5, 1.], -1., 1.).astype(np.float32)
        self.last = dict(phase=self.phase, phase_tick=self.ticks, leg=self.leg,
                         aim=waypoint.tolist(), destination=self.goal.tolist(), tcp_target=target.tolist(),
                         tcp_error=error.tolist(), angle_error_rad=float(np.linalg.norm(angular)),
                         flank_deg=float(active_angle),
                         actual_edge_gap_m=float(((fingers[:, :2]-plate) @ normal)[
                             fingers[:, 2] <= fingers[:, 2].min() + max(.004, s["rim_z"] - table)].min() - rim_extent),
                         desired_edge_gap_m=float(gap), plate_to_waypoint_m=float(distance),
                         lowest_finger_above_table_m=float(s["finger_low"] - table))
        return command

    def early_stop(self):
        if self.phase == "push" and self.no_gain >= 26:
            return "plate made less than 3 mm of waypoint progress in 26 push controls"
        if self.phase != "push" and self.ticks >= (75 if self.phase == "hover" else 45):
            return f"{self.phase} exceeded its control budget"
        return None
