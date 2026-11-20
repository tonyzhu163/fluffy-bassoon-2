"""Same-arm reset evidence for the native (time, qpos, qvel) state."""

import hashlib
import math
import struct

from generated_split import FIXTURES, MOVABLES, canonical

# 1% of the native cross-body pose limits. At 20 Hz the velocity limits
# permit at most half the respective position/angle allowance in one control.
LIMITS = dict(position_m=1e-5, angle_rad=1e-4, linear_m_s=1e-4, angular_rad_s=1e-3)
WIDTHS = {0: (7, 6), 1: (4, 3), 2: (1, 1), 3: (1, 1)}  # free, ball, slide, hinge
HASH_KEYS = ("scene_fingerprint", "generated_row_fingerprint", "seed", "init_state_id",
             "post_settle_scene_fingerprint", "initial_state_fingerprint")


def ensure(ok, message):
    if not ok:
        raise ValueError(message)


def capture(env, info, poses, public_state):
    import numpy as np
    from libero_remote.reproducibility import numerical_state_fingerprint

    model, data = env.sim.model, env.sim.data
    p = info["environment_reproducibility"]
    flat = np.asarray(env.get_sim_state(), dtype="<f8")
    ensure(not model.na and flat.shape == (1 + model.nq + model.nv,) and np.isfinite(flat).all()
           and numerical_state_fingerprint(flat) == p["initial_state_fingerprint"],
           "Native state changed since the reset receipt")
    joints = []
    for i in range(model.njnt):
        kind = int(model.jnt_type[i])
        nq, nv = WIDTHS[kind]
        q, v = int(model.jnt_qposadr[i]), int(model.jnt_dofadr[i])
        joints.append(dict(name=model.joint_id2name(i), body=model.body_id2name(int(model.jnt_bodyid[i])),
                           kind=kind, qpos=data.qpos[q:q+nq].tolist(), qvel=data.qvel[v:v+nv].tolist()))
    a = {k: p[k] for k in HASH_KEYS}
    a.update(time=float(data.time), joints=joints, poses=poses, public_state=public_state.tolist(),
             raw_state_sha256=hashlib.sha256(flat.tobytes()).hexdigest(),
             model_identity=info["model_identity"], renderer=info["renderer"],
             settled_collision_check=p["settled_collision_check"], initial_success=bool(env.check_success()),
             perturbation_plan_fingerprint=p["perturbation_plan_fingerprint"])
    validate(a)
    return a


def validate(a):
    joints = a["joints"]
    identity = a["model_identity"]
    names = {j["name"] for j in joints}
    ensure(joints and None not in names and len(names) == len(joints) and
           set(identity["joint_names"] + identity["gripper_joints"]) <= names, "Incomplete named joint state")
    values = [a["time"]]
    for key, i, total in (("qpos", 0, "model_nq"), ("qvel", 1, "model_nv")):
        ensure(all(j["body"] and len(j[key]) == WIDTHS[j["kind"]][i] for j in joints),
               f"Invalid native joint {key} dimensions")
        coordinates = [v for j in joints for v in j[key]]
        ensure(len(coordinates) == identity[total], f"Incomplete native {key}")
        values.extend(coordinates)
    raw = struct.pack(f"<{len(values)}d", *values)
    prefix = ('{"dtype":"<f8","shape":[' + str(len(values)) + ']}').encode() + bytes(1)
    ensure(all(map(math.isfinite, values)) and hashlib.sha256(raw).hexdigest() == a["raw_state_sha256"]
           and hashlib.sha256(prefix + raw).hexdigest() == a["initial_state_fingerprint"],
           "Named coordinates do not reconstruct the complete raw state")
    ensure(a["poses"] and canonical(a["poses"]) == a["post_settle_scene_fingerprint"],
           "Named poses differ from native settled pose fingerprint")
    for pose in a["poses"].values():
        ensure(len(pose["xyz"]) == 3 and all(map(math.isfinite, pose["xyz"])), "Invalid object position")
        unit(pose["quat_xyzw"])
    obs = a["public_state"]
    ensure(len(obs) == 8 and all(map(math.isfinite, obs)), "Invalid public wrist/jaw state")
    unit(obs[3:7])
    collision = a["settled_collision_check"]
    ensure(a["initial_success"] is False and collision["entities"] == list(MOVABLES + FIXTURES)
           and collision["deep_contacts"] == collision["non_support_contacts"] == collision["welded_fixture_overlaps"] == 0
           and collision["tolerance_m"] == 0.002 and collision["ncon"] >= 0, "Invalid native admission")


def unit(q):
    n = math.hypot(*q)
    ensure(len(q) == 4 and math.isfinite(n) and abs(n - 1) <= 1e-6, "Invalid unit quaternion")
    return tuple(x / n for x in q)


def angle(a, b):
    a, b = unit(a), unit(b)
    opposite = tuple(-x for x in b)
    d1, d2 = sorted((math.dist(a, b), math.dist(a, opposite)))
    return 4 * math.atan2(d1, d2)


class ResetMismatch(ValueError):
    def __init__(self, report):
        self.report = report
        super().__init__(f"Post-settle same-arm state exceeds limits: {report['max_errors']}")


def audit_pair(reference, observed):
    """Compare complete named states. Both raw/native digests survive even on a failed tolerance."""
    for a in (reference, observed):
        validate(a)
    for k in ("scene_fingerprint", "generated_row_fingerprint", "seed", "init_state_id", "model_identity",
              "time", "perturbation_plan_fingerprint"):
        ensure(reference[k] == observed[k], f"Paired reset differs at {k}")
    ensure(reference["poses"].keys() == observed["poses"].keys(), "Different settled object bodies")
    errors = {k: {} for k in LIMITS}
    for a, b in zip(reference["joints"], observed["joints"], strict=True):
        ensure(all(a[k] == b[k] for k in ("name", "body", "kind")), "Different compiled joint identity/order")
        k, name = a["kind"], "joint/" + a["name"]
        q, r, v, w = a["qpos"], b["qpos"], a["qvel"], b["qvel"]
        if k in (0, 2):
            errors["position_m"][name] = math.dist(q[:3], r[:3])
            errors["linear_m_s"][name] = math.dist(v[:3], w[:3])
        if k in (0, 1, 3):
            errors["angle_rad"][name] = abs(q[0] - r[0]) if k == 3 else angle(q[-4:], r[-4:])
            errors["angular_rad_s"][name] = math.dist(v[-3:], w[-3:])
    for name, a in reference["poses"].items():
        b = observed["poses"][name]
        errors["position_m"]["body/" + name] = math.dist(a["xyz"], b["xyz"])
        errors["angle_rad"]["body/" + name] = angle(a["quat_xyzw"], b["quat_xyzw"])
    a, b = reference["public_state"], observed["public_state"]
    errors["position_m"]["public/wrist"] = math.dist(a[:3], b[:3])
    errors["position_m"]["public/jaw"] = abs(a[7] - b[7])
    errors["angle_rad"]["public/wrist"] = angle(a[3:7], b[3:7])
    maxima = {k: max(v.values(), default=0.) for k, v in errors.items()}
    report = dict(limits=LIMITS, max_errors=maxima, errors=errors,
                  hashes=[{k: a[k] for k in ("initial_state_fingerprint", "post_settle_scene_fingerprint",
                                             "raw_state_sha256")} for a in (reference, observed)])
    if any(maxima[k] > limit for k, limit in LIMITS.items()):
        raise ResetMismatch(dict(report, reference=reference, observed=observed))
    return report


def check_admission(row, arm, reset):
    a, p = reset["admission"], reset["environment_reproducibility"]
    ref = row["preflight"][arm]
    ensure(all(a[k] == p[k] for k in (*HASH_KEYS, "settled_collision_check", "perturbation_plan_fingerprint"))
           and a["model_identity"] == reset["model_identity"] and a["model_identity"]["arm"] == arm
           and a["renderer"] == reset["renderer"] and len(a["renderer"]) == 2
           and "NVIDIA" in a["renderer"][0].upper() and a["renderer"][1] and ref["renderer"] == [],
           "Rendered reset metadata differs from native receipt")
    report = audit_pair(ref, a)
    ensure(reset["paired_reset_audit"] == report, "Saved paired reset errors differ from numerical readback")
    return report
