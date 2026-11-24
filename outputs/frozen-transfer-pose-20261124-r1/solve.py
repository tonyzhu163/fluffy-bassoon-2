"""Native-model FK witnesses for policy goals under joint limits and an external one-period rate envelope."""
import argparse
import gzip
import json
import os
import subprocess
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import lsq_linear, minimize
from scipy.spatial.transform import Rotation

from arm_env import ArmEnv
from geometry import Geometry
from panel import cases, key
from replay import digest, save

# Manufacturer envelopes, in rad/s, in the native proximal-to-distal joint order.
SPEED = {"Kinova3": [1.3963]*4 + [1.2218]*3, "Panda": [2.175]*4 + [2.610]*3}
SOURCES = {
    "Kinova3": "https://github.com/Kinovarobotics/ros_kortex/blob/noetic-devel/kortex_description/arms/gen3/7dof/urdf/GEN3-7DOF-VISION_ARM_URDF_V12.urdf",
    "Panda": "https://github.com/frankarobotics/franka_ros/blob/develop/franka_description/robots/panda/joint_limits.yaml",
}
SCALE = np.array([.05]*3 + [.5]*3)
# Convergence of normalized error, for scheduling numerical refinements only.
CONVERGED = 2e-6


class Problem:
    def __init__(self, g, p, R):
        self.g, self.p, self.R = g, p, R

    def value(self, q, jac=False):
        g = self.g
        g.d.qpos[g.adr] = q
        mujoco.mj_kinematics(g.m, g.d)
        site_rot = g.d.site_xmat[g.site].reshape(3,3)
        phi = Rotation.from_matrix(site_rot @ self.R.T).as_rotvec()
        e = np.r_[g.d.site_xpos[g.site]-self.p, phi] / SCALE
        if not jac:
            return e
        mujoco.mj_comPos(g.m, g.d)
        mujoco.mj_jacSite(g.m, g.d, g.jacp, g.jacr, g.site)
        x, y, z = phi
        skew = np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
        t = np.linalg.norm(phi)
        coeff = 1/12 if t < 1e-4 else (1-.5*t/np.tan(.5*t))/t**2
        left_inv = np.eye(3) - skew/2 + coeff * (skew @ skew)
        J = np.vstack([g.jacp[:,g.dofs], left_inv @ g.jacr[:,g.dofs]]) / SCALE[:,None]
        return e, J

    def newton(self, seed, lo, hi, n):
        q = np.clip(seed, lo, hi)
        linear = None
        for _ in range(25):
            err, J = self.value(q, True)
            e, A = err[:n], J[:n]
            d = np.linalg.lstsq(A, -e, rcond=1e-10)[0]
            if np.any(q+d < lo) or np.any(q+d > hi):
                d = lsq_linear(A, -e, bounds=(lo-q, hi-q), method="bvls", tol=1e-10).x
            if linear is None:
                proposed = np.clip(q+d, lo, hi)
                linear = np.r_[err+J@(proposed-q), self.value(proposed)] * np.tile(SCALE,2)
            if np.linalg.norm(e) < CONVERGED or np.linalg.norm(d) < 1e-11:
                break
            old = np.linalg.norm(e)
            for h in (1., .5, .25, .125, .0625, .03125, .015625):
                candidate = np.clip(q+h*d, lo, hi)
                if np.linalg.norm(self.value(candidate)[:n]) < old-1e-12:
                    q = candidate
                    break
            else:
                break
        return q, linear

    def minimize_at_angle(self, seed, lo, hi, angle):
        # Equality at the requested attitude, or inequality at SAME-STEP native observed angular error.
        def cost(q):
            e,J = self.value(q, True)
            return .5 * np.dot(e[:3],e[:3]), e[:3] @ J[:3]
        if angle == 0:
            con = dict(type="eq", fun=lambda q:self.value(q)[3:], jac=lambda q:self.value(q,True)[1][3:])
        else:
            def tolerance_jac(q):
                e,J = self.value(q, True)
                return -2 * e[3:] @ J[3:]
            con = dict(type="ineq", fun=lambda q:(angle/SCALE[3])**2-self.value(q)[3:]@self.value(q)[3:],
                       jac=tolerance_jac)
        fit = minimize(cost, np.clip(seed,lo,hi), jac=True, bounds=list(zip(lo,hi)), constraints=[con],
                       method="SLSQP", options=dict(maxiter=75,ftol=1e-12))
        return np.clip(fit.x,lo,hi)

    def metrics(self, q):
        e = self.value(q)*SCALE
        return np.array([np.linalg.norm(e[:3]),np.linalg.norm(e[3:])])

    def poses(self, q, lo, hi, native_angle, multiseed):
        fixed, linear = self.newton(q,lo,hi,6)
        seed_options = [fixed]
        if multiseed and np.linalg.norm(self.value(fixed)) > CONVERGED:
            # Deterministic alternate starts in addition to tracking the preceding/observed pose.
            for seed in (self.g.qreset, self.g.qreset+np.array([.45,-.4,.3,-.4,.3,.25,-.45]),
                         self.g.qreset-np.array([.45,-.4,.3,-.4,.3,.25,-.45])):
                candidate,_ = self.newton(seed,lo,hi,6)
                seed_options.append(candidate)
                if np.linalg.norm(self.value(candidate)) < CONVERGED:
                    break
            fixed = min(seed_options, key=lambda x:np.linalg.norm(self.value(x)))
        pf, af = self.metrics(fixed)
        if pf > CONVERGED*SCALE[0] or af > CONVERGED*SCALE[3]:
            exact = self.minimize_at_angle(fixed,lo,hi,0)
            if self.metrics(exact)[1] <= CONVERGED*SCALE[3]:
                fixed = exact
        free = fixed.copy()
        if self.metrics(fixed)[0] > CONVERGED*SCALE[0]:
            trials = [self.newton(s,lo,hi,3)[0] for s in (fixed, q)]
            free = min([free]+trials, key=lambda x:self.metrics(x)[0])
        matched = fixed
        if native_angle is not None:
            # Start from the free solution when it already satisfies the measured angular allowance.
            # The observed endpoint itself supplies another legal candidate only if inside this box.
            candidates = [v for v in (fixed, free) if self.metrics(v)[1] <= native_angle + 1e-6]
            if not candidates or min(self.metrics(v)[0] for v in candidates) > self.metrics(free)[0] + 1e-7:
                candidates.append(self.minimize_at_angle(fixed,lo,hi,native_angle))
            valid = [v for v in candidates if self.metrics(v)[1] <= native_angle + 1e-6]
            matched = min(valid,key=lambda x:self.metrics(x)[0]) if valid else fixed
        for v in (fixed,free,matched):
            assert np.all(v >= lo-1e-10) and np.all(v <= hi+1e-10)
        return dict(full_q=fixed, free_q=free, matched_q=matched, full=self.metrics(fixed), free=self.metrics(free),
                    matched=self.metrics(matched), linear_and_fk=linear)


class Chain:
    """Evaluate the earlier fixed hinge-axis-point certificates on new targets."""
    def __init__(self, g, arm, evidence):
        offsets = [np.zeros(len(g.joints))]
        for f in sorted(evidence.glob("*-readback.json.gz")):
            for row in json.loads(gzip.decompress(f.read_bytes()))["rows"]:
                if row["arm"] == arm and "bound" in row:
                    offsets.append(row["bound"]["axis_offsets_m"])
        g.d.qpos[g.adr] = g.qreset
        g.forward()
        pts = g.d.xanchor[g.joints] + np.array(offsets)[:,:,None]*g.d.xaxis[g.joints]
        self.base = pts[:,0]
        self.length = np.linalg.norm(np.diff(pts,axis=1),axis=2).sum(axis=1)
        T = g.tcp()
        self.lever = (pts[:,-1]-T[:3,3]) @ T[:3,:3]
        self.offsets = np.array(offsets)

    def lower(self,p,R,theta):
        d = np.linalg.norm(p + self.lever @ R.T - self.base,axis=1)
        e = d-self.length-2*np.linalg.norm(self.lever,axis=1)*np.sin(theta/2)
        return max(0.,e.max()), int(e.argmax())


def run(spec, out, source, previous, arm):
    name = f"{key(spec)}-{arm}"
    target_meta = json.loads((source / f"{name}.json").read_bytes())
    dt = target_meta["sample_dt"]
    prior = Path(__file__).resolve().parents[1] / "frozen-vla-reachability-20261123-r1/evidence"
    with ArmEnv(arm,spec["suite"],spec["task_id"]) as env:
        reset = env.reset(spec["init_state_id"],spec["seed"])
        assert reset["environment_reproducibility"] == target_meta["reset"]
        g = Geometry(env._env)
        chain = Chain(g,arm,prior)
        for tape_arm in (arm, "Panda" if arm=="Kinova3" else "Kinova3"):
            fn = f"{key(spec)}-{tape_arm}"
            path = source / f"{fn}.npz"
            meta = json.loads(path.with_suffix(".json").read_bytes())
            assert digest(path)==meta["data_sha256"]
            data = np.load(path,allow_pickle=False)
            same = tape_arm == arm
            result = []
            seed = g.qreset
            for i,(p,R) in enumerate(zip(data["goal_p"],data["goal_R"],strict=True)):
                prob = Problem(g,p,R)
                q = data["q"][i] if same else seed
                if same:
                    native_err = prob.value(data["q_end"][i])*SCALE
                    start_err = prob.value(q)*SCALE
                    assert np.linalg.norm(data["end_p"][i]-p-native_err[:3]) < 1e-9
                    assert abs(np.linalg.norm(native_err[3:])-
                              Rotation.from_matrix(data["end_R"][i]@R.T).magnitude()) < 1e-9
                    if i == 0:
                        e,J = prob.value(q,True)
                        eps = 1e-6
                        fd = np.array([(prob.value(q+d)-prob.value(q-d))/(2*eps) for d in np.eye(len(q))*eps]).T
                        jac_diff = np.max(abs(fd-J))
                        assert jac_diff < 1e-6
                lower, index = chain.lower(p,R,.01)
                fixed = prob.poses(q,g.lo,g.hi,None,True)
                seed = fixed["full_q"]
                record = {f"static_{k}":v for k,v in fixed.items()}
                record.update(chain_lb_at_0p01_m=lower, chain_index=index, chain_lb_exact_m=chain.lower(p,R,0)[0])
                if same:
                    travel = np.array(SPEED[arm])*dt
                    lo,hi = np.maximum(g.lo,q-travel), np.minimum(g.hi,q+travel)
                    # A soft-limit overshoot larger than the hardware travel budget makes the intersection empty.
                    assert np.all(hi>lo)
                    native_angle = np.linalg.norm(native_err[3:])
                    rate = prob.poses(q,lo,hi,native_angle,False)
                    record.update({f"rate_{k}":v for k,v in rate.items()})
                    record.update(native_error=native_err, requested_from_start=-start_err,
                                  native_max_speed_fraction=np.max(abs(data["q_end"][i]-q)/travel),
                                  native_joint_margin_rad=np.minimum(q-g.lo,g.hi-q).min())
                result.append(record)
            stem = f"{fn}-on-{arm}"
            dest = out / f"{stem}.npz"
            np.savez_compressed(dest, **{k:np.array([r[k] for r in result]) for k in result[0]})
            save(dest.with_suffix(".json"),dict(tape_arm=tape_arm,model_arm=arm,tape_sha256=meta["tape_sha256"],
                 replay_sha256=meta["data_sha256"],data_sha256=digest(dest),n=len(result),case=spec,
                 speed_rad_s=SPEED[arm],speed_source=SOURCES[arm],dt_s=dt,chain_offsets_m=chain.offsets.tolist(),
                 jac_max_diff_scaled=jac_diff if same else None,script_sha256=digest(Path(__file__)),
                 source_revision=previous,affinity=sorted(os.sched_getaffinity(0)),gpu_visible=os.environ["CUDA_VISIBLE_DEVICES"]))
            print(stem,len(result),flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("arm")
    ap.add_argument("out",type=Path)
    args=ap.parse_args()
    assert not os.environ["CUDA_VISIBLE_DEVICES"] and set(os.sched_getaffinity(0)) <= {46,47}
    revision=subprocess.check_output(["git","-C",os.environ["POSE_CODE"],"rev-parse","HEAD"],text=True).strip()
    for case in cases(args.arm):
        run(case,args.out,Path(os.environ["POSE_REPLAY"]),revision,args.arm)
