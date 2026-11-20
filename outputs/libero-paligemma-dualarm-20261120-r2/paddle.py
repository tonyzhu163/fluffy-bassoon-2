"""Goal 5 native whole-gripper pushing with search on source training layouts."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, asdict
from itertools import product
import json
import multiprocessing
import os
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from expert import parse_layouts, session, write_json
from expert_sweep import rotate, unit
from demos import save_tape


@dataclass(frozen=True)
class Route:
    mode: str
    tilt: int
    yaw: int
    flank: int
    bottom: float = .006


def cloud(geom):
    m, d = geom.model._model, geom.data._data
    hand = m.geom("gripper0_hand_collision").id
    mid = m.geom_dataid[hand]
    start, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
    points = m.mesh_vert[start:start+n] @ d.geom_xmat[hand].reshape(3, 3).T + d.geom_xpos[hand]
    return np.vstack((points, geom._points("fingers")))


class Paddle:
    def __init__(self, route, scene):
        self.route = route
        self.orientation = (Rotation.from_euler("x", -route.tilt, degrees=True)
                            * Rotation.from_euler("z", route.yaw, degrees=True)
                            * Rotation.from_euler("x", 180, degrees=True))
        self.goal = np.array(scene["goal"][:2]) + .43*np.array(scene["goal_half"][:2])
        self.phase = 0
        self.tick = 0
        self.total = 0
        self.integral = np.zeros(3)
        self.origin = np.array(scene["plate"][:2])
        self.last = {}

    def action(self, geometry, s):
        c = self.route
        xyz = np.array(s["tcp"])
        plate = np.array(s["plate"][:2])
        current = Rotation.from_matrix(geometry.data.site_xmat[geometry.tcp_id].reshape(3, 3))
        vertices = cloud(geometry)
        # Track the actual lowest surface. For tilted grasps the palm can be
        # several cm below both fingertips in the compiled Sawyer model.
        low = vertices[:, 2].min()
        grip = vertices[vertices[:, 2] < low + .009]
        direction = unit(self.goal - plate)
        travel = rotate(direction, c.flank) if self.phase < 2 else direction
        lead = (grip[:, :2] @ travel).max()
        surface = grip[(grip[:, :2] @ travel) >= lead-.006, :2].mean(0)
        depth = s["plate_radius"]
        if c.mode == "rim":
            finger = -depth - .010 if self.phase < 2 else -.48*depth
        elif c.mode == "hook":
            finger = .38*depth if self.phase < 2 else 1.28*depth
        else:
            finger = -.25*depth if self.phase < 2 else .46*depth
        target = plate + finger*travel
        height = s["table_z"] + (.108 if self.phase == 0 else c.bottom)
        error = np.r_[target-surface, height-low]
        self.tick += 1
        self.total += 1
        # Orientation can relax while sliding to accommodate the Sawyer's base
        # friction and the cabinet. Force is always issued through native OSC.
        angular = (self.orientation * current.inv()).as_rotvec()
        if self.phase == 0 and ((np.linalg.norm(error) < .023 and np.linalg.norm(angular) < .32) or self.tick >= 62):
            self.phase, self.tick = 1, 0
            self.integral.fill(0)
        elif self.phase == 1 and ((low < height+.007 and np.linalg.norm(error[:2]) < .030) or self.tick >= 53):
            self.phase, self.tick = 2, 0
            self.integral.fill(0)
        self.integral = np.clip(self.integral + .14*error, -.042, .042)
        linear = (error + self.integral) / .035
        if self.phase == 2:
            linear[:2] += .34*direction
        a = np.r_[np.clip(linear, -1, 1), np.clip(angular/(.55 if self.phase < 2 else 1.0), -.9, .9), 1.]
        self.last = dict(phase=self.phase, low=float(low), tcp=xyz.round(4).tolist(), plate=plate.round(4).tolist(),
                         error=error.round(4).tolist(), angle=float(np.linalg.norm(angular)))
        return a.astype(np.float32)


def trial(job):
    out, pose, seed, index, route, camera = job
    stem = f"{pose}-{seed}-{index}"
    target = out / ("Sawyer" if camera else "drafts")
    with session("Sawyer", camera) as env:
        obs = env.reset(pose, seed)
        s = env.expert_scene()
        geometry = env.native.geometry if camera else env.geometry
        expert = Paddle(route, s)
        initial_d = np.linalg.norm(expert.goal-expert.origin)
        best, last_progress = 0., 0
        frames, acts, motions, rewards, samples = [], [], [], [], []
        for t in range(300):
            a = expert.action(geometry, s)
            if camera:
                frames.append(obs)
            obs, motion, reward, done, success = env.step(a)
            s = env.expert_scene()
            acts.append(a)
            motions.append(motion)
            rewards.append(reward)
            gain = initial_d-np.linalg.norm(expert.goal-np.array(s["plate"][:2]))
            if gain > best+.002:
                best, last_progress = float(gain), t
            if t % 20 == 0:
                samples.append(dict(t=t, **expert.last))
            if done or (expert.phase == 2 and expert.tick >= 65 and t-last_progress > 65):
                break
        ok = bool(done and success and reward > 0)
        row = dict(body="Sawyer", pose=pose, seed=seed, recipe=index, params=asdict(route), steps=len(acts),
                   native_success=bool(success), terminal_reward=float(reward), done=bool(done), accepted=bool(ok and camera),
                   search_success=ok, goal_gain_m=float(gain), best_gain_m=best, trace=samples, last=expert.last,
                   reset=env.reset_info)
        if ok and camera:
            dest=target / (stem+".npz")
            save_tape(dest, frames, acts, motions, rewards, obs["state"], obs["prompt"], "Sawyer", pose, seed)
            row["file"] = str(dest.relative_to(out))
        elif ok:
            np.save(target/(stem+".npy"), np.stack(acts))
        write_json(out / "records" / (stem+".json"), row)
        print(stem, route, "best", round(best,3), "native", ok, flush=True)
        return row


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--poses", type=parse_layouts, default=[0])
    p.add_argument("--routes", type=lambda v:[int(i) for i in v.split(",")])
    p.add_argument("--count", type=int, default=36)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--output", type=Path, default=Path("bank"))
    p.add_argument("--camera", action="store_true")
    p.add_argument("--seed", type=int, default=20261120)
    args=p.parse_args()
    grid=[Route(m,t,y,f) for m,t,y,f in product(("rim","hook","press"), (-55,0,28,48,67,85),(-90,0,90,180),(-35,0,35))]
    np.random.default_rng(2039).shuffle(grid)
    ids=args.routes if args.routes is not None else list(range(args.count))
    if any(p >= 28 for p in args.poses) and not (args.camera and args.routes):
        raise ValueError("Tune routes only on source training layouts")
    out=args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    for sub in ("records","Sawyer","drafts"):
        (out/sub).mkdir()
    jobs=[(out,pose,args.seed+pose,i,grid[i],args.camera) for pose in args.poses for i in ids]
    index=dict(suite="libero_goal",task_id=5,case="BASE",bodies=["Sawyer"],poses=args.poses,
               code_revision=os.environ.get("MOTOR_CODE_REVISION"),attempts=[])
    with ProcessPoolExecutor(args.workers,mp_context=multiprocessing.get_context("spawn")) as pool:
        for r in pool.map(trial,jobs):
            index["attempts"].append(r)
            write_json(out/"index.json",index)


if __name__ == "__main__":
    main()
