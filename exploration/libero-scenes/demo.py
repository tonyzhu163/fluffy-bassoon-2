"""Scripted pick-and-place demonstrations on admitted scenes. Never loads a policy.

For every admitted scene and every behaviour in its manifest (a bowl placed on
a target), a waypoint script grasps the bowl by its rim, carries it, lowers it
until it rests, and lets go. Each episode is restored from the admitted state,
so all behaviours of a scene start from identical physics.

    python demo.py --source <behavior_prompting checkout> --scenes <dir> \
        --admitted <admit.py output> --output <new dir>

Actions are relative end-effector commands, the convention of the released
demonstrations. The arm follows a slowly moving position reference, the servo
that reached native goals in `outputs/libero-gen-prompt-repair-20261111-r1`
(teacher05). The rim grasp is new and is the uncertain part: `--rim`,
`--rim-radius` and `--depth` set where the fingers close, so try them on a few
scenes with `--limit` before a full collection. Run inside an allocated Linux
job. The control flow was exercised against a kinematic stand-in only; this
file has never run in the simulator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

OPEN, CLOSE = -1.0, 1.0
MAX_CONTROLS = 900
ROTATION_STEP = 0.025  # radians of orientation correction per control


@dataclass
class Params:
    rim: float = 0.85  # grasp point, as a fraction of the bowl's reported horizontal radius
    rim_radius: float | None = None  # or in metres, overriding `rim`
    depth: float = 0.02  # fingertips this far below the bowl's reported top
    carry: float = 0.15  # carry height above the grasp height
    travel: float = 0.005  # reference motion per control (20 Hz), metres: 0.10 m/s
    descent: float = 0.0015  # reference motion per control while lowering onto the target
    tolerance: float = 0.008  # arrival distance
    grip_controls: int = 15


class Failed(Exception):
    """The script gave up at a named stage. The episode is still recorded."""


class LiberoSim:
    """The few simulator reads and writes the script needs."""

    def __init__(self, env):
        self.env, self.obs, self.orientation = env, None, None

    @property
    def controller(self):
        return self.env.robots[0].controller

    def restore(self, state, seed):
        self.env.seed(int(seed))
        self.env.env.reset()  # clears robosuite's episode counter before restoring
        raw = self.env.regenerate_obs_from_state(state)
        c = self.controller
        c.update(force=True)
        c.goal_pos, c.goal_ori = c.ee_pos.copy(), c.ee_ori_mat.copy()
        self.orientation = c.ee_ori_mat.copy()  # held for the whole episode
        self.obs = self.env.prepare_observation(raw)
        return self.obs

    def step(self, reference, grip):
        """One control toward `reference`; returns the observation and the action actually sent."""
        from robosuite.utils.transform_utils import mat2quat, quat2axisangle
        c = self.controller
        c.update(force=True)
        turn = quat2axisangle(mat2quat(self.orientation @ c.ee_ori_mat.T))
        turn = turn * min(1.0, ROTATION_STEP / max(float(np.linalg.norm(turn)), 1e-12))
        delta = np.r_[np.asarray(reference) - c.ee_pos, turn]
        # The controller's own input scaling, inverted, then its native bounds.
        scaled = ((delta - (c.output_max + c.output_min) / 2)
                  / ((c.output_max - c.output_min) / (c.input_max - c.input_min))
                  + (c.input_max + c.input_min) / 2)
        action = np.r_[np.clip(scaled, c.input_min, c.input_max), grip]
        self.obs = self.env.step(action)[0]
        return self.obs, action

    def eef(self):
        self.controller.update(force=True)
        return self.controller.ee_pos.copy()

    def body(self, name):
        return self.env.sim.data.body_xpos[self.env.env.obj_body_id[name]].copy()

    def target(self, name):
        """World position of a placement target: an object, or a named region."""
        if name in self.env.env.obj_body_id:
            return self.body(name)
        return np.array(self.env.sim.data.get_site_xpos(name), dtype=float)

    def shape(self, name):
        """The object's own horizontal radius and the height of its top above its body origin."""
        obj = self.env.env.objects_dict[name]
        return float(obj.horizontal_radius), float(obj.top_offset[2])

    def finger_axis(self):
        """Unit table-plane direction from one fingertip to the other."""
        sim, gripper = self.env.sim, self.env.robots[0].gripper
        a, b = (sim.data.geom_xpos[sim.model.geom_name2id(gripper.important_geoms[k][0])]
                for k in ("left_fingerpad", "right_fingerpad"))
        axis = (a - b)[:2]
        return axis / np.linalg.norm(axis)

    def holds(self, predicate):
        return bool(self.env.env._eval_predicate(list(predicate)))

    def state(self):
        return self.env.get_sim_state().copy()


class Episode:
    def __init__(self, sim, params: Params, first_obs):
        self.sim, self.p = sim, params
        self.command = sim.eef()  # the persistent position reference
        self.observations, self.actions, self.states = [first_obs], [], [sim.state()]
        self.references, self.trace, self.watch = [], [], None

    def act(self, grip):
        if len(self.actions) >= MAX_CONTROLS:
            raise Failed("out_of_controls")
        obs, action = self.sim.step(self.command, grip)
        self.observations.append(obs)
        self.actions.append(action)
        self.references.append(self.command.copy())
        self.states.append(self.sim.state())
        if self.watch is not None:
            self.trace.append(int(self.sim.holds(self.watch)))

    def move(self, goal, grip, stage):
        """Walk the commanded position to `goal` at a bounded speed, then wait for the arm to arrive."""
        goal = np.asarray(goal, dtype=float)
        while np.linalg.norm(goal - self.command) > 1e-9:
            delta = goal - self.command
            self.command = self.command + delta * min(1.0, self.p.travel / np.linalg.norm(delta))
            self.act(grip)
        for _ in range(60):
            if np.linalg.norm(self.sim.eef() - goal) < self.p.tolerance:
                return
            self.act(grip)
        raise Failed(f"{stage}_not_reached")

    def grip(self, value):
        for _ in range(self.p.grip_controls):
            self.act(value)

    def lower_until_rest(self, bowl, floor):
        """Lower the held bowl until it stops following the command, or the hand reaches `floor`."""
        heights = [self.sim.body(bowl)[2]]
        while self.command[2] > floor:
            self.command = self.command - np.array([0.0, 0.0, self.p.descent])
            self.act(CLOSE)
            heights.append(self.sim.body(bowl)[2])
            # Twelve controls command 1.8 cm of descent; a resting bowl moves almost none of it.
            if len(heights) > 24 and heights[-13] - heights[-1] < 0.002:
                return
            if self.sim.eef()[2] - self.command[2] > 0.03:  # the hand is blocked
                return


def demonstrate(sim, behaviour, others, params: Params, first_obs):
    """Run one pick-and-place. Returns the episode and the stage it failed at, or None."""
    ep = Episode(sim, params, first_obs)
    ep.watch = behaviour["predicate"]
    bowl, target = behaviour["bowl"], behaviour["target"]
    start = sim.body(bowl)
    radius, top = sim.shape(bowl)
    reach = params.rim_radius if params.rim_radius is not None else params.rim * radius
    axis = sim.finger_axis()
    # Grasp the rim on whichever side of the bowl is farther from other objects.
    sides = [start[:2] + s * reach * axis for s in (1.0, -1.0)]
    grasp_xy = max(sides, key=lambda xy: min((np.linalg.norm(xy - sim.body(o)[:2]) for o in others), default=1.0))
    z_grasp = start[2] + top - params.depth
    z_carry = max(z_grasp + params.carry, sim.target(target)[2] + 0.15)
    measured = dict(bowl_radius=radius, bowl_top=top, grasp_reach=reach, finger_axis=axis.tolist(),
                    z_grasp=z_grasp, z_carry=z_carry)
    try:
        ep.move([*ep.command[:2], z_carry], OPEN, "rise")
        ep.move([*grasp_xy, z_carry], OPEN, "above_bowl")
        ep.move([*grasp_xy, z_grasp], OPEN, "at_rim")
        ep.grip(CLOSE)
        ep.move([*grasp_xy, z_carry], CLOSE, "lift")
        if sim.body(bowl)[2] - start[2] < 0.03:
            raise Failed("grasp_slipped")
        # Keep the measured hand-to-bowl offset, so the bowl's centre lands on the target's.
        held = sim.eef()[:2] - sim.body(bowl)[:2]
        ep.move([*(sim.target(target)[:2] + held), z_carry], CLOSE, "above_target")
        if sim.body(bowl)[2] - start[2] < 0.03:
            raise Failed("dropped_in_transit")
        ep.lower_until_rest(bowl, floor=z_grasp - 0.005)
        ep.grip(OPEN)
        ep.move([*ep.command[:2], z_carry], OPEN, "retreat")
        for _ in range(20):
            ep.act(OPEN)
    except Failed as failure:
        return ep, str(failure), measured
    return ep, None, measured


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source", type=Path, required=True, help="behavior_prompting checkout")
    ap.add_argument("--scenes", type=Path, required=True, help="output directory of scenes.py")
    ap.add_argument("--admitted", type=Path, required=True, help="output directory of admit.py")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--size", type=int, choices=(128, 256), default=256, help="camera pixels")
    ap.add_argument("--limit", type=int, help="only the first N admitted scenes of this shard")
    ap.add_argument("--shard", default="0/1", help="i/n: scenes whose index is i modulo n")
    for name, value in asdict(Params()).items():
        ap.add_argument(f"--{name.replace('_', '-')}", type=type(value) if value is not None else float, default=value)
    args = ap.parse_args()
    params = Params(**{k: getattr(args, k) for k in asdict(Params())})
    part, parts = map(int, args.shard.split("/"))
    args.output.mkdir(parents=True, exist_ok=False)

    import yaml
    base = args.source / "behavior_prompting/train_network"
    libero = args.source / "deps/LIBERO/libero/libero"
    (args.output / "libero_config").mkdir()
    (args.output / "libero_config/config.yaml").write_text(yaml.safe_dump({
        "benchmark_root": str(libero), "assets": str(libero / "assets"),
        "bddl_files": str(base / "env/libero/bddl_files"),
        "init_states": str(base / "env/libero/init_files"), "datasets": str(args.output)}))
    os.environ["LIBERO_CONFIG_PATH"] = str(args.output / "libero_config")
    from behavior_prompting.train_network.env.libero.env_wrapper import ControlEnvForRunner
    from behavior_prompting.train_network.utils.libero_util import discover_and_register_benchmarks
    discover_and_register_benchmarks()

    scenes = {s["index"]: s for s in json.loads((args.scenes / "manifest.json").read_text())["scenes"]}
    admitted = [r for r in json.loads((args.admitted / "result.json").read_text())["records"]
                if r["admitted"] and r["index"] % parts == part][: args.limit]
    records = []
    for admission in admitted:
        scene = scenes[admission["index"]]
        assert admission["bddl_sha256"] == scene["bddl_sha256"], "scene changed since admission"
        state = np.load(args.admitted / f"scene-{scene['index']:04d}/state.npy")
        assert hashlib.sha256(np.ascontiguousarray(state).tobytes()).hexdigest() == admission["state_sha256"]
        folder = args.output / f"scene-{scene['index']:04d}"
        folder.mkdir()
        env = ControlEnvForRunner(bddl_file_name=str(args.scenes / scene["bddl"]), camera_heights=args.size,
                                  camera_widths=args.size, has_offscreen_renderer=True, has_renderer=False,
                                  use_camera_obs=True, control_delta=True, render_gpu_device_id=0,
                                  ignore_done=True, horizon=1000)
        try:
            env.env.hard_reset = False
            sim = LiberoSim(env)
            assert sim.controller.use_delta
            movable = [o["name"] for o in scene["objects"]]
            for k, behaviour in enumerate(scene["behaviours"]):
                first = sim.restore(state, scene["seed"])
                assert np.array_equal(sim.state(), state), "restore did not reproduce the admitted state"
                others = [n for n in movable if n != behaviour["bowl"]]
                ep, failure, measured = demonstrate(sim, behaviour, others, params, first)
                # Every behaviour's native predicate on the final state: the outcome under each hypothesis.
                outcomes = [int(sim.holds(b["predicate"])) for b in scene["behaviours"]]
                arrays = {key: np.stack([o[key] for o in ep.observations]) for key in ep.observations[0]}
                arrays.update(actions=np.stack(ep.actions), references=np.stack(ep.references),
                              states=np.stack(ep.states), intended_predicate=np.array(ep.trace))
                np.savez_compressed(folder / f"demo-{k:02d}.npz", **arrays)
                record = dict(scene=scene["index"], behaviour=k, bowl=behaviour["bowl"], target=behaviour["target"],
                              target_kind=next(t["kind"] for t in scene["targets"] if t["name"] == behaviour["target"]),
                              controls=len(ep.actions), failed_at=failure, success=bool(outcomes[k]),
                              first_true_control=int(np.argmax(ep.trace)) if any(ep.trace) else None,
                              only_intended=outcomes[k] == 1 and sum(outcomes) == 1, outcomes=outcomes, **measured)
                (folder / f"demo-{k:02d}.json").write_text(json.dumps(record, indent=2) + "\n")
                records.append(record)
                print("DEMO", scene["index"], k, behaviour["target"], failure or "completed", outcomes[k], flush=True)
        finally:
            env.close()

    def tally(key):
        out: dict[str, list[int]] = {}
        for r in records:
            out.setdefault(str(r[key]), [0, 0])
            out[str(r[key])][0] += int(r["success"])
            out[str(r[key])][1] += 1
        return out

    summary = dict(policy_loaded=False, params=asdict(params), shard=args.shard, scenes=len(admitted),
                   episodes=len(records), successes=sum(r["success"] for r in records),
                   success_by_target_kind=tally("target_kind"), episodes_by_failed_stage=tally("failed_at"))
    (args.output / "result.json").write_text(json.dumps(dict(summary, records=records), indent=2) + "\n")
    print("DEMO_RESULT", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
