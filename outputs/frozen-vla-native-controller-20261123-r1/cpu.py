"""Paired native pulse responses, matched-state differentials and frozen-tape replays."""
import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from native import Decoder, JOINT, OSC, OSC_SCALE, JOINT_SCALE, Pair, delta, digest, restore, snapshot

TAPES = Path("/mowenjie/fluffy-bassoon-2/outputs/libero-meta-learning-20261117-r1/devref01/reference")
CASES = {"libero_spatial": (171117, 106, "1b902c87c88d196589e2d9648c130a694aa86110a8e4e3fbb25f836116543ea8"),
         "libero_goal": (171118, 87, "3516847fe14e917bef630c89cf13738ac4020e913502e1dc2500ee2fa9c1f0a2")}


def save(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False, default=lambda x: x.tolist()) + "\n")
    temp.replace(path)


def sensed(raw):
    return raw["robot0_eef_pos"].copy(), Rotation.from_quat(raw["robot0_eef_quat"]).as_matrix()


def probe(pair, name, state, action):
    env = pair.envs[name]
    restore(env, state)
    raw = env.begin_perturbation_rollout()
    before = sensed(raw)
    after, reward = pair.step(name, action)
    return delta(*before, *sensed(after)) / OSC_SCALE


def pulses(pair, name, state, grip):
    n = 6 if name == OSC else 7
    zero = probe(pair, name, state, np.r_[np.zeros(n), grip])
    cols = []
    samples = []
    for axis in np.eye(n):
        plus = probe(pair, name, state, np.r_[axis * 0.5, grip])
        minus = probe(pair, name, state, np.r_[-axis * 0.5, grip])
        cols.append(plus - minus)
        samples.append([minus, plus])
    return np.stack(cols, axis=1), zero, samples


def calibrate():
    pair = Pair("libero_goal", 1, 861701)
    try:
        b, z1, s1 = pulses(pair, OSC, pair.opening[OSC], -1)
        a, z2, s2 = pulses(pair, JOINT, pair.opening[JOINT], -1)
        sv = np.linalg.svd(a, compute_uv=False)
        assert np.linalg.matrix_rank(a) == 6 and np.linalg.matrix_rank(b) == 6
        return dict(osc_matrix=b.tolist(), joint_matrix=a.tolist(), offset=(z1-z2).tolist(),
                    damping=float(0.02 * sv[0]), singular_values=sv.tolist(),
                    osc_singular_values=np.linalg.svd(b, compute_uv=False).tolist(),
                    source="libero_goal/1/BASE/layout1/861701", reset=pair.full_hash,
                    signed_probes={OSC: s1, JOINT: s2}, zeros={OSC: z1, JOINT: z2},
                    pulse_amplitude=0.5, controls={OSC: 13, JOINT: 15})
    finally:
        pair.close()


def differential(pair, state, cmd, calib, index, osc_goal):
    # Independent one-step counterfactuals from exactly the same integrated source posture.
    env = pair.envs[JOINT]
    restore(env, state)
    raw = env.begin_perturbation_rollout()
    q = raw["robot0_joint_pos"].copy()
    jac = pair.chain.jac(q)
    eps = 1e-5
    fd = np.stack([delta(*pair.chain.fk(q - e * eps), *pair.chain.fk(q + e * eps)) / (2 * eps)
                   for e in np.eye(7)], axis=1)
    reference = probe(pair, OSC, state, cmd)
    results = {}
    matrix, zero, _ = pulses(pair, JOINT, state, cmd[-1])
    for method in ("hold", "ik", "calibrated", "oracle"):
        if method == "oracle":
            # Query-only, non-deployable achieved OSC motion and local joint response.
            damping = 0.02 * np.linalg.svd(matrix, compute_uv=False)[0]
            a = np.r_[pair.chain.solve(q, matrix, reference - zero, damping=damping), cmd[-1]]
        else:
            d = Decoder(pair.chain, q, method, calib)
            d.goal_ori = osc_goal.copy()
            a = d.action(q, cmd)
        y = probe(pair, JOINT, state, a)
        err = (y - reference) * OSC_SCALE
        results[method] = dict(native=a, response=y, error_mm=1000*np.linalg.norm(err[:3]),
                               error_deg=np.rad2deg(np.linalg.norm(err[3:])))
    return dict(control=index, q=q, canonical=cmd, source_response=reference, methods=results,
                fd_max_abs=np.max(np.abs(fd-jac), axis=1), singular_values=np.linalg.svd(jac, compute_uv=False))


def main(out):
    import mujoco
    from importlib.metadata import version
    start = time.monotonic()
    cal = calibrate()
    save(out/"calibration.json", cal)
    summary = {}
    for suite, (seed, n, sha) in CASES.items():
        path = TAPES / (suite + "-actions.npy")
        assert hashlib.sha256(path.read_bytes()).hexdigest() == sha
        archive = json.loads((TAPES / (suite + ".json")).read_text())
        assert archive["success"] and archive["controls"] == n
        assert archive["case"] == dict(suite=suite, variant="BASE", seed=seed, task_id=1, init_state_id=0)
        original = np.load(path)
        tape = np.clip(original, -1, 1)
        assert tape.shape == (n, 7)
        pair = Pair(suite, 0, seed)
        try:
            sample_at = set(np.linspace(0, n - 2, 8, dtype=int).tolist())
            states, probes, traces = {}, [], {}
            robot = pair.envs[OSC].robots[0]
            for method in (OSC, "hold", "ik", "calibrated"):
                name = OSC if method == OSC else JOINT
                obs = pair.start(name)
                decoder = Decoder(pair.chain, obs["robot0_joint_pos"], method, cal)
                qpath = [obs["robot0_joint_pos"].copy()]
                poses = [sensed(obs)[0]]
                actions, rewards = [], []
                for i, cmd in enumerate(tape):
                    if name == OSC and i in sample_at:
                        states[i] = snapshot(pair.envs[name])
                    a = decoder.action(obs["robot0_joint_pos"], cmd)
                    obs, reward = pair.step(name, a)
                    actions.append(a)
                    rewards.append(reward)
                    qpath.append(obs["robot0_joint_pos"].copy())
                    poses.append(sensed(obs)[0])
                    if reward:
                        break
                traces[method] = dict(controller=name, native_actions=actions, rewards=rewards, q=qpath,
                                      tcp_xyz=poses, native_success=bool(rewards[-1]), controls=len(rewards))
            # Independent feasibility diagnostic with privileged future source joint/hand trajectory.
            if traces[OSC]["native_success"]:
                qs = np.asarray(traces[OSC]["q"])
                joint = pair.envs[JOINT].robots[0].controller
                obs = pair.start(JOINT)
                rewards, actions = [], []
                # Half-speed traversal leaves time for the stock, bounded joint servo to follow the path.
                for t in range(pair.horizon):
                    times = np.arange(len(qs))
                    phase = min((t + 1) / 2, len(qs) - 1)
                    ref = np.array([np.interp(phase, times, col) for col in qs.T])
                    vref = (qs[min(int(phase)+1,len(qs)-1)] - qs[int(phase)]) / 0.10
                    a = np.r_[np.clip((ref - obs["robot0_joint_pos"] + joint.kd*vref/joint.kp) / JOINT_SCALE,
                                      -1, 1), tape[min(t//2,len(qs)-2),-1]]
                    obs, reward = pair.step(JOINT, a)
                    actions.append(a)
                    rewards.append(reward)
                    if reward:
                        break
                traces["trajectory_oracle"] = dict(controller=JOINT, native_actions=actions, rewards=rewards,
                                                   native_success=bool(rewards[-1]), controls=len(rewards))
            for i, state in states.items():
                probes.append(differential(pair, state, tape[i], cal, i, state["goal_ori"]))
            save(out/(suite+".json"), dict(tape_sha256=sha, historical=archive, instruction=pair.prompt,
                 bddl_sha256=hashlib.sha256(pair.bddl.read_bytes()).hexdigest(),
                 bddl_goal=pair.bddl.read_text().split("(:goal")[-1].strip(), horizon=pair.horizon,
                 settled_integration_hash=pair.full_hash, released_row_hash=pair.row_hash,
                 joint_order=robot.robot_joints, arm=type(robot.robot_model).__name__,
                 grip_site=robot.controller.eef_name, clipped_by_coordinate=(tape!=original).sum(axis=0),
                 controls=traces, differential=probes))
            summary[suite] = dict(source_competent=traces[OSC]["native_success"],
                replay={k:[v["native_success"],v["controls"]] for k,v in traces.items()},
                differential_mean={k:np.mean([[p["methods"][k]["error_mm"],p["methods"][k]["error_deg"]]
                                              for p in probes],axis=0) for k in probes[0]["methods"]},
                fd_max=np.max([p["fd_max_abs"] for p in probes],axis=0))
            print(suite, summary[suite]["replay"], flush=True)
        finally:
            pair.close()
    save(out/"summary.json", dict(suites=summary, seconds=time.monotonic()-start, hostname=platform.node(),
         kernel=platform.release(), affinity=sorted(os.sched_getaffinity(0)), mujoco=mujoco.__version__,
         robosuite=version("robosuite"), cuda_visible=os.environ["CUDA_VISIBLE_DEVICES"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    main(parser.parse_args().out)
