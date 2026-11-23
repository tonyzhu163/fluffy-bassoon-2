"""Privileged path-timing control using only the successful source joint trajectory."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from cpu import CASES, TAPES, save
from native import JOINT, JOINT_SCALE, OSC, Pair


def run(out):
    result = {}
    for suite, (seed,n,sha) in CASES.items():
        path = TAPES / (suite + "-actions.npy")
        assert hashlib.sha256(path.read_bytes()).hexdigest() == sha
        tape = np.clip(np.load(path), -1, 1)
        pair = Pair(suite, 0, seed)
        try:
            src = pair.start(OSC)
            qs, tcp = [src["robot0_joint_pos"].copy()], [src["robot0_eef_pos"].copy()]
            for action in tape:
                src, reward = pair.step(OSC, action)
                qs.append(src["robot0_joint_pos"].copy())
                tcp.append(src["robot0_eef_pos"].copy())
                if reward:
                    break
            if not reward:
                result[suite] = dict(eligible=False)
                continue
            ctl = pair.envs[JOINT].robots[0].controller
            freq = pair.envs[JOINT].env.control_freq
            dt = 1.0/freq
            # Predicted steady joint speed at full-scale delta command for the stock PD law.
            # A timing reference; friction, contacts, torque clipping and transients still matter.
            v = ctl.kp * JOINT_SCALE / ctl.kd
            dq = np.diff(qs, axis=0)
            duration = np.maximum(dt, np.max(np.abs(dq) / v, axis=1))
            knots = np.r_[0.0, np.cumsum(duration)]
            gripper_switch = next(i for i in range(1,len(qs)-1) if tape[i,6]>0>=tape[i-1,6])
            rows = {}
            for method, grid in (("double_duration",np.arange(len(qs))*2*dt), ("rate_limited",knots)):
                obs = pair.start(JOINT)
                rewards, qerrs, terrs, phases, actions = [], [], [], [], []
                for t in range(pair.horizon):
                    phase = np.interp((t+1)*dt,grid,np.arange(len(qs)))
                    seg = min(int(phase),len(dq)-1)
                    vref = dq[seg] / (grid[seg+1]-grid[seg]) if phase < len(dq) else np.zeros(7)
                    qref = qs[seg] + (phase-seg)*dq[seg]
                    pref = tcp[seg] + (phase-seg)*(tcp[seg+1]-tcp[seg])
                    a = np.r_[np.clip((qref-obs["robot0_joint_pos"])/JOINT_SCALE + vref/v,-1,1),
                              tape[min(int(np.interp(t*dt,grid,np.arange(len(qs)))),len(dq)-1),6]]
                    obs, reward = pair.step(JOINT, a)
                    qerrs.append(np.max(np.abs(obs["robot0_joint_pos"]-qref)))
                    terrs.append(np.linalg.norm(obs["robot0_eef_pos"]-pref)*1000)
                    phases.append(phase)
                    rewards.append(reward)
                    actions.append(a)
                    if reward:
                        break
                idx = next((i for i,a in enumerate(actions) if a[-1]>0),None)
                rows[method] = dict(native_success=bool(rewards[-1]), controls=len(rewards),
                        goal_time_s=grid[-1], reward=rewards, native_actions=actions, phase=phases,
                        max_joint_error_rad=qerrs, tcp_error_mm=terrs, first_close_index=idx,
                        error_at_close_mm=terrs[idx] if idx is not None else None)
            result[suite] = dict(eligible=True, horizon=pair.horizon, source_controls=len(qs)-1,
                    source_close_index=gripper_switch, kp=ctl.kp,kd=ctl.kd,speed_model_rad_s=v,rows=rows)
        finally:
            pair.close()
    save(out/"pace-oracle.json",result)


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("out",type=Path)
    run(p.parse_args().out)
