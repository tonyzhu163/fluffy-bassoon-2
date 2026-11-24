# Track 1 CPU diagnostic: cross-arm native JOINT_POSITION decoders

Zero-GPU CPU diagnostic on `baige-dev` (`taskset -c 50,51`, 205.8 s) replaying the 6 archived Panda OSC_POSE trajectories (`libero_spatial/1` and `libero_goal/1`, layouts `0, 2, 3`) across four robot morphologies (`Panda` 7-DoF, `Kinova3` 7-DoF continuous hinges, `IIWA` 7-DoF, `UR5e` 6-DoF non-redundant) under factory `JOINT_POSITION` (`kp=50`, `±0.05 rad` at 20 Hz). [Plan](PLAN.md) · [Raw results](results.json)

## Summary (6 trajectories per arm, mean TCP error vs. Panda OSC reference, saturation fraction)

| Arm (DoF) | `ik_naive` | `ik_dir_scaled` | `ik_pd_vel` | `ik_pose_track` |
| --- | ---: | ---: | ---: | ---: |
| Panda (7) | 0/6 (130.1 mm, sat 0.89) | 0/6 (166.1 mm, sat 0.88) | 0/6 (130.7 mm, sat 0.94) | 0/6 (**122.0 mm**, sat 0.99) |
| Kinova3 (7, 4 cont.) | 0/6 (116.2 mm, sat 0.89) | 0/6 (165.3 mm, sat 0.89) | 0/6 (118.0 mm, sat 0.92) | 0/6 (**101.1 mm**, sat 0.99) |
| IIWA (7) | 0/6 (121.7 mm, sat 0.87) | 0/6 (165.5 mm, sat 0.87) | 0/6 (121.3 mm, sat 0.93) | 0/6 (**108.6 mm**, sat 0.99) |
| UR5e (6) | 0/6 (131.4 mm, sat 0.88) | 0/6 (165.5 mm, sat 0.88) | 0/6 (149.8 mm, sat 0.93) | 0/6 (**127.9 mm**, sat 0.99) |

## What this falsifies and establishes
1. **Direction-preserving uniform scaling (`ik_dir_scaled`) is falsified:** Scaling down all joints when one joint hits `±0.05 rad` worsened TCP error by `+34 to +49 mm` on every arm. The bottleneck is joint-velocity bandwidth under `kp=50` PD, not per-joint clipping angle distortion.
2. **Persistent Cartesian pose target (`ik_pose_track`) reduced tracking error on all 4 arms** relative to `ik_naive` (by `-8.1 mm` Panda, `-15.1 mm` Kinova3, `-13.1 mm` IIWA, `-3.5 mm` UR5e), while `ik_pd_vel` without a persistent pose target did not help.
3. **Open-loop OSC replay is time-censored (0/6 everywhere):** `JOINT_POSITION` takes ~1.8–2× more controls than `OSC_POSE` to cover the same path. Ranking decoders on native task reward requires closed-loop frozen $\pi_{0.5}$ rollouts on Baige Distributed Training across `Panda`, `Kinova3`, `IIWA`, and `UR5e` (comparing `ik_naive` vs `ik_pose_track` and a saturation/posture-aware multi-step decoder).

## Closed-loop native GPU evaluation (`job-yeri98t5c50a`, 1× RTX 4090, 0.139 GPUh, exit 0)

Evaluated frozen $\pi_{0.5}$ in closed loop on unseen **Kinova3 (7-DoF continuous hinges)** and **UR5e (6-DoF non-redundant)** across the same 6 starts (`libero_spatial/1` & `libero_goal/1`, layouts `0, 2, 3`, 36 native episodes). [Live summary](live-summary.json)

| Arm (DoF) | `OSC_POSE` | `ik_naive` (`JOINT_POSITION`) | `ik_pose_track` (`JOINT_POSITION`) |
| --- | ---: | ---: | ---: |
| Kinova3 (7) | 6/6 (95.2 ctrl) | **5/6** (168.3 ctrl, sat 0.86) | 3/6 (206.3 ctrl, sat 1.00) |
| UR5e (6) | 6/6 (97.3 ctrl) | **5/6** (175.8 ctrl, sat 0.87) | 1/6 (245.0 ctrl, sat 1.00) |

**Conclusion:** Plain 1-step bounded differential IK (`ik_naive`) already achieves **5/6 (83.3%)** zero-shot on both unseen 7-DoF Kinova3 and 6-DoF UR5e, leaving minimal headroom for a learned joint-space adapter. Despite lower open-loop TCP error, `ik_pose_track` saturated ~100% of steps and regressed closed-loop native success on both arms.
