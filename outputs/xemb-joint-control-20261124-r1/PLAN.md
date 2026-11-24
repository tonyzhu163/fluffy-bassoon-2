# Track 1: Cross-embodiment native joint-position transfer (CPU diagnostic)

## Research question
When transferring a frozen Cartesian VLA ($\pi_{0.5}$, trained on Panda `OSC_POSE`) to unseen robot morphologies (`Kinova3` 7-DoF continuous hinges, `IIWA` 7-DoF, `UR5e` 6-DoF non-redundant) under their native `JOINT_POSITION` controller (`kp=50`, critical damping, `±0.05 rad` relative setpoints at 20 Hz), why did naive 1-step bounded IK (`ik_naive`) saturate 94.9% of controls on Panda and drop to 1/6 on Kinova3 source-proposal replay?

## Falsifiable hypotheses
1. **Per-joint clipping distortion:** Independent box clipping in `lsq_linear` rotates the realized end-effector velocity vector when one joint hits `±0.05 rad`. Uniformly scaling the unconstrained damped-LS step to fit the joint box (`ik_dir_scaled`) should preserve Cartesian direction and reduce path error across arms.
2. **PD-servo velocity lag:** Under `kp=50` critical damping, a 50 ms step closes only $\alpha = 1 - (1 + \omega T)e^{-\omega T} \approx 4.96\%$ of a setpoint from rest while carrying $\beta = e^{-\omega T} \approx 70.2\%$ of current velocity. Accounting for current joint velocity $v$ (`ik_pd_vel`) and/or maintaining a persistent Cartesian pose target (`ik_pose_track`, matching OSC's persistent target semantics) should reduce lag and rescue cross-arm replay successes on Kinova3, IIWA, and 6-DoF UR5e.

**What counts against these hypotheses:** Equal or worse native replay success and TCP tracking error compared to `ik_naive` across the four arms. Note: open-loop replay of Panda proposals is a cheap CPU screen (0 GPUh); any promising decoder must be confirmed in closed-loop frozen $\pi_{0.5}$ rollouts on Baige Distributed Training.

## Experimental setup
- **Arms:** `Panda` (7-DoF limited), `Kinova3` (7-DoF, 4 continuous + 3 limited), `IIWA` (7-DoF limited), `UR5e` (6-DoF limited). All use `PandaGripper` and factory `JOINT_POSITION` (`kp=50`, `±0.05 rad`, 20 Hz).
- **Trajectories:** The 6 archived successful Panda OSC_POSE source trajectories (`libero_spatial/1` and `libero_goal/1`, layouts `0, 2, 3`) from `outputs/frozen-vla-native-controller-20261123-r1/gpu01/`.
- **Decoders:** `ik_naive`, `ik_dir_scaled`, `ik_pd_vel`, `ik_pose_track`.
- **Metrics:** Native replay success (binary BDDL reward), controls to success, saturation fraction, mean/max TCP position error (mm) vs. source Panda OSC trajectory.
- **Budget:** Baige CPU only (0 GPUh, <10 CPU minutes).
