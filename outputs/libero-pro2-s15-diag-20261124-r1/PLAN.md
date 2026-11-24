# Track 2: LIBERO-Pro2.0 S15 / S16 Camera Shift Diagnosis

## Research question
On LIBERO-Pro2.0 `S15` (camera extrinsics shift), frozen $\pi_{0.5}$ collapses from ~85% on `BASE` to **0/48 (`0.0%`)** even with exact commands (zero synthetic command shift). On `S16` (camera intrinsics/distortion), it drops to **39.6%**. Why does `S15` cause a complete collapse, and how much of that failure is caused by the **external camera (`agentview`)** vs. the **wrist camera (`robot0_eye_in_hand`)**?

## Falsifiable hypotheses
1. **Wrist-camera displacement dominates `S15` collapse:** `S15` applies `camera_offset_xyz = [0.1, -0.1, 0.07]` (15.8 cm translation!) to *both* `agentview` and `robot0_eye_in_hand`. Because the wrist camera sits ~5 cm from the gripper fingertips, a 15.8 cm shift moves the gripper and target object out of the wrist field of view, breaking close-range grasping even if `agentview` is intact.
2. **2D projective rectification vs. 3D viewpoint shift:** For `agentview` (and for `S16` intrinsics/crop/shift), a calibrated planar homography or inverse-intrinsic warp may restore the frozen VLA's visual input; for a 15.8 cm wrist offset, 2D warping cannot recover occluded/out-of-frame pixels.

## Least-expensive first step (CPU inspection, 0 GPUh)
Inspect `S15` and `S16` across all 8 official held-out LIBERO-Pro2.0 tasks on Panda:
- Record exact `agentview` and `robot0_eye_in_hand` camera positions, orientations, and FOVs under `BASE`, `S15`, and `S16`.
- Render matched initial frames and measure: (a) gripper visibility in wrist image, (b) pixel-space displacement of the grip site in `agentview` and `robot0_eye_in_hand` between `BASE` and `S15`/`S16`, and (c) whether `S15` parameters are identical or task-specific across all 8 tasks.
- Save compact diagnostic JSON and a side-by-side contact sheet for visual verification before launching the factorial GPU closed-loop rollout (`BASE` / `S15_full` / `S15_ext_only` / `S15_wrist_only` / `S15_rectified`).
