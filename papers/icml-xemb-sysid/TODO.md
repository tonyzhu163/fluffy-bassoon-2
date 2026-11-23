# SymMeta ICML submission: open work

From the ICML-style audit on 2026-11-23 (soundness, presentation, significance, originality). Items are ordered by how much they change the accept/reject case. Text-only fixes already applied to `main.tex` are listed at the end.

## Blocking: experiments

- [ ] **Kinematic baseline in closed loop.** Forward kinematics plus planar alignment gets 1.93° yaw MAE on the 96 UR5e support traces, against 6.8° for SymMeta ([world control](../../outputs/symmeta-camera-command-20261123-r1/WORLD.md)). Run it as a method on the locked 96-query panels for all four arm–VLA pairs and report task success beside SymMeta. If it matches or wins, narrow the paper to the setting where joint encoders, kinematic model or base frame are unavailable or wrong, and say so in the abstract.
- [ ] **Calibrated classical visual baseline.** Horn–Schunck and DIS are uncalibrated. Add a tracker with known intrinsics/extrinsics (or a fiducial on the gripper) so the comparison shows what the learned reader adds over standard calibration.
- [ ] **Camera pose.** The reader's yaw estimate follows the camera: ±90° image roll gives 84.5° / 101.2° error, ±30° azimuth shifts the estimate by 28–30° ([result](../../outputs/symmeta-camera-command-20261123-r1/RESULT.md)). Either (a) measure task success under moderate roll/azimuth and report it as a limitation with numbers, or (b) test the scene-fixed-reference fix proposed in that result. Offline yaw error alone is not enough for the paper.
- [ ] **Faithful ReGround comparison.** The current baseline is a command-frame affine fit on the passive trace, now labelled "Affine (proprio.)". Run the released ReGround code with its own babbling budget on at least UR5e + π₀.₅, or keep the relabelled baseline and drop any sentence that implies a result about ReGround.
- [ ] **Matched adapter comparison.** Rerun SymMeta and the demonstration-fitted adapters in one job on the same 32 queries. The table currently joins two runs with identical query seeds (SymMeta 25/32, linear adapter K=1 28/32, oracle 27/32). Report the demo-collection cost; the demos use the exact inverse transform.
- [ ] **Independent scenes.** Each panel has 8 task clusters; the three reader seeds reuse the same 96 queries and support traces. Add fresh initial layouts and randomly sampled yaw/gain values (not the fixed 12 transforms) so "OOD" covers more than the fixed 12 transforms.

## Blocking: paper

- [ ] **Method reproducibility** (`TODO(method)` in `main.tex`): training loss and labels, dataset size, CNN spec, optimizer and hyperparameters, how `w_k` and `u_z` are supervised, parameter counts for both Perceiver widths, and whether the Δz correction is used in the 3D variant.
- [ ] **Statistics** (`TODO(stats)`): query-level paired permutation tests per seed, task-level effect sizes, and an appendix table accounting for every episode.
- [ ] **Page limit.** The build is 10 pages including references and the body text now ends on page 9. Move Table 2, the scaling table and the ManiSkill3 paragraph to an appendix to reach 8 body pages.
- [ ] **Pipeline figure** (one column) showing support trace → reader → Procrustes → wrapper.
- [ ] **Bibliography.** Verify the author lists of `siddique2026reground` and `yang2026icba` against arXiv, and replace the placeholder authors of `liberopro2026`. Confirm which of ICBA and ReGround count as concurrent at the deadline.
- [ ] **Appendix.** List all nine methods of the visual-perturbation run; state that the scaling sweep is offline estimation on recorded support traces.

## Should do

- [ ] Report same-attempt (no reset) adaptation per arm in a table, not only the 9.4–27.1% range.
- [ ] Reader ablations on one panel: action-blind vs. action-conditioned reader, with and without confidence weights, with and without parallax correction, measured in task success (the parallax ablation is offline only).
- [ ] 3D: either drop to an appendix or add a second camera; 38.5% vs. 88.5% oracle and 9.4% on continuous rotations does not support a 3D claim.
- [ ] ManiSkill3: raise the sample size (about 72 episodes per method now) or move to an appendix as a negative result.
- [ ] A small hardware test (one arm, a few yaw offsets) would answer the simulation-only objection.
- [ ] `DRAFT.md`, `EXTENDED_ABSTRACT.md` and the top entries of `docs/FINDINGS.md` still use the old baseline names ("Perceiver-ICBA", "ReGround") and the old Proposition 1 wording. Update or delete the two drafts.
- [ ] Add `jax`/`flax` as an optional dependency group so the reader tests run off the cluster (they skip today when the packages are missing).

## Done in this pass (text, figures and tests only; no new experiments)

- Proposition 1 now covers command-frame end-effector displacements with commands chosen independently of the rotation. The incorrect claim that joint changes carry no information is removed, and the remark names what the proposition does not cover (joint encoders with a kinematic model, world-frame proprioception, camera extrinsics, closed-loop commands).
- Proposition 2 states the fixed-`v_k`/`w_k` condition and the modulo-2π caveat. A new remark says the equivariance is to command-frame rotations for a fixed camera, not invariance to camera pose.
- Protocol paragraph corrected: Attempt 1 is recorded on initial layout 11 and Attempt 2 runs on layout 13; 0.75 s is simulated support motion and excludes reset and inference; seeds share queries and traces.
- Baselines relabelled in text, tables and regenerated figures: "Perceiver" (ours, inspired by ICBA) and "Affine (proprio.)" (ours, inspired by ReGround). Related work describes the two published methods as they are.
- "Few-shot fine-tuning / SFT" renamed to demonstration-fitted action adapters; table now includes SymMeta, Horn–Schunck and oracle on the same 32 task–transform pairs.
- Limitations gained camera pose, the kinematic alternative, and the baseline caveat. Abstract and conclusion mention the camera-pose assumption.
- π₀ quaternion canonicalization checked in code: it runs in the shared observation path for every method, including the oracle (`eval_stage2a_live_ur5e.py`, `eval_v2_multiseed_zeroshot.py`).
- ReGround bib title corrected to the arXiv title.
- Reader tests skip when `jax`/`flax` are absent, matching the other tests in those files, and still hard-fail on the qualified Linux runtime.
