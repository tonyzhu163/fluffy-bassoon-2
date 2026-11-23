# SymMeta ICML submission: open work

From the ICML-style audit on 2026-11-23 (soundness, presentation, significance, originality). Items are ordered by how much they change the accept/reject case. Text-only fixes already applied to `main.tex` are listed at the end. This file is the single list of open work; `main.tex` carries no inline TODO markers.

## Blocking: experiments

- [ ] **Kinematic baseline in closed loop.** FK and seed-41 SymMeta yaw MAE are 1.93° and 6.67° on all 96 UR5e supports, and 2.02° and 6.83° on the 64 OOD supports. Report the [registered UR5e × π₀.₅ comparison](../../outputs/symmeta-fk-ur5e-20261123-r1/PLAN.md) once verified. The other three arm–VLA panels remain open.
- [ ] **Target-calibrated classical visual baseline.** Horn–Schunck and DIS already use projections fitted on nominal source-arm probes. Compare with a tracker given target camera geometry or a gripper fiducial, and state the target calibration data and cost.
- [ ] **Camera pose.** The reader's yaw estimate follows the camera: ±90° image roll gives 84.5° / 101.2° error, ±30° azimuth shifts the estimate by 28–30° ([result](../../outputs/symmeta-camera-command-20261123-r1/RESULT.md)). Either (a) measure task success under moderate roll/azimuth and report it as a limitation with numbers, or (b) test the scene-fixed-reference fix proposed in that result. Offline yaw error alone is not enough for the paper.
- [ ] **Faithful ReGround comparison.** The current baseline is a command-frame affine fit on the passive trace, now labelled "Affine (proprio.)". Run the released ReGround code with its own babbling budget on at least UR5e + π₀.₅, or keep the relabelled baseline and drop any sentence that implies a result about ReGround.
- [ ] **Policy-noise-matched adapter comparison.** The figure and table now match the 32 task–transform pairs and simulator reset schedule. A paired outcome comparison still requires replaying the same policy-noise bank across adapters and seed-41 methods and retaining query-level outcomes.
- [ ] **Independent scenes.** Each panel has 8 task clusters; the three reader seeds reuse the same 96 queries and support traces. Add fresh initial layouts and randomly sampled yaw/gain values (not the fixed 12 transforms) so "OOD" covers more than the fixed 12 transforms.

## Blocking: paper

- [ ] **Method reproducibility:** training loss and labels, dataset size, CNN spec, optimizer and hyperparameters, how `w_k` and `u_z` are supervised, parameter counts for both Perceiver widths, and whether the Δz correction is used in the 3D variant.
- [ ] **Statistics:** Complete the appendix ledger accounting for every episode beyond the four headline panels. The [paired reanalysis](../../outputs/symmeta-paired-statistics-20261123-r1/RESULT.md) supplies per-seed query tests, task-level effects, and counts for the four panels.
- [ ] **Pipeline figure** (one column) showing support trace → reader → Procrustes → wrapper.
- [ ] **Bibliography.** Verify the author lists of `siddique2026reground` and `yang2026icba` against arXiv, and replace the placeholder authors of `liberopro2026`. Confirm which of ICBA and ReGround count as concurrent at the deadline.

## Should do

- [ ] Reader ablations on one panel: action-blind vs. action-conditioned reader, with and without confidence weights, with and without parallax correction, measured in task success (the parallax ablation is offline only).
- [ ] 3D: either drop to an appendix or add a second camera; 38.5% vs. 88.5% oracle and 9.4% on continuous rotations does not support a 3D claim.
- [ ] A small hardware test (one arm, a few yaw offsets) would answer the simulation-only objection.
- [ ] The top entries of `docs/FINDINGS.md` still use the old baseline names ("Perceiver-ICBA", "ReGround").
- [ ] Review the impact statement wording.
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

- The appendix gives same-attempt counts for each seed-41 panel, the offline scaling table, all nine visual-perturbation methods, and the ManiSkill3 result with its sample-size limitation.
- The demonstration figure uses the four adapter shifts and all eight tasks. The paper distinguishes support per query from oracle demonstrations per shift and reports retained demonstration transitions.
- The verified pdfLaTeX build fits the conclusion, impact statement and all main-text floats through page 8. References begin on page 8, and the appendix occupies pages 11–12.
