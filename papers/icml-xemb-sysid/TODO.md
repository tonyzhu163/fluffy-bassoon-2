# SymMeta ICML submission: open work

From the ICML-style audit on 2026-11-23 (soundness, presentation, significance, originality). Items are ordered by how much they change the accept/reject case. Text-only fixes already applied to `main.tex` are listed at the end. This file is the single list of open work; `main.tex` carries no inline TODO markers.

## Blocking: experiments

- [ ] **Kinematic baseline in closed loop.** FK and seed-41 SymMeta yaw MAE are 1.93° and 6.67° on all 96 UR5e supports, and 2.02° and 6.83° on the 64 OOD supports. Report the [registered UR5e × π₀.₅ comparison](../../outputs/symmeta-fk-ur5e-20261123-r1/PLAN.md) once verified. The other three arm–VLA panels remain open.
- [ ] **Target-calibrated classical visual baseline.** Horn–Schunck and DIS already use projections fitted on nominal source-arm probes. Compare with a tracker given target camera geometry or a gripper fiducial, and state the target calibration data and cost.
- [ ] **Camera pose.** The reader's yaw estimate follows the camera: ±90° image roll gives 84.5° / 101.2° error, ±30° azimuth shifts the estimate by 28–30° ([result](../../outputs/symmeta-camera-command-20261123-r1/RESULT.md)). Either (a) measure task success under moderate roll/azimuth and report it as a limitation with numbers, or (b) test the scene-fixed-reference fix proposed in that result. Offline yaw error alone is not enough for the paper.
- [ ] **Faithful ReGround comparison.** The current baseline is a command-frame affine fit on the passive trace, labelled "Affine (proprio.)". The [released code](https://github.com/umersiddique94/ReGround) (checked 2026-11-23) has paper-reproduction commands for π₀.₅ on eight LIBERO tasks with Panda, using lerobot's PyTorch π₀.₅, not our openpi stack. Its shift cells already include ours (`rot_z`, `gain`, `grip_flip`).
  1. *Reproduce.* Seed 0, cells `none/0`, `rot_z/90`, `lat/8` (400 ms), base and ReGround on the 35 test states. The question is whether their reported recovery (0.16 → 0.90 at 400 ms) holds on our cluster. About 10–15 GPU-hours, estimated from our live-wave throughput (roughly 200 π₀.₅ episodes per GPU-hour), not measured on their stack. The full recipe (5 cells, 5 seeds, 100,000 babbling plus 100,000 policy-driven transitions per cell and seed) is on the order of 120 GPU-hours by the same estimate and needs approval under the 50 GPU-hour rule.
  2. *Cross-run.* ReGround on our 12 transforms, and SymMeta on their `rot_z`, `gain` and `combo` cells. Sweep ReGround's babbling budget down towards our 15 controls, since the data-regime claim is the comparison that matters. Panda is a SymMeta training arm, so this run compares calibration methods and says nothing about unseen arms; check whether their environment can load UR5e before planning that.
  3. If neither is run, keep the relabelled baseline and drop any sentence that implies a result about ReGround.
- [ ] **Policy-noise-matched adapter comparison.** The figure and table now match the 32 task–transform pairs and simulator reset schedule. A paired outcome comparison still requires replaying the same policy-noise bank across adapters and seed-41 methods and retaining query-level outcomes.
- [ ] **Independent scenes.** Each panel has 8 task clusters; the three reader seeds reuse the same 96 queries and support traces. Add fresh initial layouts and randomly sampled yaw/gain values (not the fixed 12 transforms) so "OOD" covers more than the fixed 12 transforms.

## Blocking: what the paper claims

The adapter inverts the function that generates the benchmark shifts (one yaw, one gain, one gripper sign in `embodiment.py`), and the Procrustes step is not new. The claim the evidence supports is the combination: a learned visual measurement trained only through a fixed solver beats the classical measurement with the same solver (DIS+Procrustes 24.1° yaw, 29–42% success) and the same CNN without the solver (Perceiver 95.6°, 2–6% OOD), against 7.0° and 56–79% for SymMeta. Two experiments decide whether that is a general statement or a property of this benchmark.

- [ ] **Shifts outside the yaw/gain/sign family.** Add shear, unequal per-axis gains, an axis swap and command delay; ReGround's `reground/envs/shifts.py` also implements controller stiffness, dead zone, backlash, velocity limit and payload shifts that can be ported. Compare three inner loops on the same learned reader: Procrustes (current), an unconstrained least-squares linear map, and the Perceiver. Start with offline estimate error on new 15-control support traces, which needs no VLA rollouts; run task success only where the offline ranking is not obvious. What would count against the current design: the linear map matching Procrustes inside the family and beating it outside.
- [ ] **A fair test of a learned inner loop.** The Perceiver baseline is 8 latents, 3 self-attention blocks at width 192, reading 3 chunk tokens and regressing the same four numbers; each scaling-sweep model trains for about a minute on 1,466 traces. The headline Perceiver failure (95.6° yaw, 2–6% OOD success) is at 7 training transforms. In the [scaling sweep](../../outputs/xemb-icml-campaign-20261121-r1/evidence/stage3-scaling-summary.json) its OOD yaw error per seed is 15.9°/17.6°/14.7° at M=32 and 12.4°/101.1°/18.8° at M=128, against 5–7° for SymMeta: it improves with shift diversity, stays 2–3 times worse, and one seed collapses. That pattern points to data diversity and optimization as much as to architecture. Before the paper says learned sequence encoders fail on new combinations, train one on continuously sampled shifts (not a fixed list), with per-control tokens instead of 3 chunk tokens, a tuned learning rate and warmup, and an output that is a linear map instead of the family's parameters. Evaluate it on the wider family above. If it matches the fixed solver in the family and wins outside it, the paper's thesis changes from "fix the solver, learn the measurement" to a learned adaptation algorithm.
- [ ] **Title and framing.** The title leads with "A Geometric Inner Loop", the least novel part. Until the two items above are in, lead with what is learned and with the data regime (15 passive controls, no probing, kinematic model or camera calibration), and keep the claim at command-frame calibration for a frozen VLA.

## Blocking: paper

- [ ] **Statistics:** Complete the appendix ledger accounting for every episode beyond the four headline panels. The [paired reanalysis](../../outputs/symmeta-paired-statistics-20261123-r1/RESULT.md) supplies per-seed query tests, task-level effects, and counts for the four panels.
- [ ] **Bibliography.** Verify the author lists of `siddique2026reground` and `yang2026icba` against arXiv, and replace the placeholder authors of `liberopro2026`. Confirm which of ICBA and ReGround count as concurrent at the deadline.

## Should do

- [ ] Reader ablations on one panel: action-blind vs. action-conditioned reader, with and without confidence weights, with and without parallax correction, measured in task success (the parallax ablation is offline only).
- [ ] 3D: either drop to an appendix or add a second camera; 38.5% vs. 88.5% oracle and 9.4% on continuous rotations does not support a 3D claim.
- [ ] A small hardware test (one arm, a few yaw offsets) would answer the simulation-only objection.
- [ ] The top entries of `docs/FINDINGS.md` still use the old baseline names ("Perceiver-ICBA", "ReGround").
- [ ] Review the impact statement wording.
- [ ] Add `jax`/`flax` as an optional dependency group so the reader tests run off the cluster (they skip today when the packages are missing).

## Done in this pass (text, figures and tests only; no new experiments)

- Added a one-column inference diagram and a training appendix with the source data, labels, loss, optimizer, reader heads, scaling-model parameter counts and executed 3D correction path. [Checkpoint and implementation evidence](../../outputs/symmeta-fk-ur5e-20261123-r1/METHODS-READBACK.md)

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
