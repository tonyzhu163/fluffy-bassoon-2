# Review of `papers/icml-xemb-sysid/main.tex` (SymMeta), commit c34daa6e

## Status after the 2026-11-23 revision

**Already fixed in `main.tex`**
- Numbers now come from one source: 3-seed means, with seed-41 results
  labelled as such.
- Figures are regenerated from the evidence files.
- Claims and the abstract are rewritten.
- The propositions are restated, with proofs.
- A setup section was added (transforms, two-attempt protocol, statistics).
- Pro2, SO(3), ManiSkill and the SFT comparison are reframed.
- Four classical references were added.
- The paper now uses the official ICML 2026 template, with an impact statement.

The items below are still open.

### Open major issues

1. **Motivation versus plain calibration.** This is the biggest risk with
   reviewers. The paper now cites hand–eye calibration and the π0 quaternion
   bug as a real interface mismatch. A reviewer can still answer "read the
   manual and calibrate". Two things would settle it:
   - a physical-arm demo, even a small one;
   - a "known kinematics + camera extrinsics" baseline, or an argument for why
     the setting rules that baseline out.
2. **The two-attempt protocol.** It is now stated openly. Adapting within a
   single attempt reaches only 9–27%. Either improve it (for example,
   conservative actions until the estimate converges) or keep it as the main
   stated limitation.
3. **Length.** The text ends on page 7, but the full-width tables and figures
   float onto pages 8–10. To fit 8 pages:
   - merge Table 3 into Fig. 6, or move it to the appendix;
   - move Table 4 or Fig. 2 to the appendix;
   - move the Pro2 table to the appendix and keep Fig. 5;
   - shrink Table 2 to one column.
4. **Fine-tuning comparison.** It is not matched yet:
   - SymMeta is scored on 96 queries, but fine-tuning on 32;
   - linear fine-tuning with one demonstration (87.5%) beats SymMeta on the
     same shift;
   - there is no baseline that gets one demonstration on each new shift.
5. **Statistics.** The confidence intervals still use a bootstrap over only 8
   tasks, from seed 41 alone. Add a paired permutation test per query and per
   seed, and an appendix table that accounts for every episode. The old
   totals (5,500, 6,560, 6,768) were removed because they did not reconcile.
6. **Baselines run in different probe modes.** DIS+Procrustes is passive in
   Table 2 but active in the Pro2 table. Broyden UVS is only ever run active.
   Run each baseline in one mode, or justify the mix.
7. **3D rotations and ManiSkill** are reported honestly but are weak: 38.5%
   against an 88.5% oracle, and 23.6% against a 66.7% oracle. Consider moving
   both to the appendix as preliminary results.

### New runs needed

| Item | Why | Size |
|---|---|---|
| Unshifted frozen VLA on all 96 queries per arm and VLA | The current n=8 figures (8/8 and 7/8) are too noisy and make the oracle look higher than unshifted | 4 × 96 episodes |
| SymMeta on the 32 queries used for fine-tuning | Makes Table 3 a matched comparison | 3 seeds × 32 |
| Fine-tuning with one demonstration on each new shift | The fair cost comparison | 4 shifts × K=1 |
| A single-attempt column for SymMeta and the baselines | So Table 1 reports both protocols | 4 × 96 per method |
| Baselines rerun in one probe mode | Fair comparison | about 2 × 4 × 96 |
| Per-query results saved for every seed | Needed for the paired tests | re-export only |

### Details only the authors have (marked `% TODO` or `% VERIFY` in `main.tex`)

- **Training details:** the loss and labels; the dataset size behind the
  headline checkpoints; the 4-stage CNN spec; optimizer and other
  hyperparameters; how the confidence weights and the parallax term are
  supervised. Also add a method pipeline figure.
- **Headline training shifts:** confirm the headline checkpoints were trained
  on the 7 transforms in `TRAIN_EMBODIMENTS` (nominal, yaw ±90°/180°,
  gain 0.65/1.55, gripper inversion).
- **π0 quaternion fix:** confirm it runs in the shared observation path for
  every method, including the oracle. I could not find that code in this repo.
- **3D reader:** whether and how the vertical-parallax correction is used in
  3D, where vertical motion is not rotation-invariant.
- **Scaling sweep:** confirm it is offline estimation on recorded support
  traces, and give parameter counts for both Perceiver sizes (w192, w384).
- **ICBA status:** say whether it is prior or concurrent work at the
  submission deadline.
- **Pro2 table:** list all 9 methods; the table shows 6.
- **`liberopro2026` bib entry:** the author list is a placeholder, and the
  arXiv IDs for `yang2026icba`, `siddique2026reground` and `liberopro2026`
  need checking.
- **New bib entries:** check volume, number and page details for the four
  added references (Kabsch, Umeyama, Tsai–Lenz, Park–Martin).
- **Impact statement:** review the wording.

### Build note

`tectonic` builds the PDF. Under tectonic, the ICML style replaces the running
header with "Title Suppressed Due to Excessive Size". The official ICML
example does the same, so this is a quirk of the build tool, not of the paper.
The local TeX Live install lacks `forloop.sty`. Build the final PDF on
Overleaf or a full TeX Live install. `DRAFT.md` and `EXTENDED_ABSTRACT.md`
are stale relative to `main.tex`.

---

## Original review

Date: 2026-11-23. Simulated five-seat panel (Journal-Fit, Methodology, Domain,
Perspective, Devil's Advocate) using the ARS reviewer skill. All five seats ran
inline in one session on one model. They are separate roles, not independent
reviewers. The skill's sprint-contract checkers and its advisory scripts
(acronym, suggested-reference and claim checks) were not run. Suggested
references are listed from memory and must be checked before citing.

**Decision: Major revision before submission.** The empirical core is strong.
The frozen VLA keeps its competence and only the command interface breaks. A
geometric estimator beats learned sequence encoders by a wide margin on OOD
shifts, and the swapped-support control is convincing. The manuscript
undercuts that core in three ways:
(a) its numbers disagree across the abstract, tables and figures;
(b) it overclaims ("prove", "law", "physical robot", "every single M");
(c) it leaves out the protocol and method detail a reviewer needs to trust the
results. None of this needs new science. Most of it needs rewriting and a few
small extra runs.

---

## A. Blocking issues (fix before submitting)

### A1. Format and length will cause a desk reject
- The paper uses `\documentclass[twocolumn]{article}` with custom margins
  instead of the ICML style file (`main.tex:1-3`).
- The PDF is 10 pages, and the body alone appears to exceed the 8-page limit.
- **Fix:** move to `icml2026.sty`, then cut. Candidates are the §3 π0
  quaternion paragraph (to the appendix), Table 3 and Fig. 6 (merge them), and
  repeated numbers in §1 that also appear in §4.

### A2. Headline numbers mix single-seed and 3-seed results
Pairs that disagree:

| Claim | Abstract / §1 | Table 1 (3 seeds) | Fig. 4 |
|---|---|---|---|
| UR5e π0.5 overall / OOD | 79.2 / 81.2 | 78.1 / 79.2 | 75 fam / 81 OOD |
| IIWA π0.5 | 79.2 / 81.2 (identical to UR5e) | 78.5 / 77.6 | 75 / 81 |
| Kinova3 π0.5 | 62.5 / 59.4 | 66.3 / 64.1 | 69 / 59 |
| UR5e π0 | 56.2 / 54.7 | 55.6 / 50.5 | 59 / 55 |
| Perceiver OOD | 3.1–7.8 % | 2.1–5.7 % | 3–8 % |

The HS / DIS / Broyden gaps and their CIs (abstract, §1, §4.1 item 1, Table 2)
come from **seed 41** SymMeta: 79.2 − 58.3 = 20.8 and 56.2 − 45.8 = 10.4.
Table 1, however, shows the 3-seed mean beside single-run HS. A reader who
subtracts Table 1 gets +19.8 / +24.3 / +22.5 / +9.8, not the numbers in the
text.

Other mixing:
- The abstract's HS range "+10.0 to +25.0" includes the Pro2 result (+10.0).
  The intro range is "+10.4 to +25.0".
- The abstract's DIS range "+21.7 to +47.9" includes Pro2 (55.4 − 33.8).
  The intro range is "+26.0 to +47.9".

**Fix:** choose one canonical number per cell, preferably the 3-seed mean with
the paired difference and CI computed per seed. Use only that number
everywhere, and label every single-seed number "(seed 41)". Do not let ranges
span different benchmarks.

### A3. The unshifted UR5e baseline contradicts itself
- Table 1 reports 100.0 % (`main.tex:140`).
- Table 3 reports 87.5 % (7/8) (`main.tex:202`).
- Fig. 6a's dotted line is at 87.5 %, but Fig. 1's green line is at 100 %.
- The figure is also measured on only n = 8 episodes. The oracle (88.5 %)
  beats "unshifted" (75 %) on IIWA, which shows how noisy n = 8 is.

**Fix:** run the identity transform on all 96 queries per arm and VLA. That
makes "unshifted" directly comparable to the oracle.

### A4. The headline protocol is two attempts, but the paper presents it as one
- §6 item 3 (`main.tex:308`) shows that the 79.2 % headline reuses the
  Attempt-1 recording on **Attempt 2**. Adapting online within a single attempt
  gives 9.4–27.1 %.
- The abstract, Table 1 caption ("recovers within 0.75 s"), §1 and the
  conclusion read as single-attempt adaptation.
- **Fix:** state the two-attempt protocol in the abstract, the §4.1 setup and
  the Table 1 header. Add a Table 1 column for single-attempt streaming. Report
  what the oracle and baselines get under the same two-attempt budget; it
  appears they do, but the paper should say so.

### A5. The paper says "we prove" for claims that are not proofs
- **Abstract:** it says "We prove two structural failure modes". Failure mode
  (2), memorization and anti-scaling, is empirical. **Fix:** say "We prove (1)
  and show empirically (2)".
- **Prop. 1** (`main.tex:79-86`):
  - The invariance follows directly from the definition. If the local frame is
    rotated by the same R as the command, R cancels. Say this is a short
    observation about the shift model, not a theorem about babbling in general.
  - "Bayes-optimal error equals chance (90°)" holds only under a uniform prior
    on θ and for absolute-error loss. Under the discrete training prior, the
    Bayes estimator beats chance. State the prior and the loss.
  - It does not cover proprioception in the base or world frame, or methods
    that know the camera extrinsics. Name these exclusions.
- **Prop. 2(1):** equivariance of the atan2 estimator is immediate. Keep it,
  but say all the error comes from how accurate v_k is. Check the sign: under
  Eq. 1, rotating the commands by Rᵀ should shift the estimate by −Δθ. Verify.
- **Prop. 2(2):** this mixes a statement with an empirical number ("11.73° on
  our 3D probe fixture", "<10⁻⁴°"). Move the numbers to the experiments and
  state the result as exact recovery when D̂ = D. Add how bias grows with
  error in D̂. Say explicitly that D can be identified from local
  proprioception (Prop. 1 gives Δp_local = D ā). That is a strength the paper
  never states.
- **Prop. 3:** "linear in pixel velocity for fixed M_ψ" is undercut because
  M_ψ depends on ‖f_k‖, a function of the flow. Linearity holds only
  conditionally on the mask, and the image-plane homography only approximates
  3D motion. Reword, or prove a weaker equivariance statement.
- **Fix:** relabel these as Propositions with proofs in an appendix, or as
  Remarks. ICML reviewers will check this.

### A6. "Physical robot morphologies" while everything ran in simulation
- The abstract (`main.tex:38`) says "3 unseen physical robot morphologies".
  All results are from MuJoCo or SAPIEN simulation.
- **Fix:** write "simulated robot arms". Also call "0.75 s" simulated time at
  20 Hz.

### A7. "Anti-scaling at every single M" is false in the paper's own table
- In Table 4 (`main.tex:236-237`), w384 beats w192 on joint recovery at
  M = 2 (5 % vs 1 %) and ties at M = 7 (4 % vs 4 %).
- w192 is itself non-monotone (16° at M = 32, then 44° at M = 128), and the
  seed bands in Fig. 2 overlap heavily.
- Two counting errors:
  - "105 models × 3 seeds" (§4.2 heading, Fig. 2 caption) double-counts:
    7 M values × 5 models × 3 seeds = 105.
  - "Embodiment diversity M" counts sampled command-shift parameters on two
    source arms, not morphologies.
- **Fix:** drop "law". Say "the larger Perceiver is not better and is often
  worse". Rename M to "number of training shifts". Report the parameter counts
  for w192 and w384, and say whether the scaling evaluation is offline
  estimation or live control.

---

## B. Major issues (a reviewer will raise these)

### B1. The paper needs a real-world motivation for an unknown command rotation
- Strongest counter-argument from the Devil's Advocate: the problem is a
  calibration problem. On a real arm, the URDF, the controller convention and
  a one-time hand-eye calibration give R, g and s. The shift is injected by the
  authors. The cross-morphology transfer is done by the VLA (the oracle reaches
  77–89 %), not by SymMeta.
- **Fix:**
  - Cite concrete cases where the convention really is unknown or wrong:
    base-mount yaw, teleop-to-policy frame mismatch, vendor tool-frame
    conventions, and the π0 quaternion bug you hit yourself (§3).
  - Add a "known kinematics + camera extrinsics" baseline, or argue why the
    setting rules it out.
  - Claim cross-VLA and cross-arm robustness of the *adapter*, not of
    cross-morphology transfer.

### B2. Method and setup detail is not reproducible
Missing:
- the 12 command transforms (familiar / compositional / continuous split, and
  what "OOD" means);
- the training loss and labels for the reader;
- dataset size and composition for the headline checkpoints;
- hyperparameters and the 4-stage CNN spec;
- how w_k and u_z are supervised;
- the Attempt-1 / Attempt-2 protocol;
- the episode horizon and success definition;
- how the baselines were tuned (Perceiver width, training budget, and
  whether ICBA is a faithful reimplementation).

There is also no method figure. **Fix:** add a one-column pipeline figure (flow
→ masked v_k → dz correction → weighted Procrustes → inverse adapter around the
frozen VLA), a setup paragraph, and an appendix table of the transforms and
hyperparameters.

### B3. Statistics need work
- A task-clustered bootstrap with 8 clusters is unreliable. Use a paired
  query-level permutation or sign test, and report W/L counts as Table 5 does.
  Keep the cluster bootstrap only as a secondary check.
- "mean ± std over 3 training seeds" leaves out evaluation noise and gives no
  CI. Report a 95 % CI on the paired difference.
- Avoid statements that sit inside noise:
  - SymMeta π0 56.2 % "edges" the 54.2 % oracle (2 episodes out of 96).
  - S14 87.5 % > oracle 85.4 % (1 episode out of 48).
  - ManiSkill "#1": 23.6 % vs 18.1 % is about 17 vs 13 successes out of 72.
  Write "statistically indistinguishable from the oracle" instead.
- Episode totals disagree: 6,768 (abstract), >5,500 (§1), 6,560 (§4.1 heading
  and Fig. 1), plus 2,160 (Pro2) and 792 (ManiSkill). **Fix:** add one
  accounting table in the appendix and use a single total.

### B4. The SFT comparison is a strawman
- A fixed adapter trained on one shift scoring 0 % on a different shift is
  expected by construction.
- The SFT same-shift numbers come from 32 queries over 4 shifts. SymMeta's
  78.1 % comes from all 96 queries, so the subsets do not match.
- Linear SFT with K = 1 (87.5 %) **beats** SymMeta on the same shift. The text
  plays this down.
- **Fix:** compare on the same 32 queries. Add per-shift SFT, meaning one demo
  for each new shift, as the fair cost comparison. Frame the result as "one
  demonstration per shift vs 0.75 s of video and a reset per shift". The
  honest trade-off is still a good story.

### B5. Baselines differ in probing mode across tables
- DIS+Procrustes is passive in Table 2 but active in Table 5.
- Broyden UVS is active only.
- Table 5 says 9 methods but shows 6. Fig. 5b adds "SymMeta active,
  Pro2-retrained".
- **Fix:** run every baseline in the same mode, or label each one clearly.
  Show all 9 Pro2 methods in an appendix.

### B6. Overstated secondary results
- **SO(3):** 38.5 % vs an 88.5 % oracle, with 9.4 % on continuous SO(3) OOD.
  This is a partial result. The abstract should not present it next to the
  near-oracle claims as a successful extension.
- **ManiSkill3:** 23.6 % vs a 66.7 % oracle on about 72 episodes per method
  does not "validate cross-simulator transfer". Call it a preliminary
  second-simulator check.
- **Pro2:** "outperforms … across every single perturbation" counts S15, where
  every method scores 0 %. Leave S15 out of the success comparison and keep it
  only as a calibration-error result. Also report the overall number without
  S15.
- **EquivFlow:** it "provably eliminates cardinal quantization" yet trails
  SymMeta (66.7 vs 79.2 on UR5e). Explain why it is a contribution, for
  example robustness under active cardinal probes, or move it to an ablation.

### B7. Where the π0 fix is placed, and whether it was applied consistently
- The quaternion canonicalization is an evaluation-pipeline fix, but it sits in
  the theory section (`main.tex:106-107`) with Python inline.
- **Fix:** move it to the setup or the appendix. State that it applies to every
  method, the oracle and the baselines included. Otherwise the π0 column is not
  comparable.

### B8. The 3D case may break the dz parallax trick
- `local_dz` relies on Δz being invariant under SO(2) about z. That invariance
  does not hold under SO(3).
- **Fix:** say whether the SO(3) reader uses dz and how.

---

## C. Related work and citations
- The paper has 21 citations, which is thin for ICML.
- **Procrustes:** add Kabsch (1976) and Umeyama (1991) beside Schönemann.
- **Hand-eye and extrinsic calibration:** add Tsai & Lenz (1989) and Park &
  Martin (1994).
- **Cross-embodiment policies:** Open X-Embodiment / RT-X, Octo, CrossFormer,
  HPT.
- **Equivariant manipulation:** equivariant diffusion policy and E(2)-steerable
  CNNs.
- **Adaptive control:** MRAC / adaptive control background for gain
  estimation.
- `references.bib`:
  - `liberopro2026` has the placeholder author "LIBERO-Pro2.0 Authors".
    Replace it with the real author list.
  - Check the arXiv IDs for `yang2026icba` (2610.39280),
    `siddique2026reground` (2611.20244) and `liberopro2026` (2610.24587).
  - Your related-work framing calls ICBA "concurrent". If it is prior work by
    the submission date, call it prior work.

## D. Writing (minor, but it affects how reviewers judge the paper)
- The abstract is about 400 words in three paragraphs and lists 20+ numbers.
  Cut it to roughly 180 words with one headline number and one contrast.
- Remove the bold-everything style and hype words: "decisively", "Remarkably",
  "sharp ML scaling law", "100 % frozen", "machine precision", "every single".
- Replace code identifiers with names: `equiv_v2_dis_dz` becomes SymMeta,
  `equiv_flow_homog` becomes EquivFlow, `fam_nom` and `comp_180_lo_inv` become
  plain descriptions.
- §4.1 says "Three conclusions stand out from Table 1", but the cited numbers
  come from Table 2 and seed 41.
- Add per-factor results (yaw error, gain error, grip accuracy) for the live
  benchmark, not only the scaling study.
- Housekeeping: `DRAFT.md` line 3 contains another user's absolute home path.
  Remove it before sharing the folder. Four unreferenced figures sit in
  `figures/`.

## E. Strengths to keep
- The oracle-gap framing (a frozen VLA keeps its competence) is clean and
  persuasive.
- The swapped-support control (0–1 %) is the best causal evidence in the
  paper. Promote it.
- The Perceiver familiar-vs-OOD gap (62–91 % → 3–8 %) is a striking result and
  matches the equivariance argument.
- Covering 3 arms, 2 VLAs and Pro2 visual shifts with a BASE-trained reader is
  broad.
- The limitations section is candid. Bring its two-attempt fact forward
  (see A4).

## Suggested order of work
1. Settle the numbers (A2, A3, B3) and regenerate the figures from one results
   table.
2. Rewrite the claims (A4–A7, B6) and the abstract (D).
3. Run the small additions: unshifted on 96 queries, matched SFT subset with
   per-shift SFT, single-attempt column, baselines in matching probe modes.
4. Write the method figure and setup appendix (B2), and move to the ICML
   template (A1).
