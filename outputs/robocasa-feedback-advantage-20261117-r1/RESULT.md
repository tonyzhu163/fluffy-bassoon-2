# Native history calibration pilot

Completed November 17 at 13:35:46 Shanghai. A small DPT history reader learns
useful adaptation in this controlled drawer-opening experiment. With only a
previous probe action and success/failure feedback, held-out second-attempt
success averages **95.83%**, versus **50%** with its history cleared or with a
separately trained memoryless reader. Standard DIT reaches 50%; sharper DIT
weighting reaches 58.33%, inconsistently across seeds. In the separate
uncorrected-attempt/motion-feedback variant, DPT and standard DIT both reach 100%.

This qualifies useful history in the deliberately constructed calibration
problem. It does not establish gains for the earlier large VLA adapters or
complete the AD/DPT/DIT/IC-IQL multi-task ID/OOD comparison. The next scientific
priority is to isolate DIT's sparse-feedback value/weighting weakness and build
training trials with persistent, observable-through-history uncertainty.
These results do not justify resuming broad PPO-teacher collection yet.

## What was run

The public Human300 pi0.5 motor stays frozen. In native RoboCasa `OpenDrawer`,
a hidden signed 4 cm lateral calibration offset is applied during approach.
The small reader selects one of three corrections: left, none, right. The
current decision observation is identical across offset conditions; previous
attempt feedback can reveal the hidden sign, which persists into the next
attempt. Native simulator success determines every candidate's outcome.

The main version records a first-attempt left/right probe and its success/failure.
A second version uses an uncorrected first attempt and adds the measured
20-control end-effector displacement and mean command. It changes both the
support policy and available feedback, so it is not a one-feature ablation.
Neither reader receives the hidden sign as an input. DPT receives inverse-sign
teacher labels during training; DIT learns weights from fitted Q/V estimates.
The known-outcome control directly uses measured query success for weighting.

Collection retained 81 original panel episodes on 4090 and 224 new episodes on
four Pro6K GPUs. The original 49 candidate seeds yielded 32 eligible complete
scene panels. The primary analysis keeps the first 16 eligible scenes in the
original seed order, screening 31 scenes and including 159 native episodes;
the 16 surplus eligible panels are retained but excluded. The first 8 selected
scenes train the readers; the next 8 are held out. The separate development gate
contributed 18 episodes. These are 305 panel-collection episodes plus 18 gate
episodes, not independent learning/evaluation trials.

Each 2-layer, width 128 causal reader is trained with batch 32 for 1024 updates,
with checkpoints at 64/128/256/512/1024 and three fit seeds. Six methods across
two feedback variants give 36 actor fits. The main and drift variants contain
768 and 384 constructed training rows respectively, formed by recombining
measured scene branches. All final samplers reach 100% coverage; no post-success
query suffixes are used. The fixed 1024 endpoint is reported without held-out
checkpoint selection. The old PLAN's two-action/512-row description was
superseded by the executed three-action protocol in EXECUTION.md.

## Fixed-endpoint held-out comparison

Second-attempt success on cross-scene trials, averaged across the three fit
seeds. These are selections scored against exhaustively measured native
candidate outcomes, not new closed-loop rollouts of each trained reader.

| Method | Probe action + reward history | Uncorrected-attempt motion history |
| --- | ---: | ---: |
| DPT teacher labels | 95.83% (93.75,93.75,100) | 100% (100,100,100) |
| Uniform behavior cloning | 41.67% (50,50,25) | 16.67% (12.5,18.75,18.75) |
| DIT, tau 1 | 50% (53.125,50,46.875) | 100% (100,100,100) |
| DIT, tau 0.1 | 58.33% (75,50,50) | 97.92% (100,100,93.75) |
| Measured-outcome weighting, tau 0.1 | 92.71% (87.5,90.625,100) | 100% (100,100,100) |
| Memoryless teacher-label control | 50% (50,50,50) | 50% (50,50,50) |

The simple feedback reference achieves 100% on the held-out primary panel,
versus 50% for the best fixed correction and current-observation reference.
Its training success 90.625% exceeds both controls; this train-only condition
admitted the reader fits. The inverse-sign correction succeeds for both signs
on all 8 held-out scenes and 7/8 training scenes. The remaining training scene
has one unrescued sign and remains in the data.

For the main DPT reader, clearing history or masking the support reward lowers
success to 50%; flipping that reward lowers it to 4.17%. Supplying history from
the opposite calibration gives 5.21%. These interventions show decisions depend
on informative feedback. The donor is deliberately contradictory calibration,
not a random unrelated-history baseline, and masking is an evaluation-only
input intervention. DPT's cumulative first-or-second success is 96.875% with
own history versus 75% cleared; same-scene second-attempt success is 90.625%.

For the motion-history variant, clearing history reduces both DPT and standard
DIT from 100% to 50%, and opposite-calibration history reduces them to 0%.
Masking the reward does not reduce their primary score: motion feedback itself
carries the calibration information in this variant.

## What this says about DIT

Main-version held-out critic AUC is 0.748–0.800. The tau 1 training weight ESS
fraction remains 0.956–0.962, indicating modest reweighting; tau 0.1 lowers ESS
to 0.446–0.582 but only one of three actor seeds exceeds 50%. Directly weighting
with measured outcomes reaches 92.71%. This combination points toward estimated
values/weights and learning the action–reward relation as an unresolved
bottleneck, rather than absence of usable history information. It does not
identify a unique implementation bug or prove that critic fit alone explains
the gap. A useful next check is ranking the two nonzero corrections within
each fixed history; pooled AUC can benefit from merely rejecting the always
unsuccessful zero correction. Motion-version critic AUC rises to 0.998–1.000 and DIT succeeds there.

## Verification, limits and retained evidence

The saved audit passes 1,333 checks, including native branch pairing, final
predictions, sampler counts, milestone/optimizer/RNG states and checkpoint
hashes. All 16 exact action-tape replays match terminal physics, duration and
success. Fresh closed-loop repeats preserve success on 16/16, but match terminal
physics on 14/16; the two differing Pro6K scenes remain disclosed and included.

This is a single task family with a hand-designed calibration perturbation,
conditional on the unperturbed frozen motor already succeeding. It uses a
privileged simulator-geometry decision trigger and only three candidate
corrections. The model consumes compact robot-state/action/feedback tokens;
it is not the earlier image-conditioned large KV adapter. Training scenes
were collected on 4090 and held-out scenes onPro6K, so hardware is confounded
with that split. No cross-hardware trajectory equivalence is claimed.

The main cross-scene panel has 32 constructed trials but only 8 held-out scenes
and two connected dependency blocks. Three optimizer seeds are not independent
native datasets. Scene-block intervals are consequently coarse/descriptive;
100% on this small panel does not imply universal generalization. Cross-scene
transfer within OpenDrawer is not held-out-task OOD evaluation. The original
multi-task null results remain unchanged.

Source pin: `0c553ebade1c132b2f9a2a81dadfaa2ce28f5f65`. All four collection
shards and the reader fit exited 0. Conservative phase charge is 6.577972/8 GPUh,
including the full 5 GPUh parallel reservation; summed parallel-worker elapsed GPU time is
2.502602 GPUh plus prior 4090 use and the final fit. Complete native tapes and
all resumable reader checkpoints are retained and cold-archived; the archive
receipt is [ARCHIVE02.json](ARCHIVE02.json).

[Compact exact readout](evidence/CLOSEOUT-READOUT.json) ·
[Audit and checkpoint hashes](evidence/FIT-AUDIT.json) ·
[Execution contract](EXECUTION.md) · [Parallel launch](PARALLEL-EXECUTION.json).
