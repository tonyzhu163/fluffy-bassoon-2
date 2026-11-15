# Corrected RoboCasa three- versus six-task transformer comparison

**Recommendation: stop scaling this frozen-VLA history-pretraining recipe in its current form. Do not collect more PPO teachers for it on the strength of this campaign.** No model establishes an aggregate ID or OOD advantage over SFT or fixed PPO30, and all three-to-six-task full-success and history-change intervals include zero. This is a resource-allocation judgment from a limited pilot, not proof that all meta-learning is ineffective.

There is one positive aggregate history result: three-task IC-IQL gains **7.50 percentage points [0.83, 15.00] on ID** (14 paired wins, 5 losses, 101 ties). It is one marginal interval among many, from one fit; its OOD interval includes zero and its six-task counterpart does not retain the gain. Preserve it as an unconfirmed lead, not a basis for a new scale-up. DIT is not the strongest method consistently in this corrected comparison.

All eight fits reached the fixed 10,000 actor updates. All 13,200 physical episodes, 25 cold archives, four successful workers and every planned case were audited. There were no outcome-conditioned exclusions or additional cases selected after seeing results.

## Full-history success and retained-history effects

Values are native success on attempt six. History effect means full sixth minus episode-cleared sixth; brackets are marginal 95% paired bootstrap intervals in percentage points.

| Model | Full ID | Full OOD | History effect ID (pp) | History effect OOD (pp) |
| --- | ---: | ---: | ---: | ---: |
| AD (3 tasks) | 45.00% | 36.25% | -2.50 [-9.17, 5.00] | -2.50 [-11.25, 6.25] |
| DPT (3 tasks) | 50.83% | 37.50% | +3.33 [-4.17, 10.83] | -3.75 [-12.50, 5.00] |
| DIT (3 tasks) | 41.67% | 36.25% | -4.17 [-10.83, 2.50] | -5.00 [-13.75, 2.50] |
| IC-IQL (3 tasks) | 49.17% | 43.75% | +7.50 [0.83, 15.00] | +3.75 [-3.75, 11.25] |
| AD (6 tasks) | 45.00% | 40.00% | +0.83 [-7.50, 9.17] | -1.25 [-10.00, 7.50] |
| DPT (6 tasks) | 45.00% | 42.50% | -1.67 [-7.50, 4.17] | -2.50 [-11.25, 6.25] |
| DIT (6 tasks) | 46.67% | 40.00% | +0.83 [-5.83, 7.50] | +5.00 [-5.00, 15.00] |
| IC-IQL (6 tasks) | 47.50% | 41.25% | -1.67 [-9.17, 5.83] | +2.50 [-5.00, 10.00] |

No OOD history interval excludes zero. The three-task IC-IQL ID gain is not itself an advantage over either stage baseline. Full-versus-reset does not isolate whether the history must come from the same scene; donor-history interventions were not part of this campaign.

## Same-case SFT and PPO30 controls

On this fresh panel, SFT scores **49.17% ID / 41.25% OOD** and PPO30 scores **52.50% ID / 40.00% OOD**. PPO30 minus SFT is +3.33pp [−6.67, 12.50] ID and −1.25pp [−11.25, 8.75] OOD. These are the controls paired to the transformer evaluation, separate from the six source-task 64-case panels.

| Model | ID minus SFT | ID minus PPO30 | OOD minus SFT | OOD minus PPO30 |
| --- | ---: | ---: | ---: | ---: |
| AD (3 tasks) | -4.17 [-12.50, 4.17] | -7.50 [-17.50, 2.50] | -5.00 [-16.25, 6.25] | -3.75 [-13.75, 5.00] |
| DPT (3 tasks) | +1.67 [-6.67, 9.17] | -1.67 [-10.83, 7.50] | -3.75 [-15.00, 7.50] | -2.50 [-13.75, 8.75] |
| DIT (3 tasks) | -7.50 [-15.83, 0.83] | -10.83 [-19.17, -2.50] | -5.00 [-15.00, 5.00] | -3.75 [-13.75, 6.25] |
| IC-IQL (3 tasks) | +0.00 [-7.50, 7.50] | -3.33 [-12.50, 5.83] | +2.50 [-8.75, 13.75] | +3.75 [-7.50, 15.00] |
| AD (6 tasks) | -4.17 [-13.33, 4.17] | -7.50 [-17.50, 2.50] | -1.25 [-12.50, 10.00] | +0.00 [-12.50, 12.50] |
| DPT (6 tasks) | -4.17 [-12.50, 4.17] | -7.50 [-16.67, 1.67] | +1.25 [-10.00, 12.50] | +2.50 [-7.50, 12.50] |
| DIT (6 tasks) | -2.50 [-10.00, 5.00] | -5.83 [-15.00, 3.33] | -1.25 [-12.50, 10.00] | +0.00 [-12.50, 12.50] |
| IC-IQL (6 tasks) | -1.67 [-9.17, 5.83] | -5.00 [-14.17, 4.17] | +0.00 [-11.25, 11.25] | +1.25 [-10.00, 12.50] |

No positive model-versus-baseline contrast has a marginal interval wholly above zero. Three-task DIT is below PPO30 on ID by −10.83pp [−19.17, −2.50]. These intervals are not multiplicity-adjusted. PPO30 uses task-matched teachers for ID and the fixed OpenDrawer30 teacher for both OOD tasks: the OOD control is single-source transfer, not a target-trained oracle or a multi-task PPO policy. No GRPO comparison was run in this campaign.

The separate [six-task source verification](SIX-TASK-BASELINES.json) remains valid: Drawer and Coffee have positive marginal PPO30-versus-SFT intervals; the other four do not. Those panels have different cases and must not be pooled or paired with this readout.

## Adding the three tasks

| Method | Six minus three, full ID (pp) | Six minus three, full OOD (pp) | Change in history, ID (pp) | Change in history, OOD (pp) |
| --- | ---: | ---: | ---: | ---: |
| AD | +0.00 [-9.17, 8.33] | +3.75 [-5.00, 12.50] | +3.33 [-7.50, 14.17] | +1.25 [-11.25, 15.00] |
| DPT | -5.83 [-14.17, 2.50] | +5.00 [-5.00, 15.00] | -5.00 [-15.00, 5.83] | +1.25 [-11.25, 13.75] |
| DIT | +5.00 [-3.33, 13.33] | +3.75 [-6.25, 13.75] | +5.00 [-4.17, 14.17] | +10.00 [-5.00, 25.00] |
| IC-IQL | -1.67 [-8.33, 5.00] | -2.50 [-11.25, 6.25] | -9.17 [-19.17, 0.83] | -1.25 [-11.25, 8.75] |

All eight full-success scaling intervals and all eight history-effect scaling intervals include zero. The data do not establish that the six-task corpus helps. All six method pairs within each corpus, all per-task results and all six full-history attempt curves are retained in the [complete statistics](matched-finalize01/analysis/RESULT.json); no method was dropped from the comparison.

## Pairing and interpretation limits

ID is OpenDrawer, CloseToasterOvenDoor and CoffeeSetupMug. OOD is CloseFridge and TurnOnMicrowave, held out of both meta-training corpora but not necessarily base-VLA pretraining. The added training tasks are PickPlaceCounterToStove, TurnOffStove and TurnOnSinkFaucet. This is a narrow atomic-task pilot, not representative coverage of RoboCasa365.

There are **120 paired ID cases and 80 paired OOD cases**, 40 per task. The 13,200 physical episodes include six full-history attempts plus first/sixth reset controls across eight fits and shared stage controls; they are not 13,200 independent cases. The analysis uses 10,000 task-stratified paired bootstrap draws, equal task weighting and seed20261115. Intervals cover case uncertainty for these fitted models and tasks, not new tasks, training seeds or familywise selection among methods.

Actor architecture, logical batch8, final selection and 10,000 actor updates are matched. DIT and IC-IQL additionally use 1,000 Q/V updates, so total compute is not equal across methods. Adding tasks changes diversity, total source volume and per-task exposure together. The archived first-render summary used the imprecise phrase “fixed fitting compute”; this report and the corrected figure caption supersede that wording without changing any numerical result.

Both arms have empty history on their first attempt, yet their native success outcomes sometimes differ. The prescribed physical-state, case/noise identity, RGB-tolerance and reset-arm action-replay checks passed; this does not establish bit-identical cross-arm simulator trajectories. Initial rendering differences are permitted by the runtime, and the cause of each outcome disagreement was not isolated. These discrepancies limit clean causal attribution to history and make the isolated positive IC-IQL interval less conclusive. No discrepant case was excluded.

| Model | First full / reset ID | Disagreements /120 | First full / reset OOD | Disagreements /80 |
| --- | ---: | ---: | ---: | ---: |
| AD (3 tasks) | 41.67% / 43.33% | 6 | 38.75% / 42.50% | 5 |
| DPT (3 tasks) | 47.50% / 46.67% | 7 | 46.25% / 47.50% | 7 |
| DIT (3 tasks) | 50.00% / 48.33% | 6 | 41.25% / 42.50% | 7 |
| IC-IQL (3 tasks) | 41.67% / 41.67% | 2 | 42.50% / 46.25% | 7 |
| AD (6 tasks) | 44.17% / 43.33% | 5 | 43.75% / 38.75% | 12 |
| DPT (6 tasks) | 42.50% / 45.83% | 6 | 37.50% / 36.25% | 9 |
| DIT (6 tasks) | 44.17% / 42.50% | 6 | 40.00% / 42.50% | 2 |
| IC-IQL (6 tasks) | 42.50% / 41.67% | 3 | 46.25% / 45.00% | 9 |

The earlier [11,520-episode frozen-DIT confirmation](../robocasa-dit-paired-confirmation-20261114-r1/RESULT.md) is separate evidence: it did not reproduce the old small-panel history advantage or establish superiority over SFT, and was worse than PPO130 on ID. The old headline is retained as historical, with that nonreplication annotation. PPO130 there and PPO30 here are different checkpoints on different panels.

## Evidence and closeout

- [Coverage and archive hashes](matched-finalize01/analysis/COVERAGE.json): all25 banks; no missing planned slots. [CPU audit/archive receipt](matched-finalize01/ARCHIVE.json).
- [Performance/history figure](matched-figures02/matched-performance.pdf), [three-to-six-task figure](matched-figures02/matched-scaling.pdf), [baseline-gap figure](matched-figures02/matched-baseline-gaps.pdf). [Figure archive](matched-figures02/ARCHIVE.json).
- Corrected collector RLinf8f9bfa76; fit source4987585e; evaluation/statistics7cc56702; caption-only final rendering6619d4c1. Result SHA256: `9856ab962d7cb8fb0a95f1257103ec2d72517ee5bc86129cce20ddb100d9e674`.
- [Scheduler closeout](evidence/MATCHED-EVALUATION-CLOSEOUT.json): Succeeded at08:45:12 Shanghai, November15;29.0344 scheduler GPU-hours,29.25 conservative. Campaign closed at219.5/230 conservative GPU-hours; no further GPU job is planned. CPU analysis/rendering add0 GPU-hours.
- [Retention](STORAGE-REDUCTION.json): historical bulk is archived with restore pointers; valid corrected sources, labels, models, resumable states and evaluation evidence remain in cold storage. Shared environments and other campaigns remain protected.

Any revival should first establish reproducible, scene-specific use of history and an advantage over the stage controls before adding teachers or tasks. That would be a separately scoped experiment; none is launched here.
