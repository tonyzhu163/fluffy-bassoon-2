# Project diagnosis: qualify useful adaptation before collecting more teachers

**The present RoboCasa recipe does not justify further scale-up. The project
has demonstrated native motor competence, some useful teacher supervision, and
feedback use in a synthetic task. It has not established a reproducible robotic
benefit from the learner retaining its own experience.** The new CPU checks
identify limitations in the task/history construction and training exposure;
they do not isolate one cause of the negative native result.

Both diagnostic stages completed on Baige CPU on November 16, with **zero new
GPU-hours**, no new fit and no simulator collection. The earlier campaign stays
closed at 219.5/230 conservative GPU-hours.

## What the completed project evidence supports

| Evidence | Supported conclusion | Remaining limit |
| --- | --- | --- |
| [Corrected eight-fit comparison](../robocasa-six-task-scale-20261113-r1/RESULT.md), 13,200 physical episodes | No aggregate ID/OOD advantage over SFT or PPO30; no established three-to-six-task benefit. | One fit per configuration; three ID and two OOD task families; fixed training budget. |
| Three-task IC-IQL: ID history effect +7.50pp [0.83, 15.00] | A lead worth preserving. | One unadjusted comparison; OOD and six-task effects are inconclusive; empty-history replay discrepancies limit attribution. |
| [Old DIT confirmation](../robocasa-dit-paired-confirmation-20261114-r1/RESULT.md), 768 ID and 512 OOD cases | The small original headline did not reproduce as a reliable history or SFT advantage. | Historical PPO130 used the older collector; do not pool it with corrected PPO30. |
| [RoboCasa paired retry study](../robocasa-three-ppo-retry-20261110-r1/RESULT.md) | Teacher labels can improve a memoryless student: 60/96 retries versus frozen motor 40/96. | Own-history DPT scored 37/96 versus donor 54/96; selecting duration on separate development cases improved competence without establishing memory benefit. One native fit. |
| [Synthetic feedback intervention](../learning-histories-feedback-20261111-r1/RESULT.md) | Real feedback helped all three frozen fits; held-out accuracy 54.65% versus masked 43.94%. | This is a query intervention with perfect execution, not robot adaptation. The later [improved recipe](../learning-histories-fix-20261111-r1/RESULT.md) still has a large reference gap. |
| [RoboTwin Sidney](../robotwin-sidney-native-endpoints-20261109-r1/RESULT.md) | Correcting wrist clipping restored strong motor competence. | PPO40/GRPO40 gains of 1–3 points over SFT were inconclusive. A new benchmark does not supply an established improving teacher or meta-learning result. |
| [LIBERO prompt repair](../libero-gen-prompt-repair-20261111-r1/RESULT.md) | Repaired teacher demonstrations reach their native goals. | Frozen motor still scored 0/4 under either original or repaired prompts on these layouts; no history learner was trained. |
| [Agent teacher-update pilot](../teacher-update-pilot-20261112-r1/RESULT.md) | Native data/branch infrastructure partly qualified. | 72 abstentions produced no teacher-change targets and no trained learner. |

The strongest practical positive robot result here is improved initial/retry
competence through imitation, not demonstrated experience-specific adaptation.
Physical-robot adaptation and Flora qualification remain outside this evidence.

## The training histories do not teach the evaluated retry structure directly

The raw six-task arrays contain **3,840 distinct reset seeds per task**, with
30 different seeds in every stream and zero adjacent seed repetitions. Every
episode-start feature vector within each split is distinct. The matched native
evaluation instead gives six attempts at one saved canonical scene. Training
therefore supplies changing-scene task histories, while testing asks for
same-scene retries. Different seeds are verified; we have not proved that every
physical property differs between every pair of scenes.

This can still teach transferable task information. However, a fixed
nearest-centroid classifier already identifies the six source tasks from the
**current initial-query pooled prefix with 99.375% accuracy** on 2,880 held-out
queries, versus a balanced prior of 16.67%. The public robot state alone scores
31.08%. Normalization and centroids use only 20,160 training queries; there is
no fitted hyperparameter search. Vision and the explicit instruction contribute
to the pooled feature, so this is not a visual-only recognition result.

The implication is a missing demonstrated information advantage: we have not
shown which persistent unknown the robot needs previous episodes to resolve.
Task identity is largely visible already. This does **not** prove that history
cannot reveal scene dynamics, failure causes, or an improved strategy.

The 128 collection lanes per task are not 128 independent PPO training runs.
Each lane exposes only part of a teacher's batch-level learning process. The
offline held-out lanes were also part of source PPO collection/training; they
hold out student trajectories, not independently trained teachers. Native
evaluation uses separate cases. These distinctions constrain generalization
claims without making the archives invalid.

The six atomic task types span several manipulation actions, but they are not
a representative sample of all RoboCasa365 or RoboTwin tasks. More independent
task instances and learning runs would be needed for a broad meta-learning
claim; counting thousands of lanes/episodes does not supply that diversity.

## The fixed fit budget underexposes the expanded corpus

The exact seed-0 actor sampler was reconstructed and its final RNG state
matched **all eight saved fit checkpoints**.

| Actor exposure | Three tasks | Six tasks |
| --- | ---: | ---: |
| Sampled action queries | 80,000 | 80,000 |
| Available training queries | 120,960 | 241,920 |
| Distinct queries selected | 57,499 (47.54%) | 67,468 (27.89%) |
| Samples after first success | 17,138 (21.42%) | 15,859 (19.82%) |
| Samples with no preceding complete episode | 27,282 (34.10%) | 27,282 (34.10%) |

The first source-policy version contributes 4,284 sampled queries, while the
last contributes 711, in either corpus. Uniformly choosing a window endpoint,
then its length, then a query inside it produces this unequal version exposure.
Selecting a twelve-episode window does not imply twelve episodes precede its
selected query. DPT labels come from the fixed final teacher, so version
exposure describes its query states, not varying label-teacher quality.

This is a verified sampling property, not proof of underfitting or a defect in
random sampling. The experiment answers whether adding tasks helps at the
same 10,000 actor updates; it does not answer whether a fully trained six-task
learner benefits from diversity. DIT/IC-IQL additionally receive 1,000 value
updates, so equal actor updates are not equal total computation. More data
alone would further reduce exposure unless the training allocation changes.

## Success suffixes matter for interpreting the value diagnostic

Physical episodes contain twelve 50-control chunks. Native success rewards are
given once, and the corpus continues to the fixed physical boundary. Actor
query sampling includes the suffix after that reward. Across all raw queries,
the post-success fraction is 37.39% Drawer, 13.43% Toaster, 14.46% Coffee,
27.90% CounterToStove, 6.17% Stove and 20.18% Faucet.

The [previous diagnostic](../robocasa-dit-diagnostic-20261115-r1/RESULT.md)
labeled every query by eventual episode success, including this suffix. Those
numeric statistics are valid for that label, but a post-success query has
**zero remaining return**. Ranking it below a still-unsuccessful query is not
by itself evidence of a bad advantage estimate. Its first-query statistics
are unaffected by this issue.

Restricting the analysis to queries before or at the first rewarding chunk
reveals modest positive outcome ranking in six-task DIT on Toaster (AUC 0.573,
95% interval [0.528, 0.616]) and Faucet (0.569 [0.516, 0.625]); CounterToStove
is negative (0.424 [0.350, 0.498]). These match collection update and query
position, resample whole streams, and are exploratory unadjusted intervals.
The DIT weights still have effective sample size **99.43–99.83%** of uniform
weighting on these restricted panels. Thus the weighting remains weak even
after repairing the proxy interpretation.

Post-success actions may include useful stabilization. No masking ablation was
run, so their contribution to poor native performance is unknown. Likewise,
actual-trajectory return labels do not provide counterfactual action values.

## Critics use some action information, with little consistent own-history benefit

The second CPU stage holds the query observation and current episode fixed.
It separately substitutes the next validation stream's action, substitutes
that stream's prior complete history, masks only prior reward tokens, or uses
the earlier episode-reset prediction. Donors match task, policy version and
query position and are chosen without their outcomes. The current action
substitution changes the Q input only; the historical action record remains
unchanged. Full Q/V predictions reproduce the original panel **exactly**.

The table uses pre-success queries. A positive change means the altered input
predicts the recorded return less accurately. History contrasts exclude update
zero, where no preceding episode exists. Values are descriptive equal-task
means; they are not pooled confidence intervals or comparisons between fits.

| Frozen fit | Q mean squared error | Action substitution: change in error | Donor prior history: change in Q error | Clear prior history: change in Q error |
| --- | ---: | ---: | ---: | ---: |
| DIT3 | 0.13445 | +0.00526 | +0.00071 | +0.00115 |
| DIT6 | 0.12579 | +0.00522 | −0.00056 | +0.00321 |
| IC-IQL3 | 0.14628 | +0.00770 | +0.00063 | +0.00063 |
| IC-IQL6 | 0.13807 | +0.00740 | −0.00045 | +0.00010 |

Action substitution increases error in all eighteen task/model panels, with
nine marginal task intervals wholly positive. Own-versus-donor Q effects have
one positive and two negative intervals among eighteen; the rest include zero.
Masking prior rewards has small, mixed effects. Full per-task, first-query,
Q/V, sensitivity and interval results are in [value evidence](evidence/value01/RESULT.json).

This rules out the simple description that every critic ignores its action
input. It does **not** validate action ranking: donor actions may be unlikely
at the target state, and their actual returns were never measured. IC-IQL
values also have a different fitting objective from Monte Carlo regression;
the cross-model error table is not an algorithm leaderboard. The main finding
is limited consistent predictive benefit from the *particular* prior history.

Archived native flow checks extend the earlier DIT finding to all eight actors:
full/reset changes span 0.214–0.557% of native flow RMS, with relative imitation
loss benefit between −0.0147% and +0.0471%. These are velocities at 128 saved
queries per actor, not integrated actions or success rates. They suggest weak
history influence but cannot establish a motor-authority ceiling.

## Method and evaluation distinctions

The four names denote adaptations to a frozen VLA, not exact reproductions:

- **AD:** chronological behavior imitation is present, but improvement is
  modest and histories span only six task types. Original AD explicitly chose
  tasks whose identity was difficult to infer from a single observation and
  used many independent source learning runs. Distributed collectors are
  compatible with AD; our lane structure alone does not invalidate it.
  [AD, sections 3–4](https://arxiv.org/html/2210.14215).
- **DPT:** labels are fixed PPO30 actions at saved query states, not proven
  optimal actions. It need not imitate an improving teacher sequence, but
  teacher quality and extra label computation remain part of the comparison.
  [DPT](https://arxiv.org/html/2306.14892).
- **DIT-style:** the published offline value estimators consume prior
  state/action/**return-labeled** contexts. Our independent critics consume
  causal shifted action/reward history, using returns only as training labels.
  Both weight action fitting, but this is a consequential recipe difference.
  It does not justify putting future returns in the deployed actor's context.
  [DIT, section 4.2](https://arxiv.org/html/2601.20116v1).
- **IC-IQL-style:** the referenced implementation shares a transformer trained
  by value losses and detaches actor gradients from that backbone. Ours has
  separate actor/Q/V encoders and one Q plus an EMA target. These results do
  not isolate the published shared architecture or a double-Q alternative.
  [Offline IC-IQL, section 3.1 and Appendix B](https://arxiv.org/pdf/2502.17666v5).

The current feature is a signed masked mean of native prefix embeddings,
normalized by RMS, plus public robot state. It does not discard embedding
signs. Whether this pooling retains the detail needed to diagnose a failed
grasp remains unmeasured.

The matched campaign executes all 50 predicted controls per query, unlike the
released Human300 reference that executes five. Comparisons within the
campaign share its cadence; released benchmark scores are not interchangeable
controls. The prior [shorter-chunk benchmark](../robocasa-ppo-short-chunk-benchmark-20261104-r1/RESULT.md)
measured one update's feasibility and cost, not post-update benefit.

The two OOD task families are held out from meta-training, not necessarily from
the base VLA's pretraining. Their PPO baseline transfers one Drawer teacher;
it is not an OOD-task-trained oracle. GRPO was not included in this campaign.
The first full/reset attempts already have occasional outcome discrepancies
with empty history. The source report records these rather than excluding
cases. More paired episodes reduce sampling uncertainty; they do not remove
rendering variation, fit-seed uncertainty or a training/evaluation mismatch.

## Decision and smallest useful next experiment

**Stop spending on more PPO teachers for the current recipe. Retain the valid
data, checkpoints and negative results.** We cannot explain DIT's exact
ranking by one bug, and stronger PPO remains an untested remedy. Existing
successful behavior and the memoryless result show that teacher weakness is
not a sufficient project-wide explanation.

The next native experiment should establish an information advantage before
another four-method sweep. Use one qualified task family with a persistent
unknown that changes which attainable action succeeds. First verify that
matched previous feedback helps a reference choose among attainable actions
better than the best memoryless/current-observation reference. Hold physical
starting states and query randomness fixed, and measure empty-history repeat
variation. The existing drawer correction study, where the best fixed offset
matched the per-scene choice, does not yet supply this information advantage.

If that reference contrast exists, construct training and evaluation trials
with the same persistent variable. Compare a simple imitation/history model
under own, matched-donor, cleared and feedback-masked histories on fresh cases,
with independent fitted seeds. Retain failed attempts; distinguish success
on the next attempt from cumulative success after more chances. Establish
coverage and learning curves before expanding task count. An optional
uniform/estimated/known-outcome weighting comparison can then isolate the
DIT learning signal; known outcomes are a diagnostic reference, not deployed
inputs or true action advantages.

If the reference cannot use the supplied information to improve attainable
actions, that experiment cannot test meta-learning. If it can and a small
well-trained reader still fails, focus on the representation/objective or stop
that route. Expand to representative task families only after a reproducible
native history benefit. RoboCasa remains the lower-switching-cost venue for
this diagnosis; switching to RoboTwin would require fresh task qualification
without removing the identified learning-design questions. This is a proposed
new experiment, not a launched collection or an extension of the closed cap.

## Execution and retained evidence

- Corpus audit pin `696ee560656fb8830aecf2ab5cb2a9e6c07b9127`; 93.33 seconds.
  [Code](corpus_audit.py), [result](evidence/corpus01/RESULT.json), [exit](evidence/EXIT.json).
- Frozen-value intervention pin `c60eda37ee937461c79b265ac85ec45320a3a2a1`.
  [Code](value_interventions.py), [result](evidence/value01/RESULT.json),
  [launch](evidence/VALUE-LAUNCH.json), [exit](evidence/VALUE-EXIT.json).
- Dataset/checkpoint/manifest hashes verified; exact empty-history identities
  checked. The earlier batched-versus-production numerical qualification
  remains applicable, and all repeated full Q/V predictions agree exactly.
  Independent raw-record recomputation verified 588 statistics/count checks.
  These protect reported results; no repository bookkeeping tests were added.
  [Audit](evidence/AUDIT.json).
- All raw records, logs and receipts are preserved in 1,441,513 bytes of cold
  evidence, with every payload readback hash verified. Compact JSON evidence
  accompanies this report. [Restore inventory](evidence/ARCHIVE.json).
- All intervals condition on one fit/model and the observed tasks; they
  resample sixteen whole validation streams/task and are not multiplicity
  adjusted. A larger observational audit cannot prove a causal remedy.
