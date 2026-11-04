# OpenDrawer PPO: what the first 40 updates establish

Checked Nov 4, 2026 against the completed cluster receipts. The new PPO arm
completed all 40 updates, but did not meet Stage 1's learning gate. This is
not a causal demonstration that the new recipe is worse than historical PPO.
The historical run also had little learning by update 40.

| Measure | Historical PPO, first 40 updates | Stage 1 PPO, 40 updates |
| --- | ---: | ---: |
| Fresh TRAIN episodes / controls | 5,120 / 3,072,000 | 5,120 / 3,072,000 |
| Epochs / Adam steps per update | 4 / 24 | 1 / 6 |
| Cumulative Adam steps per actor rank | 960 | 240 |
| TRAIN first ten updates | 823/1,280 = 64.30% | 809/1,280 = 63.20% |
| TRAIN updates 30–39 | 834/1,280 = 65.16% | 825/1,280 = 64.45% |
| TRAIN change | +0.86 pp | +1.25 pp |
| Reported DEV baseline → update 40 | 44 → 44 / 64 | 34 → 26 / 64 |
| Reported DEV change | 0 pp | −12.5 pp |
| Mean / maximum logged approximate KL | .0502 / .1513 | .00912 / .02088 |
| Mean / maximum clip fraction | .1930 / .2623 | .04798 / .08253 |
| Explained variance, update 1 → 40 | .0298 → .4152 | −.9106 → .4158 |
| Value loss, update 1 → 40 | .1669 → .04923 | .4612 → .05124 |

The historical DEV numbers are the later checkpoint-selection bank, seeds
8,010,000–8,010,063. Stage 1 uses 610,100–610,163. These are different cases
and sampler streams. The historical original before panel on 610,100–610,163
was 40/64, not 44/64; do not substitute the later bank's baseline into it.
Cross-run raw DEV scores therefore do not estimate a treatment effect.
Within-run paired changes remain informative, with substantial uncertainty.

Forty new updates equal ten historical updates **only in Adam-step count**.
They use four times as many fresh rollout batches; the policy distributions,
critic and selected denoising steps also differ. The historical update-10
DEV result was 46/64 versus 44/64 initially. Its independently verified
512-case gain of +7.42 pp was measured at update 130, after 3,120 Adam steps,
not at update 40. Neither comparison predicts the new recipe's eventual gain.

## Checks of the proposed explanations

**The final DEV drop is verified, but inconclusive.** DEV40 has five wins and
13 losses versus its own baseline, exact two-sided McNemar p=.09625244.
That is an observed decline and a failed improvement gate. It neither
establishes a regression at a conventional .05 threshold nor establishes
equivalence/no effect. The paired checks verify all 64 case identities and
physical/public resets. All five DEV panels are retained, not just the final
or best score. They contain 34, 34, 36, 34 and 26 successes.

**A permanently untrained critic is not supported.** The new head starts
poorly: explained variance is −.9106 and −.3215 in updates 1 and 2, with
value clipping fractions .8333 and .5. Explained variance becomes positive
at update 3, reaches .3657 at update 10 and .4158 at update 40. Value loss
falls to approximately the historical value. These are training-fit metrics,
not independent proof of good advantages or held-out value generalization.
Explained variance near zero by itself would not imply that all advantages
are noise.

**Joint gradient clipping is confirmed; critic dominance is not.** The exact
RLinf vendor commit 8f9bfa7661928ef8c17494b04f4d7bf3125634e1 calls
`clip_grad_norm_(model.parameters(), 1.0)` in its NO_SHARD FSDP path. Actor and
value-head parameters have separate learning-rate groups in one optimizer,
but share that clipping norm. Critic warmup is zero. The misleadingly named
`actor/grad_norm` is the combined pre-clipping norm, averaged across the
update's optimizer steps. It is not an actor-only norm. Its mean is 40.41
in Stage 1 versus 44.77 historically; there is no observed increase in this
combined scalar. Separate actor/head norms were not logged, so the evidence
cannot show that the new critic consumed most of the norm. Shared scaling
also does not translate directly into an Adam parameter-step reduction.
Measuring per-group norms and actual parameter deltas is necessary before
changing clipping or adding warmup on this hypothesis.

**Several treatments changed together.** Resolved configurations confirm
four versus one epoch, action-expert versus VLM-prefix critic, and
`ignore_last=false` versus true. The vendor also corrects the historical
mean-over-denoising-values versus selected-step replay mismatch. The old
head reads width-1024 expert features through 512→256→128 hidden layers;
the new head reads width-2048 frozen prefix features through 1024→512→256.
The value inputs are detached/frozen and the VLM stays unchanged. None of
these simultaneous changes is isolated by this pilot.

**Low measured KL is expected to have multiple causes.** There are four
times fewer optimizer steps per fresh batch, a different critic/advantage
estimator and no final-denoising-step selection. The logged KL is an
approximation on sampled conditional transitions, not a full decoded-action
marginal KL. The Stage 1 conditional KL and clipping fraction were lower on the
sampled transitions. Neither statistic measures cumulative distance from SFT. H50, ten denoising
steps and noise .3 remain unchanged. Their role is a hypothesis, not a
confirmed root cause. Existing dimension-drift work also warns against
comparing conditional ratios at different denoising indices.

## Decision

It is worth finishing the already approved, bounded Stage 1 GRPO arm. It
answers whether this critic-free grouped alternative learns under the same
40-update controls budget. It does not isolate critic placement, gradient
clipping or epoch reuse because grouping and advantages also change.
The human explicitly requested running GRPO in parallel with diagnosis and
matching action steps. Both arms therefore retain 3,072,000 executed TRAIN
controls (40 × 128 × 600); Adam-step counts are reported separately. GRPO's
128 episodes contain 16 distinct scenes with eight attempts each, versus
128 distinct scenes in PPO. Equal controls do not mean equal scene diversity.

Do not extend PPO or scale to Stage 2 based on these results. The next
diagnostic comparison should evaluate SFT, historical PPO40 and new PPO40
on the same cases, eight-lane membership and explicit common sampler RNG,
with an initial-policy repeat. It should preserve decoded actions and
physical reset identities. This removes evaluation confounding without
new training. Before a future training ablation, measure actor/head gradient
norms and parameter deltas on retained batches; then change one factor
(epoch reuse or critic treatment) while holding the others fixed. A training
ablation is not authorized by this note or charged silently to Stage 1.

The rollout worker's inspected HF initialization has no explicit sampler
seed, and the runner snapshots/restores its initialized RNG within a run.
That supports a cross-run noise-bank concern; it does not by itself prove
that CUDA RNG is the sole cause of the 40/64 versus 34/64 baseline gap.
The matched evaluation must log actual RNG hashes and compare physical and
public snapshots, rather than infer equality from reset seeds or serialized
snapshot-file hashes alone. High conditional-ratio sensitivity likewise
does not prove the final DEV decline was sampling noise.

The released StructRL pi0.5 sparse-PPO launcher uses H50/K10. Its suitability
for OpenDrawer at this budget remains an empirical question. The proposed
H10/K5 profile also changes noise, minibatch size and the physical duration
represented by a per-decision discount. A single benchmark can measure its
cost and replay parity. [StructRL launcher](https://github.com/amazon-science/StructRL/blob/6caba1f9538167e0752a7338663e45aac8df965d/launch/train_pi05_sparse_rl_ppo.sbatch)

The agreed order is matched-control GRPO, the SFT, old PPO40, new PPO40 and GRPO40 rerun on the
original DEV64 with common sampler RNG, then the single-update shorter-chunk
benchmark. The separate ablation supplies the fresh TEST512 comparison.
[DEV64 contract](../robocasa-ppo40-matched-dev64-20261104-r1/PLAN.md) ·
[benchmark contract](../robocasa-ppo-short-chunk-benchmark-20261104-r1/PLAN.md)

## Evidence

- [Complete old/new cluster metrics and source hashes](RCA-METRICS-R1.json):
  all 40 updates and 5,120 TRAIN episode receipts for each run; both actor
  ranks are checked. Rank copies are synchronized metrics, not independent
  statistical replicates.
- [Exact vendor source excerpts](RCA-SOURCE-R1.txt), including clipping,
  optimizer groups and actual resolved warmup/clip settings.
- [Full Stage 1 audit](PPO-AUDIT-LIVE.json), including paired DEV, parity,
  frozen weights, native action counts and hashed source references.
- [Historical DEV curve](../robocasa-flow-sde-checkpoint-selection-20261104-r1/DEV-CURVES-LOCKED-R1.csv),
  [original native result](../robocasa-flow-sde-10m-per-task-20261103-r1/OpenDrawer/FINAL-READOUT-R1.json),
  and [512-case endpoint](../robocasa-endpoint512-acceleration-20261104-r1/RESULT.md).

No historical run, shared worktree, policy or optimizer was modified for this
analysis. The joins ran on the allocated Linux development container.

## Observed follow-up

At equal TRAIN controls, GRPO finished DEV at its starting 41/64. On matched DEV64, new PPO40 scored 40, SFT 36, old PPO40 33 and GRPO40 31. The separate PPO ablation's fresh TEST512 scored 315, 303 and 320 for new PPO40, SFT and old PPO40, respectively. The original new-PPO DEV decline did not reproduce on either common-draw comparison. The three panels do not isolate its cause. [Matched and fresh readouts](../robocasa-ppo40-matched-dev64-20261104-r1/RESULT.md).

The shorter-chunk candidate passed real-weight replay and completed 15 Adam steps at 283.2 seconds per first update. The corresponding H50/K10 first update took 156.9 seconds and six Adam steps. A source-policy endpoint for the shorter chunk was outside the one-update benchmark. [Benchmark](../robocasa-ppo-short-chunk-benchmark-20261104-r1/RESULT.md).
