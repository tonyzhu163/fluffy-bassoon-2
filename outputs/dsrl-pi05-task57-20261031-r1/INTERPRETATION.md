# Task57 native ceiling: interpretation and proposed next comparison

Read-only discussion, 2026-10-31. This note does not change the active run,
launch another experiment, select new cases or spend another budget.

## Observed

Owner receipt `first-advancement.json`, checked14:17:09 Shanghai:

- Baige `job-iudscqfntb4e` is running at source `d001556`.
- Native query states40..49: **10/10** successes,1,451 primitive actions
  plus100 settling actions. Queries are excluded from replay.
- Training: **45/45** complete episodes succeeded,707 queried chunks,
  4,280 SAC updates, all40 training state IDs visited.
- Total8,816 executed simulator actions,664MB outputs. Latest scalar losses
  are finite; sampled raw trace and full recovery hashes match. This proves
  actual collection/update/capture progress, not convergence or adaptation.

## Implication

The declared query bank has no observed binary-success headroom. The current
run cannot exceed10/10 there. Matched later panels can reveal regression,
preserved competence and completion-cost changes relative to the native
1,451 query primitives. Ten cases do not establish population-perfect success.
The45 successful training episodes currently supply no failure-to-recovery
learning history. More such records alone would not fix that deficiency.

The paper used pi0, whereas this experiment keeps the user-requested pi0.5.
Checkpoint competence, native horizon and the fixed state bank differ; the
authors' reported starting performance is not our expected baseline score.
Nothing here disproves noise steering on difficult cases or establishes
action-residual superiority. Keep these explanations separate from learner
bugs: actual SAC update progress is now verified.

## What the literature establishes

- [VLRLA §4.1](https://arxiv.org/html/2610.19517v1#S4.SS1): simp replaces the
  action expert with a recurrent action predictor; bridge adds a residual to
  the VLM's KV cache while freezing the encoder and action expert. Its bridge
  residual is not an additive physical-action residual. Simulation histories
  come from updating the native action expert, supervised later in action space.
- [DSRL C.5](https://arxiv.org/html/2506.15799v2#A3.SS5) reports noise-space
  SAC through a frozen pi0. It supports this interface's feasibility, not
  guaranteed improvement for our pi0.5 and task distribution.
- [PLD](https://arxiv.org/html/2511.00091v1) already learns physical-action
  residual specialists over a frozen VLA, then collects base-policy roll-ins
  followed by specialist recovery. Its later SFT changes the generalist and
  is not our frozen-weight in-context deployment objective.
- [VLaRL](https://arxiv.org/abs/2609.30868) reports latent-conditioned residual
  RL over frozen VLAs; [RLT](https://www.pi.website/research/rlt) studies a
  compact representation for a lightweight online actor/critic. These are
  relevant nearby interface work, not reproduced results in this repository.

A focused interface/data comparison is warranted; simply adding a residual or
combining a frozen motor with a recurrent model is not an adequate novelty claim.
No broad review or paper reproduction is introduced as a launch gate.

## Cheapest proposed discriminating screen

First inspect the already funded first10k-update panel at the next scheduled
owner check. It tests competence preservation and matched action cost without
new scope. Do not read its binary-success tie as sample-efficient improvement.

For a later interface comparison, use four predeclared development cases
with observed native failures. Run four paired draws for each of three
families: native, the declared noise family, and bounded physical-action
residuals over native. This is48 episodes, at most19,680 actions at H400+10
settling. Native gets the same four attempts; comparing another family's
best-of-four against a single native draw would confound correction with
extra search. This proposal is outside the current run and has no allocation.
Report per-draw success as well as equal-budget best-of-four reachability.

Define residuals in documented robot control units, initially retaining native
gripper behavior. Zero physical residual should preserve the native action
path; zero DSRL noise is a different motor input and is not that identity test.
Use outcome-independent candidate draws and record clipping. Equal-budget best-of-four is
an oracle reachability upper bound, not a deployable controller or adaptation
score. A small negative search cannot rule out the entire interface. Label
known-failure case selection as development; retain full90 final reporting.

## Useful source data after an interface learns

Keep all40 source goals and a declared physical-case population. Preserve the
teacher's actual chronological learning process across randomized cases,
including warmup, failures, intermediate checkpoints and later stronger
behavior. Do not reset the500-chunk warmup/replay for each K4 case. Measure
improvement on separate matched source validation cases, not only train return.
If realistic source cases remain ceiling, propose source-only variation before
another large collection rather than importing target trajectories.

Record the existing raw current/next observations, motor and steering actions,
rewards, durations, boundaries, policy/version/order and complete recoveries.
For physical residuals additionally retain the base action and delta separately.
AD predicts actual steering decisions from history; DPT requires a distinct
competent final same-motor teacher label pass. A PLD-style base roll-in followed
by a trained noise/residual teacher may collect deployment-relevant recoveries;
that is a separate proposed data recipe, not evidence already obtained here.

All task57 data remain target-online development data. They cannot enter the
all40 meta-pretraining bank. Task57 stays in full90 reporting with its prior
development exposure disclosed. Native PPO remains queued: its physical actions
are valid action-space data, but its changing-motor noise seeds are not direct
labels for the original frozen pi0.5 noise controller.
