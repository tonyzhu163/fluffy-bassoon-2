# PPO recipe diagnosis before changes

2026-11-04. Written at the human's request before changing the recipe.
Scope: the five completed native Flow-SDE sources and the ongoing checkpoint
selection sweep. No training/configuration migration is performed by this
document; the active evaluation checkout remains immutable.

## Conclusion and confidence

We have evidence of **checkpoint sensitivity**, especially Drawer-to-counter,
and a **confirmed mismatch with the recommended pi0.5 critic architecture**.
Our smaller minibatches and higher sample reuse are also credible reasons
for sensitivity. We do not yet have a causal ablation showing which setting
caused the performance gap, proof that the underlying PPO implementation is
incorrect, or fresh evidence of improvement over the initial policy.

The current setup is an adaptation of several recipes, not an exact
reproduction of StructRL's sparse PPO or the original RLinf pi0 example.
Scaling this adaptation to10M controls did not establish that its optimization
settings transfer successfully to our Human300 pi0.5 initializer and tasks.

## What the measured curve establishes

The [14:14 snapshot](PARTIAL-CURVE-R2.md) contains56/80 completed DEV panels,
with64 fixed cases per checkpoint and policy0 still pending. Drawer-to-counter
scores34/64 at80 versus21/64 at130, a20.31-point observed difference. Several
other checkpoints also beat130, but its rebound at120 means this is not a
monotonic collapse. OpenDrawer mostly plateaus; Toaster fluctuates; Coffee130
ties its best observed score; Stove remains low. There is no universal
late-training collapse across the five tasks.

These are validation observations used for selection. The largest observed
gap is not an independently confirmed effect. Finish DEV, lock the selected
model including0 as a candidate, then use the separate counterbalanced
TEST128. The original teacher gates and original130 results remain unchanged.

## Concrete recipe comparison

Our values below are confirmed in an actual source `resolved-config.json`,
not inferred only from a default YAML; allfive resolved configurations were
checked in [the configuration audit](RECIPE-DIAGNOSIS-EVIDENCE-R1.json).
Source RLinf pin is8a1233c3; the
accepted source runtime is0988b5f9. StructRL values combine its pinned YAML
**and launcher overrides** at6caba1f9.

| Setting | Our five-task source | StructRL released pi0.5 sparse-PPO launcher | Diagnosis |
| --- | --- | --- | --- |
| Critic input | Action-expert features, `value_after_vlm=false` | VLM features, `true`; detached input | Clearest architectural mismatch |
| Fresh episodes per policy update | 128, one task | 512, jointly sampled tasks | Different batch freshness/diversity; not4x more episodes per task |
| PPO epochs over that batch | 4 | 1 | Fourfold sample reuse; plausible over-optimization risk |
| Global minibatch | 256 chunk slots | 512 chunk slots | Noisier/smaller update samples are a concern |
| Actor/value learning rate | 5e-6 /1e-4 | 5e-6 /1e-4 after launcher overrides | Same rates; not evidence of a wrong scalar by itself |
| Flow-SDE noise | .3 | .5 | Exploration difference requiring qualification, not a proven error |
| Executed chunk/denoising steps | 50/10 | 50/10 in launcher | This matches released launcher, despite paper's chunk10 statement |
| Clip, gamma, lambda | .2, .99, .95 | .2, .99, .95 | Matches |
| Reference-KL/entropy coefficients | 0/0 | 0/0 | Missing KL is not the distinguishing mismatch |
| Initializer/tasks | Human300; one atomic task per policy;600 controls | Target Composite-Seen SFT; joint composite tasks;longer horizons | Different scientific conditions, not a matched performance baseline |

Sources: [StructRL sparse launcher](https://github.com/amazon-science/StructRL/blob/6caba1f9538167e0752a7338663e45aac8df965d/launch/train_pi05_sparse_rl_ppo.sbatch),
[PPO YAML](https://github.com/amazon-science/StructRL/blob/6caba1f9538167e0752a7338663e45aac8df965d/configs/robocasa365_ppo_openpi_pi05_composite_seen.yaml),
[model YAML](https://github.com/amazon-science/StructRL/blob/6caba1f9538167e0752a7338663e45aac8df965d/configs/model/pi0_5_robocasa365.yaml).

Nominal optimizer-step density is4/256 versus1/512 steps per fresh chunk
slot, an8x difference before masks and rounding. This arithmetic is not an8x
statement about actual gradient magnitude, policy KL or useful task evidence.
Our600/H50 layout has1536 padded chunk slots/update and24 optimizer steps;
successful episodes contribute fewer unmasked loss slots. Task mixtures,
horizons and normalization also differ between studies.

The original [RLinf OpenDrawer example](https://github.com/RLinf/RLinf/blob/cc6bd2588f2ef6e6e7e4da62c3fef96526a16cb1/examples/embodiment/config/robocasa365_opendrawer_ppo_openpi.yaml)
is **pi0**, executes H5 with five denoising steps, uses a2048 global minibatch,
128 episodes/update and four PPO epochs. Our pi0.5 H50/batch256 adaptation is
therefore not that same recipe. RLinf's [official pi0/pi0.5 guide](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/pi0.html)
explicitly recommends VLM-side critics for pi0.5. Its small-batch quickstart
also reduces actor LR5e-6→1e-6 when reducing batch2048→256. That is evidence
for a conservative LR candidate, not a universal scaling law for RoboCasa.

## Mechanisms worth testing, and explanations not established

1. **Critic features/advantages.** pi0.5 integrates proprioception into the
   VLM prefix. The original [piRL critic design](https://arxiv.org/html/2510.25889v1#S4.SS3.SSS2)
   uses those integrated features; expert-side values additionally depend on
   noisy denoising representations. A VLM critic is the first architectural
   correction to qualify. Our expert critic is not mathematically proven
   invalid, but it is not the recommended pi0.5 configuration.
2. **Too much optimization of each sampled batch.** Four passes with a256
   minibatch can amplify erroneous or noisy advantages. This is a stronger
   concrete suspect than claiming that an unchanged LR is automatically safe.
   The [both-rank metric audit](DRAWER-OPTIMIZATION-PRELIMINARY.json) shows no
   obvious exploding-loss or critic-collapse signature, but does not establish
   that policy updates improve native success.
3. **Exploration, execution cadence and task distribution.** These remain
   secondary ablations. H50 is coarser than the original RLinf H5, but matches
   StructRL's released launcher. Shortening executed chunks must preserve the
   checkpoint's network horizon and correct likelihood slicing, and revisit
   per-chunk discounting and compute. Changing SDE noise requires a new
   qualification of initial exploration and likelihoods.

KL=0 is shared with both references. A KL stop can be a useful future guard,
but a generic target such as.02 cannot be copied onto our signed,
chunk-conditional metric and called a validated bound. SDE training versus
ODE evaluation is also part of the reference method's design; finite-step
differences may matter, but that difference is not a discovered implementation
mistake. Native success, not Flora or another learned reward, trained these
sources.

There is no current evidence to copy StructRL's done-shift patch: the existing
[native saved-batch GAE check](../robocasa-ppo-literature-20261102-r1/GAE-CHECK.json)
matched all512 saved returns with maximum error0 and verified correct
success-boundary masking. That is a bounded check, not a proof covering all
possible collectors. Earlier vision-mode limitations remain in the original
campaign record; the accepted current runtime passed zero initial likelihood
replays and native update qualification.

StructRL's [paper](https://arxiv.org/html/2609.36352v1#S4.T1) reports pi0.5
39.3% SFT,41.1% sparse PPO and45.8% structured-reward PPO on its different
composite-task benchmark. Its sparse gain is only1.8 points; its larger
structured result does not show that our terminal-only recipe should give
the same gain. We should not silently add structured rewards and label that
an optimizer repair.

## Proposed corrections, in order

1. **Finish the existing evaluation unchanged.** It supplies the initial
   baseline and independent selected-policy comparison. Preserve all old
   sources, corpora and failed gates.
2. **Correct the next pi0.5 critic to VLM features**, keeping the VLM frozen
   and critic input detached. Qualify strict loads, initial action/likelihood
   behavior, gradient ownership and native value/advantage computation on
   allocated Linux resources. A changed critic needs its own initialization
   and checkpoints; do not resume an old optimizer into this architecture.
3. **Reduce update aggressiveness.** Test one PPO epoch first. Use a larger
   global minibatch through gradient accumulation; for a reference-aligned
   candidate use512 and more fresh episodes per update. If retaining the256
   batch, actor LR1e-6 is a documented conservative candidate. Record the
   resulting optimizer schedule and compare at equal valid environment
   controls, not equal iteration counts. These settings are candidates, not
   established improvements.
4. **Add behavior-based safeguards.** Retain the best fixed-DEV checkpoint
   including0, use predeclared validation patience and independent TEST, and
   qualify a full-batch policy-change guard before selecting a KL threshold.
   Stopping prevents keeping a bad final policy; it does not fix bad
   advantages or supply a monotonic-improvement guarantee.
5. **Resolve causes with controlled comparisons.** A small critic-placement
   by reuse-epoch comparison, followed by replicated seeds, separates those
   two leading suspects. Keep noise, chunk cadence, rewards and initializer
   fixed during that comparison; test those later only if necessary.

The existing campaign launcher intentionally asserts its old batch, epochs,
noise and KL contract. A future recipe requires a separate named configuration
and explicit launcher/contract migration, not weakened checks or edits to an
active checkout. This diagnosis authorizes no extra GPU charges against the
current22GPU-hour evaluation cap. It records the proposed fixes before their
implementation, as requested.
