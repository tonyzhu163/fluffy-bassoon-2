# Atomic task-pool and portfolio review

2026-11-04. Requested by the human: justify Faucet versus Kettle from scratch,
check an approximately 30–60% baseline range, and explain both the 18-task pool
and selections from it. This is a review of existing evidence, with 0 new GPU
seconds, simulator controls or SGD. Recommendations here are not an adopted
replacement of any active or completed experiment contract.

## Why18

The pool is the complete official `TARGET_TASKS["atomic_seen"]` at pinned
RoboCasa commit `3443cdf3dd20f9ae8ed41adba442059b506891d0`. RoboCasa's 50 target
tasks consist of 18 Atomic-Seen, 16 Composite-Seen and 16 Composite-Unseen tasks.
The broader simulator contains 65 atomic tasks. We did not obtain18 by selecting
our policy's most favorable results.

Primary sources:
- [Official benchmark split](https://robocasa.ai/docs/build/html/benchmarking/foundation_model_learning.html).
- [Official atomic task scope](https://robocasa.ai/docs/build/html/tasks/atomic_tasks.html).
- [Our original18-task screening contract](../robocasa-atomic-baseline-sweep-20261103-r1/PLAN.md).

The methodological rationale for starting here is an externally defined,
reproducible manipulation pool with individual native success predicates,
before introducing long sequences of dependent subtasks. This supports an
initial atomic-skill adaptation study; it cannot establish full 365-task coverage.
All 18 are foundation-pretraining-seen. Our ID/OOD distinction concerns later
source/PPO/meta-training exposure, not tasks unseen to the foundation model.
The 18 include 17 manipulation tasks and NavigateKitchen; excluding navigation from
a manipulation portfolio is a separate, explicit scope decision.

## Why five source tasks

The human clarified that five PPO source tasks is a practical balance for
debugging, inspired by the seen/held-out structure of the RoboTwin comparison.
It is a compute and iteration-speed choice, not a number derived from the
18-task pool or proof that five tasks suffice for broad meta-generalization.
The referenced [VLRLA simulation experiment](https://arxiv.org/html/2610.19517v1#S5)
actually uses six seen and six unseen tasks; our five-plus-five is a smaller
pilot with that general structure.

Screening the 18 with the unchanged checkpoint does not require PPO on all 18.
Likewise, the official Atomic-Seen, Composite-Seen and Composite-Unseen labels
do not require choosing five from each category for this debugging phase.
Specific source/held-out memberships still need the baseline, skill-coverage
and exposure justification below. No broader training campaign is adopted.

## Evidence for every candidate

First success is the primary endpoint. Counts below come from different fresh
banks and execution protocols; they are not pooled or treated as a single
matched 18-task experiment. A 30–60% point-estimate target was raised by the
human in this review; the earlier admission gate was5–70%.

| Task | Best relevant baseline evidence | Horizon | Assessment against preferred30–60% |
| --- | --- | --- | --- |
| CloseBlenderLid |1/20,5%|900|Far below target; weak initial success signal.|
| CloseFridge |28/50,56%|900|In range; candidate opening/closing holdout.|
| CloseToasterOvenDoor |21/64,32.8%; earlier5/20,25% at 450|600|In range at actual source horizon.|
| CoffeeSetupMug |20/64,31.3%|600|In range; source task with pick/place behavior.|
| NavigateKitchen |2/20,10%|450|Below range and different action/skill family.|
| OpenCabinet |35/50,70%;15/20,75%|1050|Above target under measured native horizon.|
| OpenDrawer |40/64,62.5%|600|Slightly above preferred upper edge; substantial remaining failures.|
| OpenStandMixerHead |34/50,68%;13/20,65%|450|Above preferred range;50-case evidence already exists.|
| PickPlaceCounterToCabinet |24/50,48% in common-before;28/50,56% in calibration|600|In range. Its historical37/50,74% was measured at 750 on another bank and is insufficient grounds to exclude it at 600.|
| PickPlaceCounterToStove |10/20,50%|600|In range, but only a coarse20-case screen.|
| PickPlaceDrawerToCounter |32/64,50%|600|In range; existing source task.|
| PickPlaceSinkToCounter |19/20,95%|900|Near ceiling under measured native horizon.|
| PickPlaceToasterToCounter |15/20,75%|600|Above target; coarse screen.|
| SlideDishwasherRack |14/20,70%|450|Above target; coarse screen, one end-state loss.|
| TurnOffStove |9/64,14.1%|600|Clear low-success outlier among existing source tasks.|
| TurnOnElectricKettle |32/50,64%;end29/50,58%|600|First success slightly above preferred range; all 32 first successes already occurred by 450.|
| TurnOnMicrowave |8/50,16%;separate6/20,30%|450|Uncertain low-success candidate; larger existing bank is below target.|
| TurnOnSinkFaucet |18/50,36%;end18/50,36%|600|In range; latest independent qualification complete.|

Sources: [source policy-zero64-case panels](../robocasa-flow-sde-10m-per-task-20261103-r1/RESULT.md),
[600-control qualification and50-case common-before panels](../robocasa-joint-h600-preparation-20261102-r4/RESULT.md),
[remaining12-task screen](../robocasa-atomic-baseline-sweep-20261103-r1/RESULT.md),
and [new Faucet/Kettle qualification](../robocasa-buttons-replacement-qualification-20261104-r1/RESULT.md).
Existing50-case OOD results are relevant even though the newer portfolio plan
quotes only the 20-case screen for some tasks. Variation across banks is
uncertainty, not permission to select whichever estimate falls inside a cutoff.

## Faucet versus Kettle, reconsidered

| Criterion | Faucet | Kettle |
| --- | --- | --- |
| Fresh first success |18/50,36%;Wilson95%24.1–49.9%|32/50,64%;Wilson95%50.1–75.9%|
| Maximum possible absolute improvement from this baseline |64percentage points|36percentage points|
| Binary outcome variance p(1-p) |0.2304|0.2304|
| Positive examples per 50 episodes |18|32|
| Final success at 600 |18/50|29/50;3first successes lost by the end|
| Native goal |Faucet handle state makes water_on true|Press lever; native turned_on state|
| Native horizon |600|450; prospectively extended to600for qualification|
| Existing portfolio consequence |Moves a previous OOD task into ID, requiring another OOD choice|Preserves Faucet as a held-out task|

For the human's preferred baseline band and improvement-headroom objective,
Faucet is the stronger first candidate. Kettle supplies more positive episodes,
and the equal binary variances mean there is no Bernoulli-variance argument
that Faucet automatically trains a better critic. First-success reward also
means Kettle's three lost terminal successes do not invalidate its first-success
qualification. No improving teacher or Flora reward agreement is established
for either candidate. All32Kettle successes precede450, so the new 64% estimate
cannot be attributed to the extra 150 controls.

The earlier claim that Kettle would make Microwave transfer nearly ID was too
strong. Kettle uses a lever, Microwave a button; shared switch-actuation
semantics is a hypothesis about transfer distance, not measured transfer or a
reason by itself to reject Kettle.

## What makes a defensible next selection

1. Fix the pool and claims first: official 18 Atomic-Seen tasks; manipulation-only
   exclusions explicit; OOD means held out from the particular new learning stage.
2. Qualify native first success at each task's declared deployment horizon,
   same motor/runtime/batching protocol, at least 50 fresh cases per candidate;
   use intervals and retain all failures. Prefer 30–60%, with near-boundary
   cases labeled rather than silently rounding into the band.
3. Preserve skill/fixture coverage with explicit coarse groups, and avoid
   counting close variants as independent kinds of transfer. Our 3 coarse
   categories are a portfolio convenience, not RoboCasa's complete taxonomy.
4. Select using native baseline/coverage evidence, before inspecting the new
   learner's transfer gains. Freeze new ID/OOD roles before source training.
5. Separately require each trained source to improve on fresh evaluation;
   baseline headroom alone does not qualify a teacher.

There is no fully qualified 10-task set known to be inside a strict30–60% band.
With Faucet as the proposed Stove replacement, ID point estimates become
31.3–62.5%. OOD still has Microwave 16–30% and Mixer 65–68%, and moving Faucet
leaves an OOD slot. CounterToCabinet merits reconsideration at 600, but it was
used in older steering-PPO pilots; a future run may hold it out only with an
explicit exposure boundary and exclusion of those older histories from fitting.
Do not automatically relabel it as never previously used.

The current 5+5 selection is therefore provisional under this tighter goal.
A smaller well-qualified portfolio or a prospectively justified broader pool
is preferable to forcing 10 tasks through inconsistent eligibility evidence.
No new task sweep, source training, relabeling of active evaluations, or
student fitting is authorized by this review.
