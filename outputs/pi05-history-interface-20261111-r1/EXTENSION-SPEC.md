# Extending LIBERO-Gen for hidden-rule sessions

A specification, not a launch. Written 2026-11-11 from main `2f148a02`. It says what the Spatial Combination release lacks for the history experiment, what to add, and in which order. It builds on the [custom-layout stage](../libero-gen-custom-layouts-20261111-r1/PLAN.md) owned by the LIBERO-Gen pilot and changes nothing in it.

## What the release contains

Tallied from the static [release audit](../libero-gen-history-pilot-20261111-r1/RELEASE-AUDIT.json), 193 task files:

- Every goal is `On(bowl, target)`. The picked object is always one of two black bowl instances (97 and 96 tasks).
- There are 12 targets: plate, ramekin, cookie box, stove cook region, cabinet top, five table regions, and each bowl on the other.
- Object positions are fixed per task file. The language names objects by their fixed neighbours ("the black bowl next to the ramekin").

So a scene offers at most 2 bowls × about 10 targets, roughly 20 behaviours. The pick is binary. A rule cannot transfer across layouts because the layouts do not vary and the descriptions are tied to them.

## What to add

### 1. Rules as functions of the scene

A rule is a pick rule composed with a place rule. Each is computed from observable geometry, so it names a behaviour in any layout.

| Part | Family | Count |
| --- | --- | ---: |
| Pick | Leftmost, rightmost, uppermost, lowermost bowl in the fixed agent-view image | 4 |
| Pick | Bowl nearest to, or farthest from, a landmark (plate, ramekin, cookie box, stove, cabinet) | 10 |
| Place | A fixed target (the release's ten non-bowl targets) | 10 |
| Place | A relational target: the receptacle nearest the picked bowl, farthest from it, leftmost, rightmost | 4 |

That is 14 × 14 = 196 rules over 20 behaviours per scene. Hold out a quarter of the compositions, with every pick rule and every place rule still seen in training, as in the toy. Image-frame relations use the projection the custom-layout stage already audits (pixel margins, no ties).

With two bowls, any two pick rules either agree or disagree in a scene, so the pick contributes one bit per scene. The place rule carries most of the ambiguity. A third bowl would widen the pick but needs a motor that has never seen three bowls, so it is deferred to step 4.

### 2. A layout generator

Sample bowl and receptacle positions under the checks the custom-layout stage defines: no initial contact, settling drift within tolerance, image-axis separation, no goal initially true. Add one check per session: the scenes chosen for attempts and query must keep at least three hypotheses apart, given the demonstration.

Sampled layouts replace the current hand-placed templates once those pass. Geometry01's failures (contact, drift, a cabinet collision) show that free sampling needs rejection on simulator checks, not on static coordinates alone. Rejection must use geometry only, never a policy outcome.

### 3. Outcomes for every hypothesis

Record one execution and evaluate all 20 native `On` predicates on its final state through the official evaluator. This is the sidecar route the custom-layout stage qualifies for two goals, extended to 20. The hidden rule then only selects which recorded bit the learner sees. Stopping, horizon and padding stay tied to the executed skill, as in that stage's recording contract.

### 4. Motor coverage on new layouts

This is the main cost and the main risk. The motor must execute each of the 20 behaviours wherever the generator puts the objects. Neither released motor has been run on a moved layout yet. Three outcomes, in order of preference:

1. The released Behavior Prompting motor holds at 85% or more per behaviour under position changes. No new training.
2. It holds for a subset of targets. Restrict the rule family to those targets.
3. It does not hold. Collect new demonstrations on generated layouts and fine-tune the motor before any history work. Scripted pick-and-place can produce them, since every goal is a bowl placed on a known target.

A third bowl or new objects fall under case 3 by default.

## Order of work

| Step | What | Simulator or GPU | Pass condition |
| --- | --- | --- | --- |
| A | Rule family in geometry only: sample positions, compute all 196 rules, enumerate hypotheses exactly as the toy does | None | With one demonstration and K attempts, the enumeration reference rises from near the demonstration-only level to at least 80%. This fixes K and the rule family before any rollout |
| B | Custom-layout stage as planned: two pick rules, one target, 16 layouts | Owned by the LIBERO-Gen pilot | That stage's own admission |
| C | Two bowls × four targets on sampled layouts, about 50 layouts, every behaviour executed once | About 400 episodes | At least 85% native success per behaviour; all predicates match the primary environment |
| D | Full 20 behaviours, several hundred layouts | 6,000 to 10,000 episodes | Same, per behaviour; held-out compositions and layouts frozen before collection |
| E | Third bowl or new objects | New demonstrations and motor fine-tuning | Only if D succeeds and the paper needs a wider pick |

Step A needs no cluster and can run on the Mac; it reuses `exploration/learning-histories/sessions.py` with a different rule function.

## Cost

Panel03 ran 40 episodes in 380 seconds of process time, about 9.5 seconds per episode on one GPU. At that rate step C is about 1 GPU-hour and step D is 16 to 26 GPU-hours. Recording every episode to the fixed 562-control horizon will raise this by an unmeasured factor; measure it in step C before sizing D. Step D alone can approach the 50 GPU-hour approval line with the gates of the [main plan](PLAN.md), so ask before launching it.

## Known limits

- A single success bit over 20 behaviours is weak evidence per attempt: a failure removes one behaviour. Step A will show whether K stays practical. If it does not, add teacher corrections or a two-part outcome (right bowl, right target) and say so in the claim.
- All rules stay within "put a bowl on something". The claim this supports is inference of unseen compositions of seen relations in one task family, not unseen rule families.
- Fresh layouts are position changes with the same objects, camera and background.
