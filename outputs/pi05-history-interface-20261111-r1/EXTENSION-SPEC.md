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

A rule is one choice on each of a few independent axes, each computed from observable geometry, so it names a behaviour in any layout. Each axis is a short list; extending the family means appending to one list, with no new motor behaviour.

| Axis | Choices | Count |
| --- | --- | ---: |
| Pick | Leftmost, rightmost, uppermost, lowermost bowl in the image; nearest to or farthest from each of five landmarks | 14 |
| Anchor | A named receptacle (five), or a relation: nearest the picked bowl, farthest from it, leftmost, rightmost | 9 |
| Context | None, or a scene predicate that flips the pick when true (three predicates such as "plate left of ramekin") | 4 |
| Offset | On the receptacle or beside it | 2 |

That is 1,008 rules over 20 behaviours. [Step A](STEP-A-RESULT.md) checked this in geometry only. It showed that an axis adds difficulty only when its value changes with the scene: context quadrupled the ambiguity left after a demonstration, while a constant offset added none. New axes should therefore be scene-dependent. Candidates, in rough order of motor cost: a context predicate on the anchor as well as the pick; an offset that depends on the scene ("beside it, on the side facing the other bowl"); a second step with the other bowl, which the release's Goal Chain split already covers.

Hold out a quarter of the compositions, with every choice on every axis still seen in training. With two bowls the pick is one bit per scene; a third bowl is deferred to step 4.

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
| A | Rule family in geometry only, enumerated exactly as the toy does | None | **Passed** with the context axis: 44% after the demonstration, 82% at six attempts, 94% at eight. Use K=8. [Result](STEP-A-RESULT.md) |
| B | Custom-layout stage as planned: two pick rules, one target, 16 layouts | Owned by the LIBERO-Gen pilot | That stage's own admission |
| C | Two bowls × four targets on sampled layouts, about 50 layouts, every behaviour executed once | About 400 episodes | At least 85% native success per behaviour; all predicates match the primary environment |
| D | Full 20 behaviours, several hundred layouts | 6,000 to 10,000 episodes | Same, per behaviour; held-out compositions and layouts frozen before collection |
| E | Third bowl or new objects | New demonstrations and motor fine-tuning | Only if D succeeds and the paper needs a wider pick |

Step A ran on the Mac with `exploration/learning-histories/placement.py`.

## Cost

Panel03 ran 40 episodes in 380 seconds of process time, about 9.5 seconds per episode on one GPU. At that rate step C is about 1 GPU-hour and step D is 16 to 26 GPU-hours. Recording every episode to the fixed 562-control horizon will raise this by an unmeasured factor; measure it in step C before sizing D. Step D alone can approach the 50 GPU-hour approval line with the gates of the [main plan](PLAN.md), so ask before launching it.

## Known limits

- One demonstration alone gives 44% in step A, because named anchors are easy to identify. Rebalance the anchor choices before fixing the family.
- All rules stay within "put a bowl on something". The claim this supports is inference of unseen compositions of seen relations in one task family, not unseen rule families.
- Fresh layouts are position changes with the same objects, camera and background.
