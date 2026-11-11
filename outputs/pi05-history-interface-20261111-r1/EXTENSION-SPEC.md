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
| Pick-if | None, or a scene predicate that switches to the other bowl | 4 |
| Anchor | A named receptacle (five), or one of eight relations (nearest the picked bowl, leftmost, nearest the other bowl, and so on) | 13 |
| Anchor-if | None, or a scene predicate that switches to a partner anchor | 4 |
| Offset | On the receptacle; beside it on a fixed side; beside it on the side facing a bowl's start | 9 |
| Then | Nothing, or the other bowl goes on a named or relational receptacle | 10 |

[Step A](STEP-A-RESULT.md) enumerated these in geometry only. Two findings shape the design:

- An axis adds difficulty only when its value changes with the scene. A constant offset doubled the rule count and changed nothing.
- Each scene-dependent axis is a dial. With a success bit only, the exact reference needs about 8 attempts for 728 rules (the first three axes), 12 for 2,912 (four axes) and more than 16 for all six (262,080). With a teacher correction per attempt, eight attempts suffice at every size.

So the family can grow by appending to an axis or adding one, and the session length tells us how far to go. Start with the first four axes: they use only the 20 behaviours the release already contains. Offset and the second step add placements the motor has not been shown (beside a receptacle on a chosen side; two moves in one episode, which the release's Goal Chain split partly covers).

Hold out a quarter of the compositions, with every choice on every axis still seen in training. With two bowls the pick is one bit per scene; a third bowl is deferred to step 4.

### 2. A procedural scene generator

Scenes get the same treatment as rules: a scene is one draw on each of a few axes, each a short list or range. LIBERO already has the machinery at the pinned commit `8f1084e3`: a scene template class declares fixture and object counts and an initial region (centre and half-width) per object, tasks are registered against it with goal predicates, and `generate_bddl_from_task_info()` writes the task files ([upstream example](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/8f1084e3132a39270c3a13ebe37270a43ece2a01/scripts/create_libero_task_example.py)). The released tasks are fixed outputs of that route. The custom-layout stage instead writes object positions into the simulator state directly, which also works and needs no new task file.

| Scene axis | Draws | New motor behaviour? |
| --- | --- | --- |
| Positions | Bowl and receptacle positions sampled in the reachable table area | No, if the motor tolerates moved objects (unmeasured) |
| Receptacles present | Any three or four of plate, ramekin, cookie box, basket, frying pan | New targets for basket and pan |
| Fixtures | Stove and cabinet present or absent, at one of a few positions | No |
| Bowl appearance | Black, red or white bowl (all three ship with LIBERO) | Probably small; it opens pick-by-colour rules |
| Bowl count | Two or three (an upstream kitchen scene has three) | Unqualified: no released demonstration has three |
| Camera | The release's two views | No, both are in the release |
| Workspace | Kitchen, living-room or study table | Yes: new backgrounds and heights |

Every sampled scene passes the custom-layout stage's checks in the simulator: no initial contact, settling drift within tolerance, image separation, no goal initially true. Rejection uses geometry only, never a policy outcome. One more check per session: after the demonstration, the attempt and query scenes still separate at least three hypotheses.

Rules and scenes share one vocabulary. A rule refers to roles ("the bowl", "a receptacle", "a landmark"), not instances, so adding a receptacle type or a bowl colour to the scene lists extends every rule that mentions that role. Held-out scenes can then be defined per axis: unseen positions (easy), unseen receptacle subsets, an unseen bowl colour, the other camera.

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
| A | Rule family in geometry only, enumerated exactly as the toy does | None | **Passed.** Four axes, 2,912 rules: 31% after the demonstration, 73% at eight attempts, 95% at twelve. [Result](STEP-A-RESULT.md) |
| B | Custom-layout stage as planned: two pick rules, one target, 16 layouts | Owned by the LIBERO-Gen pilot | That stage's own admission |
| C | Two bowls × four targets on sampled layouts, about 50 layouts, every behaviour executed once | About 400 episodes | At least 85% native success per behaviour; all predicates match the primary environment |
| D | Full 20 behaviours, several hundred layouts | 6,000 to 10,000 episodes | Same, per behaviour; held-out compositions and layouts frozen before collection |
| E | Third bowl or new objects | New demonstrations and motor fine-tuning | Only if D succeeds and the paper needs a wider pick |

Step A ran on the Mac with `exploration/learning-histories/placement.py`.

## Cost

Panel03 ran 40 episodes in 380 seconds of process time, about 9.5 seconds per episode on one GPU. At that rate step C is about 1 GPU-hour and step D is 16 to 26 GPU-hours. Recording every episode to the fixed 562-control horizon will raise this by an unmeasured factor; measure it in step C before sizing D. Step D alone can approach the 50 GPU-hour approval line with the gates of the [main plan](PLAN.md), so ask before launching it.

## Known limits

- One demonstration alone gives 22 to 41% in step A depending on the family. Report gains relative to that floor, not to chance.
- All rules stay within "put a bowl on something". The claim this supports is inference of unseen compositions of seen relations in one task family, not unseen rule families.
- Fresh layouts are position changes with the same objects, camera and background.
