# Procedural LIBERO scenes

Exploratory. Generates tabletop scenes for the hidden-rule experiment described in the [extension spec](../../outputs/pi05-history-interface-20261111-r1/EXTENSION-SPEC.md). Scenes are built the same way as the [rule family](../learning-histories/placement.py): one draw on each of a few axes, each a short list.

**Status:** `scenes.py` and `histories.py` have produced a candidate scene/session bank. Eight fixed training layouts passed simulator admission and 48 dropped-bowl predicate controls on Baige. These controls establish goal semantics, not robot reachability. Realized positions changed 3,040 of 23,296 nominal rule labels; three bowls were partly cropped despite their centres being visible. The full bank remains nominal and policy execution is pending. [Geometry evidence](../../outputs/libero-procedural-qual-20261111-r1/GEOMETRY-RESULT.json) · [First case bank and limits](../../outputs/libero-history-cases-20261111-r1/RESULT.md).

## Why this exists

The frozen Behavior Prompting motor scored 20/20 per candidate on the release's fixed layouts and 0/8 on four hand-moved layouts. The same recorder then succeeded 2/2 on a native state ([result](../../outputs/libero-gen-custom-layouts-20261111-r1/RESULT.md)). A motor for varied scenes may need training on varied scenes, and that needs a scene generator and executable demonstrations.

## Scene axes

| Axis | Options in `Axes` | Preset that varies it |
| --- | --- | --- |
| Positions | Uniform in the table rectangle the release uses, spaced by footprint | all |
| Bowl count | 2 or 3 | `objects`, `extended` |
| Bowl kind | Black, or a black and white mix | `objects`, `extended` |
| Receptacles | Subsets of plate, ramekin, cookie box | `objects` |
| Receptacles | Subsets that also include basket and frying pan | `extended` |
| Fixtures | Cabinet and stove, either, or neither, at the release's poses | `objects`, `extended` |
| Beside-targets | An 8 cm table region on each free side of each receptacle | `objects`, `extended` |

To extend an axis, append to its tuple in `PRESETS`. To add an object kind, add it to `BOWLS` or `RECEPTACLES` with a footprint radius. The camera and the workspace are not axes here: the release selects its two views outside the task file, and other tables need other fixture poses.

Every scene lists its behaviours: each bowl placed on each target (receptacle, fixture surface or beside-region), with the native predicate that scores it. `positions` gives 10 behaviours per scene; the other presets gave 4 to 30 across 300 sampled scenes each.

## Use

```bash
python scenes.py --preset positions --n 64 --seed 0 --out <scenes dir>
```

This writes one task file per scene, in the layout of the release's own files, and `manifest.json`. Then, inside an allocated job with the qualified Behavior Prompting runtime:

```bash
python admit.py --source <behavior_prompting checkout> --scenes <scenes dir> --output <new dir>
```

It builds each scene, lets LIBERO place the objects, holds the arm still for 20 controls and checks initial contacts, finite simulator state, movable object centres inside the image, and whether a behaviour is already satisfied. Settling drift and differences from requested region centres are recorded for review without numerical rejection cutoffs. For receptacle targets it also drops each bowl on each receptacle and checks that only the matching predicate turns true. Rejections are recorded, never retried with a different seed. These drop controls do not establish robot reachability. No policy is loaded. Use `--image-size 256` for π0.5 inspection; the default is 128.

## What is assumed and not yet verified

- Footprint radii and fixture keep-out distances in `scenes.py` are estimates. Admission is the real collision check, and its rejection rate will show whether they need changing.
- `white_bowl`, `basket` and `chefmate_8_frypan` exist as LIBERO assets, but whether a bowl placed on a basket or pan satisfies `On` is unknown. Start with `positions`.
- Beside-regions and fixture surfaces have no positive control in `admit.py`.
- LIBERO's reset is assumed to place each object inside its 2 cm region. The requested-layout check measures this.

## Next pieces

1. Scripted or planned pick-and-place demonstrations on admitted scenes, with native success measured for each behaviour. LIBERO-Gen's [existing collector](https://github.com/real-stanford/behavior_prompting/blob/58125e682acceefef3d60c071212dc483a4db1c9/behavior_prompting/train_network/scripts/libero/generate_demonstrations.py) is a starting point, but its task metadata and replay assumptions need adapting to new scenes.
2. Recompute rule labels on realized resets and repeat the ambiguity checks. The candidate history bank below uses nominal table coordinates, not measured camera coordinates.

## Candidate history-learning cases

`histories.py` binds the existing four-axis `placement.py` family to `positions` scenes. It does not change the simulator. Each session has one demonstration, K attempts on distinct scenes, and a query on another scene. The hidden rule persists across the session; the instruction stays generic.

```bash
python scenes.py --preset positions --n 256 --seed 110000 --out <scene-dir>
python histories.py --scenes <scene-dir> --out <new-case-dir> \
  --train 1024 --validation 256 --test 256 --attempts 12 --seed 0
```

- `schedule/` contains scene references and a generic instruction. These are orchestration records, not additional student input tokens.
- `evaluator/` contains hidden rules and nominal target predicates. Keep it outside the student observation/history path. No executed actions, reward labels or demonstrations are fabricated.
- `scenes/` contains the original BDDL scene files. Their fixed placeholder goal is independent of the session's hidden rule; it must not determine stopping or enter student observations.
- `summary.json` describes the generated bank, not simulator or learner performance.

Layouts and rule compositions are disjoint across training, validation and test. All primitive axis choices remain represented in training. Rules with identical behaviour signatures across the generated bank are grouped before splitting. This is an empirical alias check on these scenes, not a proof of global rule equivalence. Test jointly changes layouts and compositions; it does not isolate those two transfer effects.

A candidate demonstration must leave at least three distinct continuations, and at least two answers on its query. The correct behaviours on the attempt scenes must identify the query answer within the full rule family. These constraints prevent trivial demonstrations and sessions that cannot disambiguate the query even with full corrections. They do **not** guarantee identification from binary feedback, successful execution, or a learning gain. Ambiguity is measured without knowledge of the hidden rule's split. Proposals are uniform before these geometry-based constraints; the resulting bank is conditional on them.

The rules here use table XY coordinates: `min_x` does not mean image-left. Fixture landmarks use their requested table positions. LIBERO samples initial placements within regions, so real resets can change near ties or conditional rules. Before collection, bind labels to the settled reset, check observable distinctions in rendered images, and repeat the session checks. Record failures rather than silently resampling based on policy success.

For an actual learning experiment, collect demonstrations and attempted trajectories, evaluate native predicates on their physical states, then compare the same frozen learner with own history, cleared history and matched donor history; intervene on outcome bits separately. All variants use the same case bank and execution budget. Outcome-independent episode horizons and padding are required so masking rewards does not leak the hidden goal through termination. The simulator/collector integration and history training are not implemented by this compiler.
