# Procedural LIBERO scenes

Exploratory. Generates tabletop scenes for the hidden-rule experiment described in the [extension spec](../../outputs/pi05-history-interface-20261111-r1/EXTENSION-SPEC.md). Scenes are built the same way as the [rule family](../learning-histories/placement.py): one draw on each of a few axes, each a short list.

**Status:** `scenes.py` runs and its output was checked for structure on a Mac. `admit.py` has only been syntax-checked. No generated scene has been loaded in a simulator.

## Why this exists

The frozen Behavior Prompting motor scored 20/20 on the release's fixed layouts and 0/8 on four hand-moved layouts ([result](../../outputs/libero-gen-custom-layouts-20261111-r1/RESULT.md); a control for the new reset path was still pending when this was written). If that holds, a motor for varied scenes has to be trained on varied scenes, and that needs a scene generator before it needs anything else.

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

It builds each scene, lets LIBERO place the objects, holds the arm still for 20 controls and rejects a scene for any of: failure to build or place, initial contact, settling drift above 5 mm, an object centre within 8 pixels of the image edge, a placement that differs from the requested layout, or a behaviour already satisfied. For receptacle targets it also drops each bowl on each receptacle and checks that only the matching predicate turns true. Rejections are recorded, never retried with a different seed. No policy is loaded.

## What is assumed and not yet verified

- Footprint radii and fixture keep-out distances in `scenes.py` are estimates. Admission is the real collision check, and its rejection rate will show whether they need changing.
- `white_bowl`, `basket` and `chefmate_8_frypan` exist as LIBERO assets, but whether a bowl placed on a basket or pan satisfies `On` is unknown. Start with `positions`.
- Beside-regions and fixture surfaces have no positive control in `admit.py`.
- LIBERO's reset is assumed to place each object inside its 2 cm region. The requested-layout check measures this.

## Next pieces

1. Scripted pick-and-place demonstrations on admitted scenes, one per behaviour. Every goal is a bowl on a known target, so a waypoint script should do.
2. Rule labels: evaluate `placement.py`'s rule family on the admitted scenes' projected positions instead of on uniform table coordinates.
