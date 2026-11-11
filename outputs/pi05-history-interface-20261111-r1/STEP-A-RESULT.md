# Step A: the rule family in geometry only

Run 2026-11-11 on a Mac CPU with [placement.py](../../exploration/learning-histories/placement.py), `--K 12 --n 4000`, seed 1. No simulator, motor or learner. Positions are uniform in a unit square with a minimum separation; image relations use table coordinates as a stand-in for the camera projection. [Raw output](step-a-enumeration.txt).

A rule is one choice on each of four axes: pick (14), anchor (9), offset (2), context (4). All families use the same 20 motor behaviours. The table gives query accuracy of the exact enumeration reference, making its own greedy attempts and seeing one success bit per attempt. Chance is 5%.

| Family | Rules | Rules left after the demonstration | k=0 | k=2 | k=4 | k=6 | k=8 | k=12 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pick × anchor | 126 | 15.5 | 0.516 | 0.675 | 0.832 | 0.948 | 0.980 | 0.995 |
| + offset | 252 | 15.5 | 0.516 | 0.675 | 0.832 | 0.948 | 0.980 | 0.995 |
| + context | 504 | 61.9 | 0.441 | 0.545 | 0.664 | 0.824 | 0.935 | 0.990 |
| + offset + context | 1008 | 61.9 | 0.441 | 0.545 | 0.664 | 0.824 | 0.935 | 0.990 |

A two-part outcome (right bowl, right place) or a teacher correction is worth about 8 to 11 points at k=4 and almost nothing by k=8.

## Reading

- **Pass.** With context, the reference rises from 44% to 82% at six attempts and 94% at eight. K=8 gives a 49-point range for a learner to capture.
- **An axis adds difficulty only if its value changes with the scene.** Offset doubled the rule count and changed nothing: one demonstration shows "on" or "beside" and that holds everywhere. Context quadrupled the ambiguity left after the demonstration, because a demonstration shows what to do in one context only. A count of rules is therefore not a measure of difficulty; rules left after the demonstration is.
- **The demonstration alone is worth 44 to 52%.** Named anchors ("always the plate") are five of nine anchor choices and one demonstration nearly identifies them. Sampling rules evenly across named and relational anchors, or dropping some named ones, lowers this floor.
- **No rule pair is redundant.** The closest pair agrees in 78% of scenes.

## Extension: scene-dependent axes

Second run the same day, `--K 16`, seed 1, [raw output](step-a-six-axes.txt). The script at this commit has six axes; the four-axis table above came from the previous version (`726e178a`). Changes: eight relational anchors instead of four; a predicate that switches the anchor to a partner (anchor-if); offsets beside the receptacle on a fixed side or on the side facing a bowl's start; an optional second step that puts the other bowl on a receptacle. Behaviours rise from 20 to at most 300, because offsets and second steps are new placements.

Success bit only, exact enumeration, greedy attempts:

| Family | Rules | Left after the demonstration | k=0 | k=4 | k=8 | k=12 | k=16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pick, pick-if, anchor | 728 | 99 | 0.371 | 0.593 | 0.886 | 0.987 | 0.997 |
| same, named and relational anchors equally likely | 728 | 95 | 0.407 | 0.614 | 0.894 | 0.986 | 0.997 |
| + anchor-if | 2,912 | 382 | 0.309 | 0.434 | 0.730 | 0.951 | 0.991 |
| + anchor-if + offset | 26,208 | 813 | 0.242 | 0.371 | 0.625 | 0.845 | 0.954 |
| + anchor-if + second step | 29,120 | 785 | 0.251 | 0.403 | 0.639 | 0.856 | 0.963 |
| all six | 262,080 | 1,697 | 0.217 | 0.253 | 0.477 | 0.710 | 0.867 |

With a teacher correction after each attempt, every family reaches 95 to 97% by k=8 (72 to 77% at k=4, except 68% for all six).

- **Each scene-dependent axis is a difficulty dial.** It lowers the demonstration-only floor and lengthens the session a success bit needs: about 8 attempts for 728 rules, 12 for 2,912, 16 for about 27,000, more than 16 for 262,080.
- **Corrections do not get harder.** Eight corrected attempts suffice from 728 rules to 262,080. The success bit is what scales badly, since a failure removes only the rules naming one behaviour.
- **Rebalancing anchors did not lower the floor.** It moved from 37% to 41%. The earlier guess that named anchors caused the high floor was wrong; adding relational anchors and predicates is what lowers it.
- The all-six row uses 300 sessions, so its entries carry roughly ±3 points.

A family for the robot experiment should be chosen by session length: 728 to 2,912 rules for success-bit sessions of 8 to 12 attempts, larger families only with corrections.

## Limits

Greedy attempts do not seek information, so these curves are a reference, not an optimum. Uniform positions and table-frame relations are not the real camera or the reachable workspace. "Beside" a stove or cabinet is not a behaviour the release contains. Nothing here says a motor can execute the 20 behaviours on moved layouts.
