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

## Limits

Greedy attempts do not seek information, so these curves are a reference, not an optimum. Uniform positions and table-frame relations are not the real camera or the reachable workspace. "Beside" a stove or cabinet is not a behaviour the release contains. Nothing here says a motor can execute the 20 behaviours on moved layouts.
