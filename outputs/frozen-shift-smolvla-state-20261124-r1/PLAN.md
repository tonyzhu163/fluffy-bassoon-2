# Does the hand-rotation sign branch stop frozen SmolVLA on Kinova3?

Registered November 24 before any episode of this run. Exploratory, CPU native reward, frozen SmolVLA `a24f1f38`. This says nothing about π₀.₅ until the same adapter runs there. Part of the [proposal-shift program](../frozen-vla-proposal-shift-20261124-r1/PLAN.md).

## Observation that prompted it

In the [SmolVLA arm comparison](../frozen-shift-smolvla-20261124-r1/RESULT.md) the policy state's rotation X was −3.57 to −2.81 at every replan of all 14 Kinova3 episodes and +2.72 to +3.20 at every replan of all 26 Panda and IIWA episodes. The rotations are nearly the same; the simulator reports the Kinova3 hand quaternion with the opposite sign, and the axis-angle conversion turns that into a 2π shift. Kinova3 scored 0/12, Panda and IIWA 5/18. A sign error in this channel already took frozen π₀ on UR5e to 0/96 ([FINDINGS](../../docs/FINDINGS.md)).

## Question and prediction

If the sign branch is what stops Kinova3, giving the policy the same rotation on the source branch should produce Kinova3 successes on the six layouts, and putting Panda on the Kinova3 branch should remove Panda's. Against it: mapped Kinova3 stays at 0/12, or flipped Panda keeps its success rate.

## Arms

LIBERO-Pro2.0 BASE `libero_goal/3`, layouts 14, 15, 17, 36, 37, 38, LIBERO-release wording, reset seed 42710300 + layout, noise offsets 0 and 1000003, the two used in the arm comparison. All arms rerun on this VM so comparisons are within one machine.

| Arm | State given to the policy |
| --- | --- |
| Kinova3 raw | As reported |
| Kinova3 mapped | Rotation vector `r` with angle θ replaced by `r·(θ − 2π)/θ` when its X is negative: same rotation, source branch. Nothing else changes |
| Panda raw | As reported |
| Panda flipped | The same formula applied when X is positive: same rotation, Kinova3 branch |

The adapter reads only the public state vector. Images, prompt, noise and commands are untouched. Report successes per arm, paired rescues and losses per start and noise seed, and exact intervals. With 12 episodes per arm and a source success rate near 25%, only a large effect is visible.
