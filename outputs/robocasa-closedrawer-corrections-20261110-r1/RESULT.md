# CloseDrawer corrections on eight native scenes

Choosing the best action separately on each scene gave 10/16 native closures. The unchanged policy also gave 10/16. All three recurrent failures survived 18 trials with physical base motion and six with a stationary prelude. The forward move lost 3 native successes.

| Fixed strategy | Closures / 16 | Gained vs native | Lost vs native |
| --- | ---: | ---: | ---: |
| Native policy | 10 | 0 | 0 |
| Stationary prelude | 10 | 0 | 0 |
| Negative lateral move | 10 | 0 | 0 |
| Positive lateral move | 10 | 0 | 0 |
| Forward move | 7 | 0 | 3 |

| Scene | Native | Wait | Negative Y | Positive Y | Forward |
| --- | ---: | ---: | ---: | ---: | ---: |
| 983000 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 |
| 983001 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| 983002 | 2/2 | 2/2 | 2/2 | 2/2 | 1/2 |
| 983003 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 |
| 983004 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 |
| 983005 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| 983006 | 2/2 | 2/2 | 2/2 | 2/2 | 0/2 |
| 983007 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 |

Choosing an action from the other noise seed produced 5/8 and 5/8 in the two directions. The best fixed action selected on the other seed also gave 5/8 and 5/8. The prespecified tie break prefers native when it already succeeds. These same-scene upper bounds measure the opportunity for action selection. The experiment did not train a learner or transfer a support episode to a fresh pose. [Scores and arm choices](SUMMARY.json) · [Readout](readout.py).

Measured peak base displacement ranged from 6.8 to 13.2 cm for negative Y, 4.2 to 13.2 cm for positive Y, and 2.9 to 13.2 cm for forward. The respective numbers of moves reaching 12 cm were 14, 8 and 6 out of 16. The cutoff stops the command and allows the base to coast. The forward move lost scene 983006 on both noise seeds and scene 983002 on repeat 1.

## Where the native failures occurred

The native evaluator requires all drawer joints to be at most 5% open. Closest approach is the distance between the end-effector site and the handle center. Contact counts refer to gripper contact with the drawer fixture. The two views in each saved frame are the exterior camera and the wrist camera. [Camera comparisons](NATIVE-FAILURES.jpg).

| Scene | What happened across five strategies and two seeds | Native video |
| --- | --- | --- |
| 983000 | Made gripper contact on 10/10 attempts. Starting from 82.5% open, the drawer got at best to 79.3%. | [Video](r0a/983000-native.mp4) |
| 983003 | Made contact on 10/10. From 89.0% open, the best opening on each attempt lay between 59.3% and 62.9%. | [Video](r0a/983003-native.mp4) |
| 983004 | Never made gripper contact. The drawer stayed at 90.3% open across all ten attempts. | [Video](r0a/983004-native.mp4) |

All three failures and four consistently successful scenes received the same public instruction, “Close the left drawer.” The eighth scene used the right-drawer instruction and always succeeded under the native policy. The language alone does not separate the native failures. The motion records show that 983003 makes partial progress, while 983004 needs a way to reach the drawer. A targeted next diagnostic is to branch a sustained inward push from matched first contact on 983003, using 983000 and a normally successful drawer as controls. The larger history collection remains on hold.

## Execution

Two Baige Pro6K jobs ran source `32686b09c2d470fcf8fc7b714e50eb2a148df2d1`. Each loaded the frozen Human300 π0.5 checkpoint, passed its single-GPU probe, reproduced all eight archived native outcomes and evaluated every fixed strategy. All 80 branches matched the archived simulator state and public observations before acting. Fresh native inference reproduced 15/16 archived terminal hashes. The remaining run, 983004 on repeat 1, still timed out at 600 controls without drawer contact. In total the panel delivered 32,246 controls. [Protocol](PLAN.md) · [Repeat 0 records](r0a/events.jsonl) · [Repeat 1 records](r1a/events.jsonl).

| Worker | Job | Worker time | Scheduler allocation |
| --- | --- | ---: | ---: |
| r0a | job-4j8fm4kzy1o5 | 1,683 s | 28m11s |
| r1a | job-y8pphq83rz7t | 1,564 s | 26m10s |

Total was 0.902 worker GPU-hours and 0.906 allocated GPU-hours. [Scheduler receipt](completed.jpg).
