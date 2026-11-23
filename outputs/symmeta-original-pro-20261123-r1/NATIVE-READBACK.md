# Original LIBERO-PRO with the native Panda command interface

The frozen π₀.₅ policy succeeded in 19 of 32 single attempts. Adding the frozen seed-41 streaming reader reduced success to 3 of 32 under the identical task, start state, simulator seed, policy noise and native action interface. All three reader successes were also base successes. Sixteen pairs regressed, zero were rescued, and thirteen failed under both methods. The paired difference was −50.0 percentage points. Its 95% interval was −59.4 to −40.6 points from the registered enumeration of all 256 task resamples within suites. The eight selected tasks and one start per task limit generalization. [Full 32-pair summary](evidence/gpu01-summary.json) · [Registered task and reset selection](PROTOCOL.md)

| Released perturbation | Pairs | Base successes | Streaming reader successes |
| --- | ---: | ---: | ---: |
| Semantic rewording | 8 | 7 | 1 |
| Object appearance | 8 | 7 | 1 |
| Position swap | 8 | 4 | 1 |
| Changed task or goal | 8 | 1 | 0 |

Both arms used the released Panda controller, with no added yaw, gain or gripper perturbation. Each ran once. The first ten issued controls agreed in every pair. The reader subsequently changed commands in all 32 episodes, including 9,805 of the 10,125 controls it issued. This supports limiting the retry result to deliberate interface shifts. The reader applied yaw and gain corrections on every episode. The median across episodes of the largest absolute applied yaw was 64.6° (range 12.4° to 165.2°). It never flipped the gripper. Assigning the losses to yaw or gain requires separate interventions. The historical five-family leaderboard includes Environment and covers many more tasks and resets. This four-family pilot estimates a paired effect on its own sample. [Historical denominator and runner details](PROTOCOL.md)

The separate Baige Distributed Training job `job-9me3dekaeqaj` exited zero after 963 seconds on its one Pro6K GPU. An independent readback checked all 32 saved pair records against both native outcome files and the merged rows, including the common prefix and first changed control. Raw records are in `/mowenjie/fluffy-bassoon-2/outputs/symmeta-original-pro-20261123-r1/gpu01/eval/case-*/` on Baige. [Producing revision and job](LAUNCH.md)

The manuscript reports this result in the abstract, main results and conclusion, and distinguishes the pilot from the published historical total. The rebuilt 13-page PDF passed final reference and box checks. The benchmark result appears on pages 1 and 4–5. A separate offline result on map recovery, reported on page 6, comes from a three-seed comparison of the geometric solver, ridge and a continuously trained Perceiver. [Independent map-recovery evidence](../symmeta-wider-family-20261123-r1/RESULT.md)
