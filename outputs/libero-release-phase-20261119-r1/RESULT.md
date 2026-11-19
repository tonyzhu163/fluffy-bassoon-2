# Does wrist motion before release help the robot choose a better retry?

On new practice poses, the reader with wrist motion aligned to jaw opening reached **237/320** native query successes versus **224/320** for a clock-based reader of the same histories. The advantage came from histories involving one placement where the intervention never fired. On the 256 assignments where both timing rules fired on both practice inputs, the wrist reader reached **181** and the clock reader **182**. Each reader benefited from seeing which execution had actually occurred.

## What the reader could observe

All three ridge models learned to predict whether a practice reached its instructed destination on the same 30 source executions. The summary reader used six jaw and wrist statistics. The clock reader also saw when the jaw widened most during intended closure and how long the gripper had been commanded closed. The phase reader added the wrist's height change during closure, its descent from the highest point during closure, and signed vertical travel in the ten controls preceding the jaw-widening window. All weights stayed fixed during evaluation. The simulator supplied the native completion labels for fitting and diagnostics. At query time the controller saw the two public rewards and the two public practice records. It chose a destination instruction for the same motor. [Plan](PLAN.md) · [Model and physical replay](phase.py)

The ten nominal practice commands at poses 15–19 had been recorded previously. New early- and late-opening simulator branches shared those commands and resets. Each policy's choices were saved before replaying the matched query actions at poses 24–27. The test crossed all five cyclic support pairs with each of the four query poses, both assignment orders, interruption choices and hidden goals. This yielded 320 replay assignments per panel from 30 practice and eight query executions in total. Each support pair appears with the same four query poses. The nominal query outcomes existed before this study. The new disturbed trajectories use fixed recorded commands without live post-intervention VLA replanning.

## Paired query results

| Practice alternatives | Reward table | Prior completion reader | New six-feature reader | Six features plus clock | Clock plus wrist motion | Wrist reader with reassigned motion | Motor ceiling |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Early or late opening, primary | 216 | 222 | 230 | 224 | **237** | 210.5 | 280 |
| Normal or early opening | 216 | 249 | 243 | 230 | 241 | 214.25 | 280 |
| Normal or late opening | 280 | 268 | 268 | 268 | 269.5 | 269.5 | 280 |

All rows are out of 320. The three rows share practices and queries. The prior reader used the earlier S02 training data. The comparison between the two rightmost trained readers uses identical BASE data and training labels. Four query poses gave eight candidate motions. Seven reached the intended destination. Allowing the best physical choice for each query gives the 280/320 ceiling. A perfect practice-completion decoder with uniform tie-breaking reached 245/320 in the primary panel. A diagnostic supplied with the actual intervention stage and source-estimated completion rates reached 243. These diagnostics had access to evaluator signals. [Native, final and goal-choice outcomes](phase.json) · [Privileged references and support-pair breakdown](diagnostic.json)

The wrist reader gained 17 and lost four assignments against the clock reader. By first support pose the net differences were 0, +7, +7, 0 and −1. The adjacent positive pairs both contain pose 17. The plate placement there already succeeded without ever meeting either trigger. On the 64 assignments involving that specific practice the wrist reader reached 56 versus 42 for the clock reader. On the remaining 256, the clock reader gained one. This decomposition is descriptive and reuses the same five support poses.

## When both releases actually occurred

Nine of ten early and nine of ten late interventions fired. Own-destination practice completion was 9/10 normal, 1/10 with early opening and 9/10 with late opening. Among the 18 early/late tapes for which both triggers were possible at that start, the wrist reader's completion Brier score was **0.067**, versus **0.085** with the clock and **0.213** with summaries alone. Lower is better. The better probability estimate supplied zero additional net native successes over the clock reader on the trigger-matched queries.

| Early versus late assignments with both triggers possible | Physical trials reused | Six-feature reader | Clock reader | Wrist reader |
| --- | --- | ---: | ---: | ---: |
| Actual execution histories | 18 eligible practice tapes, eight query episodes | 174/256 | 182/256 | 181/256 |
| Independently reassigned execution histories | Same poses, commands and delivered rewards | 161.5/256 | 160/256 | 160.5/256 |

The clock comparator includes execution-dependent jaw timing. Independent reassignment averages all four combinations of early/late motion at the two starts while keeping commands and rewards fixed. On the 144 eligible two-zero assignments the wrist reader reached 81, the clock reader 82 and the six-feature reader 74. An absolute advantage over a reward table, or better completion prediction, would overstate what the added wrist features contributed to native success. [Trigger-matched diagnostic](diagnostic.json)

## Paper decision

Use stage-sensitive completion inference as an interpretable adaptation baseline. Retain the clock comparator, the same-start execution reassignment and all missed triggers. Report the primary unconditional result beside the subset in which a real early/late distinction exists. For the next independent experiment, vary the speed of the approach or transport so elapsed time and placement stage can disagree. Freeze these readers and compare native retry success on fresh bottle or manipulation tasks with live post-fault replanning. The current study supports using public execution history on this bowl layout. A transferable benefit from adding wrist phase above jaw timing remains open.

Baige CPU worker `phase01` at `9a63c5ad` completed 38 simulator episodes. Diagnostics at `376e411b` and `b157f448` reproduced every native choice total and reported the support-pair and donor effects. All three workers exited successfully. Zero incremental GPU-hours. Raw tapes, the choices saved before query replay and the fitted model (`cedcd0620faca0c2ad7acca97215d4fdb37ab9fa442e724fc05b748d66f46746`) are under `/mowenjie/fluffy-bassoon-2/outputs/libero-release-phase-20261119-r1/phase01/`. Training poses 10–14 supplied the practices in the [preceding release-timing experiment](../libero-retry-value-20261119-r1/RESULT.md).
