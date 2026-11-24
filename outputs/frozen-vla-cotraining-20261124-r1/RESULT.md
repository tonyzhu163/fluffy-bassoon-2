# Joint source training helped Panda and lost a stock success on Kinova3

All three evaluations used locked policies. On three tasks held out of the additional Panda training, joint π₀.₅ and adapter training solved 7/12 Kinova3 starts and stock solved 8/12. The joint model rescued no stock failures and lost drawer row 46. On Panda, joint rescued cream-cheese row 49 and preserved all five stock successes. The adapter trained on its own had the highest target point estimate. Its single rescue over stock on Kinova3 and its single loss on Panda occurred on the same moka-pot layout, row 49. [Individual native outcomes](evaluation/NATIVE.md) · [Paired stock comparisons](evidence/stock-pairs.json)

| Frozen deployment on tasks held out of this fit | Kinova3 successes / 12 | Panda successes / 6 |
| --- | ---: | ---: |
| Original π₀.₅ | 8 | 5 |
| Action projection trained alone | 8 | 5 |
| History adapter trained alone | 9 | 4 |
| Jointly trained action projection and history adapter | 7 | 6 |
| Joint model with history cleared | 6 | 4 |
| Joint model with history delayed five controls | 9 | 4 |

On Kinova3 the joint-minus-stock success difference was −8.3 percentage points (95% paired layout-bootstrap interval −25.0 to 0.0). The exploratory adapter-only-minus-stock difference was +8.3 points (0.0 to +25.0). Resampling applies within the three fixed tasks and holds each noise bank fixed. These are few starts and one trained checkpoint. Joint with delayed history scored 9/12 versus 7/12 with current history. The point estimates do not establish a target benefit from co-training or from using the freshest measurements. [Prespecified comparisons](evaluation/evidence/native-results.json) · [Exact resampling for comparisons with stock](evidence/stock_pairs.py)

On two tasks that appeared in the additional fit, every setting succeeded on all four fresh layouts on both arms. In a separate 24-episode control all three paths through the same frozen joint model also solved 8/8. Real and zero history offsets ran on the same compiled program after a common opening. Switching the input changed proposed actions on all eight starts. All eight full-episode outcomes remained successes. These 24 and the earlier 40 episodes used the same eight arm/layout cases. [Task-matched readback](matched-task/NATIVE.md) · [Matched-computation readback](../frozen-vla-matched-history-sham-20261124-r1/NATIVE.md)

## What was trained and what can change on a test arm

The adapter receives four completed controls. For each it reads the seven commands that were actually issued, measured tool translation and jaw movement, current jaw width and a validity bit. A small neural network turns that history into offsets for the next ten proposed actions. The joint version also fits a rank-four update to π₀.₅’s existing final action projection. The two additions together use 10,176 trainable scalars. Images and instructions still go through the stock policy. A new measurement can change the next proposal through a forward pass. Every test attempt starts with an empty buffer. [Implementation and dimensions](model/README.md) · [Observation contract](data/API.md)

The source fit used 192 windows from 21 successful Panda demonstrations across eight tasks. It reserved eight whole demonstrations for validation. All three trained settings used identical expert actions, draws, ordering and update budget. The joint model reduced training flow loss from 0.002682 to 0.002462 but increased held-out source loss from 0.003582 to 0.003657. The paired increase was 0.0000746 (95% demonstration-bootstrap interval 0.0000326 to 0.0001185). The published π₀.₅ checkpoint has its own LIBERO pretraining exposure. Holding tasks and layouts out of this additional fit makes no claim about exposure in the original checkpoint. [Offline fit](evidence/fit/fit.json) · [Data lineage](data/RESULT.md)

## What the comparisons identify

Policy-only changes the final action projection. Adapter-only learns from history against the same expert actions, leaving the original projection intact. Joint trains both together. Zeroing or delaying history then intervenes on the same joint model. The original stock policy supplies the native reference. A separate runtime check found that asking the GPU compiler to add an exact zero could change some issued action coordinates by up to 0.000957 on a fixed input. The held-out evaluator therefore restores exact native arithmetic for zero corrections and records the forced-add difference. Live outcomes comparing joint with zero history measure the deployed program. A further fitting-task control applies real and zero additions with exactly the same compiled sampler after the same opening. [Held-out runtime and pairing](evaluation/NATIVE.md) · [Same-computation comparison](../frozen-vla-matched-history-sham-20261124-r1/NATIVE.md)

A separate audit located 506 successful Sawyer teacher tapes for plate pushing, including 280 layouts with successful Panda tapes. That source is relevant to a follow-up in which arm varies *within* a task. The current fit used Panda alone. The plate teacher always commanded the gripper closed, so its recordings cannot establish grasp or release transfer. [Source-arm and task coverage](source-diversity/RESULT.md)

## Research decision

Keep stock π₀.₅ as the transfer baseline. Adapter-only merits independent replication because it rescued a stock failure, but the matched Panda loss and the interval reaching zero preclude an architecture claim. The next experiment that would distinguish arm-specific adaptation from learning task phase would compare Panda-only supervision against matched Panda-plus-Sawyer supervision on the same plate layouts, equalizing demonstration and update counts, and test frozen policies on fresh third-arm layouts. Include a current-state/command-only control to identify the contribution of measured physical response. This comparison is proposed; no scores for it enter this study.

The three scored jobs completed 108, 40 and 24 native episodes, for **172 terminal trajectories**, on Baige Distributed Training. All used one RTX 4090 and recorded matching parameter digests before and after evaluation. Total measured launcher time across fitting, initial failed checks, the separate numerical probe and all three evaluations was 4,279 GPU-seconds (1.19 GPU-hours). The final native job exits were recorded by 12:35 Shanghai on November 24. [Runs and fit recovery](EXECUTION.md) · [Final scheduler receipt](native-final.jpg)
