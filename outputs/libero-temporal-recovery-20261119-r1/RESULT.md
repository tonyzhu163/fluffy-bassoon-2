# Failure cause can change the useful correction for the same bottle and destination

On four of twelve practice/query pose-and-goal pairs, both practice attempts earned zero and the two faults required different executable corrections on the next reset. All four involved putting the bottle on the cabinet. A retreat and second grasp worked after the released grasp disruption (D17). The transport-recovery commands worked after the released object drop (D18). Neither recovery succeeded under the other fault on these four pairs. The nominal motor tape failed under both. Three of the four pairs also satisfied this test when success had to persist through control 300. [Physical branches](branches.jsonl) · [Case-by-case decisions](result.json)

Both support restrictions fired in all twelve pairs. The support commands were identical across faults and the public query start matched across alternatives. Ten practice pairs earned two zeros. Four of their query pairs had a successful action in both conditions. No pair had exactly equal final agent and wrist images across the two support histories. In the four paired reversals, the mean absolute camera differences were 2.31 and 5.68 intensity units out of 255. Pixel inequality alone does not establish useful decodability.

| Query program, all 12 poses and goals per fault | D17 any-time / terminal hits | D18 any-time / terminal hits |
| --- | ---: | ---: |
| Default independent π0.5 controls | 0 / 0 | 0 / 0 |
| Retreat and repeat grasp | 10 / 9 | 0 / 0 |
| Archived π0.5 transport recovery | 0 / 0 | 4 / 4 |

The grasp controller follows the nominal pose-specific plan until twelve controls after sustained closure. It opens the jaws, retreats vertically and returns to the pregrasp waypoint before resuming the nominal commands. The transport program replays the motor's commands originally collected with live replanning under D18 at that query pose. This gives the menu a favorable, condition-informed candidate. The tests above execute the same candidate programs under both native faults. They assess whether a menu contains physically different solutions. The reader selects a whole program at reset. It has no mechanism for inventing a new recovery, and a reactive controller could potentially infer the condition partway into the query.

## Can the support predict which retry succeeds?

For each public goal, fit a nearest-neighbor reader directly to the three native query outcomes. Remove every fitting pair that shares either the test support pose or the test query pose. Each fold has three remaining pairs and six fitting cases. No test fault label enters the decision. The fixed arm chooses the program with the most fitting successes. A tie favors default, then regrasp, then transport. Rows are dependent and each strict pair contributes two query choices.

| Support information | Correct on the eight strict cases | Same-start swapped history | Native successes across all 24 cases |
| --- | ---: | ---: | ---: |
| Fitting-selected fixed action | 4 | 4 | 10 |
| Last dual-camera view, spatial RGB means | 2 | 2 | 4 |
| Last wrist/jaw observation | 2 | 2 | 4 |
| Thirteen dual-camera views, spatial RGB means | 3 | 2 | 6 |
| Wrist/jaw trajectory in twelve time bins | 4 | 2 | 6 |
| Six established wrist/jaw features | 6 | 0 | 11 |

Perfect selection from the three options could reach 14/24: ten grasp recoveries and four transport recoveries. The six-feature reader selected a successful program on six of eight strict cases, five still successful at control 300. Swapping the actual support histories made it miss every strict case. The features summarize jaw widening under commanded closure, jaw width and wrist travel. This is evidence of execution-specific information in motion history and a small gain over the fixed policy across the entire panel. Exact final observations differed throughout, so this experiment does not establish that interpreting temporal order is necessary. [Frozen-fold choices and endpoint distances](result.json)

## Frozen visual representations

An exploratory comparison reused the previous frozen π0.5 LIBERO and PaliGemma PT-224 visual towers. Both saw the same cameras and fitting folds. Using final-image features alone, the native-outcome reader chose 1/8 correct retries with π0.5 and 2/8 with PaliGemma. Using temporal differences of five frame embeddings raised those to 4/8 and 3/8. On all 24 conditions the five-frame visual scores were 8 and 4, versus 10 for the fitting-selected fixed policy. A separate source-fault-supervised centroid diagnostic could already distinguish the two faults on 5/8 strict cases from π0.5 terminal features. Gold fault labels were used only for that auxiliary diagnostic's fitting folds. These small pose-disjoint probes do not warrant attributing the retry rule to robot pretraining. [π0.5 reads](robot.json) · [General PaliGemma reads](vlm.json)

## Earlier controls and execution

The earlier bowl and bottle reversal recordings also yielded zero exact final-image matches across opposite eligible histories. A source-fitted 2 × 2 terminal RGB reader transferred at 6/12 on bottles versus 7/12 for its matched five-frame RGB version. The earlier per-destination 8 × 8 terminal bowl reader had already reached 4/4. Their representations and fitting rules differ. The archive coverage audit found one fitting cadence crossover whose nominal and delayed support rewards were different. The four-cadence development panel had none on 16 pairs. [Archived endpoint audit](endpoints/readout.json) · [Motor-menu coverage](coverage/RESULT.md)

Baige development CPU executed 96 native branches with software camera rendering, six bottle poses and two destination prompts in the same Goal 2 environment. The released D18 support replays matched the archived commands, images, states and predicates, and all D18 transport replays reproduced their recorded states and native predicates. The release pin was `e5500c69`, the collection pin `eb5e109c` and the direct readout pin `777d043c`. Collection took 156 seconds. All 24 physical support tapes remain on Baige under the run's `cpu01/`. This custom development experiment used zero GPU-hours and no official evaluation tasks. [Plan](PLAN.md) · [Launch](launch.txt) · [Tape hashes](tapes.sha256)
