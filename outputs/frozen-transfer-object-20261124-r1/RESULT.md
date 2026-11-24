# Can a wrist camera tell that a different arm is carrying an object?

**Stop the fixed-center wrist-motion reader.** On the preregistered carry-versus-empty test, the source-trained camera and proprioception readers both ordered every opposing pair correctly. Proprioception had lower mean squared probability error. Giving extra weight to motion near the center of the wrist image also failed to improve on the spatial image control in the direct arm swap. This run used two Baige CPU cores and zero GPU-hours. It measured physical grasp recognition under recorded commands. Native task success from using the reader to change an action remains unmeasured. [Preregistration](PLAN.md) · [Machine-readable scores and intervals](evidence/result.json)

## Same commands on different layouts

Each of 22 untreated frozen π₀.₅ tapes ran twice on its own arm, first on its recorded layout and also on layout 20. These 44 CPU replays covered eight tasks on Kinova3, IIWA and UR5e. Five tasks and 11 command tapes yielded **146 same-tape, same-time opposing windows across 17 contiguous bouts**. The original layout supplied the carried branch 144 times and the crossed layout twice. Every model trained on Panda with the evaluated task and all target layout numbers excluded. Smaller Brier loss means a more accurate probability. [Every opposing pair](evidence/opposing.csv.gz) · [All eligible decisions](evidence/predictions.csv.gz)

| Public reader | Opposing-pair Brier | Fraction ordering the carried hand higher |
| --- | ---: | ---: |
| Elapsed time and close duration | 0.3704 | 0.500 |
| Time, measured jaw and wrist proprioception | **0.0682** | **1.000** |
| Same sensors and wrist images, four corner squares | 0.1022 | 0.997 |
| Same sensors and wrist images, center square | 0.0956 | **1.000** |

The center reader's Brier minus proprioception was **+0.0274** (95% whole-task bootstrap interval **−0.0349 to +0.0756**). Center minus corners was **−0.0066** (**−0.0624 to +0.0363**). Equalizing contiguous bouts instead of windows within each tape gave +0.0315 and −0.0045. Among the five tasks absent from the entire Panda source set, two produced opposing events, both object tasks. Center scored 0.1226 averaged over these two tasks, corners 0.0765 and proprioception 0.0452. The other three offered no opposed layout pair. [Task, arm and bout breakdowns](evidence/result.json)

| Target arm | Opposing windows | Informative tasks | Center Brier | Corners Brier | Proprioception Brier |
| --- | ---: | ---: | ---: | ---: | ---: |
| IIWA | 31 | 3 | 0.0735 | 0.1161 | **0.0407** |
| Kinova3 | 72 | 3 | 0.0945 | 0.1411 | **0.0835** |
| UR5e | 43 | 3 | 0.1137 | 0.1155 | **0.0699** |

Rows above average tasks within each arm. The primary result averages tapes within each arm/task, then arms within each task, then tasks. Ninety-four opposed windows came from the two-moka-pots task. Grouping prevents those frames from dominating the effect. Intervals resample five whole informative tasks 20,000 times and condition on these source fits, command draws and bodies. [Outcome and seed coverage](evidence/coverage.csv) · [Estimator](score.py)

**Why the distinction matters.** A post hoc check of the untrained finger-gap reading alone found a larger aperture in the carried hand in **146/146** layout-opposed windows and **14/14** arm-opposed windows. Across the complete target panel, empty hands measured 1.0–41.7 mm and loaded hands 4.1–53.8 mm. A single absolute threshold cannot separate all of those readings. The pair provides a matched reference for interpretation; a deployed robot would see just one branch. [Independent reconstruction](evidence/independent-check.json) · [Public aperture and labels](evidence/predictions.csv.gz)

Across all closed-command target windows, 364 were carried, 165 empty and 148 ambiguous. On original layouts there were **253 carried and only 3 empty** among eligible decisions. On this one-sided mixture, the center camera beat proprioception (0.0735 vs 0.1245), while corners scored 0.0554. On crossed layouts the corresponding losses were 0.1310, 0.0967 and 0.1345. The different class mixtures and control rankings explain why favorable aggregate original-layout accuracy alone would have given a poor basis for promoting the candidate. [All-decision scores](evidence/result.json) · [Window accounting](evidence/coverage.csv)

## What changed when only the arm changed

Changing the arm reversed which grasp held on two of eight tasks. Each IIWA–UR5e pair started from the same released row 14, with matching task, initial scene, seed and complete issued command tape. Both arms used their own native kinematics, initial posture and controller response, with the same simulated PandaGripper. An independent Panda-trained reader scored the resulting public wrist images and proprioception. [Preregistered body-swap protocol](ARM-CONTROL.md) · [Every matched setup and per-task result](evidence/arm-swap.json)

| Task | First opposing decision | Physical outcome under identical commands | Opposing five-control windows | Center / corners / proprioception Brier |
| --- | ---: | --- | ---: | ---: |
| Object 8 | 75 | IIWA held the chocolate pudding. UR5e's hand was empty. | 13 | 0.1011 / 0.0661 / 0.0585 |
| Spatial 0 | 90 | UR5e held the black bowl. IIWA's hand was empty. | 1 | 0.4996 / 0.4961 / 0.4490 |

Each row represents one contiguous bout. Averaging the two tasks equally, center Brier was **0.3003**, against **0.2538** for proprioception, **0.2811** for the equal-access corner model and **0.3492** for command timing. The center reader ranked the carried hand higher in the pudding bout and lower on the bowl. The corner model ranked the carried hand higher in both bouts. On the bowl it gave the empty hand a carry probability of 0.0014 and the loaded hand 0.0039. That correct ordering still yielded a Brier loss near 0.5. Six tasks supplied no strict opposing event on this arm swap. Both tasks favored the stronger controls on Brier. Two informative tasks leave the effect on other tasks poorly identified. [All opposing predictions](evidence/arm-swap-opposing.csv.gz)

## Physical labels and model access

Every decision followed five consecutive close commands. A retained object had a free joint, simultaneous contact with both opposing fingers, and no support outside the gripper at every sampled endpoint. An empty hand had no external finger or palm contact at any of those five endpoints. The scorer excluded transitions, unilateral or supported contact and terminal decisions. The criterion concerns any carried object. It measures five 20 Hz control endpoints. The endpoint trace retains contact partners, including the supports that exclude a positive. [Recorded public features, endpoint witnesses and reset lineage](evidence/trajectories.tar.gz)

The source comprised three untreated Panda policy tapes on layout 15, each issued on layouts 15, 18 and 19. The nine trajectories supplied 61 carried, 73 empty and 25 excluded windows. For each target task we omitted all source trajectories of that task, standardized on the remaining source only, and fitted the four registered models with equal trajectory weights. We froze the source-only coefficients before inspecting target scores. Final scoring checks their byte identity. The models use elapsed time and close duration, realized jaw aperture and wrist motion, or add causal flow and image-change features. The two image models share the same two frames, global image features, texture measurements, proprioception, coefficient count and penalty. One emphasizes the middle half of each axis. The other averages over four equally sized corner squares. Those four squares tile the image and each overlaps one quarter of the center, making this a test of spatial emphasis. [Source fits](evidence/models.json) · [Observation and feature definitions](public.py) · [Original registration](PLAN.md)

Im2Flow2Act predicts desired object motion and learns a robot policy to realize it. RAM and Robo-ABC retrieve contact locations from demonstrations. BID ranks alternative action chunks for temporal coherence and forward contrast. Here the commands stay fixed and public sensors estimate whether the arm has actually retained an object. The source supervision and target endpoint labels come from simulation. No target contact, object identity, pose, reward or future frame enters any prediction. [Im2Flow2Act](https://proceedings.mlr.press/v270/xu25a.html) · [RAM](https://proceedings.mlr.press/v270/kuang25a.html) · [Robo-ABC](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05748.pdf) · [BID](https://proceedings.iclr.cc/paper_files/paper/2025/hash/0d78dd998f7b9ac79604d47a2d79bb0d-Abstract-Conference.html)


## Replay fidelity and next choice

The 22 original-layout target reenactments matched archived released scene, row, perturbation and task hashes. Eighteen reproduced the complete archived reward tape and 20 agreed on whether success occurred. Two originally successful IIWA tapes ended without success in CPU replay (goal 6 and spatial 0). The largest observed replay-to-archive tool deviation was 32.2 mm on spatial 0 and the largest finger-coordinate deviation was 7.59 mm on a Kinova3 moka-pot tape. The spatial 0 arm-swap counterexample belongs to the realized CPU trajectory. Thirteen of 22 crossed runs exhausted the donor commands before native termination. All physical scores use the newly observed trajectories and all registered tapes remain in the analysis. No counterfactual command after a tape ended was supplied. [Reset, command, reward and motion lineage](evidence/lineage.json) · [Replay fidelity audit](evidence/independent-check.json)

The source-only model artifact and final fitted artifact have identical SHA-256 digests. A first arm-swap score caught a batch-size-dependent rounding difference of 1.11 × 10⁻¹⁶ in the clock's probabilities for identical inputs. Fixed-order accumulation preserved the fitted weights and made every matched clock tie exact. All 61 physics replays used software rendering. A separate reconstruction recovered each paired loss and the reported task averages. [Exact pins and reproduction commands](evidence/run.json) · [Runtime and logs](evidence/runtime-logs.tar.gz) · [Evidence checksums](evidence/SHA256SUMS)

The next distinct question is whether an arm holds the **requested object or a distractor** when both grips are loaded at comparable apertures. That contrast would require target-identity labels for both physical alternatives and a public object-specific cue. Empty closures already separate by measured jaw width within this panel's matched pairs. Close the current candidate without a native evaluation.
