# Reusing a successful carry after lift recovers delayed bottle placements

On three separate evaluation poses, frozen π0.5 missed every delayed rack placement under both original cadences. Replacing the motion after lift recovered two of the three. Both recovered placements still satisfied the native rack predicate at control 300. Under normal timing the same correction reached the rack on all three poses, although only two remained on the rack at the final control.

## What changed

The instruction specified the rack. The simulator executed an archived frozen-VLA opening until public wrist height rose 7.089 cm during a continuous closing command. At that point, the correction selected the nearest wrist position at lift among three successful normal-timing rack demonstrations and issued the remainder of that demonstration's commands. These are task-specific motion references. Selection uses the current attempt's proprioception. Commands after selection run open loop. At each fork we verified the initial physics fingerprint, every preceding public proprioceptive state and both native predicates against the original recording. Every episode ran for 300 controls. [Design and split](PLAN.md) · [Simulator experiment](experiment.py)

We fitted on delayed poses 0–2. With the faster opening, command replay reached the rack on 2/3 and feedback tracking on 1/3. With a paced opening the counts were 0/3 and 2/3. The prespecified tie-break selected the faster opening followed by command replay. Under normal timing, replaying each source's own suffix reproduced its native first hit exactly at controls 136, 154 and 146. The feedback tracker also succeeded on the three source controls. [Source checks](control01.jsonl) · [Pose-level results](result.json)

| Evaluation poses and timing | Original with matched opening | Other original cadence | Correction after lift | Corrected placements still successful at 300 |
| --- | ---: | ---: | ---: | ---: |
| 3–5, normal | 2/3 | 1/3 | 3/3 | 2/3 |
| 3–5, delayed | 0/3 | 0/3 | 2/3 | 2/3 |
| 6–11, normal, paced opening | 3/6 | unavailable | 5/6 | 5/6 |
| 6–11, delayed, paced opening | 1/6 | unavailable | 2/6 | 1/6 |

On the six additional delayed poses the correction gained two native successes and lost one. One attempt never triggered the lift detector. The original paced continuation also finished with the rack predicate true on 1/6. The gain on this subset depends on counting a placement reached at any time. The final gripper command was open on all six primary trials and 10/12 additional trials. An open command alone does not verify a stable release.

## Does an earlier attempt help select this correction?

We used the public command lag from the first 24 controls of a separate cabinet attempt at the same timing. The bowl-fitted threshold selected the correction after a slow response and the original motion otherwise. On poses 3–5 that rule achieved 4/6, compared with 5/6 for always applying the correction and 2/6 for always retaining the original motion. On the additional paced starts the counts were 5/12, 7/12 and 4/12 respectively. Even an evaluator choosing the better branch separately for every case could earn only 5/6 and 8/12 on these menus. Earlier motion can identify the timing, but this correction also helped under normal timing. [History comparison](history.json) · [Routing rule](history.py)

**Paper decision.** Use this as evidence that the available motor actions limited the previous rack test. Carry the always-applied correction into all comparisons. To test adaptation across attempts, vary a persistent property across trials for which different placement motions succeed. On fitting poses measure the gains and losses for each fixed correction first. A learned reader can improve physical outcomes only where the best fixed correction leaves recoverable cases. Keep the held-out evaluation focused on those predefined conditions and report all assigned poses.

The study used Baige CPU, 30 native correction trials and six source controls, with zero new GPU-hours. Collection pin `177180c7`, supplementary history comparison `2bf0d39a`. All evaluations used previously inspected development poses, disjoint from the three references. The original trajectories and native scores came from the earlier bottle experiment. This study provides within-task motor evidence. Transfer to a new task and an advantage from across-attempt ICL remain open.
