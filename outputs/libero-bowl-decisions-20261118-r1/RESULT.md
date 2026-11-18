# Can one failed bowl placement identify a useful correction?

Pacing the first attempt raised native retry success from 11/16 to 14/16 on fresh pose pairs. Under delay, pacing let the robot complete the attempted placement and learn from the resulting binary reward. With the first attempt paced, an outcome-trained timing selector and an always-paced retry succeeded on exactly the same 14 assignments. The selector reached the assigned goal earlier under normal timing. This is a fitting-task mechanism study with frozen OpenPI π0.5 on LIBERO-Pro2 Goal 4/8.

## Reachability and gain

The nominal screen used initial states 10 and 11. With the original motor, normal timing gave 3/4 instructed placements and delayed timing gave 0/4. On state 11, the cabinet and plate predicates were first reached at controls 90 and 78 respectively. A matched camera sequence shows the delayed robot passing the bowl while the normal-timing robot carries it onto the plate. [Camera sequence](plate-support.png) · [Eight branch records](nominal02-evidence.json)

Four fresh starts (12–15), each with both prompts and both execution choices, produced the following native instructed placements. Each entry has eight physical episodes. The same initial physics and public observations matched across every branch within each initial state, including both timings. Every rollout received 300 controls. No incorrect destination predicate fired on this panel. [32 branch records](paired01-evidence.json)

| Execution | Normal timing | S20 |
| --- | ---: | ---: |
| Released five-control replanning | 8/8 | 0/8 |
| Half translation, same replanning | 7/8 | 0/8 |

The command–motion correlation on the eight original support tapes peaked at one control for all four normal attempts and at 10–15 controls for the four delayed attempts. A nearest-median classifier calibrated on the other initial state recovered the timing label on 8/8. The lag estimate is an empirical response time. S20 configures seven controls each of action and observation delay.

## One-support retry

For a support with a known attempted destination, the rule repeats it after reward and switches after failure. It uses the public lag estimate to choose the execution option. Each measured support is scored under both possible goals, with the selected continuation looked up on four independent query starts at the same timing. This gives 64 dependent lookups from eight support trajectories and four query starts. Native placement is the endpoint. A uniform destination control averages the two measured branches. [Fixed protocol](PLAN.md) · [Full readout](paired01-retry.json)

| Query rule on states 12–15 | Normal | S20 | Overall |
| --- | ---: | ---: | ---: |
| Uniform destination, default motion | 16/32 | 0/32 | 16/64 |
| Repeat or switch, default motion | 28/32 | 0/32 | 28/64 |
| Repeat or switch, half gain | 24/32 | 0/32 | 24/64 |
| Repeat or switch, gain chosen from matched support | 28/32 | 0/32 | 28/64 |
| Best measured option with evaluator knowledge | 32/32 | 0/32 | 32/64 |

The improvement over a uniform destination came entirely from using goal feedback under normal timing. Choosing the gain from history added zero successes over the fixed default. Under delay, a support missed both destinations, so its zero reward had the same value under either goal. Switching destinations was correct on half of those balanced assignments.

## Pacing under delay

On four additional starts (16–19), the paced option executes ten predicted controls and waits 14 controls for feedback. The arm receives zero displacement during each wait and holds the last gripper command. Both strategies receive 300 controls. Pacing trades inference calls and movement time for observations that reflect the previous burst. Its delayed first hits ranged from controls 162 to 209. [Matched camera sequence](pacing-plate.png) · [32 pacing branch records](pacing01-evidence.json)

| Execution | Normal timing | S20 |
| --- | ---: | ---: |
| Released five-control replanning | 7/8 | 0/8 |
| Ten-control burst with charged waits | 6/8 | 8/8 |

Across the 64 paired lookups, choosing the execution strategy from the earlier attempt produced 41 native successes. The same destination rule with always-paced query execution produced 38, and with default execution produced 25. Giving the execution selector the opposite-timing support produced 22. The 3/64 incremental gain over always pacing came from normal timing. On S20, all default support attempts missed both destinations, so the goal rule chose the correct destination in only half of the balanced assignments even when pacing could carry it there. This motivates the direct intervention on the first attempt below. [Pacing retry comparison](pacing01-retry.json)

## Fresh two-attempt comparison

The selector learned from native continuation outcomes on 24 physical support tapes (states 10–15) and four query starts (16–19). There were 192 dependent training lookups. With the destination rule fixed, the best one-split policy used default execution for a lag of at most 5.5 controls and paced execution above it. The best history-blind execution policy always paced. The learner saw no timing labels. Both policies were frozen before fresh collection. [Frozen fit](fit01-policy.json) · [Fitting code](learn.py)

Each fresh trial pairs a support state from 20–23 with a different query state from 24–27. Four pose pairs crossed with two timings and two hidden preferences give 16 paired assignments. The robot chose a balanced exploratory destination independently of the preference. It repeated that destination after reward and switched after failure. The live selector recorded its decision before physically executing the query. It then ran the alternative execution mode from the identical query reset with the same noise. The learned policy chose the same mode as the live lag classifier on all 16 assignments under each first-attempt strategy. The four pose pairs are the independent scene blocks. [Fresh branch evidence](livepair01-evidence.json)

| First attempt | Default retry | Always-paced retry | Learned retry | Correct destination |
| --- | ---: | ---: | ---: | ---: |
| Default execution | 7/16 | 11/16 | 11/16 | 12/16 |
| Paced execution | 7/16 | 14/16 | 14/16 | 16/16 |

- **Changing the first attempt** added three native retry successes and lost none with learned or always-paced retry execution. On the four delayed support poses, default execution reached the exploratory destination 0/4 times and pacing reached it 4/4. For a completed placement, reward identified whether that destination was the hidden goal. A missed placement yielded zero reward for both possible goals. Delayed destination choices improved from 4/8 to 8/8, and native delayed query success from 4/8 to 7/8. Normal-timing native success was 7/8 in both arms.
- **Changing retry execution** from always paced to learned produced zero discordant native-success pairs under either support strategy. With paced first attempts, mean query controls to first assigned-goal hit were 161.7 for the learner and 198.9 for always pacing. This descriptive secondary measure counts misses at 300. Every query actually ran all 300 controls. All elapsed-control savings came from normal timing.
- **Paper implication.** Test whether the robot can learn to choose a first attempt that makes sparse feedback informative while preserving enough time to act. A fixed paced first attempt plus reward-based switching and a fixed paced retry already solves 14/16 here. Use it as the direct comparator. Prespecify the control-budget curve or time-to-success endpoint before collecting the transfer panel if speed is part of the claim. Measure the same fixed policies across the full panel, retain a cleared-history control and report paired physical native success with all failures included.

The four successful phases completed 152 simulator episodes (8 + 32 + 32 + 80). Raw trajectories, evaluator predicates, choices and job receipts are under `/mowenjie/fluffy-bassoon-2/outputs/libero-bowl-decisions-20261118-r1/` on Baige. The timed GPU invocations used 0.653 elapsed device-hours including the reset-loader startup failure (214 seconds, zero episodes). A separate reset/control smoke check also used the development container. Device-lock queue time is excluded. Final worker `livepair01` exited 0 on commit `70af0253`. No official test task contributed data.

## Benchmark choice for the paper

A static comparison of the released BASE task definitions found identical problem, domain, regions, fixtures, objects and initial-state declarations across all ten LIBERO-Goal tasks. This allows the benchmark to hold the world fixed while varying which native goal earns reward. The closest transfer pair is Goal 2/9, wine bottle on cabinet versus on rack. Both remain in the repository's fitting split. An experiment that reserves this pair before selector training can claim transfer to the reserved bottle pair. The official held-out Goal 3/6 tasks instead ask for the bowl in the top drawer or cream cheese in the bowl. Qualify each motor option, check initial predicate values and measure whether both predicates ever fire on one trajectory before interpreting this pair. Keep all eight official test tasks for the broader transfer evaluation. No transfer-task outcomes were collected here. [Registered splits](../libero-meta-learning-20261117-r1/SPLITS.json)
