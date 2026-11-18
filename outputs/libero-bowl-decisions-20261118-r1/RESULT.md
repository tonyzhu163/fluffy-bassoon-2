# Can one failed bowl placement identify a useful correction?

Frozen OpenPI π0.5 can place the same bowl on either the cabinet or the plate in this LIBERO-Pro2 world. Under S20, its public action and motion history exposes the timing change. A smaller translation command failed to recover a single placement under S20.

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
