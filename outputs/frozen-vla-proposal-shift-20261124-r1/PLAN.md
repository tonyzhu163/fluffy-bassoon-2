# Which frozen-policy adapter rescues a Kinova3 failure that a source arm solves?

Registered November 24 before any lane ran new policy or simulator episodes. Program for [PR #10](https://github.com/tonyzhu163/fluffy-bassoon-2/pull/10). The frozen π₀.₅ weights and motor parameters never change. Each lane has its own run directory and stop condition. Native reward decides every lane.

## What the evidence says about the failure

Frozen π₀.₅ times out on Kinova3 drawer-and-bowl layouts 14, 15, 17, 36, 37 and 38. Panda, UR5e and IIWA solve the same task, and IIWA solves 36 and 37.

| Explanation | Evidence | Status |
| --- | --- | --- |
| The arm executes the proposed motion badly | Target-calibrated correction rescued 0/3. Planar and 3D correction gave 4/6 and 3/6 against 4/6 raw. The tuned reader, plan reuse and post-grasp timing rescued nothing. [Selector](../selector-native-launch-20261124-r1/RESULT.md) · [Adaptation](../frozen-adaptation-20261124-r1/RESULT.md) | Not supported |
| The requested poses are unreachable | All 600 goals on two failed drawer runs have a joint-valid witness. Acting on feasibility lost successes. [Pose](../frozen-transfer-pose-20261124-r1/RESULT.md) | Not supported |
| The wrong object is grasped | 0 of 3 failures show a wrong-object carry. [Identity](../frozen-transfer-identity-20261124-r1/RESULT.md) | Not supported |
| One changed decision would recover | A one-time cue rescued 0 in two studies. Hindsight over continue, correct and cue equals continuing. | Not supported |
| The body cannot do the task from these scenes | IIWA commands replayed open loop on Kinova3 earned reward on layouts 37 and 14. [Donor](../frozen-transfer-donor-20261124-r1/RESULT.md) · [Validation](../frozen-transfer-donor-validation-20261124-r1/RESULT.md) | Contradicted on 37 and 14, open on the rest |
| The policy proposes different behaviour when it observes Kinova3 | Arm appearance and proprioception both move frozen proposals. Panda rendering changed all ten π₀.₅ proposals. Untreated outcomes on six matched starts ranged from 2/6 to 4/6 across jobs. [Factory](../frozen-vla-factory-20261123-r1/RESULT.md) · [Source view](../frozen-vla-source-view-20261123-r1/NATIVE.md) | Open, and the target of this program |

Working hypothesis: these are proposal failures. Observing Kinova3 puts the frozen policy on a behaviour it does not recover from, although a successful command sequence exists. Useful adapters therefore change what the policy is conditioned on or which of its samples is used for the whole episode, and leave execution alone. The hypothesis is wrong if no input, prompt or sample change moves proposals toward a known successful sequence, or if every such change leaves the six failures unsolved.

## Lanes

Every π₀.₅ lane implements `ADAPTERS` against the shared harness in this directory and reuses one untreated branch and one untreated repeat per start. A lane counts only rescues beyond the repeat's discordance.

| Lane | Directory | Architecture | Counts against it |
| --- | --- | --- | --- |
| Attribution | `frozen-shift-attribution-20261124-r1` | No adapter. Replay a successful donor tape on Kinova3 and query the frozen policy at each visited state under input swaps. Measure distance to the donor's next five controls | No swap beats a fresh-noise control |
| Body view | `frozen-shift-view-20261124-r1` | Render the source arm at the target grip site in both cameras for the whole episode | No rescue on six failures, or losses match rescues |
| State | `frozen-shift-state-20261124-r1` | Map target proprioception to the source convention from 15 target pairs. Controls drop and shuffle state | Mapped equals dropped |
| Sampling | `frozen-shift-sampling-20261124-r1` | Eight policy-noise draws per failed start, then a source-trained public critic that picks one of K proposals per replan | 0/48 untreated draws succeed, or critic equals a random draw |
| Phase prompt | `frozen-shift-phase-20261124-r1` | The camera drawer reader sets the prompt for the whole episode: open the drawer, then place the bowl | Equals the clock-switched prompt |
| Source guidance | `frozen-shift-guidance-20261124-r1` | Among K proposals, execute the one closest to a source-arm success aligned by public progress. No target expert | Equals a random draw, or needs the same-layout donor |
| Embodiment prefix | `frozen-shift-prefix-20261124-r1` | Fit a small input prefix through the frozen policy on donor replays that succeeded on Kinova3. Evaluate on other layouts | No gain over raw on held-out layouts, or source-arm success falls |
| Second motor | `frozen-shift-smolvla-20261124-r1` | Frozen SmolVLA on the public LIBERO-Pro2.0 release on CPU. Reproduce the arm gap, then test state, prompt and sampling arms with native reward | No Kinova3 gap, or Panda success is not reproduced |

[Source demonstration prefix with motor handoff](../frozen-transfer-demonstration-20261124-r1/PLAN.md) and [offline co-training](../frozen-vla-cotraining-20261124-r1/PLAN.md) are separate registered studies and are not repeated here.

## Panel and comparison

Official LIBERO-Pro2.0 BASE `libero_goal/3` on Kinova3, original instruction, seed 42710300 + layout, five-control replans, one independent 15-control calibration on layout 11. Development starts are the six known failures and the known successes 16, 31, 34, 35 and 39. They have been inspected, so results on them are exploratory. An adapter that rescues a development failure is locked and run once on layouts 40–47, shared with the registered handoff study, against raw and repeat. Panda layouts 0–3 measure retention for every locked adapter.

Report per start successes, rescues, losses, control counts and paired layout bootstrap intervals. Record the prompt, the state and image digests supplied to the policy, noise digests and commands actually consumed. Online adapter inputs are public observations and issued commands. Reward, contacts, object state and simulator state are audit data.

## Compute

One RTX 4090 job through Baige Distributed Training loads the policy once and runs all qualified adapters. The development panel is 11 starts × about 14 branches, near 1.5 GPU-hours at the selector run's rate of 0.0037 GPU-hours per episode. The sampling lane adds 48 episodes. The prefix lane needs a separate fitting job. The second-motor lane uses CPU only. This VM has no Baige session, so lanes deliver CPU-qualified code and launch scripts.
