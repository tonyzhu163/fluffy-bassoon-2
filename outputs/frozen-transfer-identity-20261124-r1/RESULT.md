# No observed wrong-object carry on the three native drawer failures

Across 22 original frozen-policy target tapes, **0 of 3 native failures had a recorded strict wrong-object pickup**. All three failures were on the same drawer-and-bowl instruction. Only one contained any saved grasp window. In that Kinova3 replay the fingers held the instructed bowl for 12 consecutive five-control windows and the native task still failed. The IIWA and UR5e failures supplied no eligible windows. **Stop the post-grasp identity-reader experiment on this archive.** There is no observed failed wrong pickup for it to detect, and no matched correct-versus-wrong load for evaluating a reader against aperture and phase. [Registered rules](PLAN.md) · [Counts](evidence/result.json) · [Episode ledger](evidence/episodes.jsonl)

## What was visible on the original native commands

These are CPU replays of each target arm's own untreated, closed-loop π₀.₅ command tape, on its original BASE start. Physical object identities come from the replay. Success and failure in the table come from the original native reward. All three failure replays matched their entire native reward tape and reached the 300-control horizon. The two layout-14 tapes share the released scene and seed; each arm issued its own commands.

| Failed drawer tape | Eligible five-control windows | Strict wrong | What the sampled contacts establish |
|---|---:|---:|---|
| IIWA, layout 14 | 0 | 0 observed | No saved post-closure identity in 300 controls. |
| UR5e, layout 14 | 0 | 0 observed | No saved post-closure identity in 300 controls. |
| Kinova3, layout 17 | 15 | 0 | Only bowl 1 touched the hand in these windows. Twelve windows ending at controls 55–110 had an unsupported bilateral hold on that bowl. The other three did not meet the strict hold rule. |

The recorder saved preterminal five-control blocks with five positive jaw commands and sampled contacts at each control endpoint. The failed tapes supplied **75 sampled endpoints out of 900 executed controls**. Wrong carry was 0/1 among failures with a documented load and 0/19 among originally successful episodes. All 19 successful tapes had at least one correct carry in replay. Across all 22 tapes, 253 of 337 eligible windows were strict correct, 0 were strict wrong, 3 were completely empty and 81 had contact without a strict hold. No saved original endpoint showed even a shorter unsupported bilateral hold on a named non-target. A missed or supported approach, or a pickup during unsaved controls, remains possible. The 12 consecutive Kinova windows describe a single observed bout. [Episode and control-level counts](evidence/episodes.jsonl)

| Task | Original tapes | Native failures | Failure tapes with eligible windows | Tapes with a strict correct pickup | Tapes with a strict wrong pickup |
|---|---:|---:|---:|---:|---:|
| 10/5, book into caddy | 2 | 0 | 0 | 2 | 0 |
| 10/8, both moka pots onto stove | 4 | 0 | 0 | 4 | 0 |
| goal/3, bowl into top drawer | 4 | 3 | 1 | 2 | 0 |
| goal/6, cheese onto bowl | 4 | 0 | 0 | 4 | 0 |
| object/1, cream cheese into basket | 2 | 0 | 0 | 2 | 0 |
| object/8, chocolate pudding into basket | 2 | 0 | 0 | 2 | 0 |
| spatial/0, bowl between plate and ramekin | 2 | 0 | 0 | 2 | 0 |
| spatial/8, bowl next to plate | 2 | 0 | 0 | 2 | 0 |

Each of IIWA (8 tapes), UR5e (8) and Kinova3 (6) contributed one failure. Seven tasks, covering 18 tapes and all three failures, declared at least one non-target loose object. In the eighth task both moka pots were required. All four pot tapes carried each of the two pots, accounting for 100 legitimate carry windows. On the spatial tasks, the initial relation and released placement predicate both select bowl 1. The similarly named bowl 2 is a different instance. A reader would need public scene context to resolve that instruction. A simulator root establishes physical identity for scoring only. [Instructions, initial relations and predicates](tasks.json)

Replay and native success agreed on 20/22 original tapes and exact reward sequences agreed on 18/22. Two originally successful IIWA tapes, goal/6 and spatial/0, ended before CPU success when their commands ran out; both replays contained correct carries. They remain in the native-success denominator. Restricting to the 20 outcome-concordant cases retains all three native failures and the same zero wrong pickups. [Replay lineage](evidence/episodes.jsonl)

## Contact with a distractor can occur without acquisition

Nine original windows included simultaneous hand contact with multiple bodies at an endpoint. Four original windows included contact with the named floor fixture. Eight included the generic body label `table`, which does not uniquely match a declared BDDL instance under the registered alias rule. Those two counts can overlap concurrent-contact counts. No unresolved name was ever in the original carried set. No original eligible window contained hand contact with a *named* loose distractor or loose receiver. Contact elsewhere in the episode was unobserved.

A changed-start example shows the distinction directly. On IIWA's cream-cheese-to-basket tape replayed on layout 20, the hand touched both the cheese and tomato sauce at the same endpoint. Neither was carried at any saved endpoint on that branch. In the source Panda drawer tape, the hand touched the wine bottle in two windows while also contacting the bowl. Only the bowl ever met the carry test. On changed target cheese-to-bowl tapes, contacts with the receiving bowl likewise did not make it the carried object. The label keeps the opposite-finger unsupported hold separate from these concurrent contacts. [Per-case touched and carried roots](evidence/episodes.jsonl)

## Alternatives and matched comparisons

All branches used BASE definitions. The alternate layouts and arm swaps changed how previously recorded actions played out. They are separate from the original native failure sample.

| Replay group | Tapes | Eligible windows | Strict correct / wrong windows | Replay successes | End of tape before termination |
|---|---:|---:|---:|---:|---:|
| Target, original commands and start | 22 | 337 | 253 / 0 | 17 | 2 |
| Target, same arm and tape on layout 20 | 22 | 340 | 111 / 0 | 6 | 13 |
| UR5e on IIWA's original start and tape | 8 | 104 | 66 / 0 | 4 | 3 |
| Source Panda, original | 3 | 53 | 41 / 0 | 3 | 0 |
| Source Panda, alternate layouts 18 and 19 | 6 | 106 | 20 / 0 | 1 | 5 |

Matching *the same tape and absolute control* produced 103 doubly loaded target layout pairs out of 337 common windows and 64 doubly loaded arm-swap pairs out of 104. Each member of every such pair carried an allowed object. The analogous source comparison had 20 doubly loaded pairs out of 106 and zero opposed identities. The 61 Panda source carry windows all had correct identities, even before holding out a task. Thus there was neither a source wrong-load class nor a held-out matched incorrect load; a retrieval, aperture or phase comparison would have no observed identity contrast. Empty closures, alternative instructions and the shorter contact events were never used as substitute negatives. No proxy was trained. [Matched controls by tape](evidence/matched.json) · [Group summaries](evidence/result.json)

## Uncertainty and interpretation

The measured exposure is **zero documented strict wrong pickups on these three failed tapes**. Each event is joined to its own native outcome. There is only one failure-bearing task and two of the three failure tapes are wholly unsighted by this sampler. A whole-task bootstrap cannot resolve an upper tail with zero observed events. Under the additional assumption that failure-bearing tasks were independent exchangeable draws with the same capture protocol, observing 0 affected tasks in 1 gives a one-sided 95% exact upper reference of 95% for *observing at least one such event on a task*. These tasks were selected, and recording depends on the issued jaw commands. The reference does not bound the probability of a wrong approach or pickup in a new native episode. It would also be invalid to count 337 windows as independent evidence about failures. [Uncertainty record](evidence/result.json)

The adjacent Kinova3 drawer layout 15 is informative about an earlier stage. In its separate replay the first plate contact occurred at control 67, with no preceding drawer contact, and the drawer stayed closed. The published trace establishes a wrong approach and plate contact; acquisition was not demonstrated. That episode is outside this study's Kinova layouts 16/17. It contributes no extra pickup or failure here. The wrist-retention and phase studies therefore leave open a public instruction-and-scene reader **before closure**. Assessing it would require an independently collected panel with approach observations before a grasp, on fresh tasks or starts. This retrospective result supports stopping the loaded-identity proxy on the current archive and supplies no confirmatory evidence about general object grounding. [Kinova drawer replay](../frozen-vla-reachability-20261123-r1/RESULT.md) · [Phase study](../frozen-vla-phase-transfer-20261123-r1/RESULT.md) · [Literature and registered decision](PLAN.md)

The readout ran on Baige CPU cores 32 and 33 with zero GPU-hours. [Run and reproduction notes](RUN.md) · [Pinned runtime](evidence/runtime.json)
