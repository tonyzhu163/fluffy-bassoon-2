# First-attempt selection from bowl to bottle

The bowl-trained support policy achieved 14/24 native bottle retry successes. Choosing the first target evenly and always pacing achieved 14.5/24 in expectation. The policy recognized when to change cadence. It also carried over a preference for the second destination, which changed from the plate in the bowl task to a wine rack that the delayed robot struggled to reach. A fixed cabinet-first probe identified the correct destination on all 24 assignments and reached the 16/24 success ceiling of the measured paced retries. [Evidence](transfer02-evidence.json) · [Budget curve](budget-curve.png)

## What transferred

The first destination was chosen before moving. After ten policy actions and fourteen charged waits, the learned selector used the public command–motion lag to choose between continued pacing and normal five-control replanning. Both native goals, both timings and both support continuations contributed bowl training outcomes. All retries used the same paced motor and the same rule of repeating after reward and switching after failure. Four bowl support starts crossed with four different query starts, two timings and two preferences produced 64 dependent training assignments. The selected rule and the best fixed rule each earned 56/64. The prespecified budget-curve tie-breaker selected the threshold of eight controls. The frozen fit was saved before the first bottle episode. [Plan](PLAN.md) · [Fit and scoring](design.py)

The bottle evaluation used six fresh support/query pose pairs, each crossed with BASE/S20 and the two hidden destinations. The first 24 issued commands and public states matched exactly across each pair of support continuations. The worker recorded its online choice at control 24 and the resulting retry choices before executing the fresh query branches. The table scores both native goals at an equal budget of 300 controls per attempt. The learned row is verified against the live choices. The remaining rows select among measured, reset-matched branches.

| First-attempt policy | Normal timing | Delayed | Overall |
| --- | ---: | ---: | ---: |
| Bowl-trained rack-first, cadence from public lag | 8/12 | 6/12 | 14/24 |
| Bowl-trained fixed rack-first, always pace | 7/12 | 6/12 | 13/24 |
| Balanced first target, always pace | 8/12 | 6.5/12 | 14.5/24 |
| Balanced first target, cadence from public lag | 8.5/12 | 6.5/12 | 15/24 |
| Cabinet-first, always pace (retrospective comparator) | 9/12 | 7/12 | 16/24 |
| Cleared feedback, uniform retry destination | 4.5/12 | 3.5/12 | 8/24 |

Fractional counts are expectations over both physically executed destination options. Against the bowl-fitted fixed strategy the learner gained one assignment, lost none and tied on the other 23. That gain came from one of six pose pairs. The learned rule switched to normal motion for all six normal-timing rack probes (estimated lag 1–3) and kept pacing for all six delayed ones (14–15). Across equal per-attempt budgets from 24 to 300, its mean native success was 19.1%, compared with 18.2% for the bowl-fitted constant and 18.4% for balanced always-pacing. The curves summarize the same six pose pairs. [Full curve](budget-curve.png)

## Why the first target matters

- On normal timing, switching cadence increased successful rack probes from 1/6 to 5/6. Under delay, both rack continuations scored 0/6. Cabinet-first pacing reached its target on 6/6 supports in each timing condition.
- A completed first placement makes binary feedback informative under either preference. The learned rack-first policy selected the correct retry destination on 17/24 assignments. A paced cabinet-first probe selected it on 24/24.
- Even given the correct destination, the paced retry reached the cabinet on 12/12 query branches and the rack on 4/12 (3/6 normal, 1/6 delayed). No wrong destination or dual predicate fired. This limits native success to 16/24 for the measured query poses, noise draws, budget and motor. Giving an evaluator the pose and timing to select the first attempt produced the same 16/24 as always probing the cabinet. The cabinet strategy was identified retrospectively on bottle.

## Decision for the paper

Keep the bowl and bottle pairs as mechanism studies. The evidence supports using motion history to select cadence and shows the cost of choosing an unreliable exploratory target. At the full budget, a learned first-attempt advantage over balanced pacing did not reproduce on bottle. For a claim about learning what to try, train a score for each candidate destination from the current scene, instruction and motor behavior across multiple source goal pairs. Test on reserved pairs where the easiest informative probe changes across scenes and the retry motor can physically reach either assigned goal. Include the best source-fitted fixed target and the balanced always-paced policy. Report support informativeness and native retry success separately. The bottle pair is reserved for this selector within the project's fitting split. The registered eight official test tasks were not opened in this experiment.

## Execution

Baige development RTX 4090, `transfer02`, commit `2fe3149f`, exit 0. The completed panel contains 16 new bowl continuations and 72 bottle episodes. A prior 16-episode bowl attempt regenerated slightly different prefix actions despite equal resets and images, and stopped before fitting or bottle evaluation. The accepted run replayed the archived opening and independently verified both command and state equality. The two timed invocations used 0.336 device-hours in total. Raw rollouts and the policy written before transfer are under `/mowenjie/fluffy-bassoon-2/outputs/libero-first-attempt-20261119-r1/transfer02/`. [Verified numerical readout](transfer02-evidence.json)
