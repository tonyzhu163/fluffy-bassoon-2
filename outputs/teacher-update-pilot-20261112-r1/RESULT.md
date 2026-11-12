# The first teacher-update pilot produced no usable update targets

Astra abstained in all 72 fresh-context judgments. Adding 50 controls of actual preceding experience did not produce an action preference, including on the qualified query where the two candidates had different native success outcomes. The paired teacher-change target is **undefined**, not zero. No student, SFT or RL fit was launched. This tests the proposed teacher-label interface; it does not establish whether a better teacher interface can transfer capability into a TTT policy.

## What ran

The frozen released Human300 pi0.5 checkpoint ran native RoboCasa OpenDrawer on four fresh pretrain scenes, with a 500-control pilot budget. It succeeded on 3/4 scenes. At controls 100 and 250, two independent policy draws supplied candidate chunks: predict 50 controls, execute the first five, then continue with common policy noise. The budget is shorter than the 750-control registry horizon, so this is not a benchmark score.

For each of eight queries, Astra received current observation only, the same observation plus the preceding 50-control history, or a nonmatching scene's history. Each condition had three fresh-context repetitions, with balanced randomized candidate ordering held fixed within each triple. The teacher selected A, B or abstain and gave a short public evidence summary. It saw no branch outcomes or privileged simulator state.

| Teacher input | Calls | Choices | Abstentions |
| --- | ---: | ---: | ---: |
| Current observation; history unavailable | 24 | 0 | 24 |
| Current observation plus actual recent history | 24 | 0 | 24 |
| Current observation plus another scene's history | 24 | 0 | 24 |

The input audit verified 72 public-only prompts, 24 matched context triples and 360 image-file hashes against the native records. No teacher tool calls occurred. Astra ran through the signed-in **Codex CLI harness**, selecting `gpt-6-astra` with medium reasoning effort. This was not a direct API experiment. Model identity is the explicit client selection, without independent backend attestation. Reported usage was 1,261,583 input tokens, including 287,232 cached tokens, and 7,907 output tokens.

## Native outcomes and incomplete comparisons

Thirteen branches qualified, including seven complete candidate-zero reproductions of the original baseline. The full sixteen-branch panel did not complete: three qualified branches remain missing. Six queries have both candidates verified in the same motor process; one additional candidate-zero branch is unpaired.

| Scene / query control | Candidate 0: success, return | Candidate 1: success, return | Already successful before query |
| --- | --- | --- | --- |
| 120700 / 100 | Yes, 277 | Yes, 279 | No |
| 120700 / 250 | Yes, 250 | Yes, 250 | Yes |
| 120701 / 100 | Yes, 253 | No, 0 | No |
| 120701 / 250 | Yes, 250 | Yes, 250 | Yes |
| 120702 / 100 | No, 0 | No, 0 | No |
| 120702 / 250 | No, 0 | No, 0 | No |

Returns sum native rewards **after the query**. There is one success contrast and two return contrasts among the six qualified pairs. Neither candidate recovers the failed baseline scene, and two qualified queries are already successful. Consequently, this bank also has no success headroom over its original candidate-zero continuations. The paired subset contains three scenes; repeated teacher calls are not additional independent scenes.

The initial qualification exposed a renderer cleanup defect: collecting old rendering contexts could corrupt camera data while physics stayed fixed. A process-local context-owned cleanup fix plus collection at closed-episode boundaries removed that corruption in the admitted run. Seven complete continuations then matched physics, RGB, public state, controllers, RNG, executed commands and native outcomes exactly. See the [forced-collection diagnostic](qualification/render-gc/original.json) and [cleanup intervention](qualification/render-gc/fixed.json).

The last scene subsequently rejected an initial reset. A fresh diagnostic matched that saved initial state exactly, but restarting the frozen motor later produced different inferred commands despite matched observations and saved noise; the maximum reported coordinate difference was 0.00192529. Its cause is unresolved. The one completed cross-process alternative is excluded, and the remaining comparisons are not claimed. Exact continuation across motor-process restarts remains unqualified. [Initial-state check](qualification/native05-initial-check.json) · [Action replay failure](qualification/native05-action-replay-failure.txt).

## Decision and retained artifacts

Close this pilot at the failed teacher-signal prerequisite. Ranking two raw five-control proposals under this prompt did not yield usable supervision. A useful next test would ask for **history-dependent subgoal or recovery decisions** on actual failure/recovery states, ground those decisions in executable policy continuations, and verify a teacher advantage before implementing the fast-weight learner. That is a follow-up hypothesis, not a demonstrated remedy or novelty claim.

The six exported diagnostic pairs preserve public history, candidate commands, executed-prefix masks and separate native outcome labels. Every teacher-update target is null and explicitly unavailable; they are not a training set for the proposed change objective. The raw archives remain outside Git. [Summary](summary.json) · [Input and runtime audit](audit.json) · [Artifact locations and hashes](ARTIFACTS.json) · [Prespecified plan and qualification amendments](PLAN.md).

Compute used 0.9003 worker GPU-hours; counting the whole reservation interval conservatively gives 1.2453 GPU-hours, within the 4-hour pilot budget. The owned simulator and policy processes have exited. Collector source: `47c24d3d`; attempted restart: `566a066a`; final audit: `1cd6afde`; teacher runner: `9bf5b77f`. RoboCasa source: `3443cdf3`; OpenPI source: `ca4c6d71`. No shared vendor library files were modified.
