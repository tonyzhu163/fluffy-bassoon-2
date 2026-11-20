# A public gate reduced losses from correction but fell below the frozen motor

The best of the three measured controllers on each source arm was the frozen Huber PaliGemma policy with correction off. The gated policy lost three Panda and two Sawyer successes and rescued zero raw failures. Each arm used 50 matched source-development layouts on simulated LIBERO Goal 5 BASE and native terminal reward. [Evaluation](evidence/native03/evaluation.json) · [Per-layout readback](evidence/native03/readback/eval-cases.csv)

| Source arm | Off | Always correct | Learned gate | Gate opened / reached decision |
| --- | ---: | ---: | ---: | ---: |
| Panda | 33/50 | 30/50 | 30/50 | 5/41 |
| Sawyer | 20/50 | 0/50 | 18/50 | 11/50 |

On Panda, always-on correction rescued five raw failures and lost eight raw successes. The gate avoided five of those eight losses but took none of the five rescues. Its tie with always-on therefore hides different successful starts. On Sawyer, the gate avoided 18 of 20 losses. No raw failure was solved by the learned correction on Sawyer. Even perfect selection between these two measured continuations could reach only 20/50 Sawyer; the corresponding hindsight ceiling on Panda was 38/50. These ceilings use outcomes unavailable to the gate.

## How the two comparisons were controlled

Each condition used the same frozen source motor, layouts, five-control replanning cadence and 300-control budget. At control 60, the learned gate used only coarse summaries of the two cameras, public wrist and gripper state, the frozen action proposal, and elapsed time. Nine Panda episodes had already terminated. The gate decision was fixed once, before any corrected outcome. Always-on and gated shared the same fitted bounded action residual. The raw prefix observations and issued actions were shared; each condition executed in its own simulation. Paired readback checked the native state and controls and verified that the gated trajectory reproduced the selected controller. Independent wrist-camera rerenders can differ on the same physical prefix. [Protocol](PLAN.md) · [Paired verification](evidence/native03/readback/readback.json)

The fit used 64 separate training layouts on each arm. The residual learned from 23 successful post-fork teacher continuations on Panda and 24 on Sawyer. The gate learned the signed difference between the privileged **teacher** and frozen-motor outcomes on training pairs. The evaluated intervention was the learned **residual**. Those are different prediction targets. Training never used the earlier 50 privileged Sawyer forks or the source-development outcomes. The present test measured public selection on two familiar arms and one development task. [Fit exposure and counts](evidence/native03/fitted/fit.json) · [Model specification](MODEL-API.md)

The earlier raw checkpoint report scored 34/50 Panda and 20/50 Sawyer. The concurrent off condition scored 33/50 and 20/50. Three individual Panda outcomes differed from the archive; all Sawyer outcomes agreed. All paired gains and losses here use the concurrent off branch. The earlier expert-from-contact experiment switched at actual first contact using privileged geometry. This run switched at a public fixed time and used a learned residual. The comparison across experiments cannot isolate the effect of either change. [Previous baseline](../embodiment-probe-order-20261120-r1/MOTOR-LIVE.md) · [Privileged experiment](../embodiment-sawyer-contact-fork-20261120-r1/RESULT.md)

## Research decision

Keep the frozen motor as the baseline. On Sawyer, the next question is whether a deployable correction can produce any reproducible native rescues. On Panda there is a separate selection question: can a gate trained on the learned correction's own out-of-sample branch outcomes identify beneficial interventions while avoiding losses? Use independent training-layout rollouts for those labels. These results provide no reason to advance this gate to UR5e.
