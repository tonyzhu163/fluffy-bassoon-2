# Expert takeover rescued 18 of 30 Sawyer failures after first plate contact

The frozen Huber PaliGemma motor solved 20/50 simulated LIBERO Goal 5 BASE starts. Switching to the fixed Sawyer expert at first hand-to-plate contact solved 31/50, a net gain of 11 starts (22 percentage points). The teacher solved 37/50 when started from reset. This is a single-task source-development result. [Native per-layout results](evidence/native01/fork/cases.json)

| Original PaliGemma outcome | Expert succeeded from contact | Expert failed from contact |
| --- | ---: | ---: |
| Failed (30) | 18 rescued | 12 still failed |
| Succeeded (20) | 13 retained | 7 lost |

The raw replay and takeover shared the complete action prefix, native physics and controller state at the decision boundary on all 50 starts. The fork occurred after 36–82 controls, leaving 218–264 controls of the original 300. The expert received privileged current simulator geometry and used one route selected from source training. The reward was the native terminal task reward. All raw replays reproduced their archived trajectories. [Control and state checks](evidence/native01/compact/receipt-audit.json) · [Teacher selection](TEACHER.md)

## What the reset control explains

Among the 30 original failures, the expert solved 23 layouts when allowed to start from reset. It rescued 17 of those same 23 when introduced at contact. On the other seven layouts it rescued one from contact. Of the 12 starts still failing after takeover, six were solvable by this expert from reset. The reset branch changes both the approach and the time available, so those six do not isolate poor contact geometry as the cause. [Three outcomes per layout](evidence/native01/compact/paired-cases.csv)

## Decision

Eighteen successful continuations establish that changing only actions after contact can rescue many source failures. Seven lost successes establish a cost to switching unconditionally. Test post-contact corrective imitation on fresh **training** layouts, keeping this development panel and all its continuations out of fitting. Collect both successful expert rescues and raw-policy continuations that already succeed. Compare the same frozen motor with identity, always-on correction, and a learned gate using only deployment-visible inputs. Evaluate native gains and lost successes on both arms. The expert's simulator state and the observed counterfactual outcomes are available for training labels only. This experiment establishes the opportunity for a gate; a trained gate's ability to select the useful branch remains open. [Registered experiment](PLAN.md) · [Baige execution](EXECUTION.md)
