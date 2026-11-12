# Draft method for distilling teacher policy changes

The proposed method transfers evidence-dependent decisions from a frontier teacher into an already competent robot policy. A small, learned update operator writes experience into fast weights that condition the existing action expert. Training supervises how the policy changes when relevant evidence is added, alongside the correctness of the resulting decision. At deployment the student uses its own observations and executed actions; Astra or Fable supplies no online supervision.

The research hypothesis is that this objective transfers useful changes in behavior more efficiently than ordinary trajectory imitation or a standard TTT action objective, while preserving the base policy's existing competence. This is supervised meta-learning. Its proposed contribution is the paired behavioral-change objective and its connection to a frozen motor; it is not a claim to eliminate gradient-based training. The mechanism and its advantage remain unimplemented and unmeasured.

## What is transferred and where it is stored

The transfer target is the effect of evidence on a teacher's decisions: which action becomes appropriate, which alternatives become inappropriate, and which decisions should remain unchanged. Teacher explanations can identify candidate evidence and provide auxiliary labels, but physical outcomes determine whether the resulting guidance is useful.

| Component | Offline training | Deployment |
| --- | --- | --- |
| Competent VLA parameters theta0 | Frozen, including original action transforms | Frozen |
| Update and readout parameters phi | Learned across support and query sequences | Frozen |
| Shared fast-weight initialization W0 | Starts at zero and may be meta-learned | Reset to this learned initialization at each new trial |
| Per-trial fast weights Wt | Updated causally while unrolling each sequence | Updated from the student's own experience |
| Teacher annotations and decisions | Targets and provenance | Absent |

W0 can absorb shared teacher knowledge; Wt carries facts specific to the current interaction. A control with W fixed at the learned W0 isolates shared post-training gains from gains caused by the learned update. Clearing memory means restoring W0, not zeroing an adapter that was trained with a nonzero prior.

Freezing theta0 preserves its parameters, not necessarily its behavior under a modifier. Retention must therefore be measured on native tasks where the base is already competent. Disabling the entire modifier must recover the qualified base exactly.

## Teacher signal from matched queries

For a fixed query q, hold the instruction, current RGB views, proprioception, action candidates and action budget constant. Construct two teacher inputs:

- H-minus: the available past with a designated evidence segment omitted and explicitly marked unavailable.
- H-plus: the same past including that segment, entirely preceding q.

Use separate teacher contexts so the omitted evidence cannot survive in conversation state. Include repeated judgments and same-length irrelevant-history controls to distinguish evidence sensitivity from sampling variation or extra context length. Do not fabricate a different physical past and present it as observed history.

The first implementation uses a pair of native action chunks proposed by the competent base at q. The teacher selects a preferred candidate or abstains in each context and supplies a short observation-grounded explanation. Randomize candidate order across repetitions and preserve the same ordering within a matched pair. Repeated non-abstaining choices estimate preference frequencies pT-minus and pT-plus; these are empirical choices, not access to model logits or calibrated probabilities.

The supervision of interest is:

    deltaT = pT-plus - pT-minus

Preserve both changed and unchanged judgments, abstentions, and teacher disagreements. Do not define a zero target merely because a noisy pair happened to agree. Native continuation results and repeated judgments determine which pairs support an informative target; report uncertainty and sensitivity rather than inventing an admission cutoff.

The candidate restriction establishes whether the teacher can improve selection among behaviors the base can execute. Measure candidate coverage independently. If neither candidate can succeed, the sample cannot establish transferable teacher advantage through selection. Teacher-generated actions can later expand the proposal set, but must be evaluated separately: their inclusion may itself disclose the answer to the teacher without history.

## Verifying usefulness in the simulator

Replay each candidate from the same saved query state and use the same continuation policy, remaining control budget and paired random draws. Evaluate native success and native return; record execution time separately. A preference change is a useful transfer target only to the extent that the preferred behavior has better physical consequences. Retain unsuccessful and tied branches as evidence without inventing a strict preference.

Snapshot equality must include scene, robot/controller state, elapsed time, task internals and RNG state. The current recorder saves MuJoCo state, XML and metadata but does not yet provide a qualified full-state restore API. Exact replay is a prerequisite for this comparison, not an existing collector capability.

This is an intervention on information supplied to the teacher. It does not by itself identify the physical cause of a failure. Physical explanations such as obstruction, slip or insufficient force require additional evidence before they are labeled as facts.

## Student update mechanism

Let zq be the frozen VLA features of the current query. Each completed transition contributes a public evidence token:

    xt = encode(observation_before, executed_action,
                observation_after, available_native_feedback, boundaries)

Use the same feedback convention during training and deployment. The initial prototype can use observations and actions with native rewards retained only for offline validation, matching the successful Astra collection. The current query target, candidate outcomes, teacher preference and retrospective explanation never enter xt.

A fixed-size matrix W stores fast memory. Start with a normalized delta-rule update:

    kt = Kphi(xt)
    vt = Vphi(xt)
    Wnext = W + eta_phi(xt) * (vt - W kt) kt^T / (epsilon + ||kt||^2)

Use eta_phi = sigmoid(g_phi), giving a step size between zero and one, with positive epsilon for the zero-key case. For a fixed key/value this makes the prediction error along that key contract by a factor between zero and one. This is a gradient step on a local squared prediction loss with learned key/value projections. The delta rule is an established component, not the novelty claim. This single-step property does not bound all accumulated memory states; measure numerical behavior over the actual sequence lengths.

The query reads memory and modifies selected keys and values of the existing action expert:

    mq = W Qphi(zq)
    deltaKV = Pphi(mq)
    action_chunk = frozen_action_expert(q, baseKV + deltaKV)

Pphi is linear without bias, so W = 0 gives an exactly zero modifier. Use the repository's existing KV bridge boundary; keep the action expert in the computation graph for input gradients while its parameters stay frozen. This retains the native action generator rather than replacing it with an unconstrained action head.

The first candidate has one fixed memory matrix. Its per-step memory cost does not grow with history length. Total sequence processing still grows linearly, and neither fixed storage nor bounded updates guarantees that information survives arbitrarily long histories.

## Objective on the effect of an update

Process H-minus and H-plus from the same W0 to obtain W-minus and W-plus. Place a separate query after the support sequence: its action labels and outcomes supervise the update but are never written into memory before querying. These training queries are distinct from the held-out evaluation split.

For a flow action expert, define a candidate compatibility score using the negative flow-matching error at shared noise and interpolation samples:

    score(q, A, W) = -E[masked_flow_error(q, A, W)]

With x_tau = (1 - tau) A + tau epsilon, the repository's target velocity is epsilon - A. Apply the checkpoint's action normalization, binary-coordinate conversion and delivered-prefix masks. An Astra chunk shorter than the native model horizon does not supply labels for its unexecuted suffix.

Convert the two candidate scores to a finite-set preference pS with a shared temperature. This is a ranking surrogate, not the policy's action likelihood. Validate its relationship to sampled robot behavior rather than interpreting a better score as successful control.

    deltaS = pS(W-plus) - pS(W-minus)
    Lchange = squared_error(deltaS, deltaT)
    Lchoice = cross_entropy(pT-plus, pS(W-plus))
    Lkeep = flow_velocity_distance(modified_policy, original_base)
            on a separate bank of already competent native cases

    L = Lchoice + lambda_change Lchange + lambda_keep Lkeep

Lchoice prevents a student from matching a direction of change while choosing the wrong action in both conditions. Lchange directly supervises the response to evidence. Lkeep measures interference with existing motor behavior, using matched noisy actions and interpolation times on cases where the original behavior remains appropriate. Anchor trials use their own legitimate histories and reset to W0; do not carry another task's memory into an anchor implicitly. Choose loss weights and score temperature using source-development data, then freeze them before held-out evaluation.

Differentiate through the support updates and frozen action expert to train phi and W0. Truncated backpropagation can retain W across segment boundaries while detaching its graph. Teacher explanations can add a separately ablated prediction head, but the core objective must work with behavioral judgments and validated outcomes alone.

At deployment there are no candidate scores, teacher labels or simulator branches in the control loop. W updates from actual experience and the conditioned action expert produces its usual action chunk. Consequently, simulator evaluation is essential: compatibility on a small candidate set does not guarantee useful free-running actions.

## Distinction from SFT and adjacent methods

Ordinary SFT supplies individual action targets. This method additionally requires matched information interventions, differentiable support-to-query updates, and a target for the policy's change under those interventions. Removing Lchange and the paired construction yields an important sequence-distillation baseline. It is not legitimate to call that ablation a new method simply because it contains fast weights.

There is no theorem that SFT cannot learn the same behavior. The empirical claim must be better capability transfer, retention, or data/teacher efficiency at matched cost.

| Prior work | Established component | Proposed distinction to test |
| --- | --- | --- |
| [Algorithm Distillation](https://arxiv.org/abs/2210.14215) | Action prediction from learning histories can distill an improvement process | Explicit matched evidence effects rather than only chronological action targets |
| [TTT layers](https://arxiv.org/abs/2407.04620) | Fast weights updated through local learning objectives | Teacher-verified behavioral-change supervision for the update and motor interface |
| [RoboTTT](https://research.nvidia.com/labs/gear/robottt/) | Robot fast-weight memory and failure-to-correction sequence training | Added evidence-intervention objective on a competent frozen motor; must beat its action-only counterpart |
| [TTCD](https://arxiv.org/abs/2608.01672) | Long-context teachers supervise short-context fast weights through hidden-state discrepancies | Black-box behavioral comparisons across different teacher/student architectures, with physical verification |
| [ARC](https://arc-robot-reasoning.github.io/) and [ERVLA](https://arxiv.org/abs/2606.03784) | Action-grounded textual supervision and transfer to robot policies | Behavioral update target and teacher-free fast adaptation, tested against matched context-supervised training |

These distinctions are a research hypothesis, not an established novelty claim. In particular, TTCD makes generic teacher-guided fast-weight distillation an insufficient claim. A full comparison of its method and available implementation is required before making an originality claim.

## Scalability commitments

Use the competent base for ordinary rollout generation. Query a teacher at selected states with informative evidence; each matched pair needs two contexts plus any repetitions, not a frontier-model call for every control. Cache shared query material and run independent episodes in separate simulator processes. The existing single-session MCP server remains serial for simulator ownership and graphics-context correctness.

Native branch verification is a real collection cost. Account for all candidate continuations, unsuccessful teachers, repeated judgments, images/tokens, simulator controls, GPU time and retained datasets. Compare improvement per teacher cost and per physical control as coverage grows. Parallel collection alone does not establish favorable scaling.

Keep teacher outputs in a common schema so Astra and Fable can be compared or combined without requiring identical hidden states. Establish each teacher's usefulness separately before pooling them. Do not silently resolve conflicting judgments into a target. At deployment the fixed-size memory and existing motor determine latency; measure the additional cost over the competent base.

## Smallest experiment that can reject the idea

First establish a panel with a competent motor and a measurable teacher advantage on matched queries. Include cases where prior experience contains relevant information absent from the current views, cases where it is irrelevant, and cases where the base already chooses correctly. The successful seed-7 drawer demonstration alone does not establish such a panel.

Then compare the original base, ordinary action SFT/LoRA, context-supervised SFT, a matched TTT model with Lchoice only, and the same model with Lchange. Add a candidate-ranking control to expose whether the motor modifier supplies an advantage over simple selection. The update-objective comparison must share teacher calls, source trajectories, trainable capacity, initialization, optimizer budget and deployment inputs. Report any unmatched cost in broader SFT or RL comparisons rather than hiding it.

Evaluate on held-out scenes and task conditions with the student's own rollouts. Measure native task success, retention on original competent tasks, recovery behavior, teacher cost and deployment latency. At paired saved queries, also evaluate full history, evidence omitted, irrelevant donor history, and memory fixed at W0. Query-state matching requires qualified replay; complete rollout results must be reported separately because histories change visited states.

Use independent fits and uncertainty paired by task/scene/trial; controls from one episode are not independent samples. Choose sampling and compute in the executable pilot contract based on baseline variability and available budget. Do not infer a sample size or an acceptance cutoff from this one demonstration.

The method loses its intended justification if teacher evidence effects are mostly noise, useful actions fall outside motor coverage, the paired objective fails to improve over matched action-only TTT, effects vanish on student histories, gains come entirely from W0, or retention and deployment cost erase the transfer benefit. A stronger static policy would remain useful but would support a narrower claim.

## Repository implementation boundary

The current collector provides native RGB/state/action trajectories, pre-action annotations and native outcomes. The source run is documented in [the Astra demonstration result](../astra-robocasa-demo-20261112-r1/RESULT.md). Existing [flow bridge training](../../src/fluffy_bassoon/meta_rl/flow_bridge_train.py) provides action-derived supervision through a frozen expert; [flow history](../../src/fluffy_bassoon/meta_rl/flow_history.py) supplies the KV interface and numerical conventions.

New work is required for qualified exact restoration, a base-policy proposal tool, matched read-only teacher judgments, evidence-pair records, the fast-weight updater, and the paired outer objective. Neither Fable control nor this training procedure has been qualified. The code-inspection reference is commit `25197908`; this note introduces no model computation or accelerator job.

Implementation should begin with the paired teacher and native verification experiment. If that produces useful evidence, implement the smallest delta-rule/KV module and the action-only versus paired-objective comparison before expanding the corpus or adding explanatory-text heads.
