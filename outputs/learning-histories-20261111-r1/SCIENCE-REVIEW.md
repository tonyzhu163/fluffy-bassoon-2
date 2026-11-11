# Interpretation and the possible LIBERO-Gen bridge

The toy isolates whether an offline-trained Transformer can use a history of
choices and success/failure to improve query decisions with frozen weights.
Fresh scenes, a persistent hidden rule and perfect execution make history useful
without conflating a wrong decision with failed motor execution. It is a useful
diagnostic, not evidence of robotic adaptation or learned active exploration.

## Interpretation constraints found during the run

- The supervised objective predicts the correct query action. Greedy evaluation
  attempts do not demonstrate an information-seeking exploration policy.
- `history_logw` starts from an equal prior over all 126 rules. `sample` draws
  rules from either the 99 training rules or the 27 held-out rules. The reader
  is exact under the full-family prior, but is not the split-specific Bayes
  optimum. Its greedy action policy is not a globally optimal experiment design.
- Training attempts depend on prior rewards. Later actor choices can therefore
  indirectly reveal feedback in the blind condition. Evaluation instead uses
  the learner's own attempts, so this is also a train/evaluation history-policy
  shift. A strong causal claim needs outcomes removed/shuffled on the same
  frozen learner and the same histories; the current grid does not contain this
  intervention.
- The held-out rules are new combinations within a fixed family, not new rule
  families. Full corrections already reveal the correct object; linked failures
  add no task information conditional on that full answer in this toy.

The immutable grid remains the originally specified first-pass experiment.
Read its reference gaps descriptively. Additional evaluation can use saved
checkpoints; it must receive its own output identity and explicit contract.

## Proposed next platform, not launched

Assuming LIBERO-Gen means the Behavior Prompting Policy benchmark (Spatial
Combination and Goal Chain), it offers task generation, motion-planned
demonstrations and pretrained checkpoints. Its standard protocol tests
demonstration-conditioned generalization, not learning from failed attempts.

Sources inspected November 11, 2026:
- https://arxiv.org/html/2606.30457v1
- https://github.com/real-stanford/behavior_prompting/blob/main/docs/liberogen.md

Recommended progression: finish this toy and test causal feedback use, then a
small Spatial Combination pilot with a qualified motor for every candidate
action. A hidden rule must persist across fresh scenes, and the initial demo
must leave meaningful ambiguity. Use native task success and compare demo-only,
correct feedback, shuffled feedback and an informed controller. Start Goal
Chain only after that simpler physical experiment shows an effect. A standard
LIBERO-Gen score increase alone does not establish the project's ICRL claim.
