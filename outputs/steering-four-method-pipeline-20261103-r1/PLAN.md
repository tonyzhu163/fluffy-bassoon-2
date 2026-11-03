# Four-method frozen-steering history pipeline

User request: "yes so fix the pipeline" following confirmation of AD, DPT,
DIT and value-based offline ICRL. Implement the missing exporters, learners,
full checkpoint/resume and frozen inference compatibility for both independent
velocity and all-layer K/V sources. No scientific PPO run or converged student
fit is launched by this implementation qualification.

All four consume identical task-pure actual physical histories. DPT alone
changes query labels to the same-interface final teacher. DIT uses executed
behavior actions, complete discounted Monte Carlo returns and independently
learned causal Q/V estimates with conventional TD regularization. In-context
IQL uses Bellman Q fitting, upper-expectile V and frozen-value weighted policy
extraction on the existing decoder; it does not claim S-ICQL's world-model
prompt. Reward/boundary semantics are physical-end with observed first-success
once and zero terminal bootstrap. Future returns are critic labels only.

Retain per-update producer archives/checkpoints. Fit source normalization only
on the training partition. Preserve both value and actor Adam states, target-Q,
best value candidate, sampler/Python/NumPy/Torch/CUDA RNG, stage, plateau and
source/data contracts at operational boundaries. Evaluate each stage on fixed
held-out windows; value fitting completes before actor extraction. Use the
existing actor export and native five-attempt readout without online updates.

Qualification: allocated Linux core numerical tests, both32/1152 producer
serialization through all eight corpus files, scalar-reference returns/TD/
expectile/weights, microbatch equivalence, causal padding/label checks,
next-Adam recovery, all eight fit/export paths and interrupted value-stage
resume. macOS performs only static checks and editing. Record exact commits,
Linux allocation/runtime, failures/repairs and final results here. Do not treat
synthetic qualification as simulator success or source PPO improvement.
