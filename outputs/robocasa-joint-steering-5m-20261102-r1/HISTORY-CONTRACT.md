# Producer history binding

The source contract distinguishes collection rounds from behavior-policy versions. TRAIN collection rounds are1..103. Round r is produced by policy version r-1, so `policy_iteration` in both valid logical rows and raw physical chunks is0..102. The fixed final teacher is checkpoint `iteration=103`; it has not produced an additional TRAIN bank. `case.slot=(r-1)*16+slot_within_task` and the raw chunk `iteration` is r. A history adapter must not discard version0 or treat it as round1's post-update policy.

Checkpoints keep `spec` as the minimal InterfaceSpec constructor arguments; the new `interface` field is the complete `InterfaceSpec.record()` including deterministic map metadata. Raw physical chunks also keep the complete `interface` record. Compare those full records when binding a final DPT teacher; do not compare full physical metadata against the constructor-only `spec`.

The initial review found that the independently owned adapter assumed versions1..final_update and compared teacher `spec` against full physical `interface`. Its owner has since published zero-based producer-clock/full-interface alignment and producer-port validation. Actual source admission remains unbound; synthetic adapter qualification alone does not establish a completed scientific source. No student fit is launched here.

Both arms have2064 pre-action public features. Velocity has32 coefficients. Independent all-layer K/V has1152 coefficients with explicit18-layer mapping. Native Gaussian noise is fixed-law and separately retained; it is not a learned1600-coordinate action. Steering Gaussian log probabilities alone enter PPO. Feature normalization is fitted once from the first TRAIN bank and stored in each policy checkpoint. The original pre-normalization critic values remain in raw episodes; no behavior log-probability is relabeled.

PPO logical rows stop at first success or timeout. Raw physical chunks continue to the declared full horizon; after-success reward is zero and `valid_ppo=false`. Complete source episode checkpoints retain both the logical prefix and references to every physical chunk, full current/next observations, actual mapped commands and native three-view videos. Later AD/DPT adapters must explicitly choose their physical-history/feedback treatment while preserving this provenance.

## Shared source for all four meta-learning methods

The human confirmed AD, DPT, DIT and value-based offline ICRL on 3 November
2026. Each interface supplies its own corpus to all four methods. The velocity
and all-layer K/V policies, teachers, steering spaces and optimizer states stay
independent. The four-method collection audit is
[`META-LEARNING-COLLECTION-AUDIT.md`](../robocasa-rollout-capacity-3248-20261103-r1/META-LEARNING-COLLECTION-AUDIT.md).

Retain every raw physical chunk, episode/request reference, chronological
behavior version, task/case identity, actual steering sample, Gaussian mean/std
and log probability, reward, current/next observation, first-success/logical
boundary and physical boundary. Keep failures as well as successes. Preserve
every pre-SGD bank and every post-update recovery, including actor, critic,
normalization, optimizer and RNG state. Saving/evaluating every five updates
does not permit deleting intermediate behavior histories or checkpoints.

DPT query labels come from the same interface's frozen final teacher at the
recorded feature. DIT and value-based offline ICRL use the executed steering
actions and observed rewards; neither should silently consume DPT teacher
labels. The current source reward is simulator first-success, not Flora.

Value-data adapters must qualify nonterminal next-feature joins against the
next physical chunk's saved observation/feature. Raw next observations support
re-encoding if needed. They must explicitly choose the first-success logical
MDP or the complete physical-episode MDP, preserve both boundary fields, and
apply the corresponding bootstrap mask consistently. Do not bootstrap across
a reset or infer a next feature from the following episode. Discount/return
construction and causal separation of offline return targets from actor inputs
need numerical qualification before claiming either value method is ready.
