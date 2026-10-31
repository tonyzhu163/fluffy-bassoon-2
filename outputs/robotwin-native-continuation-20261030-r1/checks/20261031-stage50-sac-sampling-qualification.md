# Stage50 SAC sampling clarification

Both serial stage50 conditions sample the stochastic actor: `1.5*tanh(mu + exp(clamp(log_std,-20,2))*epsilon)`, with `deterministic=False`. Each task and frozen/online condition restores the same source learner and private RNG streams. This differs from the deterministic mean action used in the separate first frozen-baseline panel.

The source stage50 checkpoint is SHA256 `5f6c30831bfcfff6442a8d20392fb158008c219936518f3d90c3a585330f1caa`; its initial actor identity is `ebf6654c0b89560d03c548aa8ab306eca60601e80d70c4a89242bf545c635b77`. Saved episode1 first decisions match exactly between frozen and online on both targets: initial actor/features, interaction RNG, sampled steering noise, issued motor targets and next observation.

Full episode1 equality is not required. Online SAC uses fresh target replay, samples with replacement from its first transition, and performs20 gradient updates after every executed chunk. Before episode1’s second decision the online copy is already at314,520 updates versus314,500 frozen. The saved interaction RNG states still match, while actor, steering-noise and motor-action hashes differ. Optimizer randomness uses a separate stream, so this is the expected effect of within-episode learning, not evidence of an interaction-RNG mismatch.

Source counts stay502,356 actions /314,500 updates /943,500 optimizer steps. Online target clones separately add1,340 /4,020 update/optimizer steps on bottle and1,920 /5,760 on stapler; these do not advance the source quota. Frozen target copies add no SGD.

Native before/after are0/1 per task on separate seed21161030. Target episode1 uses22261030, so these native references are unpaired with the target K8 case bank.

[Detailed qualification and pinned-code references](20261031-stage50-sac-sampling-qualification.json). [Saved first-decision receipts](20261031-stage50-sac-first-decisions.json). No models, rollouts or tests were run for this clarification.
