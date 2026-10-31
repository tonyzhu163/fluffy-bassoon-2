# Offline context probe

This diagnostic checks whether a fixed AD500 controller responds to support
history in the completed one-source-goal target comparison. It consumes the
hash-bound support and query traces of an existing AD full group. It runs on
one allocated Linux GPU for at most ten minutes and executes zero new simulator
actions. No OpenPI backbone, simulator client, optimizer or learner is loaded.

Every variant replays the same recorded support public features, observations,
executed 32-dimensional noises and episode boundaries. Recomputed support
proposals never replace the noise that was actually executed. The variants are:

- Genuine feedback and persistent context.
- The same weights with context reset at the query episode boundary.
- All support reward inputs replaced by zero.
- Only the final support reward input flipped between the recorded cost values
  minus one and zero. Done remains true.

Each variant then consumes the same recorded first query feature, with its
receipt's paired query RNG seed. No query reward or later query feature is fed
back. The genuine replay must reproduce every support proposal and the first
query proposal and clipped noise within `1e-5`. The probe reports mean, proposal
and clipped-noise differences with the same Gaussian draw, context norms, and
hash-bound context/RNG snapshots before and after the query. Policy weights
must remain identical and no gradients may be populated.

Changes in these predictions demonstrate sensitivity to the specified inputs.
They do not establish counterfactual closed-loop robot success, a causal reward
benefit, reliable adaptation, or generalization beyond the observed case.

Launch after the exact source is deployed to its fresh worktree:

```bash
bash scripts/diagnostics/vla_source40_context_probe.sh EXACT_COMMIT FRESH_RUN \
  libero40-subset-scales-20261030-r2 libero40-subset-target-20261030-r2 group-008
```

The input manifests, policy and trace files are read only. Output is a fresh
`experiment/index.json` plus full context/RNG snapshot artifact. The source
training cost belongs to the original experiment; this probe adds GPU time and
zero robot interactions.
