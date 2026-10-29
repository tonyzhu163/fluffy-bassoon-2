# Zero-interaction motor numerical replay

This bounded diagnostic investigates differing first motor actions from two
completed runs with the same actual first observations, native H10 noise and
query seed. It loads no simulator and spends zero new robot interactions.
It cannot compare controller performance or establish a robot success rate.

The pinned OpenPI [`Policy.infer`](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/policies/policy.py)
splits the motor key on every call. Its [`Pi0.sample_actions`](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/pi0.py#L216-L240)
only consumes the RNG when `noise is None`. With explicit noise, key values
therefore do not influence the mathematical computation. Typed and legacy keys
still have different JIT argument types and may select different executable
signatures. That is a hypothesis to measure, not an established cause.

The local prefix attachment creates separate frozen `module_jit` wrappers and
does not replace the motor sampler. The pinned jaxrl2 source contains no
`jax.config.update`, x64/default-matmul or XLA-flag mutation. OpenPI loads BF16
weights. Numerical compilation/kernel choices and process initialization remain
possible confounds even when the physical input arrays are identical.

The input manifest binds both original native-group indexes and both new DSRL
case indexes, including contracts and episode trace checksums. The loader
requires exact equality of image/wrist/proprio arrays, noise, query seed,
physical state identity and instruction. It preserves both recorded first H10
motor outputs, including their differences.

A new motor process replays both input origins for tasks0/1, with typed and
legacy keys and two repetitions, at four stages:

- Plain pixel-interface motor before prefix/SAC construction.
- Prefix encoder attached without qualification.
- Prefix qualification completed on both recorded cases.
- Fresh official SAC initialized with zero replay and zero optimizer updates.

The motor sampler callable must remain identical throughout, and a full hash
of frozen NNX state before/after must match. Each stage saves every predicted
H10 action, exact-repeat/typed-versus-legacy/stage-versus-plain differences,
comparisons with both recorded outputs, JAX/runtime/precision settings, relevant
environment flags and the actual GPU probe. The original `module_jit` closure
is lowered for both key types when available, recording HLO fingerprints and
sample-key data without replacing its function. HLO text equality/difference
alone does not establish optimized executable identity or a causal mechanism.

Run only inside an existing one-GPU allocation of at most ten minutes:

```bash
bash scripts/diagnostics/vla_motor_replay.sh EXACT_COMMIT FRESH_RUN \
  libero40-subset-target-20261030-r2 libero40-target-online-dsrl-20261030-r2 typed-first
```

An explicit `legacy-first` option reverses initial key compilation order in a
separate fresh run if a further bounded check is warranted. The first run does
not automatically launch that comparison. Completed reference artifacts and
active experiment worktrees remain read only.
